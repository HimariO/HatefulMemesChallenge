"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

UNITER finetuning for VQA
"""
import json
import os
import sys
import shutil
import argparse
from os.path import abspath, dirname, exists, join
from time import time
from functools import partial
from easydict import EasyDict as edict

import torch
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.optim import Adam, Adamax

import ray
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining

import pandas as pd
from apex import amp
from horovod import torch as hvd
from tqdm import tqdm
from loguru import logger

from data import (TokenBucketSampler, PrefetchLoader,
                  TxtTokLmdb, ImageLmdbGroup, ConcatDatasetWithLens,
                  MemeDataset, MemeEvalDataset,
                  meme_collate, meme_eval_collate, meme_eval_itm_ot_collate)
from model_villa.vqa import UniterForVisualQuestionAnswering, UniterForITM
from optim import AdamW, get_lr_sched

# from utils.logger import LOGGER, TB_LOGGER, RunningMeter, add_log_to_file
from utils.distributed import (all_reduce_and_rescale_tensors, all_gather_list,
                               broadcast_tensors)
from utils.save import ModelSaver, save_training_meta
from utils.misc import NoOp, parse_with_config, set_dropout, set_random_seed
from utils.const import BUCKET_SIZE, IMG_DIM


def build_dataloader(dataset, collate_fn, is_train, opts):
    batch_size = (opts.train_batch_size if is_train
                  else opts.val_batch_size)
    if is_train:
        train_sampler = WeightedRandomSampler(
            dataset.weights_by_class,
            len(dataset),
            replacement=True)
        dataloader = DataLoader(dataset, sampler=train_sampler,
                                num_workers=opts.n_workers,
                                batch_size=32,
                                pin_memory=opts.pin_mem, collate_fn=collate_fn)
    else:
        sampler = TokenBucketSampler(dataset.lens, bucket_size=BUCKET_SIZE,
                                    batch_size=batch_size, droplast=is_train)
        dataloader = DataLoader(dataset, batch_sampler=sampler,
                            num_workers=opts.n_workers,
                            pin_memory=opts.pin_mem, collate_fn=collate_fn)
    dataloader = PrefetchLoader(dataloader)
    return dataloader


def build_optimizer(model, opts):
    """ vqa linear may get larger learning rate """
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    param_optimizer = [(n, p) for n, p in model.named_parameters()
                       if 'vqa_output' not in n]
    param_top = [(n, p) for n, p in model.named_parameters()
                 if 'vqa_output' in n]
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_top
                    if not any(nd in n for nd in no_decay)],
         'lr': opts.learning_rate,
         'weight_decay': opts.weight_decay},
        {'params': [p for n, p in param_top
                    if any(nd in n for nd in no_decay)],
         'lr': opts.learning_rate,
         'weight_decay': 0.0},
        {'params': [p for n, p in param_optimizer
                    if not any(nd in n for nd in no_decay)],
         'weight_decay': opts.weight_decay},
        {'params': [p for n, p in param_optimizer
                    if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]

    # currently Adam only
    if opts.optim == 'adam':
        OptimCls = Adam
    elif opts.optim == 'adamax':
        OptimCls = Adamax
    elif opts.optim == 'adamw':
        OptimCls = AdamW
    else:
        raise ValueError('invalid optimizer')
    optimizer = OptimCls(optimizer_grouped_parameters,
                         lr=opts.learning_rate, betas=opts.betas)
    return optimizer


def main(opts, checkpoint_dir=None, tuning=False):
    from utils.logger import LOGGER, TB_LOGGER, RunningMeter, add_log_to_file
    with logger.catch(reraise=True):
        logger.info(f"{opts}")
        if isinstance(opts, dict):
            opts = edict(opts)

        hvd.init()
        n_gpu = hvd.size()
        device = torch.device("cuda", hvd.local_rank())
        torch.cuda.set_device(hvd.local_rank())
        rank = hvd.rank()
        opts.rank = rank
        LOGGER.info("device: {} n_gpu: {}, rank: {}, "
                    "16-bits training: {}".format(
                        device, n_gpu, hvd.rank(), opts.fp16))

        if opts.gradient_accumulation_steps < 1:
            raise ValueError("Invalid gradient_accumulation_steps parameter: {}, "
                            "should be >= 1".format(
                                opts.gradient_accumulation_steps))

        set_random_seed(opts.seed)
        
        """
        # load DBs and image dirs
        """
        all_img_dbs = ImageLmdbGroup(opts.conf_th, opts.max_bb, opts.min_bb,
                                    opts.num_bb, opts.compressed_db)
        
        # train
        LOGGER.info(f"Loading Train Dataset "
                    f"{opts.train_txt_dbs}, {opts.train_img_dbs}")
        train_datasets = []
        for txt_path, img_path in zip(opts.train_txt_dbs, opts.train_img_dbs):
            img_db = all_img_dbs[img_path]
            txt_db = TxtTokLmdb(txt_path, opts.max_txt_len)
            train_datasets.append(MemeDataset(1, txt_db, img_db))
        train_dataset = ConcatDatasetWithLens(train_datasets)
        train_dataloader = build_dataloader(train_dataset, meme_collate, True, opts)
        
        # val
        LOGGER.info(f"Loading Train Dataset {opts.val_txt_db}, {opts.val_img_db}")
        val_img_db = all_img_dbs[opts.val_img_db]
        val_txt_db = TxtTokLmdb(opts.val_txt_db, -1)
        val_dataset = MemeEvalDataset(1, val_txt_db, val_img_db)
        val_dataloader = build_dataloader(val_dataset, meme_eval_itm_ot_collate,
                                        False, opts)
        
        # test_img_db = val_img_db
        # test_txt_db = TxtTokLmdb(opts.test_txt_db, -1)
        # test_dataset = MemeEvalDataset(1, test_txt_db, test_img_db)
        # test_dataloader = build_dataloader(test_dataset, meme_eval_collate,
        #                                 False, opts)
        """
        # Prepare model
        """
        if opts.checkpoint:
            checkpoint = torch.load(opts.checkpoint)
        else:
            checkpoint = {}

        all_dbs = opts.train_txt_dbs + [opts.val_txt_db]

        model = UniterForITM.from_pretrained(
            opts.model_config, checkpoint,
            img_dim=IMG_DIM, num_answer=1)
        model.to(device)
        # make sure every process has same model parameters in the beginning
        broadcast_tensors([p.data for p in model.parameters()], 0)
        set_dropout(model, opts.dropout)

        """
        # Prepare optimizer
        """
        optimizer = build_optimizer(model, opts)
        model, optimizer = amp.initialize(model, optimizer,
                                        enabled=opts.fp16, opt_level='O2')
        global_step = 0
        if rank == 0:
            save_training_meta(opts)
            TB_LOGGER.create(join(opts.output_dir, 'log'))
            pbar = tqdm(total=opts.num_train_steps)
            model_saver = ModelSaver(join(opts.output_dir, 'ckpt'))
            # json.dump(ans2label,
            #           open(join(opts.output_dir, 'ckpt', 'ans2label.json'), 'w'))
            os.makedirs(join(opts.output_dir, 'results'), exist_ok=tuning)  # store VQA predictions
            add_log_to_file(join(opts.output_dir, 'log', 'log.txt'))
        else:
            LOGGER.disabled = True
            pbar = NoOp()
            model_saver = NoOp()
        
        LOGGER.info(f"***** Running training with {n_gpu} GPUs *****")
        LOGGER.info("  Num examples = %d", len(train_dataset) * hvd.size())
        LOGGER.info("  Batch size = %d", opts.train_batch_size)
        LOGGER.info("  Accumulate steps = %d", opts.gradient_accumulation_steps)
        LOGGER.info("  Num steps = %d", opts.num_train_steps)

        running_loss = RunningMeter('loss')
        model.train()
        n_examples = 0
        n_epoch = 0
        
        if checkpoint_dir is not None and tuning:
            checkpoint = os.path.join(checkpoint_dir, "checkpoint")
            (model_state, optimizer_state, n_epoch, n_examples) = torch.load(checkpoint)
            model.load_state_dict(model_state)
            optimizer.load_state_dict(optimizer_state)

            LOGGER.info(f"***** Resume from ray tune checkpoint : {checkpoint_dir} *****")
            LOGGER.info("  n_examples = %d", n_examples)
            LOGGER.info("  n_epoch = %d", n_epoch)

            # shutil.rmtree(checkpoint_dir)
        
        start = time()
        # quick hack for amp delay_unscale bug
        optimizer.zero_grad()
        optimizer.step()
        while True:
            for step, batch in enumerate(train_dataloader):
                if global_step > 2000:
                    logger.error('Force stop at global step 2000')
                    sys.exit(0)
                n_examples += batch['input_ids'].size(0)
                
                if opts.adv_training:
                    # NOTE: reverse label like what we do in UniterForITM
                    targets = batch['targets']
                    targets = (targets > 0.5).long()
                    targets = torch.abs(targets - 1)
                    batch['targets'] = targets

                    # initialize delta
                    txt_embeds_init = model.uniter.embeddings.word_embeddings(
                        batch['input_ids'])
                    img_embeds_init = batch['img_feat']

                    # for simplicity, we initialize the delta as zero vectors, which performs
                    # very simliar as initializing randomly using norm or uniform distributions
                    txt_delta = torch.zeros_like(txt_embeds_init)
                    img_delta = torch.zeros_like(img_embeds_init)

                    # calculate the prob. scores for clean samples
                    gt_answer_scores = model(batch, compute_loss=False)
                    gt_answer_prob = F.softmax(gt_answer_scores, dim=1)
                    gt_answer_logprob = F.log_softmax(gt_answer_scores, dim=1)

                    # the main loop
                    for astep in range(opts.adv_steps):
                        # (0) forward
                        if opts.adv_modality == ["text"]:
                            txt_delta.requires_grad_()
                            img_delta = torch.zeros_like(img_embeds_init)
                        elif opts.adv_modality == ["image"]:
                            img_delta.requires_grad_()
                            txt_delta = torch.zeros_like(txt_embeds_init)
                        else:
                            txt_delta.requires_grad_()
                            img_delta.requires_grad_()
                        
                        if "alter" not in opts.adv_modality:
                            answer_scores = model(
                                batch, adv_training=True,
                                adv_modality=opts.adv_modality,
                                adv_delta_txt=txt_delta,
                                adv_delta_img=img_delta, compute_loss=False)

                            # CE loss
                            ce_loss = F.cross_entropy(
                                answer_scores, batch['targets'].squeeze(-1),
                                reduction='mean')

                            # KL loss
                            answer_prob = F.softmax(answer_scores, dim=1)
                            answer_logprob = F.log_softmax(answer_scores, dim=1)
                            kl_loss = F.kl_div(
                                answer_logprob, gt_answer_prob, reduction='none') + \
                                F.kl_div(
                                    gt_answer_logprob, answer_prob,
                                    reduction='none')
                            kl_loss = kl_loss.mean()

                            # (1) backward
                            loss = (ce_loss + opts.adv_kl_weight * kl_loss
                                    ) / opts.adv_steps
                        else:
                            answer_scores_1 = model(
                                batch, adv_training=True,
                                adv_modality=["text"],
                                adv_delta_txt=txt_delta,
                                adv_delta_img=None, compute_loss=False)

                            # CE loss
                            ce_loss_1 = F.cross_entropy(
                                answer_scores, batch['targets'].squeeze(-1),
                                reduction='mean')

                            answer_scores_2 = model(
                                batch, adv_training=True,
                                adv_modality=["image"],
                                adv_delta_txt=None,
                                adv_delta_img=img_delta, compute_loss=False)

                            # CE loss
                            ce_loss_2 = F.cross_entropy(
                                answer_scores, batch['targets'].squeeze(-1),
                                reduction='mean')

                            # KL loss
                            answer_prob_1 = F.softmax(answer_scores_1, dim=1)
                            answer_logprob_1 = F.log_softmax(answer_scores_1, dim=1)
                            answer_prob_2 = F.softmax(answer_scores_2, dim=1)
                            answer_logprob_2 = F.log_softmax(answer_scores_2, dim=1)
                            kl_loss_1 = F.kl_div(
                                answer_logprob_1, gt_answer_prob, reduction='none') + \
                                F.kl_div(
                                    gt_answer_logprob, answer_prob_1,
                                    reduction='none')
                            kl_loss_1 = kl_loss_1.mean()
                            kl_loss_2 = F.kl_div(
                                answer_logprob_2, gt_answer_prob, reduction='none') + \
                                F.kl_div(
                                    gt_answer_logprob, answer_prob_2,
                                    reduction='none')
                            kl_loss_2 = kl_loss_2.mean()

                            # (1) backward
                            loss = (ce_loss_1 + ce_loss_2 + opts.adv_kl_weight * (kl_loss_1+kl_loss_2)
                                    ) / (opts.adv_steps*2)

                        delay_unscale = (
                            (step+1) % opts.gradient_accumulation_steps != 0
                            ) or ((astep+1) % opts.adv_steps != 0)
                        with amp.scale_loss(
                                loss, optimizer, delay_unscale=delay_unscale
                                ) as scaled_loss:
                            scaled_loss.backward(retain_graph=True)
                            if not delay_unscale:
                                # gather gradients from every processes
                                # do this before unscaling
                                # to make sure every process uses
                                # the same gradient scale
                                grads = [p.grad.data for p in model.parameters()
                                        if p.requires_grad and p.grad is not None]
                                all_reduce_and_rescale_tensors(grads, float(1))

                        running_loss(loss.item())

                        if astep == opts.adv_steps - 1:
                            # further updates on delta
                            break

                        # (2) get gradient on delta
                        # fix fp16 problem
                        amp_scale = scaled_loss.item() // loss.item()
                        if "text" in opts.adv_modality:
                            txt_delta_grad = txt_delta.grad.clone().detach()
                            txt_delta_grad = txt_delta_grad.float() / amp_scale
                        if "image" in opts.adv_modality:
                            img_delta_grad = img_delta.grad.clone().detach()
                            img_delta_grad = img_delta_grad.float() / amp_scale

                        # (3) update and clip for txt delta
                        if "text" in opts.adv_modality:
                            if opts.norm_type == "l2":
                                denorm = torch.norm(
                                    txt_delta_grad.view(
                                        txt_delta_grad.size(0), -1),
                                    dim=1).view(-1, 1, 1)
                                denorm = torch.clamp(denorm, min=1e-8)
                                txt_delta_step = (
                                    opts.adv_lr_txt * txt_delta_grad
                                    / denorm).to(txt_delta)
                                txt_delta = (txt_delta + txt_delta_step).detach()
                                if opts.adv_max_norm > 0:
                                    delta_norm = torch.norm(
                                        txt_delta.view(txt_delta.size(0), -1),
                                        p=2, dim=1).detach()
                                    exceed_mask = (
                                        delta_norm > opts.adv_max_norm).to(
                                            txt_embeds_init)
                                    reweights = (opts.adv_max_norm / delta_norm
                                                * exceed_mask + (1-exceed_mask)
                                                ).view(-1, 1, 1)
                                    txt_delta = (
                                        txt_delta * reweights).detach()
                            elif opts.norm_type == "linf":
                                denorm = torch.norm(
                                    txt_delta_grad.view(
                                        txt_delta_grad.size(0), -1),
                                    dim=1, p=float("inf")).view(-1, 1, 1)
                                denorm = torch.clamp(denorm, min=1e-8)
                                txt_delta_step = (opts.adv_lr_txt * txt_delta_grad
                                                / denorm).to(txt_delta)
                                txt_delta = (txt_delta + txt_delta_step).detach()
                                if opts.adv_max_norm > 0:
                                    txt_delta = torch.clamp(
                                        txt_delta, -opts.adv_max_norm,
                                        opts.adv_max_norm).detach()

                        # (4) update and clip for image delta
                        if "image" in opts.adv_modality:
                            if opts.norm_type == "l2":
                                denorm = torch.norm(
                                    img_delta_grad.view(
                                        img_delta_grad.size(0), -1),
                                    dim=1).view(-1, 1, 1)
                                denorm = torch.clamp(denorm, min=1e-8)
                                img_delta_step = (opts.adv_lr_img * img_delta_grad
                                                / denorm).to(img_delta)
                                img_delta = (img_delta + img_delta_step).detach()
                                if opts.adv_max_norm > 0:
                                    delta_norm = torch.norm(
                                        img_delta.view(img_delta.size(0), -1),
                                        p=2, dim=1).detach()
                                    exceed_mask = (delta_norm > opts.adv_max_norm
                                                ).to(img_embeds_init)
                                    reweights = (opts.adv_max_norm / delta_norm
                                                * exceed_mask + (1-exceed_mask)
                                                ).view(-1, 1, 1)
                                    img_delta = (img_delta * reweights).detach()
                            elif opts.norm_type == "linf":
                                denorm = torch.norm(
                                    img_delta_grad.view(
                                        img_delta_grad.size(0), -1),
                                    dim=1, p=float("inf")).view(-1, 1, 1)
                                denorm = torch.clamp(denorm, min=1e-8)
                                img_delta_step = (opts.adv_lr_img * img_delta_grad
                                                / denorm).to(img_delta)
                                img_delta = (img_delta + img_delta_step).detach()
                                if opts.adv_max_norm > 0:
                                    img_delta = torch.clamp(
                                        img_delta, -opts.adv_max_norm,
                                        opts.adv_max_norm).detach()
                else:
                    loss = model(batch, compute_loss=True)
                    loss = loss.mean()
                    delay_unscale = (step+1) % opts.gradient_accumulation_steps != 0
                    with amp.scale_loss(loss, optimizer, delay_unscale=delay_unscale
                                        ) as scaled_loss:
                        scaled_loss.backward()
                        if not delay_unscale:
                            # gather gradients from every processes
                            # do this before unscaling to make sure every process uses
                            # the same gradient scale
                            grads = [p.grad.data for p in model.parameters()
                                    if p.requires_grad and p.grad is not None]
                            all_reduce_and_rescale_tensors(grads, float(1))

                    running_loss(loss.item())

                """
                loss compute end
                log & step start
                """

                if (step + 1) % opts.gradient_accumulation_steps == 0:
                    global_step += 1

                    # learning rate scheduling
                    lr_this_step = get_lr_sched(global_step, opts)
                    for i, param_group in enumerate(optimizer.param_groups):
                        if i == 0 or i == 1:
                            param_group['lr'] = lr_this_step * opts.lr_mul
                        elif i == 2 or i == 3:
                            param_group['lr'] = lr_this_step
                        else:
                            raise ValueError()
                    TB_LOGGER.add_scalar('lr', lr_this_step, global_step)

                    # log loss
                    # NOTE: not gathered across GPUs for efficiency
                    TB_LOGGER.add_scalar('loss', running_loss.val, global_step)
                    TB_LOGGER.step()

                    # update model params
                    if opts.grad_norm != -1:
                        grad_norm = clip_grad_norm_(amp.master_params(optimizer),
                                                    opts.grad_norm)
                        TB_LOGGER.add_scalar('grad_norm', grad_norm, global_step)
                    optimizer.step()
                    optimizer.zero_grad()
                    pbar.update(1)

                    if global_step % 100 == 0:
                        # monitor training throughput
                        LOGGER.info(f'============Step {global_step}=============')
                        tot_ex = sum(all_gather_list(n_examples))
                        ex_per_sec = int(tot_ex / (time()-start))
                        LOGGER.info(f'{tot_ex} examples trained at '
                                    f'{ex_per_sec} ex/s')
                        TB_LOGGER.add_scalar('perf/ex_per_s',
                                            ex_per_sec, global_step)
                        LOGGER.info(f'===========================================')

                    if global_step % opts.valid_steps == 0:
                        val_log, results = validate(model, val_dataloader, None)
                        
                        with open(f'{opts.output_dir}/results/'
                                f'results_{global_step}_'
                                f'rank{rank}.json', 'w') as f:
                            json.dump(results, f)
                        pd.DataFrame.from_dict(results).to_csv(
                            f'{opts.output_dir}/results/'
                            f'results_{global_step}_'
                            f'rank{rank}.csv', index=False)
                        
                        # _, test_results = test(model, test_dataloader, global_step)
                        # pd.DataFrame.from_dict(test_results).to_csv(
                        #     f'{opts.output_dir}/results/'
                        #     f'test_{global_step}.csv',
                        #     index=False)
                        
                        TB_LOGGER.log_scaler_dict(val_log)
                        model_saver.save(model, global_step)

                        if tuning:
                            with tune.checkpoint_dir(step=n_epoch) as checkpoint_dir:
                                logger.info(f'***** Save tune ckpt: {checkpoint_dir} *****')
                                path = os.path.join(checkpoint_dir, "checkpoint")
                                torch.save(
                                    (model.state_dict(), optimizer.state_dict(), n_epoch, n_examples), path)
                            tune.report(
                                loss=(val_log['valid/loss']),
                                accuracy=val_log['valid/acc'],
                                auroc=val_log['valid/auroc'],
                            )
                if global_step >= opts.num_train_steps:
                    break
            if global_step >= opts.num_train_steps:
                break
            n_epoch += 1
            LOGGER.info(f"finished {n_epoch} epochs")
            """
            END of training loop
            """
        
        if opts.num_train_steps % opts.valid_steps != 0:
            val_log, results = validate(model, val_dataloader, None)
            with open(f'{opts.output_dir}/results/'
                    f'results_{global_step}_'
                    f'rank{rank}.json', 'w') as f:
                json.dump(results, f)
            pd.DataFrame.from_dict(results).to_csv(
                f'{opts.output_dir}/results/'
                f'results_{global_step}_'
                f'rank{rank}.csv', index=False)
            TB_LOGGER.log_scaler_dict(val_log)
            model_saver.save(model, global_step)


@torch.no_grad()
def validate(model, val_loader, label2ans):
    from utils.logger import LOGGER, TB_LOGGER, RunningMeter, add_log_to_file
    from sklearn.metrics import roc_auc_score

    LOGGER.info("start running validation...")
    model.eval()
    val_loss = 0
    tot_score = 0
    n_ex = 0
    st = time()
    results = []
    
    for i, batch in enumerate(val_loader):
        scores = model(batch, compute_loss=False)
        targets = batch['targets']
        targets = (targets > 0.5).long()
        targets = torch.abs(targets - 1)
        targets = torch.squeeze(targets, dim=-1)
        
        loss = F.cross_entropy(scores, targets, reduction='sum')
        val_loss += loss.item()
        tot_score += compute_score_with_logits(scores, targets).item()
        # answers = [label2ans[i]
        #            for i in scores.max(dim=-1, keepdim=False)[1].cpu().tolist()]
        answers = torch.nn.functional.softmax(scores)[:, 1]
        for qid, answer, target in zip(batch['qids'], answers, targets):
            results.append({
                'id': qid,
                'proba': answer.cpu().item(),
                'label': target.cpu().item(),
                'delta': abs(answer.cpu().item() - target.cpu().item()),
            })
        n_ex += len(batch['qids'])
    
    val_loss = sum(all_gather_list(val_loss))
    tot_score = sum(all_gather_list(tot_score))
    n_ex = sum(all_gather_list(n_ex))
    tot_time = time()-st
    val_loss /= n_ex
    val_acc = tot_score / n_ex
    val_log = {'valid/loss': val_loss,
               'valid/acc': val_acc,
               'valid/ex_per_s': n_ex/tot_time}

    y_true = [r['label'] for r in results]
    y_scores = [r['proba'] for r in results]
    auroc = roc_auc_score(y_true, y_scores)
    val_log['valid/auroc'] = auroc
    
    model.train()
    LOGGER.info(
        f"validation finished in {int(tot_time)} seconds, "
        f"score: {val_acc*100:.2f}, "
        f"auroc: {auroc:.4f}, "
        f"loss: {val_loss}, "
    )
    return val_log, results


@torch.no_grad()
def test(model, val_loader, step):
    from utils.logger import LOGGER, TB_LOGGER, RunningMeter, add_log_to_file
    LOGGER.info("start running test...")
    model.eval()
    
    val_loss = 0
    tot_score = 0
    n_ex = 0
    st = time()
    results = []
    
    for i, batch in enumerate(val_loader):
        scores = model(batch, compute_loss=False)
        answers = torch.nn.functional.softmax(scores)[:, 1]
        for qid, answer in zip(batch['qids'], answers):
            results.append({
                'id': qid,
                'proba': answer.cpu().item(),
                'label': int(answer.cpu().item() > 0.5)
            })
        n_ex += len(batch['qids'])

    n_ex = sum(all_gather_list(n_ex))
    tot_time = time()-st
    val_log = {'test/ex_per_s': n_ex/tot_time}
    
    model.train()
    LOGGER.info(
        f"test finished in {int(tot_time)} seconds, "
    )
    return val_log, results


def compute_score_with_logits(logits, labels):
    pred = torch.nn.functional.softmax(logits)[:, 1] > 0.5
    labels = labels > 0.5
    match = (pred == labels).sum()
    return match



def run_main_train(args):
    # if exists(args.output_dir) and os.listdir(args.output_dir):
    #     raise ValueError("Output directory ({}) already exists and is not "
    #                      "empty.".format(args.output_dir))

    # options safe guard
    if args.conf_th == -1:
        assert args.max_bb + args.max_txt_len + 2 <= 512
    else:
        assert args.num_bb + args.max_txt_len + 2 <= 512

    main(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument('--compressed_db', action='store_true',
                        help='use compressed LMDB')
    parser.add_argument("--model_config",
                        default=None, type=str,
                        help="json file for model architecture")
    parser.add_argument("--checkpoint",
                        default=None, type=str,
                        help="pretrained model")

    parser.add_argument(
        "--output_dir", default=None, type=str,
        help="The output directory where the model checkpoints will be "
             "written.")

    # Prepro parameters
    parser.add_argument('--max_txt_len', type=int, default=60,
                        help='max number of tokens in text (BERT BPE)')
    parser.add_argument('--conf_th', type=float, default=0.2,
                        help='threshold for dynamic bounding boxes '
                             '(-1 for fixed)')
    parser.add_argument('--max_bb', type=int, default=100,
                        help='max number of bounding boxes')
    parser.add_argument('--min_bb', type=int, default=10,
                        help='min number of bounding boxes')
    parser.add_argument('--num_bb', type=int, default=36,
                        help='static number of bounding boxes')

    # training parameters
    parser.add_argument("--train_batch_size", default=4096, type=int,
                        help="Total batch size for training. "
                             "(batch by tokens)")
    parser.add_argument("--val_batch_size", default=4096, type=int,
                        help="Total batch size for validation. "
                             "(batch by tokens)")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=16,
                        help="Number of updates steps to accumualte before "
                             "performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=3e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--lr_mul", default=10.0, type=float,
                        help="multiplier for top layer lr")
    parser.add_argument("--valid_steps", default=1000, type=int,
                        help="Run validation every X steps")
    parser.add_argument("--num_train_steps", default=100000, type=int,
                        help="Total number of training updates to perform.")
    parser.add_argument("--optim", default='adam',
                        choices=['adam', 'adamax', 'adamw'],
                        help="optimizer")
    parser.add_argument("--betas", default=[0.9, 0.98], nargs='+',
                        help="beta for adam optimizer")
    parser.add_argument("--dropout", default=0.1, type=float,
                        help="tune dropout regularization")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="weight decay (L2) regularization")
    parser.add_argument("--grad_norm", default=2.0, type=float,
                        help="gradient clipping (-1 for no clipping)")
    parser.add_argument("--warmup_steps", default=4000, type=int,
                        help="Number of training steps to perform linear "
                             "learning rate warmup for. (invsqrt decay)")

    # device parameters
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit float precision instead "
                             "of 32-bit")
    parser.add_argument('--n_workers', type=int, default=4,
                        help="number of data workers")
    parser.add_argument('--pin_mem', action='store_true', help="pin memory")

    # adversarial training related
    parser.add_argument('--adv_training', action='store_true',
                        help="Whether to use adversarial training or not")
    parser.add_argument("--adv_modality", default=['text'],
                        help="add pertubation on text or image modality")
    parser.add_argument('--adv_lr_txt', type=float, default=0)
    parser.add_argument('--adv_lr_img', type=float, default=0)
    parser.add_argument('--adv_steps', type=int, default=1, help="should be at least 1")
    parser.add_argument('--norm_type', type=str, default="l2", choices=["l2", "linf"])
    parser.add_argument('--adv_max_norm', type=float, default=0, help="set to 0 to be unlimited")
    parser.add_argument('--adv_kl_weight', type=float, default=0, help="set to 0 to be unlimited")

    # can use config files
    parser.add_argument('--config', help='JSON config files')

    parser.add_argument('--tune_num_trial', type=int, default=64,
                        help="")

    args = parse_with_config(parser)
    
    run_main_train(args)

"""
rm -rf /storage/ && python3 train_meme_itm.py --config config/train-meme-large-1gpu.json
/home/ron_zhu/Disk3/uniter_data/finetune/meme_val_6940/large/results/results_500_rank0.csv
"""
