"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

UNITER finetuning for VQA
"""
import argparse
import json
import os
import shutil
from os.path import abspath, dirname, exists, join
from time import time
from functools import partial
from easydict import EasyDict as edict

import torch
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
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
from model.vqa import UniterForVisualQuestionAnswering,UniterForITM
from model.pretrain import UniterForPretraining
from optim import AdamW, get_lr_sched
from train_meme_itm import build_optimizer

# from utils.logger import LOGGER, TB_LOGGER, RunningMeter, add_log_to_file
from utils.distributed import (all_reduce_and_rescale_tensors, all_gather_list,
                               broadcast_tensors)
from utils.save import ModelSaver, save_training_meta
from utils.misc import NoOp, parse_with_config, set_dropout, set_random_seed
from utils.const import BUCKET_SIZE, IMG_DIM, IMG_LABEL_DIM


HERE = os.path.abspath(os.path.dirname(__file__))

def reorder_csv_rows(test_jsonl, src):
    with logger.catch(reraise=True):
        # src = pd.read_csv(sub_csv)
        val_mtx = src.T.to_dict()
        id2row = {}
        for row in val_mtx.values():
            id2row[int(row['id'])] = row
        
        test_row_idx = []
        with open(test_jsonl, mode='r') as f:
            for row in f:
                test_row_idx.append(json.loads(row)['id'])
        
        reorder = [
            {
                'id': int(id2row[i]['id']),
                'proba': id2row[i]['proba'],
                'label': int(id2row[i]['proba'] > 0.5),
            } 
            for i in test_row_idx
        ]
        dst = pd.DataFrame.from_dict(reorder)[['id','proba', 'label']]
        return dst


def build_dataloader(dataset, collate_fn, is_train, opts):
    batch_size = (opts.train_batch_size if is_train
                  else opts.val_batch_size)
    sampler = TokenBucketSampler(dataset.lens, bucket_size=BUCKET_SIZE,
                                 batch_size=batch_size, droplast=is_train)
    dataloader = DataLoader(dataset, batch_sampler=sampler,
                            num_workers=opts.n_workers,
                            pin_memory=opts.pin_mem, collate_fn=collate_fn)
    dataloader = PrefetchLoader(dataloader)
    return dataloader


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
        
        # val
        LOGGER.info(f"Loading Val Dataset {opts.val_txt_db}, {opts.val_img_db}")
        val_img_db = all_img_dbs[opts.val_img_db]
        val_txt_db = TxtTokLmdb(opts.val_txt_db, -1)
        val_dataset = MemeEvalDataset(1, val_txt_db, val_img_db)
        val_dataloader = build_dataloader(val_dataset, meme_eval_collate,
                                        False, opts)
        val_itm_dataloader = build_dataloader(val_dataset, meme_eval_itm_ot_collate,
                                             False, opts)
        
        test_img_db = val_img_db
        test_txt_db = TxtTokLmdb(opts.test_txt_db, -1)
        test_dataset = MemeEvalDataset(1, test_txt_db, test_img_db)
        test_dataloader = build_dataloader(test_dataset, meme_eval_collate,
                                        False, opts)
        """
        # Prepare model
        """
        if opts.checkpoint:
            logger.info(f"Load checkpoint: {opts.checkpoint}")
            checkpoint = torch.load(opts.checkpoint)
        else:
            checkpoint = {}
        
        all_dbs = opts.train_txt_dbs + [opts.val_txt_db]

        model = UniterForITM.from_pretrained(
            opts.model_config, checkpoint,
            img_dim=IMG_DIM, num_answer=1)
        model.to(device)

        if hasattr(opts, 'tune_checkpoint') and isinstance(model, UniterForITM):
            model_state = torch.load(opts.tune_checkpoint)[0]
            model.load_state_dict(model_state)
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
        
        LOGGER.info(f"***** Running training with {n_gpu} GPUs *****")
        LOGGER.info("  Num examples = %d", len(val_dataset) * hvd.size())
        LOGGER.info("  Batch size = %d", opts.train_batch_size)
        LOGGER.info("  Accumulate steps = %d", opts.gradient_accumulation_steps)
        LOGGER.info("  Num steps = %d", opts.num_train_steps)

        model.eval()
        
        val_log, results = validate(model, val_dataloader, None)
        
        with open(f'{opts.output_dir}/results/'
                f'results_{global_step}_'
                f'rank{rank}.json', 'w') as f:
            json.dump(results, f)
        pd.DataFrame.from_dict(results).to_csv(
            f'{opts.output_dir}/results/'
            f'results_{global_step}_'
            f'rank{rank}.csv', index=False)

        test_log, results = test(model, test_dataloader, None)
        
        os.makedirs(f'{opts.output_dir}/results/', exist_ok=True)
        with open(f'{opts.output_dir}/results/'
                f'results_{global_step}_'
                f'test.json', 'w') as f:
            json.dump(results, f)
        
        test_csv = pd.DataFrame.from_dict(results)[['id', 'proba', 'label']]
        test_csv = reorder_csv_rows(
            os.path.join(HERE, 'asset', 'test_unseen.jsonl'),
            test_csv,
        )
        test_csv.to_csv(
            f'{opts.output_dir}/'
            f'test.csv', index=False)
        output_path = (
            f'{opts.output_dir}/'
            f'test.csv'
        )
        print('Save test predict to: ', output_path)
        if opts.checkpoint:
            try:
                shutil.copy(
                    opts.checkpoint,
                    os.path.join(opts.output_dir, 'final.pt')
                )
            except shutil.SameFileError:
                logger.info('Rerun of the same chekcpoint, not re-copy it as final.pt')


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
def test(model, val_loader, label2ans):
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
        answers = torch.nn.functional.softmax(scores)
        for qid, answer in zip(batch['qids'], answers):
            results.append({
                'id': qid,
                'proba': answer[0].cpu().item(),
                'label': int(answer[0].cpu().item() > 0.5)
            })
        n_ex += len(batch['qids'])

    n_ex = sum(all_gather_list(n_ex))
    tot_time = time()-st
    val_log = {'valid/ex_per_s': n_ex/tot_time}
    model.train()
    LOGGER.info(
        f"validation finished in {int(tot_time)} seconds, "
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

    # can use config files
    parser.add_argument('--config', help='JSON config files')

    parser.add_argument('--tune_num_trial', type=int, default=64,
                        help="")

    args = parse_with_config(parser)
    
    run_main_train(args)

"""
rm -rf /storage/ && python3 train_meme.py --config config/train-meme-large-1gpu.json
/home/ron_zhu/Disk3/uniter_data/finetune/meme_val_6940/large/results/results_500_rank0.csv
"""
