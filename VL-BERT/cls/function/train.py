import os
import pprint
import shutil
import inspect
import random

from tensorboardX import SummaryWriter
import imgaug
import numpy as np
import torch
import torch.nn
import torch.optim as optim
import torch.distributed as distributed
from torch.nn.parallel import DistributedDataParallel as DDP
from loguru import logger
from torch.optim.swa_utils import SWALR

from common.utils.create_logger import create_logger
from common.utils.misc import summary_parameters, bn_fp16_half_eval
from common.utils.load import smart_resume, smart_partial_load_model_state_dict
from common.trainer import train
from common.metrics.composite_eval_metric import CompositeEvalMetric
from common.metrics import cls_metrics
from common.callbacks.batch_end_callbacks.speedometer import Speedometer
from common.callbacks.epoch_end_callbacks.validation_monitor import ValidationMonitor
from common.callbacks.epoch_end_callbacks.checkpoint import Checkpoint
from common.lr_scheduler import WarmupMultiStepLR
from common.nlp.bert.optimization import AdamW, WarmupLinearSchedule
from common.losses import RocStarLoss
from cls.data.build import make_dataloader, build_dataset, build_transforms
from cls.modules import *
from cls.function.val import do_validation

try:
    from apex import amp
    from apex.parallel import DistributedDataParallel as Apex_DDP
except ImportError:
    pass
    #raise ImportError("Please install apex from https://www.github.com/nvidia/apex if you want to use fp16.")


def train_net(args, config):
    # setup logger
    logger, final_output_path = create_logger(config.OUTPUT_PATH, args.cfg, config.DATASET.TRAIN_IMAGE_SET,
                                              split='train')
    model_prefix = os.path.join(final_output_path, config.MODEL_PREFIX)
    if args.log_dir is None:
        args.log_dir = os.path.join(final_output_path, 'tensorboard_logs')

    # pprint.pprint(args)
    # logger.info('training args:{}\n'.format(args))
    # pprint.pprint(config)
    # logger.info('training config:{}\n'.format(pprint.pformat(config)))

    # manually set random seed
    if config.RNG_SEED > -1:
        random.seed(a=config.RNG_SEED)
        np.random.seed(config.RNG_SEED)
        torch.random.manual_seed(config.RNG_SEED)
        torch.cuda.manual_seed_all(config.RNG_SEED)
        torch.backends.cudnn.deterministic = True
        imgaug.random.seed(config.RNG_SEED)

    # cudnn
    torch.backends.cudnn.benchmark = False
    if args.cudnn_off:
        torch.backends.cudnn.enabled = False

    if args.dist:
        model = eval(config.MODULE)(config)
        local_rank = int(os.environ.get('LOCAL_RANK') or 0)
        config.GPUS = str(local_rank)
        torch.cuda.set_device(local_rank)
        master_address = os.environ['MASTER_ADDR']
        master_port = int(os.environ['MASTER_PORT'] or 23456)
        world_size = int(os.environ['WORLD_SIZE'] or 1)
        rank = int(os.environ['RANK'] or 0)

        if rank == 0:
            pprint.pprint(args)
            logger.info('training args:{}\n'.format(args))
            pprint.pprint(config)
            logger.info('training config:{}\n'.format(pprint.pformat(config)))
        
        if args.slurm:
            distributed.init_process_group(backend='nccl')
        else:
            try:
                distributed.init_process_group(
                    backend='nccl',
                    init_method='tcp://{}:{}'.format(master_address, master_port),
                    world_size=world_size,
                    rank=rank,
                    group_name='mtorch')
            except RuntimeError:
                pass
        print(f'native distributed, size: {world_size}, rank: {rank}, local rank: {local_rank}')
        torch.cuda.set_device(local_rank)
        config.GPUS = str(local_rank)
        model = model.cuda()
        if not config.TRAIN.FP16:
            model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

        if rank == 0:
            summary_parameters(model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model,
                               logger)
            shutil.copy(args.cfg, final_output_path)
            shutil.copy(inspect.getfile(eval(config.MODULE)), final_output_path)

        writer = None
        if args.log_dir is not None:
            tb_log_dir = os.path.join(args.log_dir, 'rank{}'.format(rank))
            if not os.path.exists(tb_log_dir):
                os.makedirs(tb_log_dir)
            writer = SummaryWriter(log_dir=tb_log_dir)

        batch_size = world_size * (sum(config.TRAIN.BATCH_IMAGES)
                                   if isinstance(config.TRAIN.BATCH_IMAGES, list)
                                   else config.TRAIN.BATCH_IMAGES)
        if config.TRAIN.GRAD_ACCUMULATE_STEPS > 1:
            batch_size = batch_size * config.TRAIN.GRAD_ACCUMULATE_STEPS
        base_lr = config.TRAIN.LR * batch_size
        optimizer_grouped_parameters = [{'params': [p for n, p in model.named_parameters() if _k in n],
                                         'lr': base_lr * _lr_mult}
                                        for _k, _lr_mult in config.TRAIN.LR_MULT]
        optimizer_grouped_parameters.append({'params': [p for n, p in model.named_parameters()
                                                        if all([_k not in n for _k, _ in config.TRAIN.LR_MULT])]})
        if config.TRAIN.OPTIMIZER == 'SGD':
            optimizer = optim.SGD(optimizer_grouped_parameters,
                                  lr=config.TRAIN.LR * batch_size,
                                  momentum=config.TRAIN.MOMENTUM,
                                  weight_decay=config.TRAIN.WD)
        elif config.TRAIN.OPTIMIZER == 'Adam':
            optimizer = optim.Adam(optimizer_grouped_parameters,
                                   lr=config.TRAIN.LR * batch_size,
                                   weight_decay=config.TRAIN.WD)
        elif config.TRAIN.OPTIMIZER == 'AdamW':
            optimizer = AdamW(optimizer_grouped_parameters,
                              lr=config.TRAIN.LR * batch_size,
                              betas=(0.9, 0.999),
                              eps=1e-6,
                              weight_decay=config.TRAIN.WD,
                              correct_bias=True)
        else:
            raise ValueError('Not support optimizer {}!'.format(config.TRAIN.OPTIMIZER))
        total_gpus = world_size

        train_loader, train_sampler = make_dataloader(config,
                                                      mode='train',
                                                      distributed=True,
                                                      num_replicas=world_size,
                                                      rank=rank,
                                                      expose_sampler=True)
        val_loader = make_dataloader(config,
                                     mode='val',
                                     distributed=True,
                                     num_replicas=world_size,
                                     rank=rank)

    else:
        pprint.pprint(args)
        logger.info('training args:{}\n'.format(args))
        pprint.pprint(config)
        logger.info('training config:{}\n'.format(pprint.pformat(config)))


        #os.environ['CUDA_VISIBLE_DEVICES'] = config.GPUS
        model = eval(config.MODULE)(config)
        summary_parameters(model, logger)
        shutil.copy(args.cfg, final_output_path)
        shutil.copy(inspect.getfile(eval(config.MODULE)), final_output_path)
        num_gpus = len(config.GPUS.split(','))
        # assert num_gpus <= 1 or (not config.TRAIN.FP16), "Not support fp16 with torch.nn.DataParallel. " \
        #                                                  "Please use amp.parallel.DistributedDataParallel instead."
        if num_gpus > 1 and config.TRAIN.FP16:
            logger.warning("Not support fp16 with torch.nn.DataParallel.")
            config.TRAIN.FP16 = False
        
        total_gpus = num_gpus
        rank = None
        writer = SummaryWriter(log_dir=args.log_dir) if args.log_dir is not None else None

        if hasattr(model, 'setup_adapter'):
            logger.info('Setting up adapter modules!')
            model.setup_adapter()

        # model
        if num_gpus > 1:
            model = torch.nn.DataParallel(model, device_ids=[int(d) for d in config.GPUS.split(',')]).cuda()
        else:
            torch.cuda.set_device(int(config.GPUS))
            model.cuda()

        # loader
        # train_set = 'train+val' if config.DATASET.TRAIN_WITH_VAL else 'train'
        train_loader = make_dataloader(config, mode='train', distributed=False)
        val_loader = make_dataloader(config, mode='val', distributed=False)
        train_sampler = None

        batch_size = num_gpus * (sum(config.TRAIN.BATCH_IMAGES) if isinstance(config.TRAIN.BATCH_IMAGES, list)
                                 else config.TRAIN.BATCH_IMAGES)
        if config.TRAIN.GRAD_ACCUMULATE_STEPS > 1:
            batch_size = batch_size * config.TRAIN.GRAD_ACCUMULATE_STEPS
        base_lr = config.TRAIN.LR * batch_size
        optimizer_grouped_parameters = [{'params': [p for n, p in model.named_parameters() if _k in n],
                                         'lr': base_lr * _lr_mult}
                                        for _k, _lr_mult in config.TRAIN.LR_MULT]
        optimizer_grouped_parameters.append({'params': [p for n, p in model.named_parameters()
                                                        if all([_k not in n for _k, _ in config.TRAIN.LR_MULT])]})

        if config.TRAIN.OPTIMIZER == 'SGD':
            optimizer = optim.SGD(optimizer_grouped_parameters,
                                  lr=config.TRAIN.LR * batch_size,
                                  momentum=config.TRAIN.MOMENTUM,
                                  weight_decay=config.TRAIN.WD)
        elif config.TRAIN.OPTIMIZER == 'Adam':
            optimizer = optim.Adam(optimizer_grouped_parameters,
                                   lr=config.TRAIN.LR * batch_size,
                                   weight_decay=config.TRAIN.WD)
        elif config.TRAIN.OPTIMIZER == 'AdamW':
            optimizer = AdamW(optimizer_grouped_parameters,
                              lr=config.TRAIN.LR * batch_size,
                              betas=(0.9, 0.999),
                              eps=1e-6,
                              weight_decay=config.TRAIN.WD,
                              correct_bias=True)
        else:
            raise ValueError('Not support optimizer {}!'.format(config.TRAIN.OPTIMIZER))

    # partial load pretrain state dict
    if config.NETWORK.PARTIAL_PRETRAIN != "":
        pretrain_state_dict = torch.load(config.NETWORK.PARTIAL_PRETRAIN, map_location=lambda storage, loc: storage)['state_dict']
        prefix_change = [prefix_change.split('->') for prefix_change in config.NETWORK.PARTIAL_PRETRAIN_PREFIX_CHANGES]
        if len(prefix_change) > 0:
            pretrain_state_dict_parsed = {}
            for k, v in pretrain_state_dict.items():
                no_match = True
                for pretrain_prefix, new_prefix in prefix_change:
                    if k.startswith(pretrain_prefix):
                        k = new_prefix + k[len(pretrain_prefix):]
                        pretrain_state_dict_parsed[k] = v
                        no_match = False
                        break
                if no_match:
                    pretrain_state_dict_parsed[k] = v
            pretrain_state_dict = pretrain_state_dict_parsed
        smart_partial_load_model_state_dict(model, pretrain_state_dict)


    # pretrained classifier
    # if config.NETWORK.CLASSIFIER_PRETRAINED:
    #     print('Initializing classifier weight from pretrained word embeddings...')
    #     answers_word_embed = []
    #     for k, v in model.state_dict().items():
    #         if 'word_embeddings.weight' in k:
    #             word_embeddings = v.detach().clone()
    #             break
    #     for answer in train_loader.dataset.answer_vocab:
    #         a_tokens = train_loader.dataset.tokenizer.tokenize(answer)
    #         a_ids = train_loader.dataset.tokenizer.convert_tokens_to_ids(a_tokens)
    #         a_word_embed = (torch.stack([word_embeddings[a_id] for a_id in a_ids], dim=0)).mean(dim=0)
    #         answers_word_embed.append(a_word_embed)
    #     answers_word_embed_tensor = torch.stack(answers_word_embed, dim=0)
    #     for name, module in model.named_modules():
    #         if name.endswith('final_mlp'):
    #             module[-1].weight.data = answers_word_embed_tensor.to(device=module[-1].weight.data.device)

    # metrics
    train_metrics_list = [
        cls_metrics.Accuracy(allreduce=args.dist, num_replicas=world_size if args.dist else 1)
    ]
    val_metrics_list = [
        cls_metrics.Accuracy(allreduce=args.dist, num_replicas=world_size if args.dist else 1),
        cls_metrics.RocAUC(allreduce=args.dist, num_replicas=world_size if args.dist else 1)
    ]
    for output_name, display_name in config.TRAIN.LOSS_LOGGERS:
        train_metrics_list.append(
            cls_metrics.LossLogger(output_name, display_name=display_name, allreduce=args.dist,
                                   num_replicas=world_size if args.dist else 1))

    train_metrics = CompositeEvalMetric()
    val_metrics = CompositeEvalMetric()
    for child_metric in train_metrics_list:
        train_metrics.add(child_metric)
    for child_metric in val_metrics_list:
        val_metrics.add(child_metric)

    # epoch end callbacks
    epoch_end_callbacks = []
    if (rank is None) or (rank == 0):
        epoch_end_callbacks = [Checkpoint(model_prefix, config.CHECKPOINT_FREQUENT)]
    validation_monitor = ValidationMonitor(do_validation, val_loader, val_metrics,
                                           host_metric_name='RocAUC',
                                           label_index_in_batch=config.DATASET.LABEL_INDEX_IN_BATCH,
                                           model_dir=os.path.dirname(model_prefix))

    # optimizer initial lr before
    for group in optimizer.param_groups:
        group.setdefault('initial_lr', group['lr'])

    # resume/auto-resume
    if rank is None or rank == 0:
        smart_resume(model, optimizer, validation_monitor, config, model_prefix, logger)
    if args.dist:
        begin_epoch = torch.tensor(config.TRAIN.BEGIN_EPOCH).cuda()
        distributed.broadcast(begin_epoch, src=0)
        config.TRAIN.BEGIN_EPOCH = begin_epoch.item()

    # batch end callbacks
    batch_size = len(config.GPUS.split(',')) * config.TRAIN.BATCH_IMAGES
    batch_end_callbacks = [Speedometer(batch_size, config.LOG_FREQUENT,
                                       batches_per_epoch=len(train_loader),
                                       epochs=config.TRAIN.END_EPOCH - config.TRAIN.BEGIN_EPOCH)]

    # setup lr step and lr scheduler
    if config.TRAIN.LR_SCHEDULE == 'plateau':
        print("Warning: not support resuming on plateau lr schedule!")
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                                  mode='max',
                                                                  factor=config.TRAIN.LR_FACTOR,
                                                                  patience=1,
                                                                  verbose=True,
                                                                  threshold=1e-4,
                                                                  threshold_mode='rel',
                                                                  cooldown=2,
                                                                  min_lr=0,
                                                                  eps=1e-8)
    elif config.TRAIN.LR_SCHEDULE == 'triangle':
        lr_scheduler = WarmupLinearSchedule(optimizer,
                                            config.TRAIN.WARMUP_STEPS if config.TRAIN.WARMUP else 0,
                                            t_total=int(config.TRAIN.END_EPOCH * len(train_loader) / config.TRAIN.GRAD_ACCUMULATE_STEPS),
                                            last_epoch=int(config.TRAIN.BEGIN_EPOCH * len(train_loader) / config.TRAIN.GRAD_ACCUMULATE_STEPS)  - 1)
    elif config.TRAIN.LR_SCHEDULE == 'step':
        lr_iters = [
            int(epoch * len(train_loader) / config.TRAIN.GRAD_ACCUMULATE_STEPS)
            for epoch in config.TRAIN.LR_STEP]
        lr_scheduler = WarmupMultiStepLR(optimizer, milestones=lr_iters,
                                        gamma=config.TRAIN.LR_FACTOR,
                                         warmup_factor=config.TRAIN.WARMUP_FACTOR,
                                         warmup_iters=config.TRAIN.WARMUP_STEPS if config.TRAIN.WARMUP else 0,
                                         warmup_method=config.TRAIN.WARMUP_METHOD,
                                         last_epoch=int(config.TRAIN.BEGIN_EPOCH * len(train_loader) / config.TRAIN.GRAD_ACCUMULATE_STEPS)  - 1)
    else:
        raise ValueError("Not support lr schedule: {}.".format(config.TRAIN.LR_SCHEDULE))
    

    if config.TRAIN.SWA:
        assert config.TRAIN.SWA_START_EPOCH < config.TRAIN.END_EPOCH
        if not config.TRAIN.DEBUG:
            true_epoch_step = len(train_loader) / config.TRAIN.GRAD_ACCUMULATE_STEPS
        else:
            true_epoch_step = 50
        step_per_cycle = config.TRAIN.SWA_EPOCH_PER_CYCLE * true_epoch_step
        
        # swa_scheduler = torch.optim.lr_scheduler.CyclicLR(
        #     optimizer,
        #     base_lr=config.TRAIN.SWA_MIN_LR * batch_size,
        #     max_lr=config.TRAIN.SWA_MAX_LR * batch_size,
        #     cycle_momentum=False,
        #     step_size_up=10,
        #     step_size_down=step_per_cycle - 10)
        
        anneal_steps = max(1, (config.TRAIN.END_EPOCH - config.TRAIN.SWA_START_EPOCH) // 4) * step_per_cycle
        anneal_steps = int(anneal_steps)
        swa_scheduler = SWALR(
            optimizer,
            anneal_epochs=anneal_steps,
            anneal_strategy='linear',
            swa_lr=config.TRAIN.SWA_MAX_LR * batch_size
        )
    else:
        swa_scheduler = None
    
    if config.TRAIN.ROC_STAR:
        assert config.TRAIN.ROC_START_EPOCH < config.TRAIN.END_EPOCH
        roc_star = RocStarLoss(
            delta=2.0,
            sample_size=config.TRAIN.ROC_SAMPLE_SIZE,
            sample_size_gamma=config.TRAIN.ROC_SAMPLE_SIZE * 2,
            update_gamma_each=config.TRAIN.ROC_SAMPLE_SIZE,
        )
    else:
        roc_star = None

    # broadcast parameter and optimizer state from rank 0 before training start
    if args.dist:
        for v in model.state_dict().values():
            distributed.broadcast(v, src=0)
        # for v in optimizer.state_dict().values():
        #     distributed.broadcast(v, src=0)
        best_epoch = torch.tensor(validation_monitor.best_epoch).cuda()
        best_val = torch.tensor(validation_monitor.best_val).cuda()
        distributed.broadcast(best_epoch, src=0)
        distributed.broadcast(best_val, src=0)
        validation_monitor.best_epoch = best_epoch.item()
        validation_monitor.best_val = best_val.item()

    # apex: amp fp16 mixed-precision training
    if config.TRAIN.FP16:
        # model.apply(bn_fp16_half_eval)
        model, optimizer = amp.initialize(model, optimizer,
                                          opt_level='O2',
                                          keep_batchnorm_fp32=False,
                                          loss_scale=config.TRAIN.FP16_LOSS_SCALE,
                                          min_loss_scale=32.0)
        if args.dist:
            model = Apex_DDP(model, delay_allreduce=True)

    # NOTE: final_model == model if not using SWA, else final_model == AveragedModel(model)
    final_model = train(
        model, optimizer, lr_scheduler,
        train_loader, train_sampler, train_metrics,
        config.TRAIN.BEGIN_EPOCH,
        config.TRAIN.END_EPOCH,
        logger,
        fp16=config.TRAIN.FP16,
        rank=rank,
        writer=writer,
        batch_end_callbacks=batch_end_callbacks,
        epoch_end_callbacks=epoch_end_callbacks,
        validation_monitor=validation_monitor,
        clip_grad_norm=config.TRAIN.CLIP_GRAD_NORM,
        gradient_accumulate_steps=config.TRAIN.GRAD_ACCUMULATE_STEPS,
        ckpt_path=config.TRAIN.CKPT_PATH,
        swa_scheduler=swa_scheduler,
        swa_start_epoch=config.TRAIN.SWA_START_EPOCH,
        swa_cycle_epoch=config.TRAIN.SWA_EPOCH_PER_CYCLE,
        swa_use_scheduler=config.TRAIN.SWA_SCHEDULE,
        roc_star=roc_star,
        roc_star_start_epoch=config.TRAIN.ROC_START_EPOCH,
        roc_interleave=config.TRAIN.ROC_INTERLEAVE,
        debug=config.TRAIN.DEBUG,
    )

    return rank, final_model
