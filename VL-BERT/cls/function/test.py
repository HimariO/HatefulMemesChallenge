import os
import pprint
import shutil

import json
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm, trange
from sklearn.metrics import roc_auc_score, accuracy_score

from common.utils.load import smart_load_model_state_dict
from common.trainer import to_cuda
from common.utils.create_logger import create_logger
from cls.data.build import make_dataloader
from cls.modules import *


@torch.no_grad()
def test_net(args, config, ckpt_path=None, save_path=None, save_name=None):
    print('test net...')
    pprint.pprint(args)
    pprint.pprint(config)
    device_ids = [int(d) for d in config.GPUS.split(',')]
    # os.environ['CUDA_VISIBLE_DEVICES'] = config.GPUS

    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if ckpt_path is None:
        _, train_output_path = create_logger(config.OUTPUT_PATH, args.cfg, config.DATASET.TRAIN_IMAGE_SET,
                                             split='train')
        model_prefix = os.path.join(train_output_path, config.MODEL_PREFIX)
        ckpt_path = '{}-best.model'.format(model_prefix)
        print('Use best checkpoint {}...'.format(ckpt_path))
    if save_path is None:
        logger, test_output_path = create_logger(config.OUTPUT_PATH, args.cfg, config.DATASET.TEST_IMAGE_SET,
                                                 split='test')
        save_path = test_output_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    test_ckpt_path = '{}_test_ckpt_{}.model'.format(config.MODEL_PREFIX, config.DATASET.TASK)
    try:
        shutil.copy2(ckpt_path,
                    os.path.join(save_path, test_ckpt_path))
    except shutil.SameFileError:
        print(f'Test checkpoints is alredy exist: {test_ckpt_path}')

    # get network
    model = eval(config.MODULE)(config)

    if hasattr(model, 'setup_adapter'):
        model.setup_adapter()

    # if len(device_ids) > 1:
    #     model = torch.nn.DataParallel(model, device_ids=device_ids).cuda()
    # else:
    torch.cuda.set_device(min(device_ids))
    model = model.cuda()
    
    checkpoint = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
    smart_load_model_state_dict(model, checkpoint['state_dict'])

    # loader
    test_loader = make_dataloader(config, mode='test', distributed=False)
    test_dataset = test_loader.dataset
    test_database = test_dataset.database

    # test
    predicts = []
    model.eval()
    cur_id = 0
    for nbatch, batch in zip(trange(len(test_loader)), test_loader):
    # for nbatch, batch in tqdm(enumerate(test_loader)):
        bs = test_loader.batch_sampler.batch_size if test_loader.batch_sampler is not None else test_loader.batch_size
        batch = to_cuda(batch)
        outputs = model(*batch[:-1])
        if outputs['label_logits'].shape[-1] == 1:
            prob = torch.sigmoid(outputs['label_logits'][:, 0]).detach().cpu().tolist()
        else:
            prob = torch.softmax(outputs['label_logits'], dim=-1)[:, 1].detach().cpu().tolist()
        sample_ids = batch[-1].cpu().tolist()
        for pb, id in zip(prob, sample_ids):
            predicts.append({
                'id': int(id),
                'proba': float(pb),
                'label': int(pb > 0.5)
            })

    cfg_name = os.path.splitext(os.path.basename(args.cfg))[0]
    output_name = cfg_name if save_name is None else save_name
    result_json_path = os.path.join(save_path, f'{output_name}_cls_{config.DATASET.TEST_IMAGE_SET}.json')
    result_csv_path = os.path.join(save_path, f'{output_name}_cls_{config.DATASET.TEST_IMAGE_SET}.csv')
    
    with open(result_json_path, 'w') as f:
        json.dump(predicts, f)
    print('result json saved to {}.'.format(result_json_path))

    pd.DataFrame.from_dict(predicts).to_csv(result_csv_path, index=False)
    return result_json_path

@torch.no_grad()
def val_net(args, config, ckpt_path=None, save_path=None, save_name=None):
    print('test net...')
    pprint.pprint(args)
    pprint.pprint(config)
    device_ids = [int(d) for d in config.GPUS.split(',')]
    # os.environ['CUDA_VISIBLE_DEVICES'] = config.GPUS

    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if ckpt_path is None:
        _, train_output_path = create_logger(config.OUTPUT_PATH, args.cfg, config.DATASET.TRAIN_IMAGE_SET,
                                             split='train')
        model_prefix = os.path.join(train_output_path, config.MODEL_PREFIX)
        ckpt_path = '{}-best.model'.format(model_prefix)
        print('Use best checkpoint {}...'.format(ckpt_path))
    # if save_path is None:
    #     logger, test_output_path = create_logger(config.OUTPUT_PATH, args.cfg, config.DATASET.TEST_IMAGE_SET,
    #                                              split='test')
    #     save_path = test_output_path
    # if not os.path.exists(save_path):
    #     os.makedirs(save_path)
    # shutil.copy2(ckpt_path,
    #              os.path.join(save_path, '{}_test_ckpt_{}.model'.format(config.MODEL_PREFIX, config.DATASET.TASK)))

    # get network
    model = eval(config.MODULE)(config)

    if hasattr(model, 'setup_adapter'):
        model.setup_adapter()

    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids).cuda()
    else:
        torch.cuda.set_device(device_ids[0])
        model = model.cuda()
    checkpoint = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
    smart_load_model_state_dict(model, checkpoint['state_dict'])

    # loader
    test_loader = make_dataloader(config, mode='val', distributed=False)
    test_dataset = test_loader.dataset
    test_database = test_dataset.database

    # test
    predicts = []
    model.eval()
    cur_id = 0
    for nbatch, batch in zip(trange(len(test_loader)), test_loader):
    # for nbatch, batch in tqdm(enumerate(test_loader)):
        bs = test_loader.batch_sampler.batch_size if test_loader.batch_sampler is not None else test_loader.batch_size
        batch = to_cuda(batch)
        outputs = model(*batch[:-1])
        if outputs['label_logits'].shape[-1] == 1:
            prob = torch.sigmoid(outputs['label_logits'][:, 0]).detach().cpu().tolist()
        else:
            prob = torch.softmax(outputs['label_logits'], dim=-1)[:, 1].detach().cpu().tolist()
        
        sample_ids = batch[-1].cpu().tolist()
        targets = batch[config.DATASET.LABEL_INDEX_IN_BATCH]
        for pb, id, tg in zip(prob, sample_ids, targets):
            predicts.append({
                'id': int(id),
                'proba': float(pb),
                'label': int(pb > 0.5),
                'target': float(tg)
            })

    pred_probs = [p['proba'] for p in predicts]
    pred_labels = [p['label'] for p in predicts]
    targets = [p['target'] for p in predicts]
    
    roc_auc = roc_auc_score(targets, pred_probs)
    print(f"roc_auc: {roc_auc}")

    max_accuracy = 0.0
    best_threshold = 1e-2
    for th in range(1, 100):
        targets_idx = [int(p['target'] > 1e-2 * th) for p in predicts]
        accuracy = accuracy_score(targets_idx, pred_labels)
        if accuracy > max_accuracy:
            max_accuracy = accuracy
            best_threshold = th * 1e-2
    print(f"max accuracy: {max_accuracy}, best_threshold: {best_threshold}")
    
