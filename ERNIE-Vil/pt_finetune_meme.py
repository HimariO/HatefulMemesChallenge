#    Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#    http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" finetuning vison-language task """

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from random import shuffle
import sys
import time
import datetime
import argparse
import multiprocessing
import json
import random
import pickle
import shutil

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from reader.meme_finetuning import MemeDataJointReader, MemeDataReader
from reader.pl_meme_data import LitHatefulMeme
from model.pt_ernie_vil import ErnieVilModel, ErnieVilConfig, LitErnieVil
from utils.args import print_arguments
from args.finetune_args import parser
from batching.finetune_batching import prepare_batch_data

from loguru import logger
from sklearn.metrics import roc_auc_score, accuracy_score


args = parser.parse_args()
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)


def test_submit(model: pl.LightningModule, test_loader, output_path):
    with torch.no_grad():
        model.eval()

        predicts = []
        cur_id = 0
        for nbatch, batch in enumerate(test_loader):
            # bs = test_loader.batch_sampler.batch_size if test_loader.batch_sampler is not None else test_loader.batch_size
            batch = [b.cuda() for b in batch]
            cls_logit = model(batch)
            if cls_logit.shape[-1] == 1:
                prob = torch.sigmoid(cls_logit[:, 0]).detach().cpu().tolist()
            else:
                prob = torch.softmax(cls_logit, dim=-1)[:, 1].detach().cpu().tolist()
            sample_ids = batch[-4].cpu().tolist()

            for pb, id in zip(prob, sample_ids):
                predicts.append({
                    'id': int(id),
                    'proba': float(pb),
                    'label': int(pb > 0.5)
                })

        result_pd = pd.DataFrame.from_dict(predicts)
        result_pd.to_csv(output_path, index=False)
        model.train()
        return result_pd


def main(args):
    """
       Main func for downstream tasks
    """
    with open(args.task_group_json) as f:
        task_group = json.load(f)
        print('task: ', task_group)
    
    print("finetuning tasks start")
    ernie_config = ErnieVilConfig(args.ernie_config_path)
    ernie_config.print_config()

    # pl_ckpt_path = './pl_ernie_checkpoints'
    pl_ckpt_path = args.checkpoints
    os.makedirs(pl_ckpt_path, exist_ok=True)
    # paddle_weight_path = '/home/ron_zhu/Disk2/ernie/ernie-vil-large-vcr.npz'
    pl_ernie_vil = LitErnieVil(
        args,
        fusion_dropout=task_group[0]['dropout_rate'],
        cls_head='linear',
    )
    if args.resume_ckpt:
        logger.warning(f"REsume model and trainer from: {args.resume_ckpt} !!")
    else:
        pl_ernie_vil.load_paddle_weight(args.init_checkpoint)

    dump_args_path = os.path.join(pl_ckpt_path, 'args.pickle')
    with open(dump_args_path, mode='wb') as f:
        pickle.dump(args, f)
    shutil.copy(
        args.task_group_json,
        os.path.join(pl_ckpt_path, os.path.basename(args.task_group_json))
    )
    shutil.copy(
        args.ernie_config_path,
        os.path.join(pl_ckpt_path, os.path.basename(args.ernie_config_path))
    )

    if args.do_train:
        lit_dataset = LitHatefulMeme(
            task_group,
            vocab_path=args.vocab_path,
            batch_size=args.batch_size,
            epoch=args.epoch,
            balance_cls=args.balance_cls,
            random_seed=args.seed
        )
        
        resume_ckpt = args.resume_ckpt
        checkpoint = ModelCheckpoint(
            filepath=pl_ckpt_path,
            save_last=False,
            save_top_k=-1,
            monitor='val_auroc_epoch',
            mode='max',
            save_weights_only=True,
        )
        trainer = pl.Trainer(
            fast_dev_run=False,
            accumulate_grad_batches=args.accumulate_grad_batches,
            val_check_interval=1.0,
            checkpoint_callback=checkpoint,
            callbacks=[],
            default_root_dir=pl_ckpt_path,
            gpus=1,
            # num_nodes=2,
            # distributed_backend='ddp',
            precision=16,
            max_steps=min(args.num_train_steps, 15000),
            resume_from_checkpoint=resume_ckpt,
            num_sanity_val_steps=0,
        )

        trainer.fit(pl_ernie_vil, datamodule=lit_dataset)
    
    if args.do_test:
        lit_dataset = LitHatefulMeme(
            task_group,
            vocab_path=args.vocab_path,
            batch_size=args.batch_size,
            epoch=args.epoch,
            balance_cls=args.balance_cls,
            random_seed=args.seed
        )
        test_loader = lit_dataset.test_dataloader()
        test_submit(pl_ernie_vil, test_loader)


if __name__ == '__main__':
    with logger.catch():
        # print_arguments(args)
        main(args)

