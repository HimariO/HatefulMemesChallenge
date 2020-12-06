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
            if nbatch % 50 == 0:
                print(f"[{nbatch}/{len(test_loader)}]")
            # bs = test_loader.batch_sampler.batch_size if test_loader.batch_sampler is not None else test_loader.batch_size
            batch = [b.cuda() for b in batch]
            cls_logit = model(*batch[:8])
            if cls_logit.shape[-1] == 1:
                prob = torch.sigmoid(cls_logit[:, 0]).detach().cpu().tolist()
            else:
                prob = torch.softmax(cls_logit, dim=-1)[:, 1].detach().cpu().tolist()
            sample_ids = batch[-4].cpu().tolist()

            for pb, id in zip(prob, sample_ids):
                predicts.append({
                    'id': int(id[0]),
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
    print("finetuning tasks start")
    pl_ernie_vil = LitErnieVil(args)
    # pl_ernie_vil.load_from_checkpoint(args.test_ckpt)
    w = torch.load(args.test_ckpt, map_location='cpu')
    pl_ernie_vil.load_state_dict(w['state_dict'])
    pl_ernie_vil = pl_ernie_vil.cuda()

    with open(args.task_group_json) as f:
        task_group = json.load(f)
        print('task: ', task_group)
    
    lit_dataset = LitHatefulMeme(
        task_group,
        vocab_path=args.vocab_path,
        batch_size=args.batch_size,
        epoch=args.epoch,
        balance_cls=args.balance_cls,
        random_seed=args.seed
    )
    test_loader = lit_dataset.test_dataloader()
    csv_path = os.path.join(args.checkpoints, 'test_set.csv')
    test_pd = test_submit(pl_ernie_vil, test_loader, csv_path)
    try:
        shutil.copy(args.test_ckpt, os.path.join(args.checkpoints, 'final.ckpt'))
    except shutil.SameFileError:
        logger.info('You already get the same final.ckpt, skip copy...')


if __name__ == '__main__':
    with logger.catch():
        # print_arguments(args)
        main(args)

