import os
import json
import random
from functools import reduce
from os.path import join
from typing import List, Dict, Union

import torch 
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from loguru import logger

from cls.data.build import make_dataloader, build_dataset, build_transforms


class LitHatefulMeme(pl.LightningDataModule):
    
    def __init__(self, config):
        super().__init__()
        self.config = config
    
    def train_dataloader(self):
        # HACK: PL will hand distribution configs, so we only return non-distributed loader.
        dataset, batch_sampler, collect_func = make_dataloader(
            self.config,
             mode='train',
             distributed=False,
             return_component=True
        )
        loader = DataLoader(
            dataset=dataset,
            batch_sampler=None,
            num_workers=self.config.NUM_WORKERS_PER_GPU,
            pin_memory=False,
            collate_fn=collect_func,
            batch_size=self.config.TRAIN.BATCH_IMAGES,
        )
        return loader
    
    def val_dataloader(self):
        # HACK: PL will hand distribution configs, so we only return non-distributed loader.
        dataset, batch_sampler, collect_func = make_dataloader(
            self.config,
             mode='val',
             distributed=False,
             return_component=True
        )
        loader = DataLoader(
            dataset=dataset,
            batch_sampler=None,
            num_workers=self.config.NUM_WORKERS_PER_GPU,
            pin_memory=False,
            collate_fn=collect_func,
            batch_size=self.config.VAL.BATCH_IMAGES,
        )
        return loader
    
    def test_dataloader(self):
        # HACK: PL will hand distribution configs, so we only return non-distributed loader.
        dataset, batch_sampler, collect_func = make_dataloader(
            self.config,
             mode='test',
             distributed=False,
             return_component=True
        )
        loader = DataLoader(
            dataset=dataset,
            batch_sampler=None,
            num_workers=self.config.NUM_WORKERS_PER_GPU,
            pin_memory=False,
            collate_fn=collect_func,
            batch_size=self.config.TEST.BATCH_IMAGES,
        )
        return loader