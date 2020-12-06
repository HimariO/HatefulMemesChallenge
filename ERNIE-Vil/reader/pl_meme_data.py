import functools

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from batching.finetune_batching import prepare_batch_data
from reader.meme_finetuning import MemeDataJointReader, MemeDataReader

prepare_batch_data = functools.partial(prepare_batch_data, to_tensor=True)

class LitHatefulMeme(pl.LightningDataModule):
    
    def __init__(self,
                 task_group,
                 vocab_path,
                 batch_size=8,
                 epoch=10,
                 balance_cls=False,
                 random_seed=1234,
                 use_aug=True,
                 num_worker=0):
        super().__init__()
        self.task_group = task_group
        self.batch_size = batch_size
        self.vocab_path = vocab_path
        self.epoch = epoch
        self.balance_cls = balance_cls
        self.random_seed = random_seed
        self.num_worker = num_worker
        self.use_aug = use_aug
    
    def train_dataloader(self):
        data_reader = MemeDataJointReader(
            self.task_group,
            split="train",
            vocab_path=self.vocab_path,
            batch_size=self.batch_size,
            epoch=self.epoch,
            balance_cls=self.balance_cls,
            use_aug=self.use_aug,
            random_seed=self.random_seed)
        assert len(data_reader.task_readers) == 1
        task_reader = data_reader.task_readers[0]

        if self.balance_cls:
            sampler = torch.utils.data.WeightedRandomSampler(
                task_reader.weights_by_class, len(task_reader), replacement=True)
        else:
            sampler = torch.utils.data.sampler.RandomSampler(task_reader)
        batch_sampler = torch.utils.data.sampler.BatchSampler(
            sampler, self.batch_size, drop_last=False
        )
        loader = torch.utils.data.DataLoader(
            task_reader,
            batch_sampler=batch_sampler,
            num_workers=self.num_worker,
            pin_memory=False,
            collate_fn=prepare_batch_data,
        )
        return loader
    
    def val_dataloader(self):
        data_reader = MemeDataJointReader(
            self.task_group,
            split="val",
            vocab_path=self.vocab_path,
            batch_size=self.batch_size,
            epoch=self.epoch,
            balance_cls=False,
            use_aug=False,
            random_seed=self.random_seed)
        assert len(data_reader.task_readers) == 1
        task_reader = data_reader.task_readers[0]

        sampler = torch.utils.data.SequentialSampler(
            task_reader)
        batch_sampler = torch.utils.data.sampler.BatchSampler(
            sampler, 1, drop_last=False
        )
        loader = torch.utils.data.DataLoader(
            task_reader,
            batch_sampler=batch_sampler,
            num_workers=self.num_worker,
            pin_memory=False,
            collate_fn=prepare_batch_data,
        )
        return loader
    
    def test_dataloader(self):
        data_reader = MemeDataJointReader(
            self.task_group,
            split="test",
            vocab_path=self.vocab_path,
            batch_size=self.batch_size,
            epoch=self.epoch,
            balance_cls=False,
            use_aug=False,
            random_seed=self.random_seed)
        assert len(data_reader.task_readers) == 1
        task_reader = data_reader.task_readers[0]

        sampler = torch.utils.data.SequentialSampler(
            task_reader)
        batch_sampler = torch.utils.data.sampler.BatchSampler(
            sampler, 1, drop_last=False
        )
        loader = torch.utils.data.DataLoader(
            task_reader,
            batch_sampler=batch_sampler,
            num_workers=self.num_worker,
            pin_memory=False,
            collate_fn=prepare_batch_data,
        )
        return loader
