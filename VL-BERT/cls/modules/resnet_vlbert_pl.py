import os
from datetime import timedelta

import torch
import torch.optim as optim
import torch.distributed as torch_distrib
import pytorch_lightning as pl
from loguru import logger
from easydict import EasyDict as edict
from pytorch_lightning.metrics.functional import accuracy

from cls.modules.resnet_vlbert_for_cls import ResNetVLBERT
from common.metrics import cls_metrics
from common.utils.load import smart_partial_load_model_state_dict
from common.nlp.bert.optimization import AdamW, WarmupLinearSchedule
from common.lr_scheduler import WarmupMultiStepLR


class LitVLBERT(pl.LightningModule):

    def __init__(self, config):
        super().__init__()
        if isinstance(config, dict):
            config = edict(config)
        self.config = config
        self.vl_bert = ResNetVLBERT(config)
        
        self.train_metrics = {
            'train_accuracy': cls_metrics.Accuracy(allreduce=False, num_replicas=1),
        }
        self.val_metrics = {
            'val_accuracy': cls_metrics.Accuracy(allreduce=False, num_replicas=1),
        }

        if config.NETWORK.PARTIAL_PRETRAIN != "":
            if os.path.exists(config.NETWORK.PARTIAL_PRETRAIN):
                self.load_torch_weight(config)
            else:
                logger.error(f"VL-BERT pretrain weight not found! : {config.NETWORK.PARTIAL_PRETRAIN}")
        self.save_hyperparameters(config)
    
    # def init_ddp_connection(self, global_rank: int, world_size: int, is_slurm_managing_tasks: bool = True) -> None:
    #     torch_backend = "nccl" if self.trainer.on_gpu else "gloo"
    #     logger.info(f"initializing ddp: GLOBAL_RANK: {global_rank}, MEMBER: {global_rank+1}/{world_size}")
    #     torch_distrib.init_process_group(torch_backend, init_method="env://", timeout=timedelta(seconds=10))
    
    def load_torch_weight(self, config):
        logger.info(f'[{self.__class__.__name__}] load_torch_weight: {config.NETWORK.PARTIAL_PRETRAIN}')
        pretrain_state_dict = torch.load(
            config.NETWORK.PARTIAL_PRETRAIN,
            map_location=lambda storage, loc: storage)['state_dict']
        prefix_change = [
            prefix_change.split('->')
            for prefix_change in config.NETWORK.PARTIAL_PRETRAIN_PREFIX_CHANGES
        ]
        
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
        smart_partial_load_model_state_dict(self.vl_bert, pretrain_state_dict)

    def forward(self, image, boxes, im_info, text, *args):
        if len(args) >= 2:
            label, sample_id = args[-2:]
            return self.vl_bert.train_forward(image, boxes, im_info, text, label, *args)
        else:
            return self.vl_bert.inference_forward(image, boxes, im_info, text)
    
    def summary_epoch_output(self, outputs, key):
        if isinstance(outputs, dict):
            if isinstance(outputs[key], list):
                epoch_val = torch.stack([x[key] for x in outputs]).mean()
            elif isinstance(outputs[key], torch.Tensor):
                epoch_val = outputs[key].mean()
            else:
                epoch_val = outputs[key]
        elif isinstance(outputs, list):
            if isinstance(outputs[0][key], (list, torch.Tensor)):
                epoch_val = torch.stack([x[key] for x in outputs]).mean()
            else:
                epoch_val = outputs[key]
        return epoch_val

    def training_step(self, batch, batch_idx):
        pred_dict, loss = self.forward(*batch)
        self.train_metrics['train_accuracy'].update(pred_dict)

        result = {}
        result['loss'] = loss
        # NOTE: metric.get() return tuple: (metric_name, value)
        result['train_accuracy'] = self.train_metrics['train_accuracy'].get()[1]
        self.log('train_accuracy', result['train_accuracy'], prog_bar=True)

        return result
    
    def training_epoch_end(self, training_step_outputs):
        for metric in self.train_metrics.values():
            metric.reset()
        
        epoch_loss = self.summary_epoch_output(training_step_outputs, 'loss')

        # return {
        #     'log': {
        #         'epoch_loss': epoch_loss,
        #     },
        #     # 'progress_bar': {'epoch_loss': epoch_loss}
        # }
    
    def validation_step(self, batch, batch_idx):
        pred_dict, loss = self.forward(*batch)
        self.val_metrics['val_accuracy'].update(pred_dict)

        result = {'val_loss': loss}
        self.log('val_loss', loss)
        return result

    def validation_epoch_end(self, validation_step_outputs):
        val_epoch = {
            k: torch.tensor(m.get()[1]) for k, m in self.val_metrics.items()
        }
        for metric in self.val_metrics.values():
            metric.reset()
        epoch_loss = self.summary_epoch_output(validation_step_outputs, 'val_loss')
        
        for k, v in val_epoch.items():
            self.log(k, v, prog_bar=True)
        self.log('epoch_val_loss', epoch_loss)
        return {
            'epoch_val_loss': epoch_loss,
            **val_epoch,
        }
    
    def test_step(self, *args, **kwargs):
        return self.validation_step(*args, **kwargs)
    
    def test_epoch_end(self, outputs):
        return self.validation_epoch_end(outputs)
    
    def get_optimizer(self, config):
        if self.trainer.is_global_zero:
            world_size = self.trainer.world_size
            batch_size = world_size * (sum(config.TRAIN.BATCH_IMAGES)
                                    if isinstance(config.TRAIN.BATCH_IMAGES, list)
                                    else config.TRAIN.BATCH_IMAGES)
        else:
            num_gpus = len(self.trainer.gpus.split(','))
            batch_size = num_gpus * (sum(config.TRAIN.BATCH_IMAGES) 
                                    if isinstance(config.TRAIN.BATCH_IMAGES, list)
                                    else config.TRAIN.BATCH_IMAGES)
        if config.TRAIN.GRAD_ACCUMULATE_STEPS > 1:
            batch_size = batch_size * config.TRAIN.GRAD_ACCUMULATE_STEPS
        base_lr = config.TRAIN.LR * batch_size
        logger.info(f"base_lr: {base_lr} = {config.TRAIN.LR} * {batch_size}")
        
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in self.vl_bert.named_parameters() if _k in n],
                'lr': base_lr * _lr_mult
            }
            for _k, _lr_mult in config.TRAIN.LR_MULT
        ]
        optimizer_grouped_parameters.append({
            'params': [
                p for n, p in self.vl_bert.named_parameters()
                if all([_k not in n for _k, _ in config.TRAIN.LR_MULT])
            ]
        })
       
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
        
        return optimizer
    
    def get_scheduler(self, config, optimizer):
        # setup lr step and lr scheduler
        train_loader = self.train_dataloader()
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
        return {"scheduler": lr_scheduler, "interval" : "step" }

    def configure_optimizers(self):
        # optimizer = torch.optim.AdamW(
        #     self.parameters(), lr=self.config.TRAIN.LR)
        
        # scheduler = torch.optim.lr_scheduler.OneCycleLR(
        #     optimizer,
        #     self.config.TRAIN.LR,
        #     steps_per_epoch=len(self.train_dataloader),
        #     epochs=self.config.TRAIN.END_EPOCH)
        # scheduler = {"scheduler": scheduler, "interval" : "step" }
        
        optimizer = self.get_optimizer(self.config)
        scheduler = self.get_scheduler(self.config, optimizer)

        return {"optimizer": optimizer, "lr_scheduler": scheduler}