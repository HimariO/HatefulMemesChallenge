import _init_paths
import os
import copy
import pickle
from functools import partial

import fire
import torch
import pandas as pd
import pytorch_lightning as pl
from loguru import logger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.cloud_io import load as pl_load

import ray
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining
from ray.tune.integration.pytorch_lightning import (
    TuneReportCallback,
    TuneReportCheckpointCallback
)

from common.trainer import to_cuda
from common.metrics import cls_metrics
from cls.modules.resnet_vlbert_pl import LitVLBERT
from cls.function.config import config, update_config
from cls.function.val import do_validation
from cls.data.datasets.cls_pl import LitHatefulMeme


def test_submit(model: pl.LightningModule, test_loader, output_path):
    with torch.no_grad():
        model.eval()

        predicts = []
        cur_id = 0
        for nbatch, batch in enumerate(test_loader):
            # bs = test_loader.batch_sampler.batch_size if test_loader.batch_sampler is not None else test_loader.batch_size
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

        result_pd = pd.DataFrame.from_dict(predicts)
        result_pd.to_csv(output_path, index=False)
        model.train()
        return result_pd


def og_validation(config, model: pl.LightningModule, data_loader, output_dir):
    metric = cls_metrics.Accuracy()
    do_validation(
        model,
        data_loader,
        metric,
        config.DATASET.LABEL_INDEX_IN_BATCH,
        model_dir=output_dir)
    accuracy = metric.get()[1]
    logger.info(f"[og_validation] Accuracy: {accuracy}")


def _train(config, pl_ckpt_path, max_epochs=None, gpus=1, nodes=1, valset_size=1.0,
            resume_ckpt=None, fast_dev_run=False, test_best=True):
    max_epochs = config.TRAIN.END_EPOCH if max_epochs is None else max_epochs
    
    vl_bert = LitVLBERT(config)
    hateful_meme = LitHatefulMeme(config)
    
    checkpoint = ModelCheckpoint(
        filepath=pl_ckpt_path,
        save_last=False,
        save_top_k=3,
        monitor='val_accuracy',
    )
    trainer = pl.Trainer(
        fast_dev_run=fast_dev_run,
        accumulate_grad_batches=config.TRAIN.GRAD_ACCUMULATE_STEPS,
        val_check_interval=valset_size,
        checkpoint_callback=checkpoint,
        callbacks=[],
        default_root_dir=pl_ckpt_path,
        gpus=gpus,
        num_nodes=nodes,
        distributed_backend='ddp',
        precision=16,
        max_epochs=max_epochs,
        resume_from_checkpoint=resume_ckpt
    )

    trainer.fit(vl_bert, datamodule=hateful_meme)
    best_weight = checkpoint.best_model_path

    if test_best:
        logger.info(f'Best weight - {checkpoint.best_model_score}: {best_weight}')
        best_model = vl_bert.load_from_checkpoint(best_weight)
        best_model = best_model.cuda()
        # tmp = trainer.test(test_dataloaders=hateful_meme.val_dataloader())
        # print(tmp)
        val_loader = hateful_meme.val_dataloader()
        og_validation(config, best_model, val_loader, pl_ckpt_path)
        submit_file_path = os.path.join(pl_ckpt_path, 'submit.csv')
        test_submit(best_model, val_loader, submit_file_path)
    
    return best_weight


def train(config_path, pl_ckpt_path, **kwargs):
    with logger.catch(reraise=True):
        update_config(config_path)
        cfg = copy.deepcopy(config)
        _train(cfg, pl_ckpt_path, **kwargs)


def _tune(tune_param_config, vl_bert_config=None, pl_ckpt_path=None, checkpoint_dir=None, num_gpus=1):
    pickle.DEFAULT_PROTOCOL = 4
    
    with logger.catch(reraise=True):
        config = copy.deepcopy(vl_bert_config)
        
        # config.TRAIN.LR = lr
        # config.TRAIN.WD = weight_decay
        
        # config.TRAIN.BATCH_IMAGES = batch_size
        # config.TRAIN.END_EPOCH = max_epoch
        
        # config.TRAIN.WARMUP_FACTOR = warmup_factor
        # config.TRAIN.WARMUP_STEPS = warmup_steps
        logger.warning(os.path.abspath('.'))
        
        checkpoint = ModelCheckpoint(
            filepath=pl_ckpt_path,
            save_last=False,
            save_top_k=3,
            monitor='val_accuracy',
        )
        tune_report = TuneReportCheckpointCallback({
            # "loss": "val_checkpoint_on",
            "mean_accuracy": "val_checkpoint_on"
        }, on="validation_end")
        adhoc_logger = TensorBoardLogger(
                save_dir=tune.get_trial_dir(),
                name="", version=".")
        
        trainer = pl.Trainer(
            # limit_train_batches=0.1,
            # limit_val_batches=0.1,
            accumulate_grad_batches=config.TRAIN.GRAD_ACCUMULATE_STEPS,
            checkpoint_callback=None,
            callbacks=[tune_report],
            logger=adhoc_logger,
            default_root_dir=pl_ckpt_path,
            gpus=num_gpus,
            num_nodes=1,
            distributed_backend='dp',
            precision=16,
            max_epochs=config.TRAIN.END_EPOCH,
            resume_from_checkpoint=None,
        )

        # vl_bert = LitVLBERT(config)
        hateful_meme = LitHatefulMeme(config)

        if checkpoint_dir:
            # Currently, this leads to errors:
            # model = LightningMNISTClassifier.load_from_checkpoint(
            #     os.path.join(checkpoint, "checkpoint"))
            # Workaround:
            ckpt = pl_load(
                os.path.join(checkpoint_dir, "checkpoint"),
                map_location=lambda storage, loc: storage)
            vl_bert = LitVLBERT._load_model_state(ckpt, config)
            trainer.current_epoch = ckpt["epoch"]
        else:
            logger.info(config)
            vl_bert = LitVLBERT(config)
        
        trainer.fit(vl_bert, datamodule=hateful_meme)


def tune_vl_bert(config_path, pl_ckpt_path, num_samples=10, num_epochs=10, gpus_per_trial=2):

    # scheduler = ASHAScheduler(
    #     metric="loss",
    #     mode="min",
    #     max_t=num_epochs,
    #     grace_period=1,
    #     reduction_factor=2)

    reporter = CLIReporter(
        parameter_columns=["lr", "weight_decay", "warmup_factor", "max_epoch", "batch_size"],
        metric_columns=["mean_accuracy", "training_iteration"])
    
    param_config = {
        "lr": 6.25e-7,
        "weight_decay": tune.loguniform(1e-5, 1e-2),
        "batch_size": 4,
        "max_epoch": tune.choice([4, 6, 8, 10]),
        "warmup_factor": tune.uniform(0, 1),
        "warmup_steps": tune.uniform(100, 800),
    }

    scheduler = PopulationBasedTraining(
        time_attr="training_iteration",
        metric="mean_accuracy",
        mode="max",
        perturbation_interval=2,
        hyperparam_mutations={
            "lr": tune.loguniform(6.25e-6, 6.25e-8),
            "batch_size": [1, 2, 3, 4],
        })
    
    update_config(config_path)
    model_base_cfg = copy.deepcopy(config)

    tune.run(
        partial(
            _tune,
            vl_bert_config=model_base_cfg,
            pl_ckpt_path=pl_ckpt_path,
            num_gpus=gpus_per_trial,
        ),
        resources_per_trial={
            "cpu": 4,
            "gpu": gpus_per_trial,
        },
        config=param_config,
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter=reporter,
        name="tune_vl_bert")


if __name__ == "__main__":    
    fire.Fire({
        'train': train,
        'tune': tune_vl_bert,
    })