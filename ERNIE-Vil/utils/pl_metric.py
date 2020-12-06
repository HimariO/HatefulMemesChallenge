from multiprocessing import Value
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.metrics import Metric
from sklearn.metrics import roc_auc_score
from loguru import logger

class LitAUROC(Metric):

    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("predicts", default=[], dist_reduce_fx=None)
        self.add_state("targets", default=[], dist_reduce_fx=None)

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        # preds, target = self._input_format(preds, target)
        assert preds.shape == target.shape, f"{preds.shape} != {target.shape}"
        assert preds.ndim == target.ndim, f"{preds.ndim} != {target.ndim}"
        
        preds = preds.detach().cpu().tolist()
        target = target.detach().cpu().tolist()
        self.predicts += preds
        self.targets += target

    def compute(self):
        if len(self.targets) == 1:
            return 0.5
        elif len(self.targets) == 0:
            raise ValueError('No predict/target pair found for auroc metric!')
        else:
            targets = [int(t > 0.5) for t in self.targets]
            try:
                return roc_auc_score(targets, self.predicts)
            except ValueError as ve:
                logger.warning(ve)
                return -1
