import torch
import torch.distributed as distributed
from .eval_metric import EvalMetric
from sklearn.metrics import roc_auc_score


def gather(tensor, tensor_list=None, root=0, group=None):
    """
        Sends tensor to root process, which store it in tensor_list.
    """
  
    rank = distributed.get_rank()
    if group is None:
        group = distributed.group.WORLD
    if rank == root:
        assert(tensor_list is not None)
        distributed.gather(tensor, gather_list=tensor_list, group=group)
    else:
        distributed.gather(tensor, dst=root, group=group)


class LossLogger(EvalMetric):
    def __init__(self, output_name, display_name=None,
                 allreduce=False, num_replicas=1):
        self.output_name = output_name
        if display_name is None:
            display_name = output_name
        super(LossLogger, self).__init__(display_name, allreduce, num_replicas)

    def update(self, outputs):
        with torch.no_grad():
            if self.output_name in outputs:
                self.sum_metric += float(outputs[self.output_name].mean().item())
            self.num_inst += 1


class Accuracy(EvalMetric):
    def __init__(self, allreduce=False, num_replicas=1):
        super(Accuracy, self).__init__('Acc', allreduce, num_replicas)

    def update(self, outputs):
        with torch.no_grad():
            cls_logits = outputs['label_logits']
            label = outputs['label']
            bs, num_classes = cls_logits.shape
            if num_classes == 1:
                pred = (torch.sigmoid(cls_logits) > 0.5).float()
                match = torch.abs(label - pred) < 1e-4
                self.sum_metric = self.sum_metric + (match).float().sum()
            else:
                # self.sum_metric += float(label[:, cls_logits.argmax(1)].sum().item())
                if label.ndim == 2:
                    label = label.squeeze(dim=-1).long()
                match = torch.argmax(cls_logits, dim=1) == label
                match = match.sum().item()
                self.sum_metric += match
            self.num_inst += cls_logits.shape[0]


class RocAUC(EvalMetric):
    def __init__(self, allreduce=False, num_replicas=1):
        super(RocAUC, self).__init__('RocAUC', allreduce, num_replicas)
        self.labels = []
        self.predicts = []

    def update(self, outputs):
        with torch.no_grad():
            cls_logits = outputs['label_logits']
            label = outputs['label']
            bs, num_classes = cls_logits.shape
            if num_classes == 1:
                prob = torch.sigmoid(cls_logits)
                pred = (prob > 0.5).float()
                match = torch.abs(label - pred) < 1e-4
                self.sum_metric = self.sum_metric + (match).float().sum()
                
                for p in prob.cpu().flatten().numpy():
                    self.predicts.append(p)
                for p in label.cpu().flatten().numpy():
                    self.labels.append(p)
                    
            else:
                # self.sum_metric += float(label[:, cls_logits.argmax(1)].sum().item())
                match = torch.argmax(cls_logits, dim=1) == label
                match = match.sum().item()
                prob = torch.softmax(outputs['label_logits'], dim=-1)
                prob = prob[:, 1].detach().cpu().tolist()
                for p in prob:
                    self.predicts.append(p)
                for p in label.cpu().flatten().numpy():
                    self.labels.append(float(p))
            self.num_inst += cls_logits.shape[0]

    def reset(self):
        """Resets the internal evaluation result to initial state."""
        self.num_inst = torch.tensor(0.)
        self.sum_metric = torch.tensor(0.)
        self.labels = []
        self.predicts = []
    
    def extract_and_filter(self, pred_tensor):
        pred_list = pred_tensor.detach().cpu().tolist()
        return [p for p in pred_list if p >= -1e-4]

    def get(self):
        """Returns the current evaluation result.
        Returns:
            names (list of str): Name of the metrics.
            values (list of float): Value of the evaluations.
        """
        if self.num_inst.item() == 0:
            return (self.name, float('nan'))
        else:
            if self.allreduce:
                self.labels = torch.tensor(self.labels)
                self.predicts = torch.tensor(self.predicts)
                
                labels = self.labels.clone().cuda()
                predicts = self.predicts.clone().cuda()
                size = distributed.get_world_size()
                label_list = [-torch.ones_like(labels) for _ in range(size)]
                pred_list = [-torch.ones_like(predicts) for _ in range(size)]
                
                # if distributed.get_rank() == 0:
                #     gather(labels, label_list)
                #     gather(predicts, pred_list)
                # else:
                #     gather(labels)
                #     gather(predicts)
                distributed.all_gather(label_list, labels)
                distributed.all_gather(pred_list, predicts)

                if distributed.get_rank() == 0:
                    for s, (l, p) in enumerate(zip(label_list, pred_list)):
                        _l = self.extract_and_filter(l)
                        _p = self.extract_and_filter(p)
                        print(f"val subset - {s} roc_acu: {roc_auc_score(_l, _p)}")

                all_labels = self.extract_and_filter(torch.cat(label_list, dim=0))
                all_preds = self.extract_and_filter(torch.cat(pred_list, dim=0))
                
                print(f"[RANK {distributed.get_rank()}] Val sample size: {len(self.labels)}/ {len(all_preds)}")
                mertric = roc_auc_score(all_labels, all_preds)
                # metric_tensor = (sum_metric / num_inst).detach().cpu()
            else:
                # metric_tensor = (self.sum_metric / self.num_inst).detach().cpu()
                mertric = roc_auc_score(self.labels, self.predicts)
            return (self.name, mertric)

