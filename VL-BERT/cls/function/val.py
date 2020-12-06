import os
import json
import torch
import pandas as pd
from collections import namedtuple
from common.trainer import to_cuda


@torch.no_grad()
def do_validation(net, val_loader, metrics, label_index_in_batch, model_dir=None, epoch_num=0):
    net.eval()
    metrics.reset()

    predicts = []
    for nbatch, batch in enumerate(val_loader):
        batch = to_cuda(batch)
        label = batch[label_index_in_batch]
        datas = [batch[i] for i in range(len(batch)) if i != label_index_in_batch % len(batch)]

        outputs = net(*datas)
        outputs.update({'label': label})
        metrics.update(outputs)

        idx = batch[-1].cpu().tolist()
        if outputs['label_logits'].shape[-1] == 1:
            prob = torch.sigmoid(outputs['label_logits'][:, 0]).detach().cpu().tolist()
        else:
            prob = torch.softmax(outputs['label_logits'], dim=-1)[:, 1].detach().cpu().tolist()
        
        if label.ndim == 2:
            if label.shape[-1] == 1:
                label = label.squeeze(dim=-1)
        label = label.cpu().tolist()
        for pb, id, lb in zip(prob, idx, label):
            predicts.append({
                'id': int(id),
                'proba': float(pb),
                'label': int(pb > 0.5),
                'target': lb,
                'error': abs(float(lb) - float(pb))
            })
        
    if model_dir is not None:
        output_path = os.path.join(model_dir, f'val_{epoch_num}.json')
        with open(output_path, 'w') as f:
            json.dump(predicts, f)
        print('>>> do_validation result JSON saved to {}.'.format(output_path))
        
        output_path = os.path.join(model_dir, f'val_{epoch_num}.csv')
        result_pd = pd.DataFrame.from_dict(predicts)
        result_pd.to_csv(output_path, index=False)
        print('>>> do_validation result CSV saved to {}.'.format(output_path))

