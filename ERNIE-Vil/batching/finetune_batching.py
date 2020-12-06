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

""" prepare data format for finetuning tasks """

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import numpy as np

from six.moves import xrange


def prepare_batch_data(batch_records, num_choice=1, pad_id=0, task_index=0, task_num=1, to_tensor=False):
    """
    prepare batch data for finetuning tasks
    """
    batch_input_ids = []
    batch_input_pos = []
    batch_seg_ids = []
    batch_input_masks = []
    num_sample = len(batch_records)
    batch_lens = [record["input_lens"] for record in batch_records]
    
    batch_labels = [record["target"] for record in batch_records]
    if to_tensor:
        labels = np.array(batch_labels).astype("int64").reshape([-1])
    else:
        labels = np.array(batch_labels).astype("int64").reshape([-1, 1])
    
    binary_labels = np.zeros([(num_choice) * num_sample, 1], dtype='float32')
    if num_choice != 1:
        for i, l in enumerate(batch_labels):
            binary_labels[i * num_choice + l] = 1.0
    
    image_features = [record["features"] for record in batch_records]
    image_boxes = [record["boxes"] for record in batch_records]
    batch_anno_ids = np.array([record["anno_id"] for record in batch_records]).astype("int64").reshape([-1, 1])
    max_len = max([max(lens) for lens in batch_lens])
    
    for i in range(len(batch_records)):
        batch_input_ids.append([inst + list([pad_id] * (max_len - len(inst))) \
            for inst in batch_records[i]["input_ids"]])
        batch_input_pos.append([inst + list([pad_id] * (max_len - len(inst))) \
            for inst in batch_records[i]["input_pos"]])
        batch_seg_ids.append([inst + list([pad_id] * (max_len - len(inst)))   \
            for inst in batch_records[i]["segment_ids"]])
        batch_input_masks.append([[1] * len(inst) + [0] * (max_len - len(inst))    \
            for inst in batch_records[i]["input_ids"]])

    image_embedding, image_mask = pad_feature_data(image_features, return_mask=True)
    
    if to_tensor:
        src_ids = np.array(batch_input_ids).astype("int64").reshape([num_choice * num_sample, max_len])
        src_pos = np.array(batch_input_pos).astype("int64").reshape([num_choice * num_sample, max_len])
        src_seg = np.array(batch_seg_ids).astype("int64").reshape([num_choice * num_sample, max_len])
    else:
        src_ids = np.array(batch_input_ids).astype("int64").reshape([num_choice * num_sample, max_len, 1])
        src_pos = np.array(batch_input_pos).astype("int64").reshape([num_choice * num_sample, max_len, 1])
        src_seg = np.array(batch_seg_ids).astype("int64").reshape([num_choice * num_sample, max_len, 1])
    src_masks = np.array(batch_input_masks).astype("float32").reshape([num_choice * num_sample, max_len, 1])
    src_task = np.zeros(src_ids.shape, dtype="int64")
    
    batch, seq_len, fea_len = image_embedding.shape
    image_embedding = np.tile(np.expand_dims(image_embedding, axis=1),    \
        (1, num_choice, 1, 1)).reshape([num_choice * batch, seq_len, fea_len])
    image_mask = np.tile(np.expand_dims(image_mask, axis=1),        \
        (1, num_choice, 1, 1)).reshape([num_choice * batch, seq_len, 1])
    
    image_loc = pad_feature_data(image_boxes)
    image_loc = np.tile(np.expand_dims(image_loc, axis=1),     \
        (1, num_choice, 1, 1)).reshape([num_choice * batch, seq_len, 5])
    
    return_list = [
        src_ids, src_pos, src_seg, src_task, src_masks,
        image_embedding, image_loc, image_mask, labels, batch_anno_ids
    ]
    return_list.append(np.array([task_index]).astype('int64'))
    return_list.append(binary_labels)
    for i in xrange(task_num):
        if i == task_index:
            return_list.append(np.array([1.0]).astype("float32"))
        else:
            return_list.append(np.array([0.0]).astype("float32"))
    if to_tensor:
        return_list = [torch.tensor(value) for value in return_list]
    return return_list


def pad_feature_data(data, pad_value=0.0, dtype="float32", return_mask=False):
    """
    pad visual features with given pad value
    """
    max_lenth=max([len(item) for item in data])
    data_width = len(data[0][0])
    out_data = np.ones((len(data), max_lenth, data_width), dtype=dtype) * pad_value
    out_mask = np.zeros((len(data), max_lenth, 1), dtype=dtype)
    for i in range(len(data)):
        out_data[i, 0: len(data[i]), :] = data[i]
        if return_mask:
            out_mask[i, 0:len(data[i]):] = 1.0
    if return_mask:
        return out_data, out_mask
    else:
        return out_data

if __name__ == "__main__":
    pass
