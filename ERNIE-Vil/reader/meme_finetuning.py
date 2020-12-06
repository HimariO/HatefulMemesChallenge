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

""" VCR Data Reader implementation """

from __future__ import print_function
from __future__ import division

import os
import base64
import pdb
import numpy as np
import re
import random
import json
import json_lines
import csv
import sys
import itertools
from collections import Counter

from reader._image_features_reader import ImageFeaturesH5Reader, ImageFeaturesPtReader
from preprocess import preprocessor
from batching.finetune_batching import prepare_batch_data
from loguru import logger


def _load_annotations(annotations_jsonpath, split):
    """
    Build an index out of FOIL annotations, mapping each image ID with its corresponding captions.
    """
    entries = []
    with open(annotations_jsonpath) as f:
        for annotation in json_lines.reader(f):
            det_names = ""
            if split == 'test':
                ans_label = 0
            else:
                ans_label = annotation["label"]
            img_id = annotation["id"]
            entries.append({
                "text": annotation['text'],
                "target": ans_label,
                "img_id": img_id,
                "anno_id": img_id,
                "det_names": [],
            })
    return entries


class MemeDataReader(object):
    """ 
    Data reader for sub VCR task
    """
    def __init__(self,
                 task_conf,
                 split,
                 vocab_path=None,
                 batch_size=4096,
                 shuffle=True,
                 epoch=100,
                 is_test=False,
                 feature_reader_dict={},
                 random_seed=None,
                 task_index=0,
                 task_num=1):

        self.task_conf = task_conf
        self.processor = getattr(preprocessor,
                                 task_conf["Proprocessor"])(
                                    tokenizer_name=self.task_conf["tokenizer_name"],
                                    vocab_path=vocab_path)
        self.vocab = self.processor.vocab
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.epoch = epoch
        self.current_epoch = 0
        self.current_file_index = 0
        self.total_file = 0
        self.current_file = None
        self.random_seed = random_seed
        self.max_seq_len = self.task_conf['max_seq_len']
        self.pad_id = self.vocab["[PAD]"]
        self.cls_id = self.vocab["[CLS]"]
        self.sep_id = self.vocab["[SEP]"]
        self.mask_id = self.vocab["[MASK]"]
        self.is_test = is_test
        self.task_index = task_index
        self.task_num = task_num
        self.global_rng = np.random.RandomState(random_seed)
        self.cache = {}

        if self.is_test:
            self.epoch = 1
            self.shuffle_files = False
        if self.shuffle:
            shufflekeep_across_task = self.task_conf.get('shufflekeep_across_task', True)
            # if shufflekeep_across_task:
            #     self.global_rng = np.random.RandomState(random_seed)
            # else:
            #     self.global_rng = np.random.RandomState()
            self.shuffle_every_epoch = self.task_conf.get('shuffle_every_epoch', False)
        task = self.task_conf['task']
        annotations_jsonpath = self.task_conf['annotations_jsonpath_' + split]
        logger.warning(f"Loading annotations {annotations_jsonpath}")
        self.num_choice = 1
        self._entries = _load_annotations(annotations_jsonpath, split)

        self._split = split
        
        # self._names = []
        # with open(self.task_conf['unisex_names_table']) as csv_file:
        #     csv_reader = csv.reader(csv_file, delimiter=',')
        #     for row in csv_reader:
        #         if row[1] != 'name':
        #             self._names.append(row[1])
        
        self._feature_reader = feature_reader_dict[self.task_conf['feature_lmdb_path']]
        self._max_region_num = self.task_conf.get('max_region_num', 37)
        print("only butd feature")
        self.tokenize()

    def get_progress(self):
        """
        Return current progress of traning data
        """
        progress_dict = {
            "current_epoch": self.current_epoch,
            "current_file_index": self.current_file_index,
            "total_file": self.total_file,
            "current_file": self.current_file
        }
        return progress_dict

    def tokenize(self):
        """
        Tokenizes the captions.
        """
        # This will add caption_tokens in each entry of the dataset.
        # -1 represents nil, and should be treated as padding_idx in embedding.
        count = 0
        for entry in self._entries:
            # det_names = entry["det_names"]
            # random_names = self.generate_random_name(det_names)
            # # replace with name
            # tokens_a, mask_a = self.replace_det_with_name(entry["question"], random_names)
            # q_str = " ".join(tokens_a)
            ids_a = self.processor.convert_sentence_to_ids_without_cls(entry['text'])

            input_ids_all = []
            segment_ids_all = []
            input_poss_all = []
            input_len_all = []

            for answer in [entry["text"]]:
                # self._truncate_seq_pair(ids_a, ids_b, self.max_seq_len - 3)
                ids_a = ids_a[:self.max_seq_len - 3]

                input_ids = []
                segment_ids = []
                input_ids.append(self.vocab["[CLS]"])
                segment_ids.append(0)

                for id in ids_a:
                    input_ids.append(id)
                    segment_ids.append(0)

                input_ids.append(self.vocab["[SEP]"])
                segment_ids.append(0)

                input_ids_all.append(input_ids)
                segment_ids_all.append(segment_ids)
                input_poss = [str(pos) for pos in range(len(input_ids))]
                input_poss_all.append(input_poss)
                input_len_all.append(len(input_ids))

            entry["input_ids"] = input_ids_all
            entry["input_poss"] = input_poss_all
            entry["segment_ids"] = segment_ids_all
            entry["input_lens"] = input_len_all

            sys.stdout.write('%d/%d\r' % (count, len(self._entries)))
            sys.stdout.flush()
            count += 1

    def parse_line(self, s_index):
        """
        Form slot info with the line information
        """
        entry = self._entries[s_index]
        image_id = entry["img_id"]
        image_fea_json = self._feature_reader[image_id]
        features = image_fea_json["features"]
        num_boxes = image_fea_json["num_boxes"]
        boxes = image_fea_json["image_location"]
        
        num_boxes = min(num_boxes, self._max_region_num)
        boxes = boxes[:num_boxes]
        features = features[:num_boxes]

        record = {
            "input_ids": entry["input_ids"],
            "input_pos": entry["input_poss"],
            "segment_ids": entry["segment_ids"],
            "input_lens": entry["input_lens"],
            "target": int(entry["target"]),
            "features": features,
            "boxes": boxes,
            "anno_id": entry["anno_id"]
        }
        return record
    
    def __len__(self):
        return len(self._entries)
    
    def __getitem__(self, index):
        try:
            return self.cache[index]
        except KeyError:
            self.cache[index] = self.parse_line(index)
            return self.cache[index]
    
    @property
    def weights_by_class(self):
        sample_indice = list(range(len(self._entries)))
        labels = [self.parse_line(i)['target'] for i in sample_indice]
        tar_to_id = {l: [] for l in labels}
        for i, l in zip(sample_indice, labels):
            tar_to_id[l].append(i)
        tar_to_w = {l: 1 / len(tar_to_id) / len(tar_to_id[l]) for l in labels}
        sample_weights = [tar_to_w[l] for l in labels]
        return sample_weights
    
    def balance_sample(self):
        sample_indice = list(range(len(self._entries)))
        labels = [self.parse_line(i)['target'] for i in sample_indice]
        sample_weights = self.weights_by_class
        
        resample_indice = self.global_rng.choice(sample_indice, p=sample_weights, size=len(sample_indice)).tolist()
        logger.info(f"balance_sample: {Counter([labels[i] for i in resample_indice])}")
        return resample_indice

    def data_generator(self, balance_cls=False):
        """ 
        Data_generator 
        """
        def wrapper():
            """
            Wrapper
            """
            sample_indice = list(range(len(self._entries)))
            for epoch_index in range(self.epoch):
                if self._split == "train":
                    self.current_example = 0
                    self.current_epoch = epoch_index
                    if balance_cls:
                        sample_indice = self.balance_sample()
                
                if self.shuffle:
                    if epoch_index == 0:
                        self.global_rng.shuffle(sample_indice)
                        print("shuffle epoch %d" % epoch_index)
                    elif self.shuffle_every_epoch:
                        self.global_rng.shuffle(sample_indice)
                        print("shuffle epoch %d" % epoch_index)
                
                batch_records = []
                for index in sample_indice:
                    batch_records.append(self.parse_line(index))
                    if len(batch_records) == self.batch_size:
                        yield prepare_batch_data(
                            batch_records, self.num_choice, self.pad_id, \
                            self.task_index, self.task_num), self.task_conf['task']
                        batch_records = []
                
                if len(batch_records) > 0:
                    yield prepare_batch_data(
                        batch_records, self.num_choice, self.pad_id, \
                        self.task_index, self.task_num), self.task_conf['task']
        return wrapper


class MemeDataJointReader(object):
    """ 
    Joint data reader for Q2A task and QA2R task
    """
    def __init__(self,
                 task_conf_group,
                 split,
                 batch_size=4096,
                 shuffle=True,
                 epoch=100,
                 vocab_path=None,
                 is_test=False,
                 random_seed=np.random.randint(1000),
                 balance_cls=False,
                 use_aug=True):
        self.balance_cls = balance_cls
        self.task_readers = []
        feature_reader_dict = {}
        self.task_dup_cnt = []
        for task_conf in task_conf_group:
            
            if 'feature_lmdb_path' in task_conf:
                if task_conf['feature_lmdb_path'] not in feature_reader_dict:
                    feature_reader_dict[task_conf['feature_lmdb_path']] =    \
                        ImageFeaturesPtReader(task_conf['feature_lmdb_path'], split, use_aug=use_aug)
            
            task_batch_size = task_conf.get('batch_size', 64)
            self.task_dup_cnt.append(max(int(task_batch_size / batch_size), 1))
        
        for task_index, task_conf in enumerate(task_conf_group):
            self.task_readers.append(
                MemeDataReader(
                    task_conf, split, vocab_path, batch_size, shuffle,
                    epoch, is_test, feature_reader_dict, random_seed,
                    task_index, len(task_conf_group)
                )
            )
        self.task_generators = [
            reader.data_generator(balance_cls=self.balance_cls)
            for reader in self.task_readers
        ]

    def get_progress(self):
        """
        Return current progress of traning data
        """
        current_epoch = max([reader.current_epoch for reader in self.task_readers])
        current_file_index = max([reader.current_file_index for reader in self.task_readers])
        total_file = max([reader.total_file for reader in self.task_readers])
        current_file = ""
        self.progress_dict = {
            "current_epoch": current_epoch,
            "current_file_index": current_file_index,
            "total_file": total_file,
            "current_file": current_file
        }
        return self.progress_dict

    def data_generator(self):
        """ 
        Data_generator 
        """
        def wrapper():
            """
            warpper
            """
            task_buffer = [[] for i in range(len(self.task_dup_cnt))]
            generators = [generator() for generator in self.task_generators]
            for data in zip(*generators):
                for i, d in enumerate(data):
                    task_buffer[i].append(d)
                    
                    if len(task_buffer[i]) >= self.task_dup_cnt[i]:
                        for t in task_buffer[i]:
                            yield t[0]
                        task_buffer[i] = []
                
                if len(task_buffer[i]) >=0:
                    # import pdb; pdb.set_trace()
                    for t in task_buffer[i]:
                        yield t[0]
                    task_buffer[i] = []

        # import pdb; pdb.set_trace()
        return wrapper


if __name__ == "__main__":
    pass
