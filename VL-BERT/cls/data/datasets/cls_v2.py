import os
import json
import re
import base64
import numpy as np
import csv
import sys
import time
import pprint
import logging
import collections
import random

import torch
from torch.utils.data import Dataset
from external.pytorch_pretrained_bert import BertTokenizer
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast, AutoTokenizer
import _pickle as cPickle
from PIL import Image

from common.utils.zipreader import ZipReader
from common.utils.create_logger import makedirsExist

# from pycocotools.coco import COCO

csv.field_size_limit(sys.maxsize)
FIELDNAMES = ['image_id', 'image_w', 'image_h', 'num_boxes', 'boxes', 'features']


class CLS2(Dataset):
    
    def __init__(self, root_path=None, image_set='train',
                 transform=None, test_mode=False,
                 zip_mode=False, cache_mode=False, cache_db=True,
                 tokenizer=None, pretrained_model_name=None,
                 add_image_as_a_box=False, mask_size=(14, 14),
                 aspect_grouping=False, **kwargs):
        """
        Visual Question Answering Dataset

        :param root_path: root path to cache database loaded from annotation file
        :param data_path: path to vcr dataset
        :param transform: transform
        :param test_mode: test mode means no labels available
        :param zip_mode: reading images and metadata in zip archive
        :param cache_mode: cache whole dataset to RAM first, then __getitem__ read them from RAM
        :param ignore_db_cache: ignore previous cached database, reload it from annotation file
        :param tokenizer: default is BertTokenizer from pytorch_pretrained_bert
        :param add_image_as_a_box: add whole image as a box
        :param mask_size: size of instance mask of each object
        :param aspect_grouping: whether to group images via their aspect
        :param kwargs:
        """
        super(CLS2, self).__init__()
        cache_dir = False
        assert not cache_mode, 'currently not support cache mode!'

        categories = ['__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
                      'boat',
                      'trafficlight', 'firehydrant', 'stopsign', 'parkingmeter', 'bench', 'bird', 'cat', 'dog', 'horse',
                      'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
                      'suitcase', 'frisbee', 'skis', 'snowboard', 'sportsball', 'kite', 'baseballbat', 'baseballglove',
                      'skateboard', 'surfboard', 'tennisracket', 'bottle', 'wineglass', 'cup', 'fork', 'knife', 'spoon',
                      'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hotdog', 'pizza', 'donut',
                      'cake', 'chair', 'couch', 'pottedplant', 'bed', 'diningtable', 'toilet', 'tv', 'laptop', 'mouse',
                      'remote', 'keyboard', 'cellphone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book',
                      'clock', 'vase', 'scissors', 'teddybear', 'hairdrier', 'toothbrush']
        self.category_to_idx = {c: i for i, c in enumerate(categories)}
        self.data_split = image_set # HACK: reuse old parameter

        self.periodStrip = re.compile("(?!<=\d)(\.)(?!\d)")
        self.commaStrip = re.compile("(\d)(\,)(\d)")
        self.punct = [';', r"/", '[', ']', '"', '{', '}',
                      '(', ')', '=', '+', '\\', '_', '-',
                      '>', '<', '@', '`', ',', '?', '!']

        self.test_mode = test_mode

        self.root_path = root_path
        
        self.box_bank = {}
        
        self.transform = transform
        self.zip_mode = zip_mode

        self.aspect_grouping = aspect_grouping
        self.add_image_as_a_box = add_image_as_a_box

        self.cache_dir = os.path.join(root_path, 'cache')
        # return_offsets_mapping
        model_name = 'bert-base-uncased' if pretrained_model_name is None else pretrained_model_name
        self.fast_tokenizer = AutoTokenizer.from_pretrained(
            'bert-base-uncased',
            cache_dir=self.cache_dir, use_fast=True, return_offsets_mapping=True)
        self.tokenizer = tokenizer if tokenizer is not None \
            else BertTokenizer.from_pretrained(
            model_name,
            cache_dir=self.cache_dir)
        self.max_txt_token = 120

        if zip_mode:
            self.zipreader = ZipReader()

        self.database = self.load_annotations()
        self.use_img_box = True
        self.random_drop_tags = False
        # if self.aspect_grouping:
        #     self.group_ids = self.group_aspect(self.database)

    @property
    def data_names(self):
        if self.use_img_box:
            if self.test_mode:
                return ['image', 'boxes', 'im_info', 'text',  'img_boxes', 'text_tags', 'id',]
            else:
                return ['image', 'boxes', 'im_info', 'text', 'img_boxes', 'text_tags', 'label', 'id']
        else:
            if self.test_mode:
                return ['image', 'boxes', 'im_info', 'text', 'id', ]
            else:
                return ['image', 'boxes', 'im_info', 'text', 'label', 'id']
    
    @property
    def weights_by_class(self):
        labels = []
        num_per_class = collections.defaultdict(lambda: 0)
        for data in self.database:
            labels.append(data['label'])
            num_per_class[data['label']] += 1
        
        weight_per_class = {k: 1 / len(num_per_class) / v for k, v in num_per_class.items()}
        sampling_weight = [weight_per_class[label] for label in labels]
        return sampling_weight

    def clip_box_and_score(self, box_and_score):
        new_list = []
        for box_sc in box_and_score:
            cliped = {k: min(max(v, 0), 1) for k, v in box_sc.items()}
            new_list.append(cliped)
        return new_list

    def __getitem__(self, index):
        idb = self.database[index]

        # image, boxes, im_info
        image = self._load_image(os.path.join(self.root_path, idb['img']))
        w0, h0 = image.size
        
        if len(idb['boxes_and_score']) == 0:
            boxes = torch.as_tensor([[0.0, 0.0, w0 - 1, h0 - 1, 0]])
        else:
            w_scale = w0 if idb['boxes_and_score'][0]['xmax'] < 1 else 1.0
            h_scale = h0 if idb['boxes_and_score'][0]['xmax'] < 1 else 1.0
            boxes = torch.as_tensor([
                [
                    box_sc['xmin'] * w_scale,
                    box_sc['ymin'] * h_scale,
                    box_sc['xmax'] * w_scale,
                    box_sc['ymax'] * h_scale,
                    box_sc['class_id'],
                ]
                for box_sc in idb['boxes_and_score']
            ])
            if self.add_image_as_a_box:
                boxes = torch.cat(
                    (torch.as_tensor([[0.0, 0.0, w0 - 1, h0 - 1, 0]]), boxes), dim=0)

        im_info = torch.tensor([w0, h0, 1.0, 1.0])
        # clamp boxes
        w = im_info[0].item()
        h = im_info[1].item()
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(min=0, max=w - 1)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(min=0, max=h - 1)
        
        flipped = False
        if self.transform is not None:
            image, boxes, _, im_info, flipped = self.transform(image, boxes, None, im_info, flipped)

        # question
        if 'token_id' not in idb:
            main_txt = idb['text']
            img_tags = [' '.join(des) for des in idb['partition_description']]
            img_tags_str = ''
            img_tags_part = []

            if not self.random_drop_tags or (self.random_drop_tags and random.random() > 0.5):
                for p, img_tag in enumerate(img_tags):
                    append_str = img_tag + ' '
                    img_tags_str += append_str
                    img_tags_part += [p] * len(append_str)

            text_with_tag = f"{main_txt} [SEP] {img_tags_str}"
            # print(f"[{index}] {text_with_tag}")
            result = self.fast_tokenizer(
                text_with_tag, return_offsets_mapping=True, add_special_tokens=False)
            token_id = result['input_ids']
            token_offset = result['offset_mapping']

            if self.use_img_box:
                text_partition = idb['text_char_partition_id']
                text_partition += [0] * len(" [SEP] ") + img_tags_part  # additinoal partition id for [SEP]
                assert len(text_partition) == len(text_with_tag), \
                    F"{len(text_partition)} != {len(text_with_tag)}"

                token_tags = []
                for a, b in filter(lambda x: x[1] - x[0] > 0, token_offset):
                    char_tags = text_partition[a: b]
                    # print(a, b, char_tags)
                    cnt = collections.Counter(char_tags)
                    token_tags.append(cnt.most_common(1)[0][0])
                
                idb['text_tags'] = token_tags
                idb['image_partition'] = np.asarray(idb['image_partition'], dtype=np.float32)[..., :4]  # HACK: remove det score from mmdet
            else:
                idb['text_tags'] = [0] * len(token_id)

            # token_id = self.tokenizer.convert_tokens_to_ids(text_tokens)
            if token_id[-1] == self.fast_tokenizer.sep_token_id:
                token_id = token_id[:-1]
                idb['text_tags'] = idb['text_tags'][:-1]
            
            if len(token_id) > self.max_txt_token:
                token_id = token_id[:self.max_txt_token]
                idb['text_tags'] = idb['text_tags'][:self.max_txt_token]

            idb['token_id'] = token_id
            assert len(idb['token_id']) == len(idb['text_tags'])
        else:
            token_id = idb['token_id']
        
        if self.use_img_box:
            if self.test_mode:
                return (
                    image, boxes, im_info, token_id,
                    idb['image_partition'], idb['text_tags'], idb['id'],
                )
            else:
                # print([(self.answer_vocab[i], p.item()) for i, p in enumerate(label) if p.item() != 0])
                label = torch.Tensor([float(idb['label'])])
                return (
                    image, boxes, im_info, token_id, 
                    idb['image_partition'], idb['text_tags'],
                    label, idb['id'],
                )
        else:
            if self.test_mode:
                return image, boxes, im_info, token_id, idb['id']
            else:
                # print([(self.answer_vocab[i], p.item()) for i, p in enumerate(label) if p.item() != 0])
                label = torch.Tensor([float(idb['label'])])
                return image, boxes, im_info, token_id, label, idb['id']

    @staticmethod
    def b64_decode(string):
        return base64.decodebytes(string.encode())

    def load_annotations(self):
        tic = time.time()
        img_name_to_annos = collections.defaultdict(list)
        
        test_json = os.path.join(self.root_path, 'test_unseen.entity.jsonl')
        dev_json = os.path.join(self.root_path, 'dev_seen.entity.jsonl')
        dev_train_json = os.path.join(self.root_path, 'dev_all.entity.jsonl')
        train_json = os.path.join(self.root_path, 'train.entity.jsonl')
        # box_annos_json = os.path.join(self.root_path, 'clean_img_boxes_gqa.json')
        box_annos_json = os.path.join(self.root_path, 'box_annos.json')
        
        test_sample = []
        dev_sample = []
        train_sample = []
        dev_train_sample = []
        
        with open(train_json, mode='r') as f:
            for line in f.readlines():
                train_sample.append(json.loads(line))
        
        with open(dev_train_json, mode='r') as f:
            for line in f.readlines():
                dev_train_sample.append(json.loads(line))

        with open(test_json, mode='r') as f:
            for line in f.readlines():
                test_sample.append(json.loads(line))
        
        with open(dev_json, mode='r') as f:
            for line in f.readlines():
                dev_sample.append(json.loads(line))
        
        with open(box_annos_json, mode='r') as f:
            box_annos = json.load(f)

        sample_sets = []
        if self.data_split == 'train':
            sample_sets.append(train_sample)
        elif self.data_split == 'val':
            sample_sets.append(dev_sample)
        elif self.data_split == 'train+val':
            sample_sets.append(train_sample)
            sample_sets.append(dev_train_sample)
        elif self.data_split == 'test':
            sample_sets.append(test_sample)
        else:
            raise RuntimeError(f"Unknown dataset split: {self.data_split}")

        for sample_set in sample_sets:
            for sample in sample_set:
                img_name = os.path.basename(sample['img'])
                img_name_to_annos[img_name].append(sample)           
        
        for box_anno in box_annos:
            img_name = box_anno['img_name']
            if img_name in img_name_to_annos:
                for sample in img_name_to_annos[img_name]:
                    sample.update(box_anno)

        print('Done (t={:.2f}s)'.format(time.time() - tic))

        flatten = []
        for annos in img_name_to_annos.values():
            flatten += annos
        return flatten

    @staticmethod
    def group_aspect(database):
        print('grouping aspect...')
        t = time.time()

        # get shape of all images
        widths = torch.as_tensor([idb['width'] for idb in database])
        heights = torch.as_tensor([idb['height'] for idb in database])

        # group
        group_ids = torch.zeros(len(database))
        horz = widths >= heights
        vert = 1 - horz
        group_ids[horz] = 0
        group_ids[vert] = 1

        print('Done (t={:.2f}s)'.format(time.time() - t))

        return group_ids

    def load_precomputed_boxes(self, box_file):
        if box_file in self.box_bank:
            return self.box_bank[box_file]
        else:
            in_data = {}
            with open(box_file, "r") as tsv_in_file:
                reader = csv.DictReader(tsv_in_file, delimiter='\t', fieldnames=FIELDNAMES)
                for item in reader:
                    item['image_id'] = int(item['image_id'])
                    item['image_h'] = int(item['image_h'])
                    item['image_w'] = int(item['image_w'])
                    item['num_boxes'] = int(item['num_boxes'])
                    for field in (['boxes', 'features'] if self.with_precomputed_visual_feat else ['boxes']):
                        item[field] = np.frombuffer(base64.decodebytes(item[field].encode()),
                                                    dtype=np.float32).reshape((item['num_boxes'], -1))
                    in_data[item['image_id']] = item
            self.box_bank[box_file] = in_data
            return in_data

    def __len__(self):
        return len(self.database)

    def _load_image(self, path):
        if '.zip@' in path:
            return self.zipreader.imread(path).convert('RGB')
        else:
            return Image.open(path).convert('RGB')

    def _load_json(self, path):
        if '.zip@' in path:
            f = self.zipreader.read(path)
            return json.loads(f.decode())
        else:
            with open(path, 'r') as f:
                return json.load(f)


if __name__ == '__main__':
    import numpy as np
    from loguru import logger
    
    def hight_light_boxes(data):
        image, boxes = data[:2]
        boxes = boxes.numpy()
        np_img = np.array(image).astype(np.int32)
        
        for box in boxes:
            x_slice = slice(int(box[0]), int(box[2]))
            y_slice = slice(int(box[1]), int(box[3]))
            np_img[y_slice, x_slice] += 20
            print(box)
        np_img = np.clip(np_img, 0, 255).astype(np.uint8)
        return np_img

    with logger.catch(reraise=True):
        pretrained_model_name = '/home/ron/Projects/VL-BERT/model/pretrained_model/bert-large-uncased'
        cls_data = CLS2(
            '/home/ron/Downloads/hateful_meme_data/',
            pretrained_model_name=pretrained_model_name
        )
        
        print(cls_data[0])
        tk_len = []
        for i in range(len(cls_data)):
            if i % 100 == 0:
                print(i)
            tmp = cls_data[i]
            print(tmp[4].shape)
            tk_len.append(len(tmp[3]))
        import pdb; pdb.set_trace()
        Image.fromarray(hight_light_boxes(cls_data[3])).show()
