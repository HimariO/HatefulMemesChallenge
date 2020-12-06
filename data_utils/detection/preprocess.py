import os
import glob
import math
import json
import shutil
import random
from functools import reduce
from collections import defaultdict

import fire
import numpy as np
from loguru import logger
from PIL import Image, UnidentifiedImageError

import visual_genome as vg
import gqa

DEFAULT_IMG_PREFIX = 'vg_'

common_attributes = set(['white','black','blue','green','red','brown','yellow',
    'small','large','silver','wooden','orange','gray','grey','metal','pink','tall',
    'long','dark'])


def split_anno_json(anno_json, output_dir, val_ratio=0.05):
    with open(anno_json, mode='r') as f:
        annos = json.load(f)
    os.makedirs(output_dir, exist_ok=True)
    random.shuffle(annos)

    val_size = math.ceil(len(annos) * val_ratio)
    train_size = len(annos) - val_size
    train_json = anno_json.replace('.json', '.train.json')
    val_json = anno_json.replace('.json', '.val.json')
    
    with open(train_json, mode='w') as f:
        json.dump(annos[:train_size], f)
    with open(val_json, mode='w') as f:
        json.dump(annos[train_size:], f)


def gqa_dataset_mmdet(scence_graph, output_path):
    with open(scence_graph, mode='r') as j:
        gqa_graph = json.load(j)
    
    all_anno = []
    for img_id, graph in gqa_graph.items():
        objs = graph['objects']
        img_annos = {
            'image_name': f"{img_id}.jpg",
            'height': graph['height'],
            'width': graph['width'],
            'boxes': [],
            'classes': [],
            'class_id': [],
            'attributes': [],
            'attribute_idx': [],
            'scores': None,
            'attr_scores': None
        }

        for obj_anno in objs.values():
            
            if obj_anno['w'] <= 8 or obj_anno['h'] <= 8:
                continue
            w = obj_anno['w']
            h = obj_anno['h']
            if obj_anno['name'] in gqa.CLASSES:
                obj_box = [
                    max(min(obj_anno['x'], graph['width']), 0),
                    max(min(obj_anno['y'], graph['height']), 0),
                    max(min(obj_anno['x'] + obj_anno['w'], graph['width']), 0),
                    max(min(obj_anno['y'] + obj_anno['h'], graph['height']), 0)
                ]
                if (obj_box[2] - obj_box[0]) <= 8 or (obj_box[3] - obj_box[1]) <= 8:
                    continue
                img_annos['boxes'].append(obj_box)
                img_annos['classes'].append(obj_anno['name'])
                img_annos['class_id'].append(gqa.CLASSES.index(obj_anno['name']))
                img_annos['attributes'].append(obj_anno['attributes'])
                img_annos['attribute_idx'].append([
                    gqa.ATTRS.index(attr) for attr in obj_anno['attributes']])
                
                assert all([atr >= 0 for atr in img_annos['attribute_idx'][-1]])
        assert all([cid >= 0 for cid in img_annos['class_id']])
        all_anno.append(img_annos)
    
    with open(output_path, mode='w') as f:
        json.dump(all_anno, f)


def generate_cliping_dataset(img_dir, output_dir, target_number=10000):

    def rand_layout(num):
        r = random.random()
        layout = {
            'layout': None,
            'aligns': []
        }
        if r > 0.4:
            layout['layout'] = 'top-down'
        else:
            layout['layout'] = 'left-right'
        layout['aligns'] = [random.choice([0, 1, 2]) for _ in range(num)]
        # NOTE: 0: left/top, 1: mid, 2: right/bottom  for top-down and left-right
        return layout

    os.makedirs(output_dir, exist_ok=True)
    img_list = glob.glob(os.path.join(img_dir, '*.jpg'))
    annotations = []
    sample_idx = list(range(target_number))

    for i in range(target_number):
        try:
            n = random.randint(1, 3)
            random.shuffle(sample_idx)
            src_imgs = [
                np.asarray(Image.open(img_list[j]).convert('RGB'))
                for j in sample_idx[:n]
            ]
            out_name = f'{i}.jpg'
            out_path = os.path.join(output_dir, out_name)
            sample_anno = {
                'image_name': out_name,
                'boxes': [],
                'classes': ['image'] * n,
                'class_id': [0] * n,
            }

            layout = rand_layout(n)
            bg_color = random.choice([
                [0, 0, 0],
                [255, 255, 255],
                [128, 128, 128],
                [172, 172, 172],
                [64, 64, 64],
            ])
            
            print(f"[{i}/{target_number}]")
            print([img_list[j] for j in sample_idx[:n]])
            print(layout)
            print('-' * 100)
            
            if layout['layout'] == 'top-down':
                w = max([img.shape[1] for img in src_imgs])
                h = sum([img.shape[0] for img in src_imgs])
                offset_y = 0

                sample_anno['width'] = w
                sample_anno['height'] = h
                
                canvas = np.zeros([h, w, 3], dtype=np.uint8)
                canvas[:, :] = bg_color
                
                for im, align in zip(src_imgs, layout['aligns']):
                    imh, imw = im.shape[:2]
                    y = offset_y
                    if align == 0:
                        x = 0
                    elif align == 1:
                        x = (w - imw) // 2
                    elif align == 2:
                        x = (w - imw)
                    
                    canvas[y: y + imh, x: x + imw, :] = im
                    sample_anno['boxes'].append([x, y, x + imw, y + imh])
                    offset_y += imh
                Image.fromarray(canvas).save(out_path)
            elif layout['layout'] == 'left-right':
                w = sum([img.shape[1] for img in src_imgs])
                h = max([img.shape[0] for img in src_imgs])
                offset_x = 0

                sample_anno['width'] = w
                sample_anno['height'] = h

                canvas = np.zeros([h, w, 3], dtype=np.uint8)
                canvas[:, :] = bg_color
                
                for im, align in zip(src_imgs, layout['aligns']):
                    imh, imw = im.shape[:2]
                    x = offset_x
                    if align == 0:
                        y = 0
                    elif align == 1:
                        y = (h - imh) // 2
                    elif align == 2:
                        y = (h - imh)
                    
                    canvas[y: y + imh, x: x + imw, :] = im
                    sample_anno['boxes'].append([x, y, x + imw, y + imh])
                    offset_x += imw
                Image.fromarray(canvas).save(out_path)
            annotations.append(sample_anno)
        except UnidentifiedImageError as e:
            pass
    
    anno_path = os.path.join(output_dir, 'annotation.json')
    with open(anno_path, mode='w') as f:
        json.dump(annotations, f)



if __name__ == "__main__":
    with logger.catch():
        fire.Fire({
            'split_anno_json': split_anno_json,
            'gqa_dataset_mmdet': gqa_dataset_mmdet,
            'generate_cliping_dataset': generate_cliping_dataset,
        })