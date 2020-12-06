import os
import glob
import shutil

import fire
import spacy
import pandas as pd
from sklearn.cluster import KMeans
nlp = spacy.load("en_core_web_lg")


import os
import glob
import json
import shutil
import pickle

import cv2
import numpy as np


from loguru import logger
from termcolor import colored
import spacy
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

nlp = spacy.load("en_core_web_lg")


def centroid_histogram(clt):
    # grab the number of different clusters and create a histogram
    # based on the number of pixels assigned to each cluster
    numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
    (hist, _) = np.histogram(clt.labels_, bins = numLabels)
    # normalize the histogram, such that it sums to one
    hist = hist.astype("float")
    hist /= hist.sum()
    # return the histogram
    return hist


def plot_colors(hist, centroids):
	# initialize the bar chart representing the relative frequency
	# of each of the colors
	bar = np.zeros((50, 300, 3), dtype = "uint8")
	startX = 0
	# loop over the percentage of each cluster and the color of
	# each cluster
	for (percent, color) in zip(hist, centroids):
		# plot the relative percentage of each cluster
		endX = startX + (percent * 300)
		cv2.rectangle(
            bar, (int(startX), 0), (int(endX), 50),
			color.astype("uint8").tolist(), -1
        )
		startX = endX
	
	# return the bar chart
	return bar


def rasicm_det(meme_anno_path, box_anno_json, img_dir, check_skin_tone=True):
    
    meme_anno = {}
    with open(meme_anno_path, 'r') as f:
        for l in f:
            data = json.loads(l)
            meme_anno[data['id']] = data
    
    with open(box_anno_json, 'r') as f:
        box_anno = json.load(f)

    box_anno_map = {
        int(a['img_name'].replace('.png', '')): a 
        for a in box_anno
    }
    keyword = ['crime', 'hang', 'rob', 'steal', 'jail', 'prison', 'slave', 'apes', 'criminal', 'gorilla']
    keyword_tok = list(nlp(' '.join(keyword)))

    cnt = 0
    rasicm_sample_idx = []
    
    for i, (id, anno) in enumerate(meme_anno.items()):
        boxes = box_anno_map[id]
        box_cls = [b['class_name'].lower() for b in boxes['boxes_and_score']]
        race_boxes = [b for b in boxes['boxes_and_score'] if b['race']]
        face_boxes = [b for b in boxes['boxes_and_score'] if b['class_name'].lower() == "human face"]
        blacks = [b for b in race_boxes if b['race'].lower() == 'black']
        
        match = any([
            any([token.similarity(kwt) > 0.6 for kwt in keyword_tok])
            for token in nlp(anno['text'])
        ])
        not_mat = not ('monkey' in box_cls)

        if len(blacks) == len(race_boxes) and len(blacks) > 0 and (match and not_mat):
            print(colored(f"[{cnt}]", color='green'))
            cnt += 1
            not_skin_color = False

            if check_skin_tone:
                img_path = os.path.join(img_dir, os.path.basename(anno['img']))
                im = cv2.imread(img_path)
                im_hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV_FULL)
                im_rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                imh, imw, _ = im.shape

                print(anno['img'], f" {len(blacks)} black face")
                for bbox in face_boxes:
                    center = [
                        (bbox['ymin'] + bbox['ymax']) / 2,
                        (bbox['xmin'] + bbox['xmax']) / 2,
                    ]
                    ymin = (bbox['ymin'] * 0.5 + center[0] * 0.5) * imh 
                    ymax = (bbox['ymax'] * 0.5 + center[0] * 0.5) * imh
                    xmin = (bbox['xmin'] * 0.5 + center[1] * 0.5) * imw
                    xmax = (bbox['xmax'] * 0.5 + center[1] * 0.5) * imw
                    yslice = slice(int(ymin), int(ymax))
                    xslice = slice(int(xmin), int(xmax))

                    face_crop = im_hsv[yslice, xslice, ...]
                    print('crop h, v channel mean: ', face_crop[..., 0].mean(), face_crop[..., 1].mean())
                    
                    clt = KMeans(n_clusters = 5)
                    clt.fit(im_hsv[yslice, xslice, ...].reshape([-1, 3]))
                    hist = centroid_histogram(clt)
                    top1 = hist.argmax()
                    
                    if 160 > clt.cluster_centers_[top1, 0] > 30:
                        not_skin_color = True
                        break

            if not not_skin_color:
                rasicm_sample_idx.append(id)
    print(f"Find {len(rasicm_sample_idx)} rasicm meme:", rasicm_sample_idx)
    return rasicm_sample_idx



def merge(dfs):
    return sum([df.proba.values for df in dfs]) / len(dfs)


def get_mean_predict(root_dir, out_path):
    ernie_csv = os.path.join(root_dir, 'checkpoints', 'ernie-vil', '**', 'test_set.csv')
    vl_bert_csv = os.path.join(root_dir, 'checkpoints', 'vl-bert', '**', '*_test.csv')
    uniter_csv = os.path.join(root_dir, 'checkpoints', 'uniter', '**', 'test.csv')
    
    csv_list = glob.glob(ernie_csv)
    csv_list += glob.glob(vl_bert_csv)
    csv_list += glob.glob(uniter_csv)
    print(f"Found {len(csv_list)} csv eval result!")

    gather_dir = os.path.join(root_dir, 'test_set_csvs')
    if os.path.exists(gather_dir):
        shutil.rmtree(gather_dir)
    os.makedirs(gather_dir, exist_ok=True)

    ensem_list = []
    All = False
    for csv_file in csv_list:
        # print(csv_file)
        if not All:
            yn = input(f"Include {csv_file} to ensemble? (y/n/all)")
        else:
            yn = 'y'
        yn = yn.strip().lower()
        if yn == 'all':
            All = True
        
        if yn == 'y' or All:
            ensem_list.append(csv_file)
            dir_name = os.path.basename(os.path.dirname(csv_file))
            shutil.copy(
                csv_file,
                os.path.join(
                    gather_dir,
                    f"{dir_name}_{os.path.basename(csv_file)}"
                )
            )
    assert len(ensem_list) >= 2, f'You must select at least two file to ensemble, only {len(ensem_list)} is picked'
    
    base = pd.read_csv(ensem_list[0])
    print(len(ensem_list))
    ensem_list = [pd.read_csv(c) for c in ensem_list]
    base.proba = merge(ensem_list)

    rasicm_idx = rasicm_det(
        os.path.join(root_dir, 'data/hateful_memes/test_unseen.jsonl'),
        os.path.join(root_dir, 'data/hateful_memes/box_annos.race.json'),
        os.path.join(root_dir, 'data/hateful_memes/img_clean'),
    )
    for i in rasicm_idx:
        base.at[int(base.index[base['id']==i].values), 'proba'] = 1.0

    base.to_csv(out_path, index=False)

if __name__ == "__main__":
    fire.Fire(get_mean_predict)