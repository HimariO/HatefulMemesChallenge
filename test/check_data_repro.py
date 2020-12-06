import os
import glob
import random
import json

import cv2
import fire
import numpy as np

HERE = os.path.dirname(__file__)
DATA = os.path.join(HERE, '../data')


def check_cleaned_image(ref_dir, tar_dir=os.path.join(DATA, 'hateful_memes/img_clean')):
    assert ref_dir != tar_dir
    assert os.path.exists(ref_dir)
    assert os.path.exists(tar_dir)
    ref_imgs = glob.glob(os.path.join(ref_dir, '*.png'))
    tar_imgs = glob.glob(os.path.join(tar_dir, '*.png'))
    assert len(ref_imgs) == len(tar_imgs), f"{len(ref_imgs)} != {len(tar_imgs)}"
    ref_name_to_imgs = {os.path.basename(p): p for p in ref_imgs}
    tar_name_to_imgs = {os.path.basename(p): p for p in tar_imgs}
    img_names = list(ref_name_to_imgs.keys())

    for img_name in random.choices(img_names, k=100):
        ref_img = ref_name_to_imgs[img_name]
        tar_img = tar_name_to_imgs[img_name]
        im1 = cv2.imread(ref_img)
        im2 = cv2.imread(tar_img)
        error = np.abs(im1 - im2)
        print(img_name, error.mean(), error.max())


def check_ocr_mask(ref_dir, tar_dir=os.path.join(DATA, 'hateful_memes/img_mask_3px')):
    assert ref_dir != tar_dir
    assert os.path.exists(ref_dir)
    assert os.path.exists(tar_dir)
    ref_imgs = glob.glob(os.path.join(ref_dir, '*.png'))
    tar_imgs = glob.glob(os.path.join(tar_dir, '*.png'))
    assert len(ref_imgs) == len(tar_imgs), f"{len(ref_imgs)} != {len(tar_imgs)}"
    ref_name_to_imgs = {os.path.basename(p): p for p in ref_imgs}
    tar_name_to_imgs = {os.path.basename(p): p for p in tar_imgs}
    img_names = list(ref_name_to_imgs.keys())

    for img_name in random.choices(img_names, k=100):
        ref_img = ref_name_to_imgs[img_name]
        tar_img = tar_name_to_imgs[img_name]
        im1 = cv2.imread(ref_img)
        im2 = cv2.imread(tar_img)
        error = np.abs(im1 - im2)
        print(img_name, error.mean(), error.max())


def check_box_annos(ref_file, tar_file=os.path.join(DATA, "hateful_memes/box_annos.json")):
    assert ref_file != tar_file
    with open(ref_file, 'r') as f:
        ref_json = json.load(f)
    with open(tar_file, 'r') as f:
        tar_json = json.load(f)
    ref_name_to_anno = {a['img_name']: a for a in ref_json}
    tar_name_to_anno = {a['img_name']: a for a in tar_json}
    names = list(ref_name_to_anno.keys())

    mismatch = 0
    for img_name in random.choices(names, k=1000):
        anno_a = ref_name_to_anno[img_name]['boxes_and_score']
        anno_b = tar_name_to_anno[img_name]['boxes_and_score']
        if len(anno_a) == len(anno_b):
            diff = [
                abs(anno_a[i][k] - anno_b[i][k])
                for i in range(len(anno_a))
                for k in ['xmin', 'xmax', 'ymin', 'ymax']
            ]
            if sum(diff) >= 1e-5:
                mismatch += 1
                print(sum(diff), img_name)
                # if sum(diff) > 1:
                #     print(anno_a)
                #     print(anno_b)
                print('-' * 100)
        else:
            mismatch += 1
            print(anno_a)
            print(anno_b)
            print(f'num boxes diff: {img_name}, {len(anno_a), len(anno_b)}')
            print('-' * 100)
    print("mismatch: ", mismatch)


if __name__ == "__main__":
    fire.Fire({
        "check_cleaned_image": check_cleaned_image,
        "check_ocr_mask": check_ocr_mask,
        "check_box_annos": check_box_annos,
    })
