import os
import glob
import json
import shutil
from multiprocessing import Pool

import fire
import easyocr
import numpy as np
import torch

from PIL import Image
from skimage import transform
from skimage.feature import canny
from skimage.color import rgb2gray, gray2rgb


def multi_boxes_mask(image, boxes, pad_crop=5):
    """
    image: np.uint8 (h, w, c)
    boxes: np.int32 (n, 4) ymin, xmin, ymax, xmax
    """
    image = image.copy()
    mask = np.zeros_like(image)
    ih, iw, ic = image.shape
    resize = lambda a, b: transform.resize(a, b, preserve_range=True).astype(np.uint8)
    import matplotlib.pyplot as plt
    
    for box in boxes:
        # image[box[0]: box[2], box[1]: box[3], :] = 0
        box[:2] = np.maximum(box[:2] - pad_crop, 0)
        box[2:] = np.minimum(box[2:] + pad_crop, image.shape[:2])
        
        patch = image[box[0]: box[2], box[1]: box[3], :]
        pure_white = (patch > 253).all(axis=-1).astype(np.uint8)
        mask[box[0]: box[2], box[1]: box[3], :] = pure_white[..., None]
        
        # plt.subplot(2, 1, 1)
        # plt.imshow(patch)
        # plt.subplot(2, 1, 2)
        # plt.imshow(pure_white)
        # plt.colorbar()
        # plt.show()
        
        print('pure_white ', pure_white.sum())
    
    shift = 3
    shifts = [
        (0, 0), (shift, 0), (-shift, 0), (0, shift), (0, -shift),
        (shift, shift), (-shift, shift), (shift, -shift), (-shift, -shift)
    ]
    # shifts = []
    for offset in shifts:
        ox, oy = offset
        _mask = mask.copy()

        slice_y = slice(max(0, 0 + oy), min(ih, ih + oy))
        slice_x = slice(max(0, 0 + ox), min(iw, iw + ox))
        print(slice_y, slice_x)
        _mask = _mask[
            max(0, 0 + oy): min(ih, ih + oy),
            max(0, 0 + ox): min(iw, iw + ox),
            :
        ]
        crop_pad = [
            (max(0, -oy), max(0, oy)),
            (max(0, -ox), max(0, ox)),
            (0, 0)
        ]
        _mask = np.pad(_mask, crop_pad)
        print(
            crop_pad,
            np.abs(_mask - mask).sum(),
            np.abs(mask - np.clip(_mask + mask, 0, 1)).sum()
        )
        mask = np.clip(_mask + mask, 0, 1)

    image = image * (1 - mask) + mask * 255 * 0
    mask *= 255
    return image, mask

def cast_pred_type(pred):
    result = []
    for tup in pred:
        coord, txt, score = tup
        coord = np.array(coord).tolist()
        score = float(score)
        result.append((coord, txt, score))
    return result


def detect(root_dir):
    reader = easyocr.Reader(['en'])
    image_dir = os.path.join(root_dir, 'img')
    images = glob.glob(os.path.join(image_dir, '*.png'))
    images += glob.glob(os.path.join(image_dir, '**', '*.png'))
    # images = images[:3]
    assert len(images) > 9000

    out_json = os.path.join(root_dir, 'ocr.json')
    out_anno = {}
    print(f"Find {len(images)} images!")

    for i, image_path in enumerate(images):
        print(F"{i}/{len(images)}")
        img_name = os.path.basename(image_path)
        pred = reader.readtext(image_path)
        out_anno[img_name] = cast_pred_type(pred)

    with open(out_json, 'w') as f:
        json.dump(out_anno, f)


def point_to_box(anno_json):
    with open(anno_json, 'r') as f:
        ocr_anno = json.load(f)
    
    boxed_anno = {}
    for k, v in ocr_anno.items():
        img_ocr_infos = []
        for txt_info in v:
            coord, txt, score = txt_info
            xmin = min([p[0] for p in coord])
            xmax = max([p[0] for p in coord])
            ymin = min([p[1] for p in coord])
            ymax = max([p[1] for p in coord])
            box = [xmin, ymin, xmax, ymax]
            img_ocr_infos.append([box, txt, score])
        boxed_anno[k] = img_ocr_infos
    
    out_path = anno_json.replace('.json', '.box.json')
    with open(out_path, 'w') as f:
        json.dump(boxed_anno, f)


def _mask_white_txt(args):
    img_name, img_boxes, img_dir, out_dir = args
    img_path = os.path.join(img_dir, img_name)
    out_path = os.path.join(out_dir, img_name)
    
    if os.path.exists(out_path):
        return
    # if img_name != '01487.png':
    #     continue
    
    print(out_path)
    img_boxes = [box_info[0] for box_info in img_boxes]
    if len(img_boxes) > 0:
        boxes = np.asarray(img_boxes, dtype=np.int32)
        # print(boxes)
        boxes = np.concatenate([boxes[:, ::-1][:, 2:], boxes[:,::-1][:, :2]], axis=1)
        # print(boxes)
        # x,y,x,y -> y,x,y,x
        img = np.array(Image.open(img_path).convert('RGB'))
        # res = inpaint_model.inpaint_multi_boxes(img, boxes)
        masked_img, mask = multi_boxes_mask(img, boxes)

        Image.fromarray(masked_img).save(out_path)
        out_path = os.path.join(out_dir, img_name.replace('.png', '.mask.png'))
        Image.fromarray(mask).save(out_path)
    else:
        img = np.asarray(Image.open(img_path).convert('RGB'))
        shutil.copy(img_path, out_path)

        mask = np.zeros_like(img)
        out_path = os.path.join(out_dir, img_name.replace('.png', '.mask.png'))
        Image.fromarray(mask).save(out_path)

def generate_mask(ocr_box_anno, img_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    with open(ocr_box_anno, 'r') as f:
        boxes_anno = json.load(f)

    # for i, (img_name, img_boxes) in enumerate(boxes_anno.items()):
    #     pass
    
    with Pool(16) as pool:
        args = [
            (img_name, img_boxes, img_dir, out_dir)
            for img_name, img_boxes in boxes_anno.items()
        ]
        pool.map(_mask_white_txt, args)


if __name__ == "__main__":
    """
    detect -[ocr.json]-> point_to_box -[ocr.box.json]->  generate_mask
    """
    # detect()
    # point_to_box('/home/ron/Downloads/hateful_meme_data/ocr.json')
    # print('hi')
    # generate_mask(
    #     '/home/ron/Downloads/hateful_meme_data/ocr.box.json',
    #     '/home/ron/Downloads/hateful_meme_data_phase2/img',
    #     '/home/ron/Downloads/hateful_meme_data_phase2/img_mask_3px'
    # )

    fire.Fire({
        "detect": detect,
        "point_to_box": point_to_box,
        "generate_mask": generate_mask,
    })