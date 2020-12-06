import os
import glob
import json
import fire
import queue
import threading
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image


def load_img(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=3)
    return img


def run_detector(detector, path):
    img = load_img(path)

    converted_img  = tf.image.convert_image_dtype(img, tf.float32)[tf.newaxis, ...]
    result = detector(converted_img)

    result = {key:value.numpy() for key,value in result.items()}
    print("Found %d objects." % len([s for s in result["detection_scores"] if s > 0.2]))
    return result


def run_detector_tensor(detector, converted_img):
    result = detector(converted_img)
    result = {key:value.numpy() for key,value in result.items()}
    print("Found %d objects." % len([s for s in result["detection_scores"] if s > 0.2]))
    return result

def pull_imgs(img_path_list, output_queue:queue.Queue):
    for img_file in img_path_list:
        img_tensor = load_img(img_file)
        converted_img = tf.image.convert_image_dtype(img_tensor, tf.float32)[
            tf.newaxis, ...]
        print("[pull_imgs] ", img_file)
        output_queue.put((converted_img, img_file))

def hateful_meme(img_dir, output_path, debug=False):
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

    with tf.device('/device:GPU:0'):
        img_pattern = os.path.join(img_dir, '*.png')
        img_files = glob.glob(img_pattern)
        img_files = sorted(img_files)

        img_queue = queue.Queue(maxsize=16)
        img_io_thread = threading.Thread(
            daemon=True,
            target=pull_imgs,
            args=[img_files, img_queue]
        )
        img_io_thread.start()

        if debug:
            print('RUn in debug mode!')
        
        module_handle = "https://tfhub.dev/google/faster_rcnn/openimages_v4/inception_resnet_v2/1" #@param ["https://tfhub.dev/google/openimages_v4/ssd/mobilenet_v2/1", "https://tfhub.dev/google/faster_rcnn/openimages_v4/inception_resnet_v2/1"]
        detector = hub.load(module_handle).signatures['default']
        det_annos = []

        i = 0
        while True:
            if debug and i > 1000:
                break
            if i >= len(img_files):
                break
            
            tensor, img_file = img_queue.get(block=True, timeout=10)
            
            print(f"[{i}/{len(img_files)}] {img_file}")
            result = run_detector_tensor(detector, tensor)
            img_queue.task_done()

            boxes_score = zip(
                result["detection_boxes"],
                result["detection_scores"],
                result["detection_class_entities"],
                result["detection_class_labels"])
            boxes_score = [
                {
                    'ymin': float(b[0]),
                    'xmin': float(b[1]),
                    'ymax': float(b[2]),
                    'xmax': float(b[3]),
                    'score': float(s),
                    'class_name': c.decode("ascii"),
                    'class_id': int(ci),
                }
                for b, s, c, ci in list(boxes_score) if s > 0.2]
            img_name = os.path.basename(img_file)
            det_anno = {
                'img_name': img_name,
                'boxes_and_score': boxes_score 
            }
            det_annos.append(det_anno)
            i += 1
        
        with open(output_path, mode='w') as output:
            json.dump(det_annos, output)


def test():
    # @param ["https://tfhub.dev/google/openimages_v4/ssd/mobilenet_v2/1", "https://tfhub.dev/google/faster_rcnn/openimages_v4/inception_resnet_v2/1"]
    module_handle = "https://tfhub.dev/google/faster_rcnn/openimages_v4/inception_resnet_v2/1"
    detector = hub.load(module_handle).signatures['default']
    np.random.seed(1234)
    img = tf.constant(np.random.normal(size=[1, 512, 512, 3]).astype(np.float32))
    out = detector(img)
    result = {key: value.numpy() for key, value in out.items()}
    print(result['detection_boxes'])


if __name__ == "__main__":
    fire.Fire(hateful_meme)
