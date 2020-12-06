SELF=$(dirname "$(realpath $0)")
MEME_ROOT_DIR="$SELF/../data/hateful_memes"
DATA="$SELF/../data/"
# echo $SELF
# echo $MEME_ROOT_DIR
# ls $SELF
# ls $MEME_ROOT_DIR

# if [ ! -e "$MEME_ROOT_DIR/ocr.json" ]; then
#     docker run --gpus all \
#         -v $SELF:/src \
#         -v $MEME_ROOT_DIR:/data \
#         dsfhe49854/vl-bert \
#         python3 /src/ocr.py detect \
#         /data
# else 
#     echo "File exists"
# fi


# testing

# docker run --gpus all \
#     -v "/home/ron/Downloads/hateful_meme_data_phase2":/data \
#     5a591b3ba1e9 \
#     python3 hateful_meme_feature.py extract_oid_boxes_feat_with_img_aug \
#     /data/box_annos.json \
#     /data \
#     /data/test.pt \
#     --random_seed 8

# sudo docker run --gpus all \
#     -v /home/ron/Projects/tmp:/pretrain_model \
#     -v /home/ron/Downloads/hateful_meme_data_phase2:/data \
#     dsfhe49854/mmdetect-mmedit \
#     python3 tools/inspect_image_clip.py \
#     /data/img_clean \
#     /data/split_img_clean \
#     /data/split_img_clean_boxes.json \
#     --config_file configs/res2net/faster_rcnn_r2_101_fpn_2x_img_clip.py \
#     --checkpoint_file /pretrain_model/faster_rcnn_r2_101_fpn_2x_img_clip/epoch_3.pth


# sudo docker run --gpus all \
#     -v $SELF:/src \
#     -v /home/ron/Downloads/hateful_meme_data_phase2:/data \
#     dsfhe49854/vl-bert \
#     python3 /src/ocr.py detect \
#     /data


# docker run --gpus all \
#     -v "$SELF/../pretrain_model":/pretrain_model \
#     -v "$MEME_ROOT_DIR":/data \
#     -it \
#     dsfhe49854/mmedit \
#     bash

# docker run --gpus all \
#         -v "$SELF/../pretrain_model":/pretrain_model \
#         -v "$MEME_ROOT_DIR":/data \
#         -it \
#         dsfhe49854/mmdetect-mmedit bash

# docker run --gpus all \
#         -v "$SELF/../pretrain_model":/pretrain_model \
#         -v "$MEME_ROOT_DIR":/data \
#         dsfhe49854/mmdetect-mmedit \
#         python3 tools/inspect_image_clip.py \
#         /data/img_clean \
#         /data/split_img_clean \
#         /data/split_img_clean_boxes.json \
#         --config_file configs/res2net/faster_rcnn_r2_101_fpn_2x_img_clip.py \
#         --checkpoint_file /pretrain_model/faster_rcnn_r2_101_fpn_2x_img_clip/epoch_3.pth

# CUDA_VISIBLE_DEVICES="0"
# docker run --gpus '"'device=$CUDA_VISIBLE_DEVICES'"' \
#     -v $SELF:/src \
#     -v $MEME_ROOT_DIR:/data \
#     dsfhe49854/vl-bert \
#     python3 /src/gen_bbox.py /data/img /data/box_annos_1126_det_3.json --debug True

# docker run --gpus '"'device=$CUDA_VISIBLE_DEVICES'"' \
#     -v $SELF:/src \
#     -v $MEME_ROOT_DIR:/data \
#     dsfhe49854/vl-bert \
#     python3 /src/gen_bbox.py /data/img /data/box_annos_1126_det_2.json --debug True
# docker run --gpus '"'device=$CUDA_VISIBLE_DEVICES'"' \
#     -v $SELF:/src \
#     -v $MEME_ROOT_DIR:/data \
#     dsfhe49854/vl-bert \
#     python3 /src/gen_bbox.py /data/img /data/box_annos_1126_nonedet_1.json --debug True

# docker run --gpus '"'device=$CUDA_VISIBLE_DEVICES'"' \
#         -v "$MEME_ROOT_DIR":/data \
#         dsfhe49854/py-bottom-up-attention \
#         python3 hateful_meme_feature.py extract_oid_boxes_feat \
#         /data/box_annos_1125.json \
#         /data \
#         /data/hateful_memes_v2_1125.pt


# docker run --gpus '"'device=$CUDA_VISIBLE_DEVICES'"' \
#         -v "$MEME_ROOT_DIR":/data \
#         dsfhe49854/py-bottom-up-attention \
#         python3 hateful_meme_feature.py extract_oid_boxes_feat_with_img_aug \
#         /data/box_annos_1125.json \
#         /data \
#         /data/hateful_memes_v2_1125.aug.0.pt \
#         --random_seed 5659

# docker run --gpus '"'device=$CUDA_VISIBLE_DEVICES'"' \
#         -v "$MEME_ROOT_DIR":/data \
#         dsfhe49854/py-bottom-up-attention \
#         python3 hateful_meme_feature.py extract_oid_boxes_feat_with_img_aug \
#         /data/box_annos_1125.json \
#         /data \
#         /data/hateful_memes_v2_1125.aug.1.pt \
#         --random_seed 6582

# docker run --gpus '"'device=$CUDA_VISIBLE_DEVICES'"' \
#         -v "$MEME_ROOT_DIR":/data \
#         dsfhe49854/py-bottom-up-attention \
#         python3 hateful_meme_feature.py extract_oid_boxes_feat_with_img_aug \
#         /data/box_annos_1125.json \
#         /data \
#         /data/hateful_memes_v2_1125.aug.2.pt \
#         --random_seed 7505

# docker run --gpus all -it \
#     -v $(pwd)/data_utils:/src \
#     -v $(pwd)/data/hateful_memes:/data \
#     nvcr.io/nvidia/tensorflow:20.11-tf2-py3

# docker run --gpus all \
#     -v $SELF:/src \
#     -v $MEME_ROOT_DIR:/data \
#     dsfhe49854/vl-bert \
#     python3 /src/gen_bbox.py

# docker run \
#     -v $SELF:/src \
#     -v $MEME_ROOT_DIR:/meme_data \
#     -v $DATA:/data \
#     dsfhe49854/vl-bert \
#     python3 /src/gcp/web_enetity.py detect_image \
#     /meme_data/img_clean/01268.png /data/test_01268.json