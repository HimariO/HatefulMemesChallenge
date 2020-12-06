# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

TXT_DB=$1
IMG_DIR=$2
OUTPUT=$3
PRETRAIN_DIR=$4

if [ -z $CUDA_VISIBLE_DEVICES ]; then
    CUDA_VISIBLE_DEVICES='all'
fi

# ------------------------------------------------------------------------
# export PATH_TO_STORAGE=/home/ron_zhu/Disk3/uniter_data/
# sh launch_container.sh $PATH_TO_STORAGE/hateful_meme/txt_db $PATH_TO_STORAGE/hateful_meme/img_db $PATH_TO_STORAGE/finetune $PATH_TO_STORAGE/pretrained
# ------------------------------------------------------------------------

# sudo nvidia-docker run --ipc=host --rm -it \
#     -v $(pwd):/src \
#     --mount src=$OUTPUT,dst=/storage,type=bind \
#     --mount src=$PRETRAIN_DIR,dst=/pretrain,type=bind,readonly \
#     --mount src=$TXT_DB,dst=/txt,type=bind,readonly \
#     --mount src=$IMG_DIR,dst=/img,type=bind,readonly \
#     -e NVIDIA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
#     -w /src asia.gcr.io/alfredlabs-model-training/uniter:latest

sudo docker run --gpus all --ipc=host --rm -it \
    -v $(pwd):/src \
    --mount src=$OUTPUT,dst=/storage,type=bind \
    --mount src=$PRETRAIN_DIR,dst=/pretrain,type=bind,readonly \
    --mount src=$TXT_DB,dst=/txt,type=bind,readonly \
    --mount src=$IMG_DIR,dst=/img,type=bind,readonly \
    -e NVIDIA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
    -w /src asia.gcr.io/alfredlabs-model-training/uniter:latest

# docker run --gpus '"'device=$CUDA_VISIBLE_DEVICES'"' --ipc=host --rm -it \
#     --mount src=$(pwd),dst=/src,type=bind \
#     --mount src=$OUTPUT,dst=/storage,type=bind \
#     --mount src=$PRETRAIN_DIR,dst=/pretrain,type=bind,readonly \
#     --mount src=$TXT_DB,dst=/txt,type=bind,readonly \
#     --mount src=$IMG_DIR,dst=/img,type=bind,readonly \
#     -e NVIDIA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
#     -w /src chenrocks/uniter
