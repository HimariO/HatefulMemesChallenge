SELF=$(dirname "$(realpath $0)")
SRC="$SELF/.."
DATA="$SELF/../../data"
MEME_ROOT_DIR="$DATA/hateful_memes"

UNITER_DIR="$DATA/uniter"
PRETRAIN="$SELF/../../pretrain_model"
CKPT="$SELF/../../checkpoints"
UNITER_CKPT="$CKPT/uniter"

TXT_DB="$UNITER_DIR/txt_db"
IMG_DIR="$UNITER_DIR/MEME_NPZ_DB"

    
IMG_NPY="$UNITER_DIR/MEME_NPZ"
NAME=$(basename $IMG_NPY)

echo "IMG_DIR: $IMG_DIR"
ls $IMG_DIR

# docker run --gpus all --ipc=host -it \
#         -v $SRC:/src \
#         --mount src=$UNITER_CKPT,dst=/storage,type=bind \
#         --mount src="$PRETRAIN/uniter",dst=/pretrain,type=bind,readonly \
#         --mount src=$TXT_DB,dst=/txt,type=bind,readonly \
#         --mount src=$IMG_DIR,dst=/img,type=bind,readonly \
#         -w /src dsfhe49854/uniter \
#         bash

if [ ! -e 'jjjsad' ]; then
    echo "1"
fi
if [ ! -e 'jjjsad' ] && [ ! -e 'jjjsad' ]; then
    echo "2"
fi
if [ ! -e 'jjjsad' ]; then
    echo "3"
fi
if [ ! -e 'jjjsad' ]; then
    echo "4"
fi