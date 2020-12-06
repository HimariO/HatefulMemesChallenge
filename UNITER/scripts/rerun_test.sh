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

mkdir -p $UNITER_CKPT

MODEL_LIST=("uniter-large-0" "uniter-base" "villa-large-1" "villa-base-1")

for MODEL in "${MODEL_LIST[@]}"; do

    if [ -d "$UNITER_CKPT/$MODEL" ]; then
        docker run --gpus all --ipc=host \
            -v "$SRC":/src \
            --mount src="$UNITER_CKPT",dst=/storage,type=bind \
            --mount src="$PRETRAIN/uniter",dst=/pretrain,type=bind,readonly \
            --mount src="$TXT_DB",dst=/txt,type=bind,readonly \
            --mount src="$IMG_DIR",dst=/img,type=bind,readonly \
            -w /src dsfhe49854/uniter \
            python3 test_meme_itm.py \
            --config "config/final_test/$MODEL.json" \
            --checkpoint="/storage/$MODEL/final.pt"
    fi;
done;
