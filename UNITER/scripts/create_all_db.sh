SELF=$(dirname "$(realpath $0)")
DATA_DIR="$SELF/../../data"
MEME_ROOT_DIR="$DATA_DIR/hateful_memes"
UNITER_DIR="$DATA_DIR/uniter"
SRC="$SELF/.."

set -e

OUT_DIR="$UNITER_DIR/MEME_NPZ_DB"
echo "converting image features ..."
if [ ! -d $OUT_DIR ]; then
    mkdir -p $OUT_DIR
    
    IMG_NPY="$UNITER_DIR/MEME_NPZ"
    NAME=$(basename $IMG_NPY)
    docker run --ipc=host --rm -it \
        --mount src="$SRC",dst=/src,type=bind \
        --mount src="$OUT_DIR",dst=/img_db,type=bind \
        --mount src="$IMG_NPY",dst=/$NAME,type=bind,readonly \
        -w /src dsfhe49854/uniter \
        python scripts/convert_imgdir.py --img_dir /$NAME --output /img_db
    echo "done"
fi;


OUT_DIR="$UNITER_DIR/txt_db"
for SPLIT in 'train_dev_all.entity' 'test_unseen.entity' 'train.entity' 'dev_seen.entity' 'dev_unseen.entity'; do
    if [ ! -d "$OUT_DIR/nlvr2_${SPLIT}.db" ]; then
        echo "preprocessing ${SPLIT} annotations..."
        mkdir -p $OUT_DIR

        docker run --ipc=host --rm -it \
            --mount src="$SRC",dst=/src,type=bind \
            --mount src="$OUT_DIR",dst=/txt_db,type=bind \
            --mount src="$MEME_ROOT_DIR",dst=/ann,type=bind,readonly \
            -w /src dsfhe49854/uniter \
            python prepro.py --annotation /ann/$SPLIT.jsonl \
                            --output /txt_db/nlvr2_${SPLIT}.db
    fi;
done;
