
SELF=$(dirname "$(realpath $0)")
MEME_ROOT_DIR="$SELF/../data/hateful_memes"
DATA="$SELF/../data/"
ENTITY_DIR="$DATA/entity_json"
echo $DATA

if [ ! -e "$DATA/entity_tags.pickle" ]; then
    echo "[web_enetity] create image entity description"
    docker run --gpus all \
        -v $SELF:/src \
        -v $MEME_ROOT_DIR:/meme_data \
        -v $DATA:/data \
        dsfhe49854/vl-bert \
        python3 /src/gcp/web_enetity.py create_description \
        --json_dir /data/entity_json \
        --out_pickle /data/entity_tags.pickle
fi
if [ ! -e "$DATA/entity_cleaned.pickle" ]; then
    echo "[web_enetity] cleaning image entity description"
    docker run --gpus all\
        -v $SELF:/src \
        -v $MEME_ROOT_DIR:/meme_data \
        -v $DATA:/data \
        dsfhe49854/vl-bert \
        python3 /src/gcp/web_enetity.py titles_cleanup \
        /data/entity_tags.pickle \
        --out_pickle /data/entity_cleaned.pickle
fi

if [ ! -e "$DATA/summary_entity_cleaned.pickle" ]; then
    echo "[web_enetity] summary image entity description"
    docker run --gpus all\
        -v $SELF:/src \
        -v $MEME_ROOT_DIR:/meme_data \
        -v $DATA:/data \
        dsfhe49854/vl-bert \
        python3 /src/gcp/web_enetity.py titles_summary \
        /data/entity_cleaned.pickle \
        /data/summary_entity_cleaned.pickle
fi

echo "Build: $MEME_ROOT_DIR/dev_all.jsonl"
docker run \
    -v $SELF:/src \
    -v $MEME_ROOT_DIR:/data \
    dsfhe49854/vl-bert \
    python3 /src/merge_dev_set.py /data

SPLIT_LIST=("train.jsonl" "test_unseen.jsonl" "test_seen.jsonl" "dev_unseen.jsonl" "dev_seen.jsonl" "dev_all.jsonl")

for SPLIT in "${SPLIT_LIST[@]}"; do
    if [ ! -e "$MEME_ROOT_DIR/$SPLIT" ]; then
        echo "Insert features to: $MEME_ROOT_DIR/$SPLIT"
        docker run \
            -v $SELF:/src \
            -v $MEME_ROOT_DIR:/meme_data \
            -v $DATA:/data \
            dsfhe49854/vl-bert \
            python3 /src/gcp/web_enetity.py insert_anno_jsonl  \
                /data/summary_entity_cleaned.pickle  \
                /meme_data/$SPLIT \
                /meme_data/split_img_clean_boxes.json  \
                /meme_data/ocr.box.json
    fi
done

cp "$MEME_ROOT_DIR/train.entity.jsonl" "$MEME_ROOT_DIR/train_dev_all.entity.jsonl"
cat "$MEME_ROOT_DIR/dev_all.entity.jsonl" >> "$MEME_ROOT_DIR/train_dev_all.entity.jsonl"