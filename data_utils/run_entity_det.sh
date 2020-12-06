SELF=$(dirname "$(realpath $0)")
MEME_ROOT_DIR="$SELF/../data/hateful_memes"
DATA="$SELF/../data"

docker run \
    -v $SELF:/src \
    -v $MEME_ROOT_DIR:/meme_data \
    dsfhe49854/vl-bert \
    python3 /src/gcp/web_enetity.py create_img_list \
    /meme_data/img_clean \
    /meme_data/img_list_all_clean \
    --split_size 20000 \
    --exclude_dir /data/split_img_clean
docker run \
    -v $SELF:/src \
    -v $MEME_ROOT_DIR:/meme_data \
    dsfhe49854/vl-bert \
    python3 /src/gcp/web_enetity.py create_img_list \
    /meme_data/split_img_clean \
    /meme_data/split_img_list_clean \
    --split_size 20000

mkdir -p "$DATA/entity_json"
docker run \
    -v $SELF:/src \
    -v $MEME_ROOT_DIR:/meme_data \
    -v $DATA:/data \
    dsfhe49854/vl-bert \
    bash /src/gcp/loop.sh \
    /meme_data/img_list_all_clean/img_clean_split.0.txt \
    /data/entity_json

mkdir -p "$DATA/entity_json_split"
docker run \
    -v $SELF:/src \
    -v $MEME_ROOT_DIR:/meme_data \
    -v $DATA:/data \
    dsfhe49854/vl-bert \
    bash /src/gcp/loop.sh \
    /meme_data/split_img_list_clean/split_img_clean_split.0.txt \
    /data/entity_json_split

cp $DATA/entity_json_split/*.json "$DATA/entity_json"

