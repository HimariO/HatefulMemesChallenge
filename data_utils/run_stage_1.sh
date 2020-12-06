SELF=$(dirname "$(realpath $0)")
MEME_ROOT_DIR="$SELF/../data/hateful_memes"
DATA_DIR="$SELF/../data"

# OCR to get text bbox and mask
# python3 ocr.py detect $MEME_ROOT_DIR
# python3 ocr.py point_to_box $MEME_ROOT_DIR/ocr.json
# python3 ocr.py generate_mask $MEME_ROOT_DIR/ocr.box.json $MEME_ROOT_DIR/img $MEME_ROOT_DIR/img_mask_3px

echo "[OCR] detect"
if [ ! -e "$MEME_ROOT_DIR/ocr.json" ]; then
    docker run --gpus all \
        -v $SELF:/src \
        -v $MEME_ROOT_DIR:/data \
        dsfhe49854/vl-bert \
        python3 /src/ocr.py detect \
        /data
fi;

echo "[OCR] convert point annotation to box"
if [ ! -e "$MEME_ROOT_DIR/ocr.box.json" ]; then
    docker run --gpus all \
        -v $SELF:/src \
        -v $MEME_ROOT_DIR:/data \
        dsfhe49854/vl-bert \
        python3 /src/ocr.py point_to_box \
        /data/ocr.json
fi;

echo "[OCR] create text segmentation mask"
if [ ! -d "$MEME_ROOT_DIR/img_mask_3px" ]; then
    docker run --gpus all \
        -v $SELF:/src \
        -v $MEME_ROOT_DIR:/data \
        dsfhe49854/vl-bert \
        python3 /src/ocr.py generate_mask \
        /data/ocr.box.json \
        /data/img \
        /data/img_mask_3px
fi;


# remove text by inpainting
echo "[mmediting] remove text using text mask"
if [ ! -d "$MEME_ROOT_DIR/img_clean" ]; then
    docker run --gpus all \
        -v "$SELF/../pretrain_model":/pretrain_model \
        -v "$MEME_ROOT_DIR":/data \
        dsfhe49854/mmedit \
        python3 /mmediting/demo/inpainting_demo.py  \
        /mmediting/configs/inpainting/deepfillv2/deepfillv2_256x256_8x2_places.py \
        /pretrain_model/deepfillv2_256x256_8x2_places_20200619-10d15793.pth \
        /data/img_mask_3px/ /data/img_clean
fi;

# Run InceptionV2 OID
echo "[TF OID] OpenImageV4 object detector"
if [ ! -e "$MEME_ROOT_DIR/box_annos.json" ]; then
    docker run --gpus all \
        -v $SELF:/src \
        -v $MEME_ROOT_DIR:/data \
        dsfhe49854/vl-bert \
        python3 /src/gen_bbox.py /data/img_clean /data/box_annos.json
fi;

# detect and extract image patchs
if [ ! -e "$MEME_ROOT_DIR/split_img_clean_boxes.json" ]; then
    echo "[mmdetection] detect is meme image compose with muliple patchs"
    docker run --gpus all \
        -v "$SELF/../pretrain_model":/pretrain_model \
        -v "$MEME_ROOT_DIR":/data \
        dsfhe49854/mmdetect-mmedit \
        python3 tools/inspect_image_clip.py \
        /data/img_clean \
        /data/split_img_clean \
        /data/split_img_clean_boxes.json \
        --config_file configs/res2net/faster_rcnn_r2_101_fpn_2x_img_clip.py \
        --checkpoint_file /pretrain_model/faster_rcnn_r2_101_fpn_2x_img_clip/epoch_3.pth
else
    echo "[mmdetection] found split_img_clean_boxes.json, skip this step~"
fi;

# Get race of face and head
echo "[FairFace]"
if [ ! -e "$MEME_ROOT_DIR/face_race_boxes.json" ]; then
    docker run --gpus all \
        -v "$MEME_ROOT_DIR":/data \
        dsfhe49854/fairface-dlib \
        python3 inference.py detect_race_mp \
        /data/box_annos.json \
        /data/img_clean \
        /data/face_race_boxes.json \
        --debug False \
        --worker 4
fi;

if [ ! -e "$MEME_ROOT_DIR/box_annos.race.json" ]; then
    docker run --gpus all \
        -v "$MEME_ROOT_DIR":/data \
        dsfhe49854/fairface-dlib \
        python3 inference.py map_race_to_person_box \
        /data/img_clean \
        /data/box_annos.json \
        /data/face_race_boxes.json
fi;
