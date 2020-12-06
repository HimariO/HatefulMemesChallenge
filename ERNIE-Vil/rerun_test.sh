SELF=$(dirname "$(realpath $0)")
PRETRAIN="$SELF/../pretrain_model"
CKPT="$SELF/../checkpoints"
DATA="$SELF/../data"
MEME_DATA="$DATA/hateful_memes"
CUDA_VISIBLE_DEVICES="0"

mkdir -p "$CKPT/ernie-vil/"

MODEL_NAME_LIST=("ernie-vil-large-vcr" "ernie-vil-large")
CKPT_NAME_LIST=("epoch=10.ckpt" "epoch=6.ckpt")
MODEL_CONF_LIST=("model_conf_meme_vcr" "model_conf_meme")

for index in ${!MODEL_NAME_LIST[*]}; do 
    MODEL_NAME="${MODEL_NAME_LIST[$index]}"
    CKPT_NAME="${CKPT_NAME_LIST[$index]}"
    MODEL_CONF="${MODEL_CONF_LIST[$index]}"
    
    echo "[$MODEL_NAME] --> [final.ckpt]"

    if [ -e "$CKPT/ernie-vil/$MODEL_NAME/final.ckpt" ]; then
        echo "********** [TSET] $CKPT/ernie-vil/$MODEL_NAME/final.ckpt **********"
        docker run --gpus '"'device=$CUDA_VISIBLE_DEVICES'"' \
            --shm-size 1G \
            -v $SELF:/src \
            -v $PRETRAIN:/pretrain_model \
            -v $CKPT:/checkpoints \
            -v $DATA:/data \
            -v $MEME_DATA:/meme_data \
            dsfhe49854/ernie-vil \
            bash /src/run_pt_test.sh meme \
            "/src/final_conf/$MODEL_NAME/$MODEL_CONF" \
            "/src/model_config/vocab.txt" \
            "/src/model_config/ernie_vil.large.json"  \
            "/pretrain_model/ernie-vil/$MODEL_NAME.npz" \
            "/checkpoints/ernie-vil/$MODEL_NAME/final.ckpt"
    else
        echo "[TSET] Checkpoint not found: $CKPT/ernie-vil/$MODEL_NAME/final.ckpt"
    fi;
done;
