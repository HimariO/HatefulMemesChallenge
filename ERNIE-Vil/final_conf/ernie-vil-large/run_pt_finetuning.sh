set -eu
set -x
SELF=$(dirname "$(realpath $0)")
#bash -x ./env.sh

TASK_NAME=$1
CONF_FILE=$2
VOCAB_PATH=$3
ERNIE_VIL_CONFIG=$4
PRETRAIN_MODELS=$5

source $CONF_FILE

e_executor=$(echo ${use_experimental_executor-'True'} | tr '[A-Z]' '[a-z]')

TASK_GROUP_JSON="$SELF/task_${TASK_NAME}_gcp.json"

gpu_cnt=`echo $CUDA_VISIBLE_DEVICES | awk -F"\t" '{len=split($0,vec,",");print len}'`
echo "gpu_cnt", $gpu_cnt

cd $SELF/../../
python3 pt_finetune_meme.py --use_cuda "True"             \
                --is_distributed "False"                                       \
                --use_fast_executor ${e_executor-"True"}                       \
                --nccl_comm_num ${nccl_comm_num:-"1"}                          \
                --batch_size $((BATCH_SIZE))                                   \
                --do_train "True"  \
                --do_test "False"     \
                --task_name ${TASK_NAME}                      \
                --vocab_path ${VOCAB_PATH}                                     \
                --task_group_json ${TASK_GROUP_JSON}                           \
                --lr_scheduler ${lr_scheduler}                                 \
                --decay_steps ${decay_steps-""}                                 \
                --lr_decay_ratio ${lr_decay_ratio-0.1}                                 \
                --num_train_steps ${num_train_steps}                           \
                --checkpoints $output_model_path                                       \
                --save_steps ${SAVE_STEPS}                                     \
                --init_checkpoint ${PRETRAIN_MODELS}                                 \
                --ernie_config_path ${ERNIE_VIL_CONFIG}                             \
                --learning_rate ${LR_RATE}                                     \
                --warmup_steps ${WARMUP_STEPS}                                               \
                --weight_decay ${WEIGHT_DECAY:-0}                              \
                --max_seq_len ${MAX_LEN}                                       \
                --validation_steps ${VALID_STEPS}                              \
                --skip_steps 10 \
                --balance_cls "True" \
                --seed  78625 \
                # --resume_ckpt /home/ron_zhu/Disk2/ernie/pl_ernie_meme_vcr/epoch=12.ckpt


