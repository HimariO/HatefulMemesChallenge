#!/usr/bin/env bash
SELF=$(dirname "$(realpath $0)")
cd $SELF/..
echo "[dist_run_multi.sh] pwd: $(pwd)"

python ./scripts/launch.py \
    --nnodes "$1" --node_rank "$2" --master_addr "$3" --nproc_per_node "$4" \
    "$5" --cfg "$6" --model-dir "$7"