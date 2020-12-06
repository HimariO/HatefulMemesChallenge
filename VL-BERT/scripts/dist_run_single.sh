#!/usr/bin/env bash
SELF=$(dirname "$(realpath $0)")
cd $SELF/..
python3 ./scripts/launch.py \
    --nproc_per_node "$1" \
    "$2" --cfg "$3" --model-dir "$4"
