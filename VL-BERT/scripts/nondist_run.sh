#!/usr/bin/env bash

python "$1" --cfg "$2" --model-dir "$3"
# python3 cls/train_end2end.py --cfg ./cfgs/cls/large_1x14G_fp32_k8s.yaml --model-dir ./