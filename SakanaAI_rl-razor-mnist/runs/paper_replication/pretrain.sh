#!/bin/bash

python scripts/pretrain.py \
    --epochs 5 \
    --lr 1e-3 \
    --n-samples 500 \
    --scheduler cosine_with_warmup \
    --weight-decay 0.0 \
    --wandb \
    --wandb-name "pretrain_lr1e-3_epoch5_wd0.0"