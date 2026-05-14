#!/bin/bash

# Variables
export PRETRAINED_RESULTS=experiments/pretrain_lr1e-3_epoch5_wd0.0_20260309_055220/results.json
export RESULTS_DIR=experiments/sweep_pretrain_epoch5_450x5
export OUTPUT_DIR=plots/sweep_pretrain_epoch5_450x5

python scripts/plot.py \
    --results-dir $RESULTS_DIR \
    --pretrained-results $PRETRAINED_RESULTS \
    --output-dir $OUTPUT_DIR