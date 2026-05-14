#!/bin/bash
export PRETRAINED_PATH=experiments/pretrain_lr1e-3_epoch5_wd0.0_20260309_055220/pretrained_model.pt
export CHECKPOINT_DIR=experiments/finetune_pretrain_epoch5

python scripts/analyze_drift_trajectory.py \
    --results-dir $CHECKPOINT_DIR \
    --pretrained-model $PRETRAINED_PATH \
    --output-dir plots/drift_trajectory \
    --probe-task old