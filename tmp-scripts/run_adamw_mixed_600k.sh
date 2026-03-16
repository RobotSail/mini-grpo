#!/bin/bash

set -eo pipefail

export WANDB_API_KEY=dcc7e9d67dd4454320776959ba154a9d285cb7db

CHECKPOINT_PREFIX="/mnt/nvme2n1/checkpoints/precision-comparison"
DATA_PATH="adamw-vs-muon-grpo-v1-artifacts/generated-data-v2/gsm8k_grpo_train.jsonl"
VALIDATION_PATH="adamw-vs-muon-grpo-v1-artifacts/generated-data-v2/gsm8k_grpo_test.jsonl"
MODEL="Qwen/Qwen2-1.5B-Instruct"

SEED=2025

echo "========================================="
echo "Mixed Precision AdamW (600K tokens, save every 10K, seed=${SEED})"
echo "========================================="
python cli.py grpo-train \
    --data-path ${DATA_PATH} --model ${MODEL} \
    --save-every-n-tokens 10000 \
    --group-size 8 --batch-size 8 --inner-batch-size 64 \
    --clip-eps 0.2 --kl 0 --temp 0.7 --max-new-tokens 512 \
    --seed ${SEED} \
    --gpu 0 --vllm-gpus 1,2,3,4,5,6,7 \
    --wandb --wandb-project gsm8k-comparison \
    --max-tokens-per-microbatch 4096 \
    --output-dir "${CHECKPOINT_PREFIX}/grpo-adamw-mixed-600k" \
    --max-tokens 600000 \
    --optimizer adamw --lr 1e-6 \
    --precision mixed \
    --wandb-run "precision_grpo-adamw-mixed-600k"

echo "========================================="
echo "Training complete!"
echo "========================================="
