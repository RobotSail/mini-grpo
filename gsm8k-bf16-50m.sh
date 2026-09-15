#!/bin/bash
set -eo pipefail

export WANDB_API_KEY="${WANDB_API_KEY:?Set WANDB_API_KEY}"

PYTHON=/mnt/4TB/workspace/oleg/mini-grpo/.venv/bin/python
DATA_PATH="adamw-vs-muon-grpo-v1-artifacts/generated-data-v2/gsm8k_grpo_train.jsonl"
VALIDATION_PATH="adamw-vs-muon-grpo-v1-artifacts/generated-data-v2/gsm8k_grpo_test.jsonl"
MODEL="Qwen/Qwen2-1.5B-Instruct"

echo "========================================="
echo "BF16 weights + fp32 optimizer | 50M tokens | lr=1e-7"
echo "========================================="
$PYTHON cli.py distributed-grpo-train \
    --data-path "${DATA_PATH}" \
    --model "${MODEL}" \
    --max-tokens 50000000 \
    --save-every-n-tokens 500000 \
    --group-size 8 \
    --batch-size 8 \
    --inner-batch-size 64 \
    --clip-eps 0.2 \
    --kl 0 \
    --temp 0.7 \
    --max-new-tokens 512 \
    --max-tokens-per-gpu 4096 \
    --optimizer adamw \
    --lr 1e-7 \
    --seed 2025 \
    --train-gpus 0,1 \
    --vllm-gpus 2,3,4,5,6,7 \
    --precision bf16 \
    --wandb \
    --wandb-project gsm8k-precision-sweep \
    --wandb-run "lr1e7_grpo-adamw-bf16-50m" \
    --validation-path "${VALIDATION_PATH}" \
    --eval-every-n-steps 50 \
    --output-dir "/mnt/nvme4n1/checkpoints/precision-sweep-lr1e7/grpo-adamw-bf16-50m"

echo "========================================="
echo "BF16 run completed!"
echo "========================================="
