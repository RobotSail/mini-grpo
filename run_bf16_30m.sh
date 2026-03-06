#!/bin/bash

set -eo pipefail

export WANDB_API_KEY=dcc7e9d67dd4454320776959ba154a9d285cb7db

CHECKPOINT_PREFIX="/mnt/nvme2n1/checkpoints/bf16-30m"
DATA_PATH="adamw-vs-muon-grpo-v1-artifacts/generated-data-v2/gsm8k_grpo_train.jsonl"
VALIDATION_PATH="adamw-vs-muon-grpo-v1-artifacts/generated-data-v2/gsm8k_grpo_test.jsonl"
MODEL="Qwen/Qwen2-1.5B-Instruct"

SEED=2025
TOKEN_BUDGET=30000000
SAVE_EVERY=100000

echo "========================================="
echo "GRPO BF16 30M: AdamW (seed=${SEED})"
echo "========================================="
python cli.py grpo-train \
    --data-path "${DATA_PATH}" \
    --model "${MODEL}" \
    --output-dir "${CHECKPOINT_PREFIX}/grpo-adamw-bf16" \
    --max-tokens ${TOKEN_BUDGET} \
    --save-every-n-tokens ${SAVE_EVERY} \
    --group-size 8 \
    --batch-size 8 \
    --inner-batch-size 64 \
    --clip-eps 0.2 \
    --kl 0 \
    --temp 0.7 \
    --max-new-tokens 512 \
    --max-tokens-per-microbatch 4096 \
    --optimizer adamw \
    --lr 1e-6 \
    --precision bf16 \
    --seed ${SEED} \
    --gpu 0 \
    --vllm-gpus "1,2,3,4,5,6,7" \
    --wandb \
    --wandb-project gsm8k-comparison \
    --wandb-run "bf16-30m_grpo-adamw" \
    --validation-path "${VALIDATION_PATH}"

echo "========================================="
echo "GRPO BF16 30M: Muon (seed=${SEED})"
echo "========================================="
python cli.py grpo-train \
    --data-path "${DATA_PATH}" \
    --model "${MODEL}" \
    --output-dir "${CHECKPOINT_PREFIX}/grpo-muon-bf16" \
    --max-tokens ${TOKEN_BUDGET} \
    --save-every-n-tokens ${SAVE_EVERY} \
    --group-size 8 \
    --batch-size 8 \
    --inner-batch-size 64 \
    --clip-eps 0.2 \
    --kl 0 \
    --temp 0.7 \
    --max-new-tokens 512 \
    --max-tokens-per-microbatch 4096 \
    --optimizer muon \
    --lr 1e-6 \
    --precision bf16 \
    --seed ${SEED} \
    --gpu 0 \
    --vllm-gpus "1,2,3,4,5,6,7" \
    --wandb \
    --wandb-project gsm8k-comparison \
    --wandb-run "bf16-30m_grpo-muon" \
    --validation-path "${VALIDATION_PATH}"

echo "========================================="
echo "All BF16 30M experiments completed!"
echo "========================================="
