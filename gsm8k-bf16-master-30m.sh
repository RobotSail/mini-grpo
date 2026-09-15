#!/bin/bash
set -eo pipefail

export WANDB_API_KEY="${WANDB_API_KEY:?Set WANDB_API_KEY}"
export WANDB_ENTITY=olegsilkin

DATA_PATH="adamw-vs-muon-grpo-v1-artifacts/generated-data-v2/gsm8k_grpo_train.jsonl"
VALIDATION_PATH="adamw-vs-muon-grpo-v1-artifacts/generated-data-v2/gsm8k_grpo_test.jsonl"
MODEL="Qwen/Qwen2-1.5B-Instruct"

echo "========================================="
echo "GRPO AdamW | bf16 master weights | 30M tokens | lr=1e-6"
echo "========================================="
python cli.py distributed-grpo-train \
    --data-path "${DATA_PATH}" \
    --model "${MODEL}" \
    --output-dir "/mnt/nvme4n1/checkpoints/grpo-adamw-bf16master-30m" \
    --max-tokens 30000000 \
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
    --lr 1e-6 \
    --precision mixed \
    --bf16-master-weights \
    --seed 2025 \
    --train-gpus "0,1,2,3" \
    --vllm-gpus "4,5,6,7" \
    --wandb \
    --wandb-entity olegsilkin \
    --wandb-project gsm8k-precision-sweep \
    --wandb-run "bf16-master_adamw-30m" \
    --validation-path "${VALIDATION_PATH}"

echo "========================================="
echo "bf16 master weights 30M run completed!"
echo "========================================="
