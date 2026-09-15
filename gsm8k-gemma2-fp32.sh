#!/bin/bash
set -eo pipefail

export WANDB_API_KEY="${WANDB_API_KEY:?Set WANDB_API_KEY}"
export HF_TOKEN="${HF_TOKEN:?Set HF_TOKEN}"
export HF_HOME=/mnt/nvme4n1/hf_cache
export VLLM_CACHE_ROOT=/mnt/nvme4n1/vllm_cache
export XDG_CACHE_HOME=/mnt/nvme4n1/cache

PYTHON=/mnt/4TB/workspace/oleg/mini-grpo/.venv/bin/python
DATA_PATH="adamw-vs-muon-grpo-v1-artifacts/generated-data-v2/gsm8k_grpo_train_nosys.jsonl"
VALIDATION_PATH="adamw-vs-muon-grpo-v1-artifacts/generated-data-v2/gsm8k_grpo_test_nosys.jsonl"
MODEL="google/gemma-2-2b-it"

echo "========================================="
echo "Gemma-2-2B-IT: fp32 weights + fp32 optimizer (mixed fwd) | lr=1e-6"
echo "========================================="
$PYTHON cli.py distributed-grpo-train \
    --data-path "${DATA_PATH}" \
    --model "${MODEL}" \
    --max-tokens 10000000 \
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
    --seed 2025 \
    --train-gpus 0,1 \
    --vllm-gpus 2,3,4,5,6,7 \
    --precision mixed \
    --wandb \
    --wandb-project gsm8k-comparison \
    --wandb-run "gemma2-2b_grpo-adamw-fp32-lr1e6-10m" \
    --validation-path "${VALIDATION_PATH}" \
    --eval-every-n-steps 50 \
    --output-dir "/mnt/nvme4n1/checkpoints/gemma2-2b/grpo-adamw-fp32"

echo "========================================="
echo "Done!"
echo "========================================="
