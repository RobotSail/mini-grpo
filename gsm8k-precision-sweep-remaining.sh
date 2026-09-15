#!/bin/bash
set -eo pipefail

export WANDB_API_KEY="${WANDB_API_KEY:?Set WANDB_API_KEY}"

PYTHON=/mnt/4TB/workspace/oleg/mini-grpo/.venv/bin/python
CHECKPOINT_PREFIX="/mnt/nvme4n1/checkpoints/precision-sweep-10m-lr1e7"
DATA_PATH="adamw-vs-muon-grpo-v1-artifacts/generated-data-v2/gsm8k_grpo_train.jsonl"
VALIDATION_PATH="adamw-vs-muon-grpo-v1-artifacts/generated-data-v2/gsm8k_grpo_test.jsonl"
MODEL="Qwen/Qwen2-1.5B-Instruct"

SEED=2025
TOKEN_BUDGET=10000000
SAVE_EVERY=100000

COMMON_ARGS=(
    --data-path "${DATA_PATH}"
    --model "${MODEL}"
    --max-tokens ${TOKEN_BUDGET}
    --save-every-n-tokens ${SAVE_EVERY}
    --group-size 8
    --batch-size 8
    --inner-batch-size 64
    --clip-eps 0.2
    --kl 0
    --temp 0.7
    --max-new-tokens 512
    --max-tokens-per-gpu 4096
    --optimizer adamw
    --lr 1e-7
    --seed ${SEED}
    --train-gpus 0,1
    --vllm-gpus 2,3,4,5,6,7
    --wandb
    --wandb-project gsm8k-precision-sweep
    --validation-path "${VALIDATION_PATH}"
    --eval-every-n-steps 50
)

echo "========================================="
echo "RUN 1/2: 10-bit mantissa weights + fp32 optimizer (mixed fwd)"
echo "========================================="
$PYTHON cli.py distributed-grpo-train \
    "${COMMON_ARGS[@]}" \
    --output-dir "${CHECKPOINT_PREFIX}/grpo-adamw-m10" \
    --precision mixed \
    --lattice-mantissa-bits 10 \
    --wandb-run "lr1e7_grpo-adamw-m10"

echo "========================================="
echo "RUN 2/2: fp32 weights + fp32 optimizer (mixed fwd)"
echo "========================================="
$PYTHON cli.py distributed-grpo-train \
    "${COMMON_ARGS[@]}" \
    --output-dir "${CHECKPOINT_PREFIX}/grpo-adamw-fp32" \
    --precision mixed \
    --wandb-run "lr1e7_grpo-adamw-fp32"

echo "========================================="
echo "Both remaining experiments completed!"
echo "========================================="
