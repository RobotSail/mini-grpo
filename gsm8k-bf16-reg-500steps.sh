#!/bin/bash

set -eo pipefail

export WANDB_API_KEY=dcc7e9d67dd4454320776959ba154a9d285cb7db

CHECKPOINT_PREFIX="/mnt/nvme2n1/checkpoints/bf16-reg-500steps"
DATA_PATH="adamw-vs-muon-grpo-v1-artifacts/generated-data-v2/gsm8k_grpo_train.jsonl"
VALIDATION_PATH="adamw-vs-muon-grpo-v1-artifacts/generated-data-v2/gsm8k_grpo_test.jsonl"
MODEL="Qwen/Qwen2-1.5B-Instruct"

SEED=2025
MAX_STEPS=500
EVAL_EVERY=15

# echo "========================================="
# echo "GRPO 500 steps: AdamW baseline (seed=${SEED})"
# echo "========================================="
# python cli.py distributed-grpo-train \
#     --data-path "${DATA_PATH}" \
#     --model "${MODEL}" \
#     --output-dir "${CHECKPOINT_PREFIX}/grpo-adamw-baseline" \
#     --max-steps ${MAX_STEPS} \
#     --group-size 8 \
#     --batch-size 8 \
#     --inner-batch-size 64 \
#     --clip-eps 0.2 \
#     --kl 0 \
#     --temp 0.7 \
#     --max-new-tokens 512 \
#     --max-tokens-per-gpu 4096 \
#     --optimizer adamw \
#     --lr 1e-6 \
#     --precision mixed \
#     --save-every-n-steps ${EVAL_EVERY} \
#     --eval-every-n-steps ${EVAL_EVERY} \
#     --seed ${SEED} \
#     --train-gpus "0,1,2,3" \
#     --vllm-gpus "4,5,6,7" \
#     --wandb \
#     --wandb-project gsm8k-comparison \
#     --wandb-run "bf16-reg_adamw-baseline" \
#     --validation-path "${VALIDATION_PATH}"

echo "========================================="
echo "GRPO 500 steps: AdamW + bf16-regularization (seed=${SEED})"
echo "========================================="
python cli.py distributed-grpo-train \
    --data-path "${DATA_PATH}" \
    --model "${MODEL}" \
    --output-dir "${CHECKPOINT_PREFIX}/grpo-adamw-bf16reg" \
    --max-steps ${MAX_STEPS} \
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
    --bf16-regularization \
    --save-every-n-steps ${EVAL_EVERY} \
    --eval-every-n-steps ${EVAL_EVERY} \
    --seed ${SEED} \
    --train-gpus "0,1,2,3" \
    --vllm-gpus "4,5,6,7" \
    --wandb \
    --wandb-project gsm8k-comparison \
    --wandb-run "bf16-reg_adamw-bf16reg" \
    --validation-path "${VALIDATION_PATH}"

echo "========================================="
echo "All bf16-regularization experiments completed!"
echo "========================================="
