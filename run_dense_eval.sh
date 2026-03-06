#!/bin/bash

set -eo pipefail

export WANDB_API_KEY=dcc7e9d67dd4454320776959ba154a9d285cb7db

CHECKPOINT_PREFIX="/mnt/nvme2n1/checkpoints/dense-eval"
DATA_PATH="adamw-vs-muon-grpo-v1-artifacts/generated-data-v2/gsm8k_grpo_train.jsonl"
VALIDATION_PATH="adamw-vs-muon-grpo-v1-artifacts/generated-data-v2/gsm8k_grpo_test.jsonl"
MODEL="Qwen/Qwen2-1.5B-Instruct"
SEED=2025
TOKEN_BUDGET=1200000

# Dense saving: every 25k for first 200k, then every 100k after
# We handle this by saving every 25k and just keeping all checkpoints
# (the dense region matters most, extra checkpoints after 200k are fine to have)
SAVE_EVERY=25000

# =========================================
# Rejection Sampling
# =========================================

echo "========================================="
echo "RS: Muon (seed=${SEED})"
echo "========================================="
MASTER_PORT=29500 python cli.py rs-train \
    --data-path "${DATA_PATH}" \
    --model "${MODEL}" \
    --output-dir "${CHECKPOINT_PREFIX}/rs-muon" \
    --max-tokens ${TOKEN_BUDGET} \
    --save-every-n-tokens ${SAVE_EVERY} \
    --samples-to-accept 64 \
    --inference-batch-size 32 \
    --inference-group-size 16 \
    --max-seq-len 8192 \
    --max-tokens-per-gpu 8192 \
    --temp 0.7 \
    --max-new-tokens 512 \
    --optimizer muon \
    --lr 1e-6 \
    --seed ${SEED} \
    --gpu 0 \
    --vllm-gpus "1,2,3,4,5,6,7" \
    --wandb \
    --wandb-project gsm8k-comparison \
    --wandb-run "dense-eval_rs-muon" \
    --validation-path "${VALIDATION_PATH}"

echo "========================================="
echo "RS: AdamW (seed=${SEED})"
echo "========================================="
MASTER_PORT=29510 python cli.py rs-train \
    --data-path "${DATA_PATH}" \
    --model "${MODEL}" \
    --output-dir "${CHECKPOINT_PREFIX}/rs-adamw" \
    --max-tokens ${TOKEN_BUDGET} \
    --save-every-n-tokens ${SAVE_EVERY} \
    --samples-to-accept 64 \
    --inference-batch-size 32 \
    --inference-group-size 16 \
    --max-seq-len 8192 \
    --max-tokens-per-gpu 8192 \
    --temp 0.7 \
    --max-new-tokens 512 \
    --optimizer adamw \
    --lr 1e-6 \
    --seed ${SEED} \
    --gpu 0 \
    --vllm-gpus "1,2,3,4,5,6,7" \
    --wandb \
    --wandb-project gsm8k-comparison \
    --wandb-run "dense-eval_rs-adamw" \
    --validation-path "${VALIDATION_PATH}"

echo "========================================="
echo "RS experiments complete. Starting GRPO..."
echo "========================================="

# =========================================
# GRPO
# =========================================

echo "========================================="
echo "GRPO: Muon (seed=${SEED})"
echo "========================================="
CUDA_VISIBLE_DEVICES=0 MASTER_PORT=29500 python cli.py train \
    --train-path "${DATA_PATH}" \
    --model-name "${MODEL}" \
    --output-dir "${CHECKPOINT_PREFIX}/grpo-muon" \
    --optimizer muon \
    --lr 1e-6 \
    --batch-size 8 \
    --group-size 8 \
    --inner-batch-size 64 \
    --token-train-budget ${TOKEN_BUDGET} \
    --save-every-n-tokens ${SAVE_EVERY} \
    --flash-attn \
    --seed ${SEED} \
    --kl 0 \
    --gpu 0 \
    --max-tokens-per-microbatch 4096 \
    --wandb \
    --wandb-project gsm8k-comparison \
    --wandb-run "dense-eval_grpo-muon"

echo "========================================="
echo "GRPO: AdamW (seed=${SEED})"
echo "========================================="
CUDA_VISIBLE_DEVICES=0 MASTER_PORT=29510 python cli.py train \
    --train-path "${DATA_PATH}" \
    --model-name "${MODEL}" \
    --output-dir "${CHECKPOINT_PREFIX}/grpo-adamw" \
    --optimizer adamw \
    --lr 1e-6 \
    --batch-size 8 \
    --group-size 8 \
    --inner-batch-size 64 \
    --token-train-budget ${TOKEN_BUDGET} \
    --save-every-n-tokens ${SAVE_EVERY} \
    --flash-attn \
    --seed ${SEED} \
    --kl 0 \
    --gpu 0 \
    --max-tokens-per-microbatch 4096 \
    --wandb \
    --wandb-project gsm8k-comparison \
    --wandb-run "dense-eval_grpo-adamw"

echo "========================================="
echo "All dense-eval experiments completed!"
echo "========================================="
