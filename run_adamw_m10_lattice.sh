#!/bin/bash
set -eo pipefail

# GRPO training with AdamW + mixed precision (FP32 master) + Mantissa-10 lattice snapping
# Matches the reference settings but uses AdamW instead of Muon, and adds lattice regularization

export WANDB_API_KEY=dcc7e9d67dd4454320776959ba154a9d285cb7db

CHECKPOINT_PREFIX="/workspace/home/oleg/mini-grpo/experiments"
DATA_PATH="adamw-vs-muon-grpo-v1-artifacts/generated-data-v2/gsm8k_grpo_train.jsonl"
VALIDATION_PATH="adamw-vs-muon-grpo-v1-artifacts/generated-data-v2/gsm8k_grpo_test.jsonl"
MODEL="Qwen/Qwen2-1.5B-Instruct"
OUTPUT_DIR="${CHECKPOINT_PREFIX}/grpo-adamw-mixed-m10lattice"

echo "========================================="
echo "AdamW + mixed precision + Mantissa-10 lattice"
echo "  500 steps, save/eval every 15 steps"
echo "  Train GPUs: 0,1  |  vLLM GPUs: 2,3,4,5,6,7"
echo "========================================="

python cli.py distributed-grpo-train \
    --data-path "${DATA_PATH}" \
    --output-dir "${OUTPUT_DIR}" \
    --model "${MODEL}" \
    --max-tokens 0 \
    --max-steps 500 \
    --inner-epochs 2 \
    --inner-batch-size 64 \
    --save-every-n-steps 15 \
    --eval-every-n-steps 15 \
    --group-size 8 \
    --batch-size 8 \
    --clip-eps 0.2 \
    --kl 0.0 \
    --format-reward 0.1 \
    --gradient-clip 1.0 \
    --update-ref-every 0 \
    --temp 0.7 \
    --max-new-tokens 512 \
    --top-p 1.0 \
    --top-k 0 \
    --max-seq-len 8192 \
    --max-tokens-per-gpu 4096 \
    --optimizer adamw \
    --lr 1e-6 \
    --beta1 0.9 \
    --beta2 0.95 \
    --wd 0.0 \
    --precision mixed \
    --lattice-mantissa-bits 10 \
    --train-gpus 0,1 \
    --vllm-gpus 2,3,4,5,6,7 \
    --seed 2025 \
    --task gsm8k \
    --wandb \
    --wandb-project gsm8k-comparison \
    --wandb-run "adamw-mixed-m10lattice" \
    --validation-path "${VALIDATION_PATH}"

echo "========================================="
echo "Training complete!"
echo "Checkpoints saved to: ${OUTPUT_DIR}"
echo "========================================="
