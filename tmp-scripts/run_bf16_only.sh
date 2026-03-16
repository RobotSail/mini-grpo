#!/bin/bash
# Each run is independent — if one is killed/fails, move on to the next.

export WANDB_API_KEY=dcc7e9d67dd4454320776959ba154a9d285cb7db

CHECKPOINT_PREFIX="/mnt/nvme2n1/checkpoints/precision-comparison"
DATA_PATH="adamw-vs-muon-grpo-v1-artifacts/generated-data-v2/gsm8k_grpo_train.jsonl"
VALIDATION_PATH="adamw-vs-muon-grpo-v1-artifacts/generated-data-v2/gsm8k_grpo_test.jsonl"
MODEL="Qwen/Qwen2-1.5B-Instruct"

SEED=2025
SAVE_EVERY=100000

COMMON_ARGS="--data-path ${DATA_PATH} --model ${MODEL} \
    --save-every-n-tokens ${SAVE_EVERY} \
    --group-size 8 --batch-size 8 --inner-batch-size 64 \
    --clip-eps 0.2 --kl 0 --temp 0.7 --max-new-tokens 512 \
    --max-tokens-per-microbatch 4096 --seed ${SEED} \
    --gpu 0 --vllm-gpus 1,2,3,4,5,6,7 \
    --wandb --wandb-project gsm8k-comparison \
    --validation-path ${VALIDATION_PATH}"

# ── BF16 AdamW (30M tokens) ──────────────────────────────────────
echo "========================================="
echo "BF16 AdamW (30M tokens, seed=${SEED})"
echo "========================================="
python cli.py grpo-train ${COMMON_ARGS} \
    --output-dir "${CHECKPOINT_PREFIX}/grpo-adamw-bf16" \
    --max-tokens 30000000 \
    --optimizer adamw --lr 1e-6 \
    --precision bf16 \
    --wandb-run "precision_grpo-adamw-bf16" || echo "AdamW bf16 exited with code $?"

# ── BF16 Muon (50M tokens, runs last) ────────────────────────────
echo "========================================="
echo "BF16 Muon (50M tokens, seed=${SEED})"
echo "========================================="
python cli.py grpo-train ${COMMON_ARGS} \
    --output-dir "${CHECKPOINT_PREFIX}/grpo-muon-bf16" \
    --max-tokens 50000000 \
    --optimizer muon --lr 1e-6 \
    --precision bf16 \
    --wandb-run "precision_grpo-muon-bf16" || echo "Muon bf16 exited with code $?"

echo "========================================="
echo "BF16 experiments completed!"
echo "========================================="
