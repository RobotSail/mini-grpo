#!/bin/bash

set -eo pipefail

# ========================================
# Countdown Experiments: AdamW vs Muon
# GRPO + SFT, aligned sample counts per gradient step
# 2 train GPUs + 2 vLLM GPUs per GRPO run
# 4 GPUs per SFT run
# All runs sequential (foreground, Ctrl+C to kill)
# ========================================

CHECKPOINT_PREFIX="/mnt/nvme2n1/checkpoints/countdown"
DATA_DIR="generated_data"
MODEL="Qwen/Qwen2-1.5B-Instruct"

# ── Step 0: Generate countdown data (50K samples, seed 67) ──
if [ ! -f "${DATA_DIR}/countdown_grpo_train.jsonl" ]; then
    echo "Generating 50K countdown samples..."
    python cli.py generate-countdown-datasets \
        --total-samples 50000 \
        --output-dir "${DATA_DIR}" \
        --val-split 0.1 \
        --test-split 0.1 \
        --seed 67
fi

GRPO_DATA="${DATA_DIR}/countdown_grpo_train.jsonl"
GRPO_VAL="${DATA_DIR}/countdown_grpo_val.jsonl"
SFT_DATA="${DATA_DIR}/countdown_sft_train.jsonl"

# ── Hyperparameters ──
# GRPO: 16 prompts × 8 rollouts = 128 samples per rollout batch
#   inner_epochs=1, inner_batch_size=128 → 1 gradient step per rollout batch
# SFT: batch_size=128 → 128 samples per gradient step (matching GRPO)
LR="1e-6"
GROUP_SIZE=8
BATCH_SIZE=16
INNER_BATCH_SIZE=128
INNER_EPOCHS=1
SFT_BATCH_SIZE=128
MAX_TOKENS=300000
SAVE_EVERY=50000
KL=0
SEED=67
MAX_TOKENS_PER_GPU=4096

# ========================================
# Run 1: GRPO AdamW
# ========================================

echo "========================================="
echo "GRPO: AdamW"
echo "========================================="

python cli.py countdown-grpo-train \
    --data-path "${GRPO_DATA}" \
    --model "${MODEL}" \
    --output-dir "${CHECKPOINT_PREFIX}/grpo/adamw" \
    --optimizer adamw \
    --lr ${LR} \
    --batch-size ${BATCH_SIZE} \
    --group-size ${GROUP_SIZE} \
    --inner-batch-size ${INNER_BATCH_SIZE} \
    --inner-epochs ${INNER_EPOCHS} \
    --max-tokens ${MAX_TOKENS} \
    --save-every-n-tokens ${SAVE_EVERY} \
    --seed ${SEED} \
    --kl ${KL} \
    --train-gpus 0,1 \
    --vllm-gpus 2,3,4,5,6,7 \
    --max-tokens-per-gpu ${MAX_TOKENS_PER_GPU} \
    --validation-path "${GRPO_VAL}" \
    --wandb --wandb-project countdown-comparison \
    --wandb-run "grpo-adamw"

echo "GRPO AdamW complete."

# ========================================
# Run 2: GRPO Muon
# ========================================

echo "========================================="
echo "GRPO: Muon"
echo "========================================="

python cli.py countdown-grpo-train \
    --data-path "${GRPO_DATA}" \
    --model "${MODEL}" \
    --output-dir "${CHECKPOINT_PREFIX}/grpo/muon" \
    --optimizer muon \
    --lr ${LR} \
    --batch-size ${BATCH_SIZE} \
    --group-size ${GROUP_SIZE} \
    --inner-batch-size ${INNER_BATCH_SIZE} \
    --inner-epochs ${INNER_EPOCHS} \
    --max-tokens ${MAX_TOKENS} \
    --save-every-n-tokens ${SAVE_EVERY} \
    --seed ${SEED} \
    --kl ${KL} \
    --train-gpus 0,1 \
    --vllm-gpus 2,3,4,5,6,7 \
    --max-tokens-per-gpu ${MAX_TOKENS_PER_GPU} \
    --validation-path "${GRPO_VAL}" \
    --wandb --wandb-project countdown-comparison \
    --wandb-run "grpo-muon"

echo "GRPO Muon complete."

# ========================================
# Run 3: SFT AdamW
# ========================================

echo "========================================="
echo "SFT: AdamW"
echo "========================================="

python cli.py countdown-sft-train \
    --data-path "${SFT_DATA}" \
    --model "${MODEL}" \
    --output-dir "${CHECKPOINT_PREFIX}/sft/adamw" \
    --optimizer adamw --precision mixed \
    --lr ${LR} \
    --batch-size ${SFT_BATCH_SIZE} \
    --max-tokens ${MAX_TOKENS} \
    --save-every-n-tokens ${SAVE_EVERY} \
    --seed ${SEED} \
    --lr-scheduler constant \
    --max-tokens-per-gpu ${MAX_TOKENS_PER_GPU} \
    --wandb --wandb-project countdown-comparison \
    --wandb-run "sft-adamw" \
    --num-gpus 8

echo "SFT AdamW complete."

# ========================================
# Run 4: SFT Muon
# ========================================

echo "========================================="
echo "SFT: Muon"
echo "========================================="

python cli.py countdown-sft-train \
    --data-path "${SFT_DATA}" \
    --model "${MODEL}" \
    --output-dir "${CHECKPOINT_PREFIX}/sft/muon" \
    --optimizer muon --precision mixed \
    --lr ${LR} \
    --batch-size ${SFT_BATCH_SIZE} \
    --max-tokens ${MAX_TOKENS} \
    --save-every-n-tokens ${SAVE_EVERY} \
    --seed ${SEED} \
    --lr-scheduler constant \
    --max-tokens-per-gpu ${MAX_TOKENS_PER_GPU} \
    --wandb --wandb-project countdown-comparison \
    --wandb-run "sft-muon" \
    --num-gpus 8

echo "SFT Muon complete."

echo "========================================="
echo "All countdown experiments completed!"
echo "========================================="
