#!/bin/bash

set -eo pipefail

# ========================================
# Countdown Experiments: AdamW vs Muon
# GRPO + SFT, aligned sample counts per gradient step
# 2 train GPUs + 6 vLLM GPUs per GRPO run
# 8 GPUs per SFT run
# All runs sequential (foreground, Ctrl+C to kill)
#
# Weight sync: NCCL (vLLM v0.16 native weight transfer)
#   - No /dev/shm checkpoint writes
#   - Weights transfer directly GPU->GPU via NCCL broadcast
#   - vLLM starts with dummy weights, real weights synced
#     before the first rollout
# ========================================

CHECKPOINT_PREFIX="/mnt/nvme2n1/checkpoints/test-nccl_countdown"
DATA_DIR="generated_data"
MODEL="Qwen/Qwen2-1.5B-Instruct"
TOTAL_SAMPLES=400000
VAL_SPLIT=0.00375
TEST_SPLIT=0.05

# ── Step 0: Generate countdown data ──
if [ ! -f "${DATA_DIR}/countdown_grpo_train.jsonl" ]; then
    echo "Generating ${TOTAL_SAMPLES} countdown samples..."
    python cli.py generate-countdown-datasets \
        --total-samples "${TOTAL_SAMPLES}" \
        --output-dir "${DATA_DIR}" \
        --val-split "${VAL_SPLIT}" \
        --test-split "${TEST_SPLIT}" \
        --seed 67 \
        --grpo-only
fi

GRPO_DATA="${DATA_DIR}/countdown_grpo_train.jsonl"
GRPO_VAL="${DATA_DIR}/countdown_grpo_val.jsonl"
SFT_DATA="${DATA_DIR}/countdown_sft_train.jsonl"

# ── Hyperparameters ──
# GRPO: 16 prompts x 8 rollouts = 128 samples per rollout batch
#   inner_epochs=1, inner_batch_size=128 -> 1 gradient step per rollout batch
# SFT: batch_size=128 -> 128 samples per gradient step (matching GRPO)
LR="3e-7"
GROUP_SIZE=8
BATCH_SIZE=16
INNER_BATCH_SIZE=128
INNER_EPOCHS=1
SFT_BATCH_SIZE=128
MAX_STEPS=2000
SAVE_EVERY_STEPS=100
TEMPERATURE=1.0
KL=0
FORMAT_REWARD=0.00
UPDATE_REF_EVERY=0
SEED=67
MAX_TOKENS_PER_GPU=8192

# ========================================
# Run 1: GRPO AdamW
# ========================================

echo "========================================="
echo "GRPO: AdamW (NCCL weight sync)"
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
    --max-steps ${MAX_STEPS} \
    --save-every-n-steps ${SAVE_EVERY_STEPS} \
    --seed ${SEED} \
    --temp ${TEMPERATURE} \
    --kl ${KL} \
    --format-reward ${FORMAT_REWARD} \
    --update-ref-every ${UPDATE_REF_EVERY} \
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
echo "GRPO: Muon (NCCL weight sync)"
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
    --max-steps ${MAX_STEPS} \
    --save-every-n-steps ${SAVE_EVERY_STEPS} \
    --seed ${SEED} \
    --temp ${TEMPERATURE} \
    --kl ${KL} \
    --format-reward ${FORMAT_REWARD} \
    --update-ref-every ${UPDATE_REF_EVERY} \
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
    --max-steps ${MAX_STEPS} \
    --save-every-n-steps ${SAVE_EVERY_STEPS} \
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
    --max-steps ${MAX_STEPS} \
    --save-every-n-steps ${SAVE_EVERY_STEPS} \
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
