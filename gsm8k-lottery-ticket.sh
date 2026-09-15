#!/bin/bash
set -eo pipefail

# ── Lottery Ticket Mask Experiment for GSM8K GRPO ─────────────────────────
#
# Tests whether bf16's sparse update is a "winning lottery ticket" or
# a precision artifact by training with masked gradients in fp32.
#
# Runs:
#   1. bf16-master baseline (to extract the mask)
#   2. Extract lottery mask + generate random controls
#   3. mixed baseline (unrestricted fp32 training)
#   4. lottery mask (fp32 training restricted to bf16's changed params)
#   5-7. random masks x3 (fp32 training with random masks, same sparsity)

CHECKPOINT_PREFIX="/mnt/nvme2n1/checkpoints/lottery-ticket-experiment"
MASK_DIR="${CHECKPOINT_PREFIX}/masks"
DATA_PATH="adamw-vs-muon-grpo-v1-artifacts/generated-data-v2/gsm8k_grpo_train.jsonl"
VALIDATION_PATH="adamw-vs-muon-grpo-v1-artifacts/generated-data-v2/gsm8k_grpo_test.jsonl"
MODEL="Qwen/Qwen2-1.5B-Instruct"

SEED=2025
MAX_STEPS=500
EVAL_EVERY=15
TRAIN_GPUS="0,1,2,3"
VLLM_GPUS="4,5,6,7"

# Common training flags
COMMON_FLAGS=(
    --data-path "${DATA_PATH}"
    --model "${MODEL}"
    --max-steps ${MAX_STEPS}
    --group-size 8
    --batch-size 8
    --inner-batch-size 64
    --clip-eps 0.2
    --kl 0
    --temp 0.7
    --max-new-tokens 512
    --max-tokens-per-gpu 4096
    --optimizer adamw
    --lr 1e-6
    --save-every-n-steps ${EVAL_EVERY}
    --eval-every-n-steps ${EVAL_EVERY}
    --seed ${SEED}
    --train-gpus "${TRAIN_GPUS}"
    --vllm-gpus "${VLLM_GPUS}"
    --wandb
    --wandb-project gsm8k-lottery-ticket
    --validation-path "${VALIDATION_PATH}"
)

# ── Step 1: bf16-master baseline ──────────────────────────────────────────
echo "========================================="
echo "Step 1: bf16-master baseline (for mask extraction)"
echo "========================================="
BF16_MASTER_DIR="${CHECKPOINT_PREFIX}/grpo-adamw-bf16master"
if [ ! -d "${BF16_MASTER_DIR}/checkpoint-initial" ]; then
    python cli.py distributed-grpo-train \
        "${COMMON_FLAGS[@]}" \
        --output-dir "${BF16_MASTER_DIR}" \
        --precision mixed \
        --bf16-master-weights \
        --wandb-run "bf16-master_baseline"
else
    echo "Skipping: ${BF16_MASTER_DIR} already exists"
fi

# ── Step 2: Extract lottery mask + random controls ────────────────────────
echo "========================================="
echo "Step 2: Extracting masks"
echo "========================================="
mkdir -p "${MASK_DIR}"

# Find the best checkpoint (latest if using best-val-ckpt-only, otherwise last)
BEST_CKPT=$(ls -d ${BF16_MASTER_DIR}/checkpoint-step* 2>/dev/null | sort -t- -k2 -n | tail -1)
if [ -z "${BEST_CKPT}" ]; then
    BEST_CKPT=$(ls -d ${BF16_MASTER_DIR}/checkpoint-best 2>/dev/null)
fi
if [ -z "${BEST_CKPT}" ]; then
    # Fall back to last numbered checkpoint
    BEST_CKPT=$(ls -d ${BF16_MASTER_DIR}/checkpoint-* 2>/dev/null | grep -v initial | sort | tail -1)
fi
echo "Using checkpoint: ${BEST_CKPT}"

if [ ! -f "${MASK_DIR}/lottery_mask.pt" ]; then
    python extract_lottery_mask.py \
        --checkpoint "${BEST_CKPT}" \
        --initial "${BF16_MASTER_DIR}/checkpoint-initial" \
        --output "${MASK_DIR}/lottery_mask.pt" \
        --random-seeds 1,2,3 \
        --random-output-prefix "${MASK_DIR}/random_mask"
else
    echo "Skipping: masks already exist"
fi

# ── Step 3: mixed baseline (unrestricted fp32) ────────────────────────────
echo "========================================="
echo "Step 3: Mixed precision baseline (unrestricted fp32)"
echo "========================================="
MIXED_DIR="${CHECKPOINT_PREFIX}/grpo-adamw-mixed"
if [ ! -d "${MIXED_DIR}/checkpoint-initial" ]; then
    python cli.py distributed-grpo-train \
        "${COMMON_FLAGS[@]}" \
        --output-dir "${MIXED_DIR}" \
        --precision mixed \
        --wandb-run "mixed_baseline"
else
    echo "Skipping: ${MIXED_DIR} already exists"
fi

# ── Step 4: lottery mask (fp32 + bf16 ticket) ─────────────────────────────
echo "========================================="
echo "Step 4: Lottery mask (fp32 training, bf16 ticket mask)"
echo "========================================="
LOTTERY_DIR="${CHECKPOINT_PREFIX}/grpo-adamw-lottery"
if [ ! -d "${LOTTERY_DIR}/checkpoint-initial" ]; then
    python cli.py distributed-grpo-train \
        "${COMMON_FLAGS[@]}" \
        --output-dir "${LOTTERY_DIR}" \
        --precision mixed \
        --gradient-mask "${MASK_DIR}/lottery_mask.pt" \
        --wandb-run "lottery_mask"
else
    echo "Skipping: ${LOTTERY_DIR} already exists"
fi

# ── Steps 5-7: random masks ──────────────────────────────────────────────
for RSEED in 1 2 3; do
    echo "========================================="
    echo "Step $((RSEED+4)): Random mask (seed=${RSEED})"
    echo "========================================="
    RANDOM_DIR="${CHECKPOINT_PREFIX}/grpo-adamw-random-seed${RSEED}"
    if [ ! -d "${RANDOM_DIR}/checkpoint-initial" ]; then
        python cli.py distributed-grpo-train \
            "${COMMON_FLAGS[@]}" \
            --output-dir "${RANDOM_DIR}" \
            --precision mixed \
            --gradient-mask "${MASK_DIR}/random_mask_seed${RSEED}.pt" \
            --wandb-run "random_mask_seed${RSEED}"
    else
        echo "Skipping: ${RANDOM_DIR} already exists"
    fi
done

echo "========================================="
echo "All lottery ticket experiments completed!"
echo "========================================="
