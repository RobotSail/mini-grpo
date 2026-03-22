#!/bin/bash
set -euo pipefail

# ── Countdown GRPO with 5-shot ICL, sequence-level averaging ─────────────
#
# Step 1: Generate data with few-shot examples (run once)
# Step 2: Train with GRPO (distributed, 2 train + 6 vLLM GPUs)

MODEL="Qwen/Qwen2-1.5B-Instruct"
DATA_DIR="generated_data/countdown_fewshot5_full_synthetic"
OUTPUT_DIR="/mnt/nvme0/experiments/grpo-countdown/countdown-fewshot5-grpo-adamw-full-synthetic-v2"

# ── Hyperparameters ──
LR=3e-7
OPTIMIZER=adamw
BATCH_SIZE=32       # prompts per rollout iteration
GROUP_SIZE=16       # rollouts per prompt
INNER_BATCH_SIZE=128
INNER_EPOCHS=2
MAX_STEPS=100_000  # we can kill it earlier if we need
SAVE_EVERY=100      # checkpoint every N steps
EVAL_EVERY=100       # validate every N steps (via save_every_n_steps)
TEMPERATURE=1.0
KL=0.0
FORMAT_REWARD=0.1
MAX_NEW_TOKENS=1024
SEED=42
NUM_TRAIN_SAMPLES=200000
NUM_VALIDATION_SAMPLES=2000
NUM_TEST_SAMPLES=20_000
FEW_SHOT=5
HARD=0
THINK=0
SYNTHETIC=1
R1_PROMPT=0
MAX_TOKENS_PER_GPU=20000

# ── GPU layout ──
TRAIN_GPUS="0,1"
VLLM_GPUS="2,3,4,5,6,7"

# ── Build flags ──
GEN_FLAGS=""
TRAIN_FLAGS=""

if [ "${HARD}" -eq 0 ]; then
    GEN_FLAGS="${GEN_FLAGS} --all"
fi

if [ "${SYNTHETIC}" -eq 1 ]; then
    GEN_FLAGS="${GEN_FLAGS} --synthetic"
fi

if [ "${R1_PROMPT}" -eq 1 ]; then
    R1_PROMPT_FLAGS="${R1_PROMPT_FLAGS} --r1-prompt"
fi

if [ "${THINK}" -eq 1 ]; then
    GEN_FLAGS="${GEN_FLAGS} --think"
    TRAIN_FLAGS="${TRAIN_FLAGS} --think"
fi

# ── Step 1: Generate synthetic countdown dataset with few-shot ICL ───────
if [ ! -f "${DATA_DIR}/countdown_hard_train.jsonl" ]; then
    echo "=== Generating synthetic countdown dataset (${FEW_SHOT}-shot ICL, hard=${HARD}, think=${THINK}) ==="
    python cli.py generate-hard-countdown \
        --n-train "${NUM_TRAIN_SAMPLES}" \
        --n-val "${NUM_VALIDATION_SAMPLES}" \
        --n-test "${NUM_TEST_SAMPLES}" \
        --few-shot "${FEW_SHOT}" \
        --seed ${SEED} \
        --output-dir ${DATA_DIR} \
        ${GEN_FLAGS}
    echo ""
fi

# ── Step 2: Train (sequence-level averaging) ─────────────────────────────
echo "=== Starting countdown GRPO training (${FEW_SHOT}-shot, hard=${HARD}, think=${THINK}) ==="
python cli.py distributed-grpo-train \
    --data-path "${DATA_DIR}/countdown_hard_train.jsonl" \
    --validation-path "${DATA_DIR}/countdown_hard_val.jsonl" \
    --output-dir "${OUTPUT_DIR}" \
    --model "${MODEL}" \
    --task countdown \
    --max-steps ${MAX_STEPS} \
    --save-every-n-steps ${EVAL_EVERY} \
    --optimizer ${OPTIMIZER} \
    --lr ${LR} \
    --batch-size ${BATCH_SIZE} \
    --group-size ${GROUP_SIZE} \
    --inner-batch-size ${INNER_BATCH_SIZE} \
    --inner-epochs ${INNER_EPOCHS} \
    --temp ${TEMPERATURE} \
    --max-new-tokens ${MAX_NEW_TOKENS} \
    --kl ${KL} \
    --format-reward ${FORMAT_REWARD} \
    --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU}" \
    --train-gpus "${TRAIN_GPUS}" \
    --vllm-gpus "${VLLM_GPUS}" \
    --seed ${SEED} \
    --wandb \
    --wandb-project "countdown-grpo" \
    --best-val-ckpt-only \
    ${TRAIN_FLAGS}
