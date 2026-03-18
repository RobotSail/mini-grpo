#!/bin/bash
set -euo pipefail

# ── Countdown GRPO (hard problems requiring * or /) ──────────────────────
#
# Step 1: Generate data   (run once)
# Step 2: Train with GRPO (distributed, 2 train + 6 vLLM GPUs)

MODEL="Qwen/Qwen2-1.5B-Instruct"
DATA_DIR="generated_data"
OUTPUT_DIR="experiments/countdown-hard-grpo-adamw"

# ── Hyperparameters ──
LR=3e-7
OPTIMIZER=adamw
BATCH_SIZE=32       # prompts per rollout iteration
GROUP_SIZE=16       # rollouts per prompt
INNER_BATCH_SIZE=128
INNER_EPOCHS=2
MAX_STEPS=5000
SAVE_EVERY=500      # checkpoint every N steps
EVAL_EVERY=25       # validate every N steps (via save_every_n_steps)
TEMPERATURE=1.0
KL=0.0
FORMAT_REWARD=0.1
MAX_NEW_TOKENS=1024
SEED=42
NUM_TRAIN_SAMPLES=10000
NUM_VALIDATION_SAMPLES=1000

# ── GPU layout ──
TRAIN_GPUS="0,1"
VLLM_GPUS="2,3,4,5,6,7"

# ── Step 1: Generate synthetic hard countdown dataset ─────────────────────
if [ ! -f "${DATA_DIR}/countdown_hard_train.jsonl" ]; then
    echo "=== Generating synthetic countdown dataset ==="
    python cli.py generate-hard-countdown \
        --n-train "${NUM_TRAIN_SAMPLES}" \
        --n-val "${NUM_VALIDATION_SAMPLES}" \
        --seed ${SEED} \
        --output-dir ${DATA_DIR}
    echo ""
fi

# ── Step 2: Train ────────────────────────────────────────────────────────
echo "=== Starting countdown GRPO training ==="
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
    --train-gpus "${TRAIN_GPUS}" \
    --vllm-gpus "${VLLM_GPUS}" \
    --seed ${SEED} \
    --wandb \
    --wandb-project "countdown-grpo" \
    --token-level-avg
