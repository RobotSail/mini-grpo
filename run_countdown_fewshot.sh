#!/bin/bash
set -euo pipefail

# ── Countdown GRPO with 5-shot ICL, sequence-level averaging ─────────────
#
# Step 1: Generate data with few-shot examples (run once)
# Step 2: Train with GRPO (distributed, 2 train + 6 vLLM GPUs)

MODEL="Qwen/Qwen2-1.5B-Instruct"
DATA_DIR="generated_data/countdown_fewshot5_full_synthetic"
OUTPUT_DIR="/mnt/nvme0n1/experiments/grpo-countdown/countdown-fewshot5-grpo-adamw-full-synthetic-v2"

# ── Hyperparameters ──
# scale LR by sqrt(ebs_new/ebs_old) = 1024/128 = sqrt(8) = 2 sqrt(2) ~= 8.5e-7 
LR='8.5e-7'
OPTIMIZER=muon
# old size: 32
BATCH_SIZE=32       # prompts per rollout iteration
# old size: 16 --> 128, so num samples: 512 * 8 = 4096 
GROUP_SIZE=128       # rollouts per prompt
# we need to keep num optimizer steps identical, so IBS=512/128=4, we need 4096/IBS=4 => IBS=4096/4=1024
# old batchsize
# INNER_BATCH_SIZE=128   
INNER_BATCH_SIZE=1024 
INNER_EPOCHS=2
MAX_STEPS=100_000  # we can kill it earlier if we need
SAVE_EVERY=100      # checkpoint every N steps
EVAL_EVERY=2      # validate every N steps (via save_every_n_steps)
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
MAX_TOKENS_PER_GPU=30000

# ── GPU layout ──
TRAIN_GPUS="0,1,2,3"
VLLM_GPUS="4,5,6,7"

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
    --wandb-run "test-bs-128-gs-4_adamw" \
    --best-val-ckpt-only \
    --overwrite-best-ckpt \
    ${TRAIN_FLAGS}
