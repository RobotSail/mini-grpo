#!/bin/bash

set -eo pipefail

export WANDB_API_KEY=dcc7e9d67dd4454320776959ba154a9d285cb7db

CHECKPOINT_PREFIX="/mnt/nvme2n1/checkpoints/lr-sweep-3m_round-2"
DATA_PATH="adamw-vs-muon-grpo-v1-artifacts/generated-data-v2/gsm8k_grpo_train.jsonl"
VALIDATION_PATH="adamw-vs-muon-grpo-v1-artifacts/generated-data-v2/gsm8k_grpo_test.jsonl"
MODEL="Qwen/Qwen2-1.5B-Instruct"

SEED=2020
TOKEN_BUDGET=3000000
SAVE_EVERY=100000

# Common args shared by all runs
COMMON_ARGS="--data-path ${DATA_PATH} --model ${MODEL} \
    --max-tokens ${TOKEN_BUDGET} --save-every-n-tokens ${SAVE_EVERY} \
    --group-size 8 --batch-size 8 --inner-batch-size 64 \
    --clip-eps 0.2 --kl 0 --temp 0.7 --max-new-tokens 512 \
    --max-tokens-per-microbatch 4096 --precision mixed --seed ${SEED} \
    --gpu 0 --vllm-gpus 1,2,3,4,5,6,7 \
    --wandb --wandb-project gsm8k-comparison \
    --validation-path ${VALIDATION_PATH}"

LRS="1e-7 2.5e-7 5e-7 7.5e-7 1e-6"

# ── Muon sweep ───────────────────────────────────────────────────
for LR in ${LRS}; do
    echo "========================================="
    echo "Muon lr=${LR} (seed=${SEED})"
    echo "========================================="
    python cli.py grpo-train ${COMMON_ARGS} \
        --output-dir "${CHECKPOINT_PREFIX}/grpo-muon-${LR}" \
        --optimizer muon --lr ${LR} \
        --wandb-run "r2_grpo-muon_lr${LR}"
done

# ── AdamW sweep ──────────────────────────────────────────────────
for LR in ${LRS}; do
    echo "========================================="
    echo "AdamW lr=${LR} (seed=${SEED})"
    echo "========================================="
    python cli.py grpo-train ${COMMON_ARGS} \
        --output-dir "${CHECKPOINT_PREFIX}/grpo-adamw-${LR}" \
        --optimizer adamw --lr ${LR} \
        --wandb-run "r2_grpo-adamw_lr${LR}"
done

echo "========================================="
echo "All LR Sweep Round 2 experiments completed!"
echo "========================================="
