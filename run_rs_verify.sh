#!/bin/bash

set -eo pipefail

# Rejection Sampling verification experiments (3 seeds)
# Matches GRPO verify seed assignments from runall.sh
# Seed 1 (1738) already done in original run_rs_train.sh

CHECKPOINT_PREFIX="/mnt/nvme2n1/checkpoints/verify-exps-variable-seeds"
DATA_PATH="adamw-vs-muon-grpo-v1-artifacts/generated-data-v2/gsm8k_grpo_train.jsonl"
VALIDATION_PATH="adamw-vs-muon-grpo-v1-artifacts/generated-data-v2/gsm8k_grpo_test.jsonl"
MODEL="Qwen/Qwen2-1.5B-Instruct"

# RS-specific hyperparameters (matching original run_rs_train.sh)
TOKEN_BUDGET=2400000
SAVE_EVERY=150000
LR=1e-6
SAMPLES_TO_ACCEPT=64
INFERENCE_BATCH_SIZE=32
INFERENCE_GROUP_SIZE=16
MAX_SEQ_LEN=8192
MAX_TOKENS_PER_GPU=8192
TEMPERATURE=0.7
MAX_NEW_TOKENS=512

# Seed assignments (matching GRPO verify in runall.sh)
# verify_1: muon=1738, adamw=1738 (already done in run_rs_train.sh)
# verify_2: muon=420,  adamw=1337
# verify_3: muon=1337, adamw=420

echo "========================================="
echo "RS Verify Seed 2: Muon (seed=420)"
echo "========================================="
MASTER_PORT=29500 python cli.py rs-train \
    --data-path "${DATA_PATH}" \
    --model "${MODEL}" \
    --output-dir "${CHECKPOINT_PREFIX}/qwen2-1.5b-gsm8k-rs-muon_verify_2" \
    --max-tokens ${TOKEN_BUDGET} \
    --save-every-n-tokens ${SAVE_EVERY} \
    --samples-to-accept ${SAMPLES_TO_ACCEPT} \
    --inference-batch-size ${INFERENCE_BATCH_SIZE} \
    --inference-group-size ${INFERENCE_GROUP_SIZE} \
    --max-seq-len ${MAX_SEQ_LEN} \
    --max-tokens-per-gpu ${MAX_TOKENS_PER_GPU} \
    --temp ${TEMPERATURE} \
    --max-new-tokens ${MAX_NEW_TOKENS} \
    --optimizer muon \
    --lr ${LR} \
    --seed 420 \
    --gpu 0 \
    --vllm-gpus "1,2,3,4,5,6,7" \
    --wandb \
    --wandb-project gsm8k-comparison \
    --wandb-run "verify-overfit_rs-muon_2" \
    --validation-path "${VALIDATION_PATH}"

echo "========================================="
echo "RS Verify Seed 2: AdamW (seed=1337)"
echo "========================================="
MASTER_PORT=29510 python cli.py rs-train \
    --data-path "${DATA_PATH}" \
    --model "${MODEL}" \
    --output-dir "${CHECKPOINT_PREFIX}/qwen2-1.5b-gsm8k-rs-adamw_verify_2" \
    --max-tokens ${TOKEN_BUDGET} \
    --save-every-n-tokens ${SAVE_EVERY} \
    --samples-to-accept ${SAMPLES_TO_ACCEPT} \
    --inference-batch-size ${INFERENCE_BATCH_SIZE} \
    --inference-group-size ${INFERENCE_GROUP_SIZE} \
    --max-seq-len ${MAX_SEQ_LEN} \
    --max-tokens-per-gpu ${MAX_TOKENS_PER_GPU} \
    --temp ${TEMPERATURE} \
    --max-new-tokens ${MAX_NEW_TOKENS} \
    --optimizer adamw \
    --lr ${LR} \
    --seed 1337 \
    --gpu 0 \
    --vllm-gpus "1,2,3,4,5,6,7" \
    --wandb \
    --wandb-project gsm8k-comparison \
    --wandb-run "verify-overfit_rs-adamw_2" \
    --validation-path "${VALIDATION_PATH}"

echo "========================================="
echo "RS Verify Seed 3: Muon (seed=1337)"
echo "========================================="
MASTER_PORT=29500 python cli.py rs-train \
    --data-path "${DATA_PATH}" \
    --model "${MODEL}" \
    --output-dir "${CHECKPOINT_PREFIX}/qwen2-1.5b-gsm8k-rs-muon_verify_3" \
    --max-tokens ${TOKEN_BUDGET} \
    --save-every-n-tokens ${SAVE_EVERY} \
    --samples-to-accept ${SAMPLES_TO_ACCEPT} \
    --inference-batch-size ${INFERENCE_BATCH_SIZE} \
    --inference-group-size ${INFERENCE_GROUP_SIZE} \
    --max-seq-len ${MAX_SEQ_LEN} \
    --max-tokens-per-gpu ${MAX_TOKENS_PER_GPU} \
    --temp ${TEMPERATURE} \
    --max-new-tokens ${MAX_NEW_TOKENS} \
    --optimizer muon \
    --lr ${LR} \
    --seed 1337 \
    --gpu 0 \
    --vllm-gpus "1,2,3,4,5,6,7" \
    --wandb \
    --wandb-project gsm8k-comparison \
    --wandb-run "verify-overfit_rs-muon_3" \
    --validation-path "${VALIDATION_PATH}"

echo "========================================="
echo "RS Verify Seed 3: AdamW (seed=420)"
echo "========================================="
MASTER_PORT=29510 python cli.py rs-train \
    --data-path "${DATA_PATH}" \
    --model "${MODEL}" \
    --output-dir "${CHECKPOINT_PREFIX}/qwen2-1.5b-gsm8k-rs-adamw_verify_3" \
    --max-tokens ${TOKEN_BUDGET} \
    --save-every-n-tokens ${SAVE_EVERY} \
    --samples-to-accept ${SAMPLES_TO_ACCEPT} \
    --inference-batch-size ${INFERENCE_BATCH_SIZE} \
    --inference-group-size ${INFERENCE_GROUP_SIZE} \
    --max-seq-len ${MAX_SEQ_LEN} \
    --max-tokens-per-gpu ${MAX_TOKENS_PER_GPU} \
    --temp ${TEMPERATURE} \
    --max-new-tokens ${MAX_NEW_TOKENS} \
    --optimizer adamw \
    --lr ${LR} \
    --seed 420 \
    --gpu 0 \
    --vllm-gpus "1,2,3,4,5,6,7" \
    --wandb \
    --wandb-project gsm8k-comparison \
    --wandb-run "verify-overfit_rs-adamw_3" \
    --validation-path "${VALIDATION_PATH}"

echo "========================================="
echo "All RS verification experiments completed!"
echo "========================================="
