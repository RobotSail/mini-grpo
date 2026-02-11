#!/bin/bash

set -eo pipefail

# Checkpoint directory prefix
CHECKPOINT_PREFIX="/mnt/nvme2n1/checkpoints/rs-train-exps"

# Data path
DATA_PATH="adamw-vs-muon-grpo-v1-artifacts/generated-data-v2/gsm8k_grpo_train.jsonl"
VALIDATION_PATH="adamw-vs-muon-grpo-v1-artifacts/generated-data-v2/gsm8k_grpo_test.jsonl"  # even though this is called test it is actually our dev set

# Common training parameters
MODEL="Qwen/Qwen2-1.5B-Instruct"
TOKEN_BUDGET=2400000
SAVE_EVERY=150000
LR=1e-6
SEED=1738
SAMPLES_TO_ACCEPT=64
INFERENCE_BATCH_SIZE=32
INFERENCE_GROUP_SIZE=16
MAX_SEQ_LEN=8192
MAX_TOKENS_PER_GPU=8192
TEMPERATURE=0.7
MAX_NEW_TOKENS=512

echo "Starting Rejection Sampling training experiments..."

# Muon on GPUs 0,1
MASTER_PORT=29500 python cli.py rs-train \
    --data-path "${DATA_PATH}" \
    --model "${MODEL}" \
    --output-dir "${CHECKPOINT_PREFIX}/qwen2-1.5b-gsm8k-rs-muon" \
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
    --seed ${SEED} \
    --gpu 0 \
    --vllm-gpus "1,2,3,4,5,6,7" \
    --wandb \
    --wandb-project gsm8k-comparison \
    --wandb-run "rs-muon" \
    --validation-path "${VALIDATION_PATH}"

# wait 

# AdamW on GPUs 0,4,5,6,7
MASTER_PORT=29510 python cli.py rs-train \
    --data-path "${DATA_PATH}" \
    --model "${MODEL}" \
    --output-dir "${CHECKPOINT_PREFIX}/qwen2-1.5b-gsm8k-rs-adamw" \
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
    --seed ${SEED} \
    --gpu 0 \
    --vllm-gpus "4,5,6,7" \
    --wandb \
    --wandb-project gsm8k-comparison \
    --wandb-run "rs-adamw" \
    --validation-path "${VALIDATION_PATH}"




# echo "Waiting for both runs to complete..."
# wait

echo "All Rejection Sampling training jobs completed!"
