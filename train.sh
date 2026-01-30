#!/bin/bash

set -eo pipefail

# Common settings
MODEL="Qwen/Qwen2-1.5B-Instruct"
SFT_DATA="generated-data-v2/gsm8k_sft_train.jsonl"
GRPO_DATA="generated-data-v2/gsm8k_grpo_train.jsonl"
TOKEN_BUDGET=1200000
SAVE_TOKENS=100000
LR=5e-6
SEED=67
OUTPUT_DIR="./checkpoints-lr-5e-6"

# # 1. SFT + AdamW
python cli.py sft-train \
    --data-path $SFT_DATA \
    --model $MODEL \
    --output-dir "${OUTPUT_DIR}/sft-adamw" \
    --optimizer adamw \
    --lr $LR \
    --lr-scheduler constant \
    --batch-size 64 \
    --max-tokens-per-gpu 8192 \
    --max-tokens $TOKEN_BUDGET \
    --save-every-n-tokens $SAVE_TOKENS \
    --seed $SEED \
    --num-gpus 8 \
    --wandb --wandb-project gsm8k-comparison --wandb-run "sft-adamw"

# # 2. SFT + Muon
python cli.py sft-train \
    --data-path $SFT_DATA \
    --model $MODEL \
    --output-dir "${OUTPUT_DIR}/sft-muon" \
    --optimizer muon \
    --lr $LR \
    --batch-size 64 \
    --lr-scheduler constant \
    --max-tokens-per-gpu 8192 \
    --max-tokens $TOKEN_BUDGET \
    --save-every-n-tokens $SAVE_TOKENS \
    --seed $SEED \
    --num-gpus 8 \
    --wandb --wandb-project gsm8k-comparison --wandb-run "sft-muon"

# 3. GRPO + AdamW
python cli.py train \
    --train-path $GRPO_DATA \
    --model-name $MODEL \
    --output-dir "${OUTPUT_DIR}/grpo-adamw" \
    --optimizer adamw \
    --lr $LR \
    --batch-size 8 \
    --group-size 8 \
    --inner-batch-size 64 \
    --token-train-budget $TOKEN_BUDGET \
    --save-every-n-tokens $SAVE_TOKENS \
    --kl 0 \
    --flash-attn \
    --seed $SEED \
    --gpu 0 \
    --max-tokens-per-microbatch 8192 \
    --wandb --wandb-project gsm8k-comparison --wandb-run "grpo-adamw"

# 4. GRPO + Muon
python cli.py train \
    --train-path $GRPO_DATA \
    --model-name $MODEL \
    --output-dir "${OUTPUT_DIR}/grpo-muon" \
    --optimizer muon \
    --lr $LR \
    --batch-size 8 \
    --group-size 8 \
    --inner-batch-size 64 \
    --token-train-budget $TOKEN_BUDGET \
    --save-every-n-tokens $SAVE_TOKENS \
    --flash-attn \
    --seed $SEED \
    --kl 0 \
    --gpu 0 \
    --max-tokens-per-microbatch 8192 \
    --wandb --wandb-project gsm8k-comparison --wandb-run "grpo-muon"
