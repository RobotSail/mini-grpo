#!/bin/bash

set -eo pipefail

BASE_PORT=29500

# # Run 3 muon jobs in parallel on GPUs 0-2
# for i in {1..3}; do
#     gpu_id=$((i - 1))
#     port=$((BASE_PORT + gpu_id))
#     (
#         CUDA_VISIBLE_DEVICES=${gpu_id} MASTER_PORT=${port} python cli.py train --train-path adamw-vs-muon-grpo-v1-artifacts/generated-data-v2/gsm8k_grpo_train.jsonl --model-name Qwen/Qwen2-1.5B-Instruct --output-dir "./verify_run_${i}/muon_grpo" --optimizer muon --lr 1e-6 --batch-size 8 --group-size 8 --inner-batch-size 64 --token-train-budget 1200000 --save-every-n-tokens 100000 --flash-attn --seed 67 --kl 0 --gpu 0 --max-tokens-per-microbatch 8192 --wandb --wandb-project gsm8k-comparison --wandb-run "grpo-muon_verify-${i}"
#     ) &
# done
# 
# # Run 3 adamw jobs in parallel on GPUs 3-5
# for i in {1..3}; do
#     gpu_id=$((i + 2))
#     port=$((BASE_PORT + gpu_id))
#     (
#         CUDA_VISIBLE_DEVICES=${gpu_id} MASTER_PORT=${port} python cli.py train --train-path adamw-vs-muon-grpo-v1-artifacts/generated-data-v2/gsm8k_grpo_train.jsonl --model-name Qwen/Qwen2-1.5B-Instruct --output-dir "./verify_run_${i}/adamw_grpo" --optimizer adamw --lr 1e-6 --batch-size 8 --group-size 8 --inner-batch-size 64 --token-train-budget 1200000 --save-every-n-tokens 100000 --flash-attn --seed 67 --kl 0 --gpu 0 --max-tokens-per-microbatch 8192 --wandb --wandb-project gsm8k-comparison --wandb-run "grpo-adamw_verify-${i}"
#     ) &
# done

# Wait for all background jobs to complete
# wait
# 
# echo "All training jobs completed!"

# SFT training commands (commented out)
# for i in {1..3}; do
#     python cli.py sft-train --data-path adamw-vs-muon-grpo-v1-artifacts/generated-data-v2/gsm8k_sft_train.jsonl --model Qwen/Qwen2-1.5B-Instruct --output-dir "./verify_run_${i}/muon_sft" --optimizer muon --lr 1e-6 --batch-size 64 --max-tokens 1200000 --save-every-n-tokens 100000  --seed 67 --lr-scheduler 'constant'  --max-tokens-per-gpu 8192 --wandb --wandb-project gsm8k-comparison --wandb-run "sft-muon_verify-${i}" --num-gpus 8 --seed 67
#     python cli.py sft-train --data-path adamw-vs-muon-grpo-v1-artifacts/generated-data-v2/gsm8k_sft_train.jsonl --model Qwen/Qwen2-1.5B-Instruct --output-dir "./verify_run_${i}/adamw_sft" --optimizer adamw --lr 1e-6 --batch-size 64 --max-tokens 1200000 --save-every-n-tokens 100000  --seed 67 --lr-scheduler 'constant'  --max-tokens-per-gpu 8192 --wandb --wandb-project gsm8k-comparison --wandb-run "sft-adamw_verify-${i}" --num-gpus 8 --seed 67
# done

# ----------------------------------------
# Run 3 muon jobs in parallel on GPUs 0-2
# for i in {1..3}; do
#     gpu_id=$((i - 1))
#     port=$((BASE_PORT + gpu_id))
#     (
#         CUDA_VISIBLE_DEVICES=${gpu_id} MASTER_PORT=${port} python cli.py train --train-path adamw-vs-muon-grpo-v1-artifacts/generated-data-v2/gsm8k_grpo_train.jsonl --model-name Qwen/Qwen2-1.5B-Instruct --output-dir "./verify_run_${i}/muon_grpo" --optimizer muon --lr 1e-6 --batch-size 8 --group-size 8 --inner-batch-size 64 --token-train-budget 1200000 --save-every-n-tokens 100000 --flash-attn --seed 67 --kl 0 --gpu 0 --max-tokens-per-microbatch 8192 --wandb --wandb-project gsm8k-comparison --wandb-run "grpo-muon_verify-${i}"
#     ) &
# done
# 
# # Run 3 adamw jobs in parallel on GPUs 3-5
# for i in {1..3}; do
#     gpu_id=$((i + 2))
#     port=$((BASE_PORT + gpu_id))
#     (
#         CUDA_VISIBLE_DEVICES=${gpu_id} MASTER_PORT=${port} python cli.py train --train-path adamw-vs-muon-grpo-v1-artifacts/generated-data-v2/gsm8k_grpo_train.jsonl --model-name Qwen/Qwen2-1.5B-Instruct --output-dir "./verify_run_${i}/adamw_grpo" --optimizer adamw --lr 1e-6 --batch-size 8 --group-size 8 --inner-batch-size 64 --token-train-budget 1200000 --save-every-n-tokens 100000 --flash-attn --seed 67 --kl 0 --gpu 0 --max-tokens-per-microbatch 8192 --wandb --wandb-project gsm8k-comparison --wandb-run "grpo-adamw_verify-${i}"
#     ) &
# done
# 
# # Wait for all background jobs to complete
# wait

# Checkpoint directory prefix
CHECKPOINT_PREFIX="/mnt/nvme2n1/checkpoints/verify-exps-variable-seeds"

# # new iteration 1
# CUDA_VISIBLE_DEVICES=0 MASTER_PORT=29400 python cli.py train \
#         --train-path adamw-vs-muon-grpo-v1-artifacts/generated-data-v2/gsm8k_grpo_train.jsonl \
#         --model-name Qwen/Qwen2-1.5B-Instruct \
#         --output-dir "${CHECKPOINT_PREFIX}/qwen2-1.5b-gsm8k-grpo-muon_verify_1" \
#         --optimizer muon \
#         --lr 1e-6 \
#         --batch-size 8 \
#         --group-size 8 \
#         --inner-batch-size 64 \
#         --token-train-budget 2400000 \
#         --save-every-n-tokens 150000 \
#         --flash-attn \
#         --seed 1738 \
#         --kl 0 \
#         --gpu 0 \
#         --max-tokens-per-microbatch 4096 \
#         --wandb \
#         --wandb-project gsm8k-comparison \
#         --wandb-run "verify-overfit_grpo-muon_1" &

# CUDA_VISIBLE_DEVICES=1 MASTER_PORT=29410 python cli.py train \
#         --train-path adamw-vs-muon-grpo-v1-artifacts/generated-data-v2/gsm8k_grpo_train.jsonl \
#         --model-name Qwen/Qwen2-1.5B-Instruct \
#         --output-dir "${CHECKPOINT_PREFIX}/qwen2-1.5b-gsm8k-grpo-adamw_verify_1" \
#         --optimizer adamw \
#         --lr 1e-6 \
#         --batch-size 8 \
#         --group-size 8 \
#         --inner-batch-size 64 \
#         --token-train-budget 2400000 \
#         --save-every-n-tokens 150000 \
#         --flash-attn \
#         --seed 1738 \
#         --kl 0 \
#         --gpu 0 \
#         --max-tokens-per-microbatch 4096 \
#         --wandb \
#         --wandb-project gsm8k-comparison \
#         --wandb-run "verify-overfit_grpo-adamw_1" &

# # iteration 2
# CUDA_VISIBLE_DEVICES=2 MASTER_PORT=29420 python cli.py train \
#         --train-path adamw-vs-muon-grpo-v1-artifacts/generated-data-v2/gsm8k_grpo_train.jsonl \
#         --model-name Qwen/Qwen2-1.5B-Instruct \
#         --output-dir "${CHECKPOINT_PREFIX}/qwen2-1.5b-gsm8k-grpo-muon_verify_2" \
#         --optimizer muon \
#         --lr 1e-6 \
#         --batch-size 8 \
#         --group-size 8 \
#         --inner-batch-size 64 \
#         --token-train-budget 2400000 \
#         --save-every-n-tokens 150000 \
#         --flash-attn \
#         --seed 420 \
#         --kl 0 \
#         --gpu 0 \
#         --max-tokens-per-microbatch 4096 \
#         --wandb \
#         --wandb-project gsm8k-comparison \
#         --wandb-run "verify-overfit_grpo-muon_2" &

# # iteration 2
# CUDA_VISIBLE_DEVICES=3 MASTER_PORT=29430 python cli.py train \
#         --train-path adamw-vs-muon-grpo-v1-artifacts/generated-data-v2/gsm8k_grpo_train.jsonl \
#         --model-name Qwen/Qwen2-1.5B-Instruct \
#         --output-dir "${CHECKPOINT_PREFIX}/qwen2-1.5b-gsm8k-grpo-adamw_verify_2" \
#         --optimizer adamw \
#         --lr 1e-6 \
#         --batch-size 8 \
#         --group-size 8 \
#         --inner-batch-size 64 \
#         --token-train-budget 2400000 \
#         --save-every-n-tokens 150000 \
#         --flash-attn \
#         --seed 1337 \
#         --kl 0 \
#         --gpu 0 \
#         --max-tokens-per-microbatch 4096 \
#         --wandb \
#         --wandb-project gsm8k-comparison \
#         --wandb-run "verify-overfit_grpo-adamw_2" &

# # iteration 2
# CUDA_VISIBLE_DEVICES=4 MASTER_PORT=29440 python cli.py train \
#         --train-path adamw-vs-muon-grpo-v1-artifacts/generated-data-v2/gsm8k_grpo_train.jsonl \
#         --model-name Qwen/Qwen2-1.5B-Instruct \
#         --output-dir "${CHECKPOINT_PREFIX}/qwen2-1.5b-gsm8k-grpo-muon_verify_3" \
#         --optimizer muon \
#         --lr 1e-6 \
#         --batch-size 8 \
#         --group-size 8 \
#         --inner-batch-size 64 \
#         --token-train-budget 2400000 \
#         --save-every-n-tokens 150000 \
#         --flash-attn \
#         --seed 1337 \
#         --kl 0 \
#         --gpu 0 \
#         --max-tokens-per-microbatch 4096 \
#         --wandb \
#         --wandb-project gsm8k-comparison \
#         --wandb-run "verify-overfit_grpo-muon_3" &

# # iteration 2
# CUDA_VISIBLE_DEVICES=5 MASTER_PORT=29450 python cli.py train \
#         --train-path adamw-vs-muon-grpo-v1-artifacts/generated-data-v2/gsm8k_grpo_train.jsonl \
#         --model-name Qwen/Qwen2-1.5B-Instruct \
#         --output-dir "${CHECKPOINT_PREFIX}/qwen2-1.5b-gsm8k-grpo-adamw_verify_3" \
#         --optimizer adamw \
#         --lr 1e-6 \
#         --batch-size 8 \
#         --group-size 8 \
#         --inner-batch-size 64 \
#         --token-train-budget 2400000 \
#         --save-every-n-tokens 150000 \
#         --flash-attn \
#         --seed 420 \
#         --kl 0 \
#         --gpu 0 \
#         --max-tokens-per-microbatch 4096 \
#         --wandb \
#         --wandb-project gsm8k-comparison \
#         --wandb-run "verify-overfit_grpo-adamw_3" &


# printf 'waiting for GRPO processes to complete\n'
# wait 
# printf 'GRPO training complete'

# # sft iteration 1
# python cli.py sft-train \
#         --data-path adamw-vs-muon-grpo-v1-artifacts/generated-data-v2/gsm8k_sft_train.jsonl \
#         --model Qwen/Qwen2-1.5B-Instruct \
#         --output-dir "${CHECKPOINT_PREFIX}/qwen2-1.5b-gsm8k-sft-muon_verify_1" \
#         --optimizer muon \
#         --lr 1e-6 \
#         --batch-size 64 \
#         --max-tokens 2400000 \
#         --save-every-n-tokens 150000  \
#         --seed 1738 \
#         --lr-scheduler 'constant'  \
#         --max-tokens-per-gpu 8192 \
#         --wandb \
#         --wandb-project gsm8k-comparison \
#         --wandb-run "verify-overfit_sft-muon_1" \
#         --num-gpus 8 \
#         --seed 67

# python cli.py sft-train \
#         --data-path adamw-vs-muon-grpo-v1-artifacts/generated-data-v2/gsm8k_sft_train.jsonl \
#         --model Qwen/Qwen2-1.5B-Instruct \
#         --output-dir "${CHECKPOINT_PREFIX}/qwen2-1.5b-gsm8k-sft-adamw_verify_1" \
#         --optimizer adamw \
#         --lr 1e-6 \
#         --batch-size 64 \
#         --max-tokens 2400000 \
#         --save-every-n-tokens 150000  \
#         --seed 1738 \
#         --lr-scheduler 'constant'  \
#         --max-tokens-per-gpu 8192 \
#         --wandb \
#         --wandb-project gsm8k-comparison \
#         --wandb-run "verify-overfit_sft-adamw_1" \
#         --num-gpus 8 \
#         --seed 67

# # sft iteration 2
# python cli.py sft-train \
#         --data-path adamw-vs-muon-grpo-v1-artifacts/generated-data-v2/gsm8k_sft_train.jsonl \
#         --model Qwen/Qwen2-1.5B-Instruct \
#         --output-dir "${CHECKPOINT_PREFIX}/qwen2-1.5b-gsm8k-sft-muon_verify_2" \
#         --optimizer muon \
#         --lr 1e-6 \
#         --batch-size 64 \
#         --max-tokens 2400000 \
#         --save-every-n-tokens 150000  \
#         --seed 1337 \
#         --lr-scheduler 'constant'  \
#         --max-tokens-per-gpu 8192 \
#         --wandb \
#         --wandb-project gsm8k-comparison \
#         --wandb-run "verify-overfit_sft-muon_2" \
#         --num-gpus 8 \
#         --seed 1337

# python cli.py sft-train \
#         --data-path adamw-vs-muon-grpo-v1-artifacts/generated-data-v2/gsm8k_sft_train.jsonl \
#         --model Qwen/Qwen2-1.5B-Instruct \
#         --output-dir "${CHECKPOINT_PREFIX}/qwen2-1.5b-gsm8k-sft-adamw_verify_2" \
#         --optimizer adamw \
#         --lr 1e-6 \
#         --batch-size 64 \
#         --max-tokens 2400000 \
#         --save-every-n-tokens 150000  \
#         --seed 1337 \
#         --lr-scheduler 'constant'  \
#         --max-tokens-per-gpu 8192 \
#         --wandb \
#         --wandb-project gsm8k-comparison \
#         --wandb-run "verify-overfit_sft-adamw_2" \
#         --num-gpus 8 \
#         --seed 1337

# # sft iteration 3
# python cli.py sft-train \
#         --data-path adamw-vs-muon-grpo-v1-artifacts/generated-data-v2/gsm8k_sft_train.jsonl \
#         --model Qwen/Qwen2-1.5B-Instruct \
#         --output-dir "${CHECKPOINT_PREFIX}/qwen2-1.5b-gsm8k-sft-muon_verify_3" \
#         --optimizer muon \
#         --lr 1e-6 \
#         --batch-size 64 \
#         --max-tokens 2400000 \
#         --save-every-n-tokens 150000  \
#         --seed 420 \
#         --lr-scheduler 'constant'  \
#         --max-tokens-per-gpu 8192 \
#         --wandb \
#         --wandb-project gsm8k-comparison \
#         --wandb-run "verify-overfit_sft-muon_3" \
#         --num-gpus 8 \
#         --seed 420

# python cli.py sft-train \
#         --data-path adamw-vs-muon-grpo-v1-artifacts/generated-data-v2/gsm8k_sft_train.jsonl \
#         --model Qwen/Qwen2-1.5B-Instruct \
#         --output-dir "${CHECKPOINT_PREFIX}/qwen2-1.5b-gsm8k-sft-adamw_verify_3" \
#         --optimizer adamw \
#         --lr 1e-6 \
#         --batch-size 64 \
#         --max-tokens 2400000 \
#         --save-every-n-tokens 150000  \
#         --seed 420 \
#         --lr-scheduler 'constant'  \
#         --max-tokens-per-gpu 8192 \
#         --wandb \
#         --wandb-project gsm8k-comparison \
#         --wandb-run "verify-overfit_sft-adamw_3" \
#         --num-gpus 8 \
#         --seed 420

# # sft iteration 1
# python cli.py sft-train \
#         --data-path adamw-vs-muon-grpo-v1-artifacts/generated-data-v2/gsm8k_sft_train.jsonl \
#         --model Qwen/Qwen2-1.5B-Instruct \
#         --output-dir "${CHECKPOINT_PREFIX}/qwen2-1.5b-gsm8k-sft-muon_verify_1_fp32" \
#         --optimizer muon \
#         --lr 1e-6 \
#         --batch-size 64 \
#         --max-tokens 2400000 \
#         --save-every-n-tokens 150000  \
#         --seed 1738 \
#         --lr-scheduler 'constant'  \
#         --max-tokens-per-gpu 8192 \
#         --wandb \
#         --wandb-project gsm8k-comparison \
#         --wandb-run "verify-overfit_sft-muon_1_fp32" \
#         --num-gpus 8 \
#         --seed 67

# python cli.py sft-train \
#         --data-path adamw-vs-muon-grpo-v1-artifacts/generated-data-v2/gsm8k_sft_train.jsonl \
#         --model Qwen/Qwen2-1.5B-Instruct \
#         --output-dir "${CHECKPOINT_PREFIX}/qwen2-1.5b-gsm8k-sft-adamw_verify_1_fp32" \
#         --optimizer adamw \
#         --lr 1e-6 \
#         --batch-size 64 \
#         --max-tokens 2400000 \
#         --save-every-n-tokens 150000  \
#         --seed 1738 \
#         --lr-scheduler 'constant'  \
#         --max-tokens-per-gpu 8192 \
#         --wandb \
#         --wandb-project gsm8k-comparison \
#         --wandb-run "verify-overfit_sft-adamw_1_fp32" \
#         --num-gpus 8 \
#         --seed 67

# # sft iteration 2
# python cli.py sft-train \
#         --data-path adamw-vs-muon-grpo-v1-artifacts/generated-data-v2/gsm8k_sft_train.jsonl \
#         --model Qwen/Qwen2-1.5B-Instruct \
#         --output-dir "${CHECKPOINT_PREFIX}/qwen2-1.5b-gsm8k-sft-muon_verify_2_fp32" \
#         --optimizer muon \
#         --lr 1e-6 \
#         --batch-size 64 \
#         --max-tokens 2400000 \
#         --save-every-n-tokens 150000  \
#         --seed 1337 \
#         --lr-scheduler 'constant'  \
#         --max-tokens-per-gpu 8192 \
#         --wandb \
#         --wandb-project gsm8k-comparison \
#         --wandb-run "verify-overfit_sft-muon_2_fp32" \
#         --num-gpus 8 \
#         --seed 1337

# python cli.py sft-train \
#         --data-path adamw-vs-muon-grpo-v1-artifacts/generated-data-v2/gsm8k_sft_train.jsonl \
#         --model Qwen/Qwen2-1.5B-Instruct \
#         --output-dir "${CHECKPOINT_PREFIX}/qwen2-1.5b-gsm8k-sft-adamw_verify_2_fp32" \
#         --optimizer adamw \
#         --lr 1e-6 \
#         --batch-size 64 \
#         --max-tokens 2400000 \
#         --save-every-n-tokens 150000  \
#         --seed 1337 \
#         --lr-scheduler 'constant'  \
#         --max-tokens-per-gpu 8192 \
#         --wandb \
#         --wandb-project gsm8k-comparison \
#         --wandb-run "verify-overfit_sft-adamw_2_fp32" \
#         --num-gpus 8 \
#         --seed 1337

# # sft iteration 3
# python cli.py sft-train \
#         --data-path adamw-vs-muon-grpo-v1-artifacts/generated-data-v2/gsm8k_sft_train.jsonl \
#         --model Qwen/Qwen2-1.5B-Instruct \
#         --output-dir "${CHECKPOINT_PREFIX}/qwen2-1.5b-gsm8k-sft-muon_verify_3_fp32" \
#         --optimizer muon \
#         --lr 1e-6 \
#         --batch-size 64 \
#         --max-tokens 2400000 \
#         --save-every-n-tokens 150000  \
#         --seed 420 \
#         --lr-scheduler 'constant'  \
#         --max-tokens-per-gpu 8192 \
#         --wandb \
#         --wandb-project gsm8k-comparison \
#         --wandb-run "verify-overfit_sft-muon_3_fp32" \
#         --num-gpus 8 \
#         --seed 420

# python cli.py sft-train \
#         --data-path adamw-vs-muon-grpo-v1-artifacts/generated-data-v2/gsm8k_sft_train.jsonl \
#         --model Qwen/Qwen2-1.5B-Instruct \
#         --output-dir "${CHECKPOINT_PREFIX}/qwen2-1.5b-gsm8k-sft-adamw_verify_3_fp32" \
#         --optimizer adamw \
#         --lr 1e-6 \
#         --batch-size 64 \
#         --max-tokens 2400000 \
#         --save-every-n-tokens 150000  \
#         --seed 420 \
#         --lr-scheduler 'constant'  \
#         --max-tokens-per-gpu 8192 \
#         --wandb \
#         --wandb-project gsm8k-comparison \
#         --wandb-run "verify-overfit_sft-adamw_3_fp32" \
#         --num-gpus 8 \
#         --seed 420


# echo "All training jobs completed!"



echo "Training the FP32 vs BF16 sparsity experiments"

DATA_PATH="adamw-vs-muon-grpo-v1-artifacts/generated-data-v2/gsm8k_grpo_train.jsonl"
MODEL="Qwen/Qwen2-1.5B-Instruct"

# Round 1: AdamW FP32 (GPUs 0-3) + AdamW BF16 (GPUs 4-7)
# echo "=== AdamW FP32 + AdamW BF16 ==="
python cli.py grpo-train \
    --data-path "${DATA_PATH}" \
    --model "${MODEL}" \
    --output-dir "${CHECKPOINT_PREFIX}/sparsity/adamw_fp32" \
    --optimizer adamw --precision fp32 \
    --lr 1e-6 --batch-size 8 --group-size 8 --inner-batch-size 64 \
    --max-tokens 1200000 --save-every-n-tokens 150000 \
    --seed 67 --kl 0 \
    --gpu 0 --vllm-gpus "1,2,3" \
    --max-tokens-per-microbatch 4096 \
    --wandb --wandb-project gsm8k-comparison \
    --wandb-run "sparsity_grpo-adamw_fp32" &

python cli.py grpo-train \
    --data-path "${DATA_PATH}" \
    --model "${MODEL}" \
    --output-dir "${CHECKPOINT_PREFIX}/sparsity/adamw_bf16" \
    --optimizer adamw --precision bf16 \
    --lr 1e-6 --batch-size 8 --group-size 8 --inner-batch-size 64 \
    --max-tokens 1200000 --save-every-n-tokens 150000 \
    --seed 67 --kl 0 \
    --gpu 4 --vllm-gpus "5,6,7" \
    --max-tokens-per-microbatch 4096 \
    --wandb --wandb-project gsm8k-comparison \
    --wandb-run "sparsity_grpo-adamw_bf16" &

wait
echo "AdamW runs complete."

# Round 2: Muon FP32 (GPUs 0-3) + Muon BF16 (GPUs 4-7)
echo "=== Muon FP32 + Muon BF16 ==="
python cli.py grpo-train \
    --data-path "${DATA_PATH}" \
    --model "${MODEL}" \
    --output-dir "${CHECKPOINT_PREFIX}/sparsity/muon_fp32" \
    --optimizer muon --precision fp32 \
    --lr 1e-6 --batch-size 8 --group-size 8 --inner-batch-size 64 \
    --max-tokens 1200000 --save-every-n-tokens 150000 \
    --seed 67 --kl 0 \
    --gpu 0 --vllm-gpus "1,2,3" \
    --max-tokens-per-microbatch 4096 \
    --wandb --wandb-project gsm8k-comparison \
    --wandb-run "sparsity_grpo-muon_fp32" &

python cli.py grpo-train \
    --data-path "${DATA_PATH}" \
    --model "${MODEL}" \
    --output-dir "${CHECKPOINT_PREFIX}/sparsity/muon_bf16" \
    --optimizer muon --precision bf16 \
    --lr 1e-6 --batch-size 8 --group-size 8 --inner-batch-size 64 \
    --max-tokens 1200000 --save-every-n-tokens 150000 \
    --seed 67 --kl 0 \
    --gpu 4 --vllm-gpus "5,6,7" \
    --max-tokens-per-microbatch 4096 \
    --wandb --wandb-project gsm8k-comparison \
    --wandb-run "sparsity_grpo-muon_bf16" &

wait
echo "Muon runs complete."

# Round 3: Mixed precision GRPO — AdamW (GPUs 0-3) + Muon (GPUs 4-7)
# Each mixed-precision run needs its own MASTER_PORT for the mock distributed process group
echo "=== AdamW Mixed + Muon Mixed ==="
MASTER_PORT=29500 python cli.py grpo-train \
    --data-path "${DATA_PATH}" \
    --model "${MODEL}" \
    --output-dir "${CHECKPOINT_PREFIX}/sparsity/adamw_mixed" \
    --optimizer adamw --precision mixed \
    --lr 1e-6 --batch-size 8 --group-size 8 --inner-batch-size 64 \
    --max-tokens 1200000 --save-every-n-tokens 150000 \
    --seed 67 --kl 0 \
    --gpu 0 --vllm-gpus "1,2,3" \
    --max-tokens-per-microbatch 4096 \
    --wandb --wandb-project gsm8k-comparison \
    --wandb-run "sparsity_grpo-adamw_mixed" &

MASTER_PORT=29501 python cli.py grpo-train \
    --data-path "${DATA_PATH}" \
    --model "${MODEL}" \
    --output-dir "${CHECKPOINT_PREFIX}/sparsity/muon_mixed" \
    --optimizer muon --precision mixed \
    --lr 1e-6 --batch-size 8 --group-size 8 --inner-batch-size 64 \
    --max-tokens 1200000 --save-every-n-tokens 150000 \
    --seed 67 --kl 0 \
    --gpu 4 --vllm-gpus "5,6,7" \
    --max-tokens-per-microbatch 4096 \
    --wandb --wandb-project gsm8k-comparison \
    --wandb-run "sparsity_grpo-muon_mixed" &

wait
echo "All GRPO sparsity experiments completed!"

# ========================================
# SFT Sparsity Experiments
# ========================================
echo "Training the SFT sparsity experiments"

SFT_DATA_PATH="adamw-vs-muon-grpo-v1-artifacts/generated-data-v2/gsm8k_sft_train.jsonl"

# SFT Round 1: FP32
echo "=== SFT AdamW FP32 + Muon FP32 ==="
CUDA_VISIBLE_DEVICES=0 python cli.py sft-train \
    --data-path "${SFT_DATA_PATH}" \
    --model "${MODEL}" \
    --output-dir "${CHECKPOINT_PREFIX}/sparsity/sft_adamw_fp32" \
    --optimizer adamw --precision fp32 \
    --lr 1e-6 --batch-size 64 \
    --max-tokens 1200000 --save-every-n-tokens 150000 \
    --seed 67 --lr-scheduler constant \
    --max-tokens-per-gpu 4096 \
    --wandb --wandb-project gsm8k-comparison \
    --wandb-run "sparsity_sft-adamw_fp32" --num-gpus 1 &

CUDA_VISIBLE_DEVICES=4 python cli.py sft-train \
    --data-path "${SFT_DATA_PATH}" \
    --model "${MODEL}" \
    --output-dir "${CHECKPOINT_PREFIX}/sparsity/sft_muon_fp32" \
    --optimizer muon --precision fp32 \
    --lr 1e-6 --batch-size 64 \
    --max-tokens 1200000 --save-every-n-tokens 150000 \
    --seed 67 --lr-scheduler constant \
    --max-tokens-per-gpu 4096 \
    --wandb --wandb-project gsm8k-comparison \
    --wandb-run "sparsity_sft-muon_fp32" --num-gpus 1 &

wait
echo "SFT FP32 runs complete."

# SFT Round 2: BF16
echo "=== SFT AdamW BF16 + Muon BF16 ==="
CUDA_VISIBLE_DEVICES=0 python cli.py sft-train \
    --data-path "${SFT_DATA_PATH}" \
    --model "${MODEL}" \
    --output-dir "${CHECKPOINT_PREFIX}/sparsity/sft_adamw_bf16" \
    --optimizer adamw --precision bf16 \
    --lr 1e-6 --batch-size 64 \
    --max-tokens 1200000 --save-every-n-tokens 150000 \
    --seed 67 --lr-scheduler constant \
    --max-tokens-per-gpu 4096 \
    --wandb --wandb-project gsm8k-comparison \
    --wandb-run "sparsity_sft-adamw_bf16" --num-gpus 1 &

CUDA_VISIBLE_DEVICES=4 python cli.py sft-train \
    --data-path "${SFT_DATA_PATH}" \
    --model "${MODEL}" \
    --output-dir "${CHECKPOINT_PREFIX}/sparsity/sft_muon_bf16" \
    --optimizer muon --precision bf16 \
    --lr 1e-6 --batch-size 64 \
    --max-tokens 1200000 --save-every-n-tokens 150000 \
    --seed 67 --lr-scheduler constant \
    --max-tokens-per-gpu 4096 \
    --wandb --wandb-project gsm8k-comparison \
    --wandb-run "sparsity_sft-muon_bf16" --num-gpus 1 &

wait
echo "SFT BF16 runs complete."

# SFT Round 3: Mixed precision
echo "=== SFT AdamW Mixed + Muon Mixed ==="
CUDA_VISIBLE_DEVICES=0 python cli.py sft-train \
    --data-path "${SFT_DATA_PATH}" \
    --model "${MODEL}" \
    --output-dir "${CHECKPOINT_PREFIX}/sparsity/sft_adamw_mixed" \
    --optimizer adamw --precision mixed \
    --lr 1e-6 --batch-size 64 \
    --max-tokens 1200000 --save-every-n-tokens 150000 \
    --seed 67 --lr-scheduler constant \
    --max-tokens-per-gpu 4096 \
    --wandb --wandb-project gsm8k-comparison \
    --wandb-run "sparsity_sft-adamw_mixed" --num-gpus 1 &

CUDA_VISIBLE_DEVICES=4 python cli.py sft-train \
    --data-path "${SFT_DATA_PATH}" \
    --model "${MODEL}" \
    --output-dir "${CHECKPOINT_PREFIX}/sparsity/sft_muon_mixed" \
    --optimizer muon --precision mixed \
    --lr 1e-6 --batch-size 64 \
    --max-tokens 1200000 --save-every-n-tokens 150000 \
    --seed 67 --lr-scheduler constant \
    --max-tokens-per-gpu 4096 \
    --wandb --wandb-project gsm8k-comparison \
    --wandb-run "sparsity_sft-muon_mixed" --num-gpus 1 &

wait
echo "All SFT sparsity experiments completed!"
echo "All sparsity experiments completed!"

