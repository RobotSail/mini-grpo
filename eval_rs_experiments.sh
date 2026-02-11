#!/bin/bash
# Evaluate rejection sampling experiments on GSM8K test set and compute KL divergence

set -e

RS_DIR="/mnt/nvme2n1/checkpoints/rs-train-exps"
CACHE_DIR="/mnt/nvme3n1/workspace/osilkin/mini-grpo/kl_cache/rs"
GPUS="${1:-0,1,2,3,4,5,6,7}"

echo "Using GPUs: $GPUS"

echo ""
echo "============================================================"
echo "STEP 1: GSM8K Test Evaluation"
echo "============================================================"

# AdamW + RS
echo ""
echo "Evaluating: AdamW + RS"
python eval_gsm8k_parallel.py \
    --checkpoint-dir "${RS_DIR}/qwen2-1.5b-gsm8k-rs-adamw" \
    --gpus "$GPUS" \
    --output "${RS_DIR}/qwen2-1.5b-gsm8k-rs-adamw/all_checkpoints_gsm8k_test.json"

# Muon + RS
echo ""
echo "Evaluating: Muon + RS"
python eval_gsm8k_parallel.py \
    --checkpoint-dir "${RS_DIR}/qwen2-1.5b-gsm8k-rs-muon" \
    --gpus "$GPUS" \
    --output "${RS_DIR}/qwen2-1.5b-gsm8k-rs-muon/all_checkpoints_gsm8k_test.json"

echo ""
echo "============================================================"
echo "STEP 2: KL Divergence Computation"
echo "============================================================"

python eval_kl_parallel.py \
    --checkpoint-dirs "${RS_DIR}/qwen2-1.5b-gsm8k-rs-adamw,${RS_DIR}/qwen2-1.5b-gsm8k-rs-muon" \
    --gpus "$GPUS" \
    --cache-dir "$CACHE_DIR" \
    --output-dir "$RS_DIR"

echo ""
echo "============================================================"
echo "ALL RS EVALUATIONS COMPLETE"
echo "============================================================"
echo ""
echo "Results saved to:"
echo "  ${RS_DIR}/qwen2-1.5b-gsm8k-rs-adamw/all_checkpoints_gsm8k_test.json"
echo "  ${RS_DIR}/qwen2-1.5b-gsm8k-rs-adamw/all_checkpoints_kl.json"
echo "  ${RS_DIR}/qwen2-1.5b-gsm8k-rs-muon/all_checkpoints_gsm8k_test.json"
echo "  ${RS_DIR}/qwen2-1.5b-gsm8k-rs-muon/all_checkpoints_kl.json"
echo ""
echo "Create combined plots with:"
echo "  python plot_combined.py --variant 1"
