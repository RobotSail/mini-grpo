#!/bin/bash
# Compute forward KL divergence KL(π || π₀) for ALL checkpoints across all experiments
set -e

VERIFY_DIR="/mnt/nvme2n1/checkpoints/verify-exps-variable-seeds"
RS_DIR="/mnt/nvme2n1/checkpoints/rs-train-exps"
GPUS="${1:-0,1,2,3,4,5,6,7}"

echo "Using GPUs: $GPUS"

# Process each variant of verify-exps separately to keep job sizes manageable
for VARIANT in 1 2 3; do
    echo ""
    echo "============================================================"
    echo "VARIANT ${VARIANT} - Forward KL"
    echo "============================================================"

    python eval_forward_kl_parallel.py \
        --checkpoint-dirs "${VERIFY_DIR}/qwen2-1.5b-gsm8k-grpo-adamw_verify_${VARIANT},${VERIFY_DIR}/qwen2-1.5b-gsm8k-grpo-muon_verify_${VARIANT},${VERIFY_DIR}/qwen2-1.5b-gsm8k-sft-adamw_verify_${VARIANT},${VERIFY_DIR}/qwen2-1.5b-gsm8k-sft-muon_verify_${VARIANT}" \
        --gpus "$GPUS"
done

echo ""
echo "============================================================"
echo "REJECTION SAMPLING - Forward KL"
echo "============================================================"

python eval_forward_kl_parallel.py \
    --checkpoint-dirs "${RS_DIR}/qwen2-1.5b-gsm8k-rs-adamw,${RS_DIR}/qwen2-1.5b-gsm8k-rs-muon" \
    --gpus "$GPUS"

echo ""
echo "============================================================"
echo "ALL FORWARD KL EVALUATIONS COMPLETE"
echo "============================================================"
echo "Results saved as all_checkpoints_forward_kl.json in each experiment directory"
