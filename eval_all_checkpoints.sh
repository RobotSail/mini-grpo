#!/bin/bash
# Evaluate ALL checkpoints on GSM8K test set and compute KL divergence
# This evaluates every checkpoint to see how they cluster in accuracy/KL space

set -e

BASE_DIR="/mnt/nvme2n1/checkpoints/verify-exps-variable-seeds"
GPUS="${1:-0,1,2,3,4,5,6,7}"

echo "Using GPUs: $GPUS"

# Function to evaluate a single experiment's checkpoints
evaluate_experiment() {
    local exp_dir="$1"
    local exp_name="$2"
    local is_sft="$3"

    if [ "$is_sft" = "true" ]; then
        ckpt_dir="${exp_dir}/hf_format"
    else
        ckpt_dir="${exp_dir}"
    fi

    echo "============================================================"
    echo "Evaluating: $exp_name"
    echo "Checkpoint dir: $ckpt_dir"
    echo "============================================================"

    # GSM8K Test evaluation
    python eval_gsm8k_parallel.py \
        --checkpoint-dir "$ckpt_dir" \
        --gpus "$GPUS" \
        --output "${exp_dir}/all_checkpoints_gsm8k_test.json"

    echo "GSM8K test results saved to: ${exp_dir}/all_checkpoints_gsm8k_test.json"
}

# Variant 1
echo ""
echo "========== VARIANT 1 =========="
evaluate_experiment "${BASE_DIR}/qwen2-1.5b-gsm8k-grpo-adamw_verify_1" "GRPO+AdamW (V1)" "false"
evaluate_experiment "${BASE_DIR}/qwen2-1.5b-gsm8k-grpo-muon_verify_1" "GRPO+Muon (V1)" "false"
evaluate_experiment "${BASE_DIR}/qwen2-1.5b-gsm8k-sft-adamw_verify_1" "SFT+AdamW (V1)" "true"
evaluate_experiment "${BASE_DIR}/qwen2-1.5b-gsm8k-sft-muon_verify_1" "SFT+Muon (V1)" "true"

# Variant 2
echo ""
echo "========== VARIANT 2 =========="
evaluate_experiment "${BASE_DIR}/qwen2-1.5b-gsm8k-grpo-adamw_verify_2" "GRPO+AdamW (V2)" "false"
evaluate_experiment "${BASE_DIR}/qwen2-1.5b-gsm8k-grpo-muon_verify_2" "GRPO+Muon (V2)" "false"
evaluate_experiment "${BASE_DIR}/qwen2-1.5b-gsm8k-sft-adamw_verify_2" "SFT+AdamW (V2)" "true"
evaluate_experiment "${BASE_DIR}/qwen2-1.5b-gsm8k-sft-muon_verify_2" "SFT+Muon (V2)" "true"

# Variant 3
echo ""
echo "========== VARIANT 3 =========="
evaluate_experiment "${BASE_DIR}/qwen2-1.5b-gsm8k-grpo-adamw_verify_3" "GRPO+AdamW (V3)" "false"
evaluate_experiment "${BASE_DIR}/qwen2-1.5b-gsm8k-grpo-muon_verify_3" "GRPO+Muon (V3)" "false"
evaluate_experiment "${BASE_DIR}/qwen2-1.5b-gsm8k-sft-adamw_verify_3" "SFT+AdamW (V3)" "true"
evaluate_experiment "${BASE_DIR}/qwen2-1.5b-gsm8k-sft-muon_verify_3" "SFT+Muon (V3)" "true"

echo ""
echo "============================================================"
echo "ALL GSM8K EVALUATIONS COMPLETE"
echo "============================================================"
echo ""
echo "Now run KL evaluation with: ./eval_all_checkpoints_kl.sh"
