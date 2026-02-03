#!/bin/bash
# Compute KL divergence for ALL checkpoints in each experiment

set -e

BASE_DIR="/mnt/nvme2n1/checkpoints/verify-exps-variable-seeds"

# Function to evaluate KL for all checkpoints in an experiment
evaluate_kl() {
    local exp_dir="$1"
    local exp_name="$2"
    local is_sft="$3"

    if [ "$is_sft" = "true" ]; then
        ckpt_dir="${exp_dir}/hf_format"
    else
        ckpt_dir="${exp_dir}"
    fi

    echo "============================================================"
    echo "KL Evaluation: $exp_name"
    echo "Checkpoint dir: $ckpt_dir"
    echo "============================================================"

    python eval_kl_all_checkpoints.py \
        --checkpoint-dir "$ckpt_dir" \
        --output "${exp_dir}/all_checkpoints_kl.json"

    echo "KL results saved to: ${exp_dir}/all_checkpoints_kl.json"
}

# Variant 1
echo ""
echo "========== VARIANT 1 =========="
evaluate_kl "${BASE_DIR}/qwen2-1.5b-gsm8k-grpo-adamw_verify_1" "GRPO+AdamW (V1)" "false"
evaluate_kl "${BASE_DIR}/qwen2-1.5b-gsm8k-grpo-muon_verify_1" "GRPO+Muon (V1)" "false"
evaluate_kl "${BASE_DIR}/qwen2-1.5b-gsm8k-sft-adamw_verify_1" "SFT+AdamW (V1)" "true"
evaluate_kl "${BASE_DIR}/qwen2-1.5b-gsm8k-sft-muon_verify_1" "SFT+Muon (V1)" "true"

# Variant 2
echo ""
echo "========== VARIANT 2 =========="
evaluate_kl "${BASE_DIR}/qwen2-1.5b-gsm8k-grpo-adamw_verify_2" "GRPO+AdamW (V2)" "false"
evaluate_kl "${BASE_DIR}/qwen2-1.5b-gsm8k-grpo-muon_verify_2" "GRPO+Muon (V2)" "false"
evaluate_kl "${BASE_DIR}/qwen2-1.5b-gsm8k-sft-adamw_verify_2" "SFT+AdamW (V2)" "true"
evaluate_kl "${BASE_DIR}/qwen2-1.5b-gsm8k-sft-muon_verify_2" "SFT+Muon (V2)" "true"

# Variant 3
echo ""
echo "========== VARIANT 3 =========="
evaluate_kl "${BASE_DIR}/qwen2-1.5b-gsm8k-grpo-adamw_verify_3" "GRPO+AdamW (V3)" "false"
evaluate_kl "${BASE_DIR}/qwen2-1.5b-gsm8k-grpo-muon_verify_3" "GRPO+Muon (V3)" "false"
evaluate_kl "${BASE_DIR}/qwen2-1.5b-gsm8k-sft-adamw_verify_3" "SFT+AdamW (V3)" "true"
evaluate_kl "${BASE_DIR}/qwen2-1.5b-gsm8k-sft-muon_verify_3" "SFT+Muon (V3)" "true"

echo ""
echo "============================================================"
echo "ALL KL EVALUATIONS COMPLETE"
echo "============================================================"
echo ""
echo "Now create plots with: python plot_all_checkpoints.py"
