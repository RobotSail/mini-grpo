#!/bin/bash
# Compute KL divergence for ALL checkpoints using parallel GPU evaluation
# Key: Base model rollouts are generated ONCE per variant and cached

set -e

BASE_DIR="/mnt/nvme2n1/checkpoints/verify-exps-variable-seeds"
CACHE_BASE="/mnt/nvme3n1/workspace/osilkin/mini-grpo/kl_cache"
GPUS="${1:-0,1,2,3,4,5,6,7}"

echo "Using GPUs: $GPUS"

# Function to evaluate KL for all experiments in a variant
evaluate_variant() {
    local variant="$1"
    local cache_dir="${CACHE_BASE}/variant_${variant}"

    echo ""
    echo "============================================================"
    echo "VARIANT ${variant} - KL Divergence (Parallel)"
    echo "Cache directory: ${cache_dir}"
    echo "============================================================"

    # All 4 experiments for this variant
    local grpo_adamw="${BASE_DIR}/qwen2-1.5b-gsm8k-grpo-adamw_verify_${variant}"
    local grpo_muon="${BASE_DIR}/qwen2-1.5b-gsm8k-grpo-muon_verify_${variant}"
    local sft_adamw="${BASE_DIR}/qwen2-1.5b-gsm8k-sft-adamw_verify_${variant}"
    local sft_muon="${BASE_DIR}/qwen2-1.5b-gsm8k-sft-muon_verify_${variant}"

    python eval_kl_parallel.py \
        --checkpoint-dirs "${grpo_adamw},${grpo_muon},${sft_adamw},${sft_muon}" \
        --gpus "$GPUS" \
        --cache-dir "$cache_dir" \
        --output-dir "${BASE_DIR}"

    echo "Variant ${variant} KL evaluation complete!"
}

# Evaluate all variants
evaluate_variant 1
evaluate_variant 2
evaluate_variant 3

echo ""
echo "============================================================"
echo "ALL KL EVALUATIONS COMPLETE"
echo "============================================================"
echo ""
echo "Results saved to each experiment's all_checkpoints_kl.json"
echo "Now create plots with: python plot_all_checkpoints.py"
