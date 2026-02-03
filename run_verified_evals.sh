#!/bin/bash
# Run GSM8K evaluation on all verified experiment checkpoints
# Usage: ./run_verified_evals.sh [GPUS]
# Example: ./run_verified_evals.sh 0,1,2,3,4,5,6,7

set -e

GPUS="${1:-0,1,2,3,4,5,6,7}"
BASE="/mnt/nvme3n1/workspace/osilkin/mini-grpo/verified-weights"
EVAL_DATA="/mnt/nvme3n1/workspace/osilkin/mini-grpo/adamw-vs-muon-grpo-v1-artifacts/generated-data-v2/gsm8k_sft_test.jsonl"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=============================================="
echo "Running verified experiment evaluations"
echo "GPUs: $GPUS"
echo "Eval data: $EVAL_DATA"
echo "=============================================="

for run in verify_run_1 verify_run_2 verify_run_3; do
    echo ""
    echo "=== $run ==="

    for exp in adamw_sft adamw_grpo muon_sft muon_grpo; do
        # SFT checkpoints are in hf_format subdir
        if [[ "$exp" == *_sft ]]; then
            ckpt_dir="$BASE/$run/$exp/hf_format"
        else
            ckpt_dir="$BASE/$run/$exp"
        fi

        output="$BASE/$run/${exp}_results.json"

        echo ""
        echo ">>> Evaluating $run/$exp"
        echo "    Checkpoint dir: $ckpt_dir"
        echo "    Output: $output"

        python "$SCRIPT_DIR/eval_gsm8k_parallel.py" \
            --checkpoint-dir "$ckpt_dir" \
            --gpus "$GPUS" \
            --eval-path "$EVAL_DATA" \
            --output "$output"

        echo ">>> Done: $run/$exp"
    done
done

echo ""
echo "=============================================="
echo "All evaluations complete!"
echo "Results saved to:"
for run in verify_run_1 verify_run_2 verify_run_3; do
    for exp in adamw_sft adamw_grpo muon_sft muon_grpo; do
        echo "  $BASE/$run/${exp}_results.json"
    done
done
echo "=============================================="
