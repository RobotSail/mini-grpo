#!/bin/bash
# Evaluate ALL checkpoints from ALL 12 precision experiments on the TEST set
# (no --eval-path = uses HuggingFace GSM8K test split)

set -e

SPARSITY_DIR="/mnt/nvme2n1/checkpoints/verify-exps-variable-seeds/sparsity"
GPUS="0,1,2,3,4,5,6,7"
RESULTS_DIR="/mnt/nvme3n1/workspace/osilkin/mini-grpo/precision_test_all_results"

mkdir -p "$RESULTS_DIR"

echo "============================================================"
echo "Evaluating ALL checkpoints on GSM8K TEST set"
echo "Temperature: 0.7, FP16 inference"
echo "GPUs: $GPUS"
echo "============================================================"

# GRPO experiments
for opt in adamw muon; do
    for prec in fp32 bf16 mixed; do
        exp_dir="${SPARSITY_DIR}/${opt}_${prec}"
        out_file="${RESULTS_DIR}/${opt}_grpo_${prec}_test.json"

        ckpts=$(ls -d "${exp_dir}"/checkpoint-[0-9]* 2>/dev/null | sort -t- -k2 -n | tr '\n' ',')
        ckpts=${ckpts%,}
        if [ -z "$ckpts" ]; then continue; fi

        echo ""
        echo ">>> ${opt}_grpo_${prec} on TEST"
        python eval_gsm8k_parallel.py \
            --checkpoints "$ckpts" \
            --gpus "$GPUS" \
            --temperature 0.7 \
            --output "$out_file"
    done
done

# SFT experiments
for opt in adamw muon; do
    for prec in fp32 bf16 mixed; do
        exp_dir="${SPARSITY_DIR}/sft_${opt}_${prec}/hf_format"
        out_file="${RESULTS_DIR}/${opt}_sft_${prec}_test.json"

        ckpts=$(ls -d "${exp_dir}"/samples_* 2>/dev/null | tr '\n' ',')
        ckpts=${ckpts%,}
        if [ -z "$ckpts" ]; then continue; fi

        echo ""
        echo ">>> ${opt}_sft_${prec} on TEST"
        python eval_gsm8k_parallel.py \
            --checkpoints "$ckpts" \
            --gpus "$GPUS" \
            --temperature 0.7 \
            --output "$out_file"
    done
done

echo ""
echo "============================================================"
echo "ALL TEST EVALUATIONS COMPLETE"
echo "Results in: $RESULTS_DIR"
echo "============================================================"
