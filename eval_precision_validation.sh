#!/bin/bash
# Evaluate ALL checkpoints from ALL 12 precision experiments on the VALIDATION set
# Uses --checkpoints flag to explicitly list paths (skipping checkpoint-initial)

set -e

SPARSITY_DIR="/mnt/nvme2n1/checkpoints/verify-exps-variable-seeds/sparsity"
EVAL_PATH="/mnt/nvme3n1/workspace/osilkin/mini-grpo/adamw-vs-muon-grpo-v1-artifacts/generated-data-v2/gsm8k_grpo_test.jsonl"
GPUS="0,1,2,3,4,5,6,7"
RESULTS_DIR="/mnt/nvme3n1/workspace/osilkin/mini-grpo/precision_validation_results"

mkdir -p "$RESULTS_DIR"

echo "============================================================"
echo "Evaluating all precision experiment checkpoints on VALIDATION set"
echo "Validation data: $EVAL_PATH"
echo "GPUs: $GPUS"
echo "============================================================"

# GRPO experiments - enumerate checkpoints explicitly, skipping checkpoint-initial
for opt in adamw muon; do
    for prec in fp32 bf16 mixed; do
        exp_dir="${SPARSITY_DIR}/${opt}_${prec}"
        out_file="${RESULTS_DIR}/${opt}_grpo_${prec}_validation.json"
        echo ""
        echo ">>> Evaluating: ${opt}_grpo_${prec}"

        # Build comma-separated list of checkpoint paths, excluding checkpoint-initial
        ckpts=""
        for d in $(ls -d "${exp_dir}"/checkpoint-* 2>/dev/null | sort -t- -k2 -n 2>/dev/null || true); do
            dirname=$(basename "$d")
            if [ "$dirname" = "checkpoint-initial" ]; then
                continue
            fi
            if [ -f "$d/config.json" ]; then
                if [ -z "$ckpts" ]; then
                    ckpts="$d"
                else
                    ckpts="$ckpts,$d"
                fi
            fi
        done

        if [ -z "$ckpts" ]; then
            echo "    WARNING: No valid checkpoints found, skipping"
            continue
        fi

        echo "    Checkpoints: $(echo "$ckpts" | tr ',' '\n' | wc -l) found"
        python eval_gsm8k_parallel.py \
            --checkpoints "$ckpts" \
            --gpus "$GPUS" \
            --eval-path "$EVAL_PATH" \
            --output "$out_file"
    done
done

# SFT experiments (hf_format subdirectory) - these use tokens_ naming, no issue
for opt in adamw muon; do
    for prec in fp32 bf16 mixed; do
        exp_dir="${SPARSITY_DIR}/sft_${opt}_${prec}/hf_format"
        out_file="${RESULTS_DIR}/${opt}_sft_${prec}_validation.json"
        echo ""
        echo ">>> Evaluating: ${opt}_sft_${prec}"

        # Build comma-separated list
        ckpts=""
        for d in $(ls -d "${exp_dir}"/samples_* 2>/dev/null | sort); do
            if [ -f "$d/config.json" ]; then
                if [ -z "$ckpts" ]; then
                    ckpts="$d"
                else
                    ckpts="$ckpts,$d"
                fi
            fi
        done

        if [ -z "$ckpts" ]; then
            echo "    WARNING: No valid checkpoints found, skipping"
            continue
        fi

        echo "    Checkpoints: $(echo "$ckpts" | tr ',' '\n' | wc -l) found"
        python eval_gsm8k_parallel.py \
            --checkpoints "$ckpts" \
            --gpus "$GPUS" \
            --eval-path "$EVAL_PATH" \
            --output "$out_file"
    done
done

echo ""
echo "============================================================"
echo "ALL VALIDATION EVALUATIONS COMPLETE"
echo "Results saved to: $RESULTS_DIR"
echo "============================================================"
