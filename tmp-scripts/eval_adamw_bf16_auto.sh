#!/bin/bash
# Re-run evals with --sampling-dtype auto

RUN_DIR="/mnt/nvme2n1/checkpoints/precision-comparison/grpo-adamw-bf16"
VALIDATION_DATA="adamw-vs-muon-grpo-v1-artifacts/generated-data-v2/gsm8k_grpo_test.jsonl"
GPUS="0,1,2,3,4,5,6,7"

CKPT_LIST=$(ls -d ${RUN_DIR}/checkpoint-[0-9]* | sort -t- -k2 -n | tr '\n' ',' | sed 's/,$//')
echo "Found $(echo "$CKPT_LIST" | tr ',' '\n' | wc -l) checkpoints"

# ── Step 1: Validation accuracy + forward KL (dtype=auto) ────
echo ""
echo "[1/2] Validation accuracy + forward KL (dtype=auto)..."
python eval_gsm8k_parallel.py \
    --checkpoints "$CKPT_LIST" \
    --gpus $GPUS \
    --eval-path "$VALIDATION_DATA" \
    --temperature 0.0 \
    --forward-kl --compute-kl \
    --kl-temperature 0.0 \
    --sampling-dtype auto \
    --kl-dataset "$VALIDATION_DATA" \
    --output "${RUN_DIR}/validation_accuracy_kl_auto.json" \
    || echo "[1/2] FAILED"

# ── Step 2: Test accuracy + forward KL (dtype=auto) ──────────
echo ""
echo "[2/2] Test accuracy + forward KL (dtype=auto)..."
python eval_gsm8k_parallel.py \
    --checkpoints "$CKPT_LIST" \
    --gpus $GPUS \
    --temperature 0.0 \
    --forward-kl --compute-kl \
    --kl-temperature 0.0 \
    --sampling-dtype auto \
    --kl-dataset gsm8k \
    --output "${RUN_DIR}/test_accuracy_kl_auto.json" \
    || echo "[2/2] FAILED"

echo ""
echo "Done!"
