#!/bin/bash
# Eval pipeline for adamw bf16 run — updated protocol.
# Greedy decoding (temp=0) for both accuracy and KL rollouts.
# Forward KL: KL(π₀ ‖ π), base model generates rollouts.

RUN_DIR="/mnt/nvme2n1/checkpoints/precision-comparison/grpo-adamw-bf16"
VALIDATION_DATA="adamw-vs-muon-grpo-v1-artifacts/generated-data-v2/gsm8k_grpo_test.jsonl"
GPUS="0,1,2,3,4,5,6,7"

CKPT_LIST=$(ls -d ${RUN_DIR}/checkpoint-[0-9]* | sort -t- -k2 -n | tr '\n' ',' | sed 's/,$//')
echo "Found $(echo "$CKPT_LIST" | tr ',' '\n' | wc -l) checkpoints"

# ── Step 1: Validation accuracy + forward KL ─────────────────
echo ""
echo "[1/4] Validation accuracy + forward KL..."
python eval_gsm8k_parallel.py \
    --checkpoints "$CKPT_LIST" \
    --gpus $GPUS \
    --eval-path "$VALIDATION_DATA" \
    --temperature 0.0 \
    --forward-kl --compute-kl \
    --kl-temperature 0.0 \
    --kl-dataset "$VALIDATION_DATA" \
    --output "${RUN_DIR}/validation_accuracy_kl.json" \
    || echo "[1/4] FAILED"

# ── Step 2: Test accuracy + forward KL ───────────────────────
echo ""
echo "[2/4] Test accuracy + forward KL..."
python eval_gsm8k_parallel.py \
    --checkpoints "$CKPT_LIST" \
    --gpus $GPUS \
    --temperature 0.0 \
    --forward-kl --compute-kl \
    --kl-temperature 0.0 \
    --kl-dataset gsm8k \
    --output "${RUN_DIR}/test_accuracy_kl.json" \
    || echo "[2/4] FAILED"

# ── Step 3 & 4: Geometry metrics for best validation checkpoint
echo ""
echo "[3/4] Geometry metrics..."
python tmp-scripts/compute_geometry.py \
    --run-dir "$RUN_DIR" \
    || echo "[3/4] FAILED"

# ── Step 5: Update norm plot ─────────────────────────────────
echo ""
echo "[4/4] Update norm trajectory plot..."
python tmp-scripts/plot_update_norms.py \
    --run-dir "$RUN_DIR" \
    || echo "[4/4] FAILED"

echo ""
echo "Done!"
