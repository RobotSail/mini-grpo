#!/bin/bash
# Post-training evaluation pipeline.
# Runs for each experiment directory under EXPERIMENT_ROOT.
# Each run is independent — failures don't block the next.

export WANDB_API_KEY=dcc7e9d67dd4454320776959ba154a9d285cb7db

EXPERIMENT_ROOT="/mnt/nvme2n1/checkpoints/precision-comparison"
VALIDATION_DATA="adamw-vs-muon-grpo-v1-artifacts/generated-data-v2/gsm8k_grpo_test.jsonl"
GPUS="0,1,2,3,4,5,6,7"

for RUN_DIR in ${EXPERIMENT_ROOT}/grpo-*; do
    RUN_NAME=$(basename "$RUN_DIR")
    echo ""
    echo "============================================================"
    echo "Evaluating: ${RUN_NAME}"
    echo "============================================================"

    # Check if there are any checkpoints
    CKPTS=$(ls -d ${RUN_DIR}/checkpoint-[0-9]* 2>/dev/null | sort -t- -k2 -n)
    if [ -z "$CKPTS" ]; then
        echo "  No checkpoints found, skipping"
        continue
    fi

    CKPT_LIST=$(echo "$CKPTS" | tr '\n' ',' | sed 's/,$//')
    echo "  Found $(echo "$CKPTS" | wc -l) checkpoints"

    # ── Step 1: Validation accuracy + forward KL ─────────────────
    if [ ! -f "${RUN_DIR}/validation_accuracy_kl.json" ]; then
        echo "  [1/4] Validation accuracy + KL..."
        python eval_gsm8k_parallel.py \
            --checkpoints "$CKPT_LIST" \
            --gpus $GPUS \
            --eval-path "$VALIDATION_DATA" \
            --temperature 0.0 \
            --compute-kl \
            --reverse-kl \
            --kl-dataset "$VALIDATION_DATA" \
            --output "${RUN_DIR}/validation_accuracy_kl.json" \
            || echo "  [1/4] FAILED"
    else
        echo "  [1/4] validation_accuracy_kl.json already exists, skipping"
    fi

    # ── Step 2: Test accuracy + forward KL ───────────────────────
    if [ ! -f "${RUN_DIR}/test_accuracy_kl.json" ]; then
        echo "  [2/4] Test accuracy + KL..."
        python eval_gsm8k_parallel.py \
            --checkpoints "$CKPT_LIST" \
            --gpus $GPUS \
            --temperature 0.0 \
            --compute-kl \
            --reverse-kl \
            --kl-dataset gsm8k \
            --output "${RUN_DIR}/test_accuracy_kl.json" \
            || echo "  [2/4] FAILED"
    else
        echo "  [2/4] test_accuracy_kl.json already exists, skipping"
    fi

    # ── Step 3 & 4: Geometry metrics for best validation checkpoint
    if [ ! -f "${RUN_DIR}/geometry_metrics.json" ]; then
        echo "  [3/4] Geometry metrics for best validation checkpoint..."
        python tmp-scripts/compute_geometry.py \
            --run-dir "$RUN_DIR" \
            || echo "  [3/4] FAILED"
    else
        echo "  [3/4] geometry_metrics.json already exists, skipping"
    fi

    # ── Step 5: Update norm plot ─────────────────────────────────
    if [ ! -f "${RUN_DIR}/update_norm_trajectory.png" ]; then
        echo "  [4/4] Update norm trajectory plot..."
        python tmp-scripts/plot_update_norms.py \
            --run-dir "$RUN_DIR" \
            || echo "  [4/4] FAILED"
    else
        echo "  [4/4] update_norm_trajectory.png already exists, skipping"
    fi

    echo "  Done: ${RUN_NAME}"
done

echo ""
echo "============================================================"
echo "All evaluations complete!"
echo "============================================================"
