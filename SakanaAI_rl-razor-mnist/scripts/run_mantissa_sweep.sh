#!/bin/bash
# Mantissa sweep: pretrain once, then finetune with each method × mantissa setting.
# Shows the tradeoff between mantissa precision, sparsity, and forgetting (KL).
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH="src:${PYTHONPATH:-}"

EXP_ROOT="experiments/mantissa_sweep"
SEED=42

# Mantissa settings: 7=bf16, 10=fp16, 13=intermediate, 16=high, 0=fp32 (no snapping)
MANTISSA_BITS=(7 10 13 16 0)
METHODS=(grpo grpo_kl sft1 sft2 oracle)

# Step 1: Pretrain (shared across all runs)
PRETRAIN_DIR="${EXP_ROOT}/pretrain"
PRETRAIN_MODEL="${PRETRAIN_DIR}/pretrained_model.pt"

if [ ! -f "$PRETRAIN_MODEL" ]; then
    echo "=== Pretraining ==="
    mkdir -p "$PRETRAIN_DIR"
    python scripts/pretrain.py \
        --epochs 50 \
        --lr 1e-3 \
        --n-samples 500 \
        --batch-size 64 \
        --scheduler cosine_with_warmup \
        --warmup-ratio 0.1 \
        --weight-decay 0.0 \
        --seed ${SEED} \
        --checkpoint-every 10 \
        --exp-dir "${PRETRAIN_DIR}"

    # Find and copy the pretrained model to a stable path
    FOUND=$(find "${PRETRAIN_DIR}" -name "pretrained_model.pt" | head -1)
    if [ "$FOUND" != "$PRETRAIN_MODEL" ]; then
        cp "$FOUND" "$PRETRAIN_MODEL"
    fi
    echo "Pretrained model: ${PRETRAIN_MODEL}"
else
    echo "Pretrained model exists: ${PRETRAIN_MODEL}"
fi

# Step 2: Finetune with each method × mantissa setting
for METHOD in "${METHODS[@]}"; do
    for MBITS in "${MANTISSA_BITS[@]}"; do
        if [ "$MBITS" -eq 0 ]; then
            LABEL="fp32"
        else
            LABEL="m${MBITS}"
        fi

        RUN_NAME="${METHOD}_${LABEL}"
        RUN_DIR="${EXP_ROOT}/${RUN_NAME}"

        if [ -f "${RUN_DIR}/results.json" ] 2>/dev/null || \
           find "${EXP_ROOT}" -path "*${RUN_NAME}*/results.json" -print -quit 2>/dev/null | grep -q .; then
            echo "SKIP ${RUN_NAME}: results exist"
            continue
        fi

        echo ""
        echo "=== ${RUN_NAME} ==="

        KL_COEF_ARG=""
        if [ "$METHOD" = "grpo_kl" ]; then
            KL_COEF_ARG="--kl-coef 0.1"
        fi

        python scripts/finetune.py \
            --pretrained-model "${PRETRAIN_MODEL}" \
            --method "${METHOD}" \
            --batch-size 64 \
            --lr 1e-4 \
            --epochs 2 \
            --scheduler cosine_with_warmup \
            --warmup-ratio 0.1 \
            --weight-decay 0.0 \
            --seed ${SEED} \
            --checkpoint-every 0.2 \
            --mantissa-bits "${MBITS}" \
            --exp-dir "${EXP_ROOT}" \
            --wandb-name "${RUN_NAME}" \
            ${KL_COEF_ARG}
    done
done

echo ""
echo "=== All runs complete ==="
echo "Results in: ${EXP_ROOT}/"
