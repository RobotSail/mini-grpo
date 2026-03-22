#!/bin/bash
#
# GSM-Plus GRPO experiment: 4 runs (2 optimizers x 2 precision modes)
# Each run is independent — if killed or crashed, the script continues to the next.
#

export WANDB_API_KEY=dcc7e9d67dd4454320776959ba154a9d285cb7db

CHECKPOINT_PREFIX="/mnt/nvme2n1/os-gsm-plus-experiment_grpo_2"
DATA_PATH="data/gsm_combined/gsm_plus/train.jsonl"
VALIDATION_PATH="data/gsm_combined/gsm_plus/validation.jsonl"
MODEL="Qwen/Qwen2-1.5B-Instruct"
SEED=2025

COMMON_ARGS=(
    --data-path "${DATA_PATH}"
    --model "${MODEL}"
    --group-size 8
    --batch-size 8
    --inner-batch-size 64
    --clip-eps 0.2
    --kl 0
    --temp 0.7
    --max-new-tokens 512
    --max-tokens-per-gpu 25000
    --lr 1e-6
    --seed ${SEED}
    --train-gpus "1,2,3,4"
    --vllm-gpus "5,6,7"
    --wandb
    --wandb-project gsm-plus-grpo
    --validation-path "${VALIDATION_PATH}"
    --num-icl 5
    --best-val-ckpt-only
)

# ─── Run 1: AdamW + mixed precision (5M tokens, eval 4x more frequent) ──────
echo "========================================="
echo "Run 1/4: AdamW + mixed precision (5M tokens)"
echo "========================================="
python cli.py distributed-grpo-train \
    "${COMMON_ARGS[@]}" \
    --output-dir "${CHECKPOINT_PREFIX}/grpo-adamw-mixed" \
    --max-tokens 5000000 \
    --save-every-n-steps 12 \
    --eval-every-n-steps 12 \
    --optimizer adamw \
    --precision mixed \
    --wandb-run "gsm-plus_grpo-adamw-mixed" \
    || echo "[WARN] Run 1 (adamw+mixed) exited with code $?"

# ─── Run 2: Muon + mixed precision (5000 steps, eval every 50) ──────────────
echo "========================================="
echo "Run 2/4: Muon + mixed precision (5000 steps)"
echo "========================================="
python cli.py distributed-grpo-train \
    "${COMMON_ARGS[@]}" \
    --output-dir "${CHECKPOINT_PREFIX}/grpo-muon-mixed" \
    --max-steps 5000 \
    --save-every-n-steps 50 \
    --eval-every-n-steps 50 \
    --optimizer muon \
    --precision mixed \
    --wandb-run "gsm-plus_grpo-muon-mixed" \
    || echo "[WARN] Run 2 (muon+mixed) exited with code $?"

# ─── Run 3: AdamW + bf16 precision (30k steps, eval every 50) ───────────────
echo "========================================="
echo "Run 3/4: AdamW + bf16 precision (30k steps)"
echo "========================================="
python cli.py distributed-grpo-train \
    "${COMMON_ARGS[@]}" \
    --output-dir "${CHECKPOINT_PREFIX}/grpo-adamw-bf16" \
    --max-steps 30000 \
    --save-every-n-steps 50 \
    --eval-every-n-steps 50 \
    --optimizer adamw \
    --precision bf16 \
    --wandb-run "gsm-plus_grpo-adamw-bf16" \
    || echo "[WARN] Run 3 (adamw+bf16) exited with code $?"

# ─── Run 4: Muon + bf16 precision (30k steps, eval every 50) ────────────────
echo "========================================="
echo "Run 4/4: Muon + bf16 precision (30k steps)"
echo "========================================="
python cli.py distributed-grpo-train \
    "${COMMON_ARGS[@]}" \
    --output-dir "${CHECKPOINT_PREFIX}/grpo-muon-bf16" \
    --max-steps 30000 \
    --save-every-n-steps 50 \
    --eval-every-n-steps 50 \
    --optimizer muon \
    --precision bf16 \
    --wandb-run "gsm-plus_grpo-muon-bf16" \
    || echo "[WARN] Run 4 (muon+bf16) exited with code $?"

echo "========================================="
echo "All GSM-Plus GRPO experiments completed!"
echo "========================================="
