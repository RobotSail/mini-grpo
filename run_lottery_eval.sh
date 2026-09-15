#!/bin/bash
set -eo pipefail

# ── Evaluation pipeline for lottery ticket experiment ─────────────────────
#
# For each run:
#   1. Eval all checkpoints on validation set (accuracy + forward KL)
#   2. Eval all checkpoints on test set (accuracy + forward KL)
#   3. Find best val checkpoint, compute geometry metrics
#   4. Plot update norm trajectory

EXPERIMENT_DIR="/mnt/nvme2n1/checkpoints/lottery-ticket-experiment"
DATA_DIR="adamw-vs-muon-grpo-v1-artifacts/generated-data-v2"
VAL_PATH="${DATA_DIR}/gsm8k_grpo_test.jsonl"   # validation split
BASE_MODEL="Qwen/Qwen2-1.5B-Instruct"
GPU=0

RUNS=(
    "grpo-adamw-bf16master"
    "grpo-adamw-mixed"
    "grpo-adamw-lottery"
    "grpo-adamw-random-seed1"
    "grpo-adamw-random-seed2"
    "grpo-adamw-random-seed3"
)

for RUN in "${RUNS[@]}"; do
    RUN_DIR="${EXPERIMENT_DIR}/${RUN}"
    echo ""
    echo "========================================="
    echo "Evaluating: ${RUN}"
    echo "========================================="

    # ── Step 1: Validation accuracy + forward KL ──
    VAL_OUTPUT="${RUN_DIR}/validation_accuracy_kl.json"
    if [ ! -f "${VAL_OUTPUT}" ]; then
        echo "  → Evaluating validation set (accuracy + forward KL)..."
        python eval_gsm8k.py \
            --checkpoint-dir "${RUN_DIR}" \
            --eval-path "${VAL_PATH}" \
            --output "${VAL_OUTPUT}" \
            --gpu ${GPU} \
            --forward-kl \
            --base-model "${BASE_MODEL}" \
            --sampling-dtype auto \
            --temperature 0.0
    else
        echo "  → Skipping validation eval (already exists)"
    fi

    # ── Step 2: Test accuracy + forward KL ──
    TEST_OUTPUT="${RUN_DIR}/test_accuracy_kl.json"
    if [ ! -f "${TEST_OUTPUT}" ]; then
        echo "  → Evaluating test set (accuracy + forward KL)..."
        python eval_gsm8k.py \
            --checkpoint-dir "${RUN_DIR}" \
            --gpu ${GPU} \
            --forward-kl \
            --base-model "${BASE_MODEL}" \
            --sampling-dtype auto \
            --output "${TEST_OUTPUT}" \
            --temperature 0.0
    else
        echo "  → Skipping test eval (already exists)"
    fi

    echo "  → Done: ${RUN}"
done

# ── Step 3 & 4: Compute geometry metrics for best val checkpoint ──
echo ""
echo "========================================="
echo "Computing geometry metrics for best checkpoints"
echo "========================================="

CUDA_VISIBLE_DEVICES="" python << 'PYEOF'
import json
import os
import torch
import numpy as np
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM

EXPERIMENT_DIR = "/mnt/nvme2n1/checkpoints/lottery-ticket-experiment"
BASE_MODEL = "Qwen/Qwen2-1.5B-Instruct"

RUNS = [
    "grpo-adamw-bf16master",
    "grpo-adamw-mixed",
    "grpo-adamw-lottery",
    "grpo-adamw-random-seed1",
    "grpo-adamw-random-seed2",
    "grpo-adamw-random-seed3",
]

# Load base model once
print("Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=torch.float32)
base_sd = base_model.state_dict()
del base_model

def load_ckpt(path):
    ckpt = {}
    for f in os.listdir(path):
        if f.startswith("model") and f.endswith(".safetensors"):
            ckpt.update(load_file(os.path.join(path, f), device="cpu"))
    return ckpt

def is_2d_weight(name):
    if any(s in name for s in [".bias", "layernorm", "layer_norm", "ln_"]):
        return False
    if "embed" in name or "lm_head" in name:
        return False
    return ".weight" in name

for run_name in RUNS:
    run_dir = os.path.join(EXPERIMENT_DIR, run_name)
    geo_path = os.path.join(run_dir, "geometry_metrics.json")

    if os.path.exists(geo_path):
        print(f"Skipping {run_name}: geometry_metrics.json already exists")
        continue

    # Find best validation checkpoint
    val_path = os.path.join(run_dir, "validation_accuracy_kl.json")
    if not os.path.exists(val_path):
        print(f"Skipping {run_name}: no validation_accuracy_kl.json")
        continue

    with open(val_path) as f:
        val_data = json.load(f)

    best_key = max(val_data, key=lambda k: val_data[k].get("accuracy", 0))
    best_entry = val_data[best_key]
    best_acc = best_entry["accuracy"]

    # Find the checkpoint path
    ckpt_name = best_key.split("_")[-1] if "_" in best_key else best_key
    best_ckpt_path = os.path.join(run_dir, ckpt_name)
    if not os.path.exists(best_ckpt_path):
        # Try matching by token count
        for d in os.listdir(run_dir):
            if d.startswith("checkpoint-") and d != "checkpoint-initial":
                best_ckpt_path = os.path.join(run_dir, d)
                break

    print(f"\n{run_name}: best val accuracy = {best_acc:.4f} at {ckpt_name}")

    # Load checkpoint
    init_path = os.path.join(run_dir, "checkpoint-initial")
    if os.path.exists(init_path):
        init_sd = load_ckpt(init_path)
    else:
        init_sd = base_sd

    ckpt_sd = load_ckpt(best_ckpt_path)

    # Compute metrics
    frob_sq = 0.0
    total_params = 0
    total_changed = 0
    k90_values = []
    num_2d = 0

    for name in init_sd:
        if name not in ckpt_sd:
            continue

        dw = ckpt_sd[name].float() - init_sd[name].float()
        n = dw.numel()
        total_params += n

        # L0 sparsity (threshold 1e-5)
        changed = (dw.abs() >= 1e-5).sum().item()
        total_changed += changed

        # Frobenius norm
        frob_sq += dw.norm().item() ** 2

        # k90 for 2D weights
        if is_2d_weight(name) and dw.dim() == 2:
            U, S, Vt = torch.linalg.svd(dw, full_matrices=False)
            cumsum = (S ** 2).cumsum(0)
            total_energy = cumsum[-1].item()
            if total_energy > 0:
                k90 = (cumsum >= total_energy * 0.9).nonzero(as_tuple=True)[0][0].item() + 1
                k90_values.append(k90)
                num_2d += 1

    frob_norm = frob_sq ** 0.5
    l0_sparsity = 1.0 - total_changed / total_params if total_params > 0 else 0
    avg_k90 = float(np.mean(k90_values)) if k90_values else 0

    metrics = {
        "best_checkpoint": best_key,
        "best_checkpoint_path": best_ckpt_path,
        "best_validation_accuracy": best_acc,
        "frobenius_norm": frob_norm,
        "avg_k90": avg_k90,
        "l0_sparsity": l0_sparsity,
        "num_layers_svd": num_2d,
        "total_parameters": total_params,
    }

    with open(geo_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"  frobenius_norm: {frob_norm:.6f}")
    print(f"  avg_k90: {avg_k90:.1f}")
    print(f"  l0_sparsity: {l0_sparsity:.6f} ({l0_sparsity*100:.2f}%)")
    print(f"  Saved to {geo_path}")

print("\nAll geometry metrics computed!")
PYEOF

echo ""
echo "========================================="
echo "All evaluations completed!"
echo "========================================="
