#!/usr/bin/env python3
"""
Plot optimization path efficiency and L0 sparsity for AdamW vs Muon
across mixed-precision and bf16 training.

Path efficiency = displacement / path length = ||ΔW||_F / Σ_t ||δW_t||_F
  - 1.0 means the optimizer walked in a straight line
  - Lower means more oscillation/wandering

L0 sparsity = fraction of parameters unchanged from init to best checkpoint.

Usage:
    python plot_path_efficiency_and_sparsity.py
    python plot_path_efficiency_and_sparsity.py --output-dir my_plots/
    python plot_path_efficiency_and_sparsity.py --recompute  # recompute from checkpoints
"""

import argparse
import json
import os
import sys

import matplotlib
matplotlib.rcParams['text.usetex'] = False
import matplotlib.pyplot as plt
import numpy as np


# ── Color scheme (matches plot_bf16_mixed_comparison_auto.py) ──────────────
COLORS = {
    "adamw_mixed": "#08519c",  # dark blue
    "adamw_bf16":  "#6baed6",  # light blue
    "muon_mixed":  "#7f0000",  # dark red
    "muon_bf16":   "#e53935",  # light red
}

# ── Checkpoint paths ───────────────────────────────────────────────────────
BASE_DIR = "/mnt/nvme2n1/checkpoints/precision-comparison"

RUNS = {
    "adamw_mixed": {
        "label": "AdamW\nMixed",
        "norms_path": f"{BASE_DIR}/grpo-adamw-mixed-600k/update_norms.jsonl",
        "best_checkpoint": f"{BASE_DIR}/grpo-adamw-mixed-600k/checkpoint-316072",
        "best_tokens": 316072,
    },
    "muon_mixed": {
        "label": "Muon\nMixed",
        "norms_path": f"{BASE_DIR}/grpo-muon-mixed/update_norms.jsonl",
        "best_checkpoint": f"{BASE_DIR}/grpo-muon-mixed/checkpoint-2100020",
        "best_tokens": 2100020,
    },
    "adamw_bf16": {
        "label": "AdamW\nBF16",
        "norms_path": f"{BASE_DIR}/grpo-adamw-bf16/update_norms.jsonl",
        "best_checkpoint": f"{BASE_DIR}/grpo-adamw-bf16/checkpoint-12906436",
        "best_tokens": 12906436,
    },
    "muon_bf16": {
        "label": "Muon\nBF16",
        "norms_path": f"{BASE_DIR}/grpo-muon-bf16/update_norms.jsonl",
        "best_checkpoint": f"{BASE_DIR}/grpo-muon-bf16/checkpoint-48152636",
        "best_tokens": 48152636,
    },
}


# ── Computation ────────────────────────────────────────────────────────────

def sum_step_norms(path: str, max_tokens: int) -> tuple[float, int, int]:
    """Sum the total Frobenius norm per step up to max_tokens.

    Returns (cumulative_norm, steps, last_token_count).
    """
    total_norm = 0.0
    steps = 0
    last_tokens = 0
    with open(path) as f:
        for line in f:
            entry = json.loads(line)
            if entry["tokens"] > max_tokens:
                break
            param_norms = list(entry["norms"].values())
            step_norm = sum(n ** 2 for n in param_norms) ** 0.5
            total_norm += step_norm
            steps += 1
            last_tokens = entry["tokens"]
    return total_norm, steps, last_tokens


def compute_displacement(checkpoint_path: str, base_model: str = "Qwen/Qwen2-1.5B-Instruct") -> float:
    """Compute ||W_best - W_init||_F."""
    import torch
    from safetensors.torch import load_file
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(base_model, dtype=torch.float32)
    init_sd = model.state_dict()
    del model

    ckpt = {}
    for f in os.listdir(checkpoint_path):
        if f.startswith("model") and f.endswith(".safetensors"):
            ckpt.update(load_file(os.path.join(checkpoint_path, f), device="cpu"))

    frob_sq = 0.0
    for name in init_sd:
        if name in ckpt:
            dw = ckpt[name].float() - init_sd[name].float()
            frob_sq += dw.norm().item() ** 2
    return frob_sq ** 0.5


def compute_l0_sparsity(checkpoint_path: str, base_model: str = "Qwen/Qwen2-1.5B-Instruct") -> float:
    """Compute L0 sparsity = fraction of unchanged parameters."""
    import torch
    from safetensors.torch import load_file
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(base_model, dtype=torch.float32)
    init_sd = model.state_dict()
    del model

    ckpt = {}
    for f in os.listdir(checkpoint_path):
        if f.startswith("model") and f.endswith(".safetensors"):
            ckpt.update(load_file(os.path.join(checkpoint_path, f), device="cpu"))

    total_params = 0
    total_changed = 0
    for name in init_sd:
        if name in ckpt:
            dw = ckpt[name].float() - init_sd[name].float()
            total_params += dw.numel()
            total_changed += (dw != 0).sum().item()
    return 1.0 - total_changed / total_params


def load_or_compute(cache_path: str, recompute: bool = False) -> dict:
    """Load cached results or compute from checkpoints."""
    if os.path.exists(cache_path) and not recompute:
        with open(cache_path) as f:
            return json.load(f)

    print("Computing metrics from checkpoints (CPU only, may take a few minutes)...")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    results = {}
    for key, run in RUNS.items():
        print(f"  {key}...")

        # Cumulative path length
        path_length, steps, tokens = sum_step_norms(run["norms_path"], run["best_tokens"])

        # Displacement
        displacement = compute_displacement(run["best_checkpoint"])

        # L0 sparsity
        l0 = compute_l0_sparsity(run["best_checkpoint"])

        results[key] = {
            "path_length": path_length,
            "displacement": displacement,
            "l0_sparsity": l0,
            "steps": steps,
            "tokens": tokens,
        }
        print(f"    path={path_length:.4f}, disp={displacement:.4f}, "
              f"l0={l0:.4f}, steps={steps}, tokens={tokens:,}")

    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Cached to {cache_path}")
    return results


# ── Plotting ───────────────────────────────────────────────────────────────

def plot(results: dict, output_dir: str = "plots"):
    order = ["adamw_mixed", "muon_mixed", "adamw_bf16", "muon_bf16"]
    labels = [RUNS[k]["label"] for k in order]

    displacements = [results[k]["displacement"] for k in order]
    efficiency = [results[k]["displacement"] / results[k]["path_length"] for k in order]
    l0 = [results[k]["l0_sparsity"] for k in order]
    bar_colors = [COLORS[k] for k in order]

    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(18, 5.5))
    fig.text(0.5, -0.02, "Evaluated at checkpoint with highest $\\bf{validation\\ accuracy}$ for each run.",
             ha="center", fontsize=10, fontstyle="italic", color="#555555")
    x = np.arange(len(labels))
    width = 0.6

    # ── Left: Update magnitude (displacement) ──
    bars0 = ax0.bar(x, displacements, width, color=bar_colors, edgecolor="black", linewidth=0.8)
    for bar, val in zip(bars0, displacements):
        ax0.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f"{val:.4f}", ha="center", va="bottom",
                 fontsize=9, fontweight="bold")

    ax0.set_ylabel(r"$\Vert\Delta W\Vert_F$", fontsize=12)
    ax0.set_title("Update Magnitude", fontsize=15, fontweight="bold")
    ax0.set_xticks(x)
    ax0.set_xticklabels(labels, fontsize=10)
    ax0.set_ylim(0, max(displacements) * 1.25)

    # ── Middle: Path efficiency ──
    bars1 = ax1.bar(x, efficiency, width, color=bar_colors, edgecolor="black", linewidth=0.8)
    for bar, val in zip(bars1, efficiency):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008,
                 f"{val:.3f}", ha="center", va="bottom",
                 fontsize=9, fontweight="bold")

    ax1.set_ylabel("Displacement / Path Length", fontsize=12)
    ax1.set_title("Optimization Path Efficiency", fontsize=15, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=10)
    ax1.set_ylim(0, max(efficiency) * 1.4)
    ax1.axhline(y=1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)

    ax1.text(0.03, 0.97,
             r"$\dfrac{\Vert\Delta W\Vert_F}{\sum_t\Vert\delta W_t\Vert_F}$"
             + "   (1.0 = straight line)",
             transform=ax1.transAxes, fontsize=11, va="top", ha="left",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="wheat", alpha=0.7))

    # ── Right: L0 sparsity ──
    bars2 = ax2.bar(x, [v * 100 for v in l0], width, color=bar_colors,
                    edgecolor="black", linewidth=0.8)
    for bar, val in zip(bars2, l0):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                 f"{val:.1%}", ha="center", va="bottom",
                 fontsize=9, fontweight="bold")

    ax2.set_ylabel(r"L0 Sparsity of $\Delta W$ (%)", fontsize=12)
    ax2.set_title(r"Weight Update Sparsity ($\Delta W$)", fontsize=15, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=10)
    ax2.set_ylim(0, 115)

    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    for ext in ("png", "pdf"):
        path = os.path.join(output_dir, f"path_efficiency_and_sparsity.{ext}")
        fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved to {output_dir}/path_efficiency_and_sparsity.{{png,pdf}}")

    # ── Summary table ──
    print()
    print(f"{'Run':<18} {'Path':>10} {'Displ':>10} {'Effic':>8} {'L0':>8} {'Steps':>8} {'Tokens':>12}")
    print("-" * 75)
    for k in order:
        r = results[k]
        eff = r["displacement"] / r["path_length"]
        print(f"{RUNS[k]['label'].replace(chr(10), ' '):<18} "
              f"{r['path_length']:>10.4f} {r['displacement']:>10.4f} "
              f"{eff:>8.3f} {r['l0_sparsity']:>7.1%} "
              f"{r['steps']:>8} {r['tokens']:>12,}")


def main():
    parser = argparse.ArgumentParser(description="Plot path efficiency and L0 sparsity")
    parser.add_argument("--output-dir", default="plots", help="Output directory for plots")
    parser.add_argument("--cache", default="plots/path_efficiency_cache.json",
                        help="Cache file for computed metrics")
    parser.add_argument("--recompute", action="store_true",
                        help="Recompute metrics from checkpoints (slow)")
    args = parser.parse_args()

    results = load_or_compute(args.cache, recompute=args.recompute)
    plot(results, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
