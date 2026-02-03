#!/usr/bin/env python3
"""
Plot accuracy vs KL for all checkpoints, showing clustering by experiment.
Creates one plot per variant showing how checkpoints evolve during training.
"""

import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


# Color scheme: Muon = red, AdamW = blue
# Marker scheme: GRPO = circle (o), SFT = square (s)
EXPERIMENT_STYLES = {
    "grpo_adamw": {"color": "#1f77b4", "marker": "o", "label": "AdamW + GRPO"},
    "grpo_muon": {"color": "#d62728", "marker": "o", "label": "Muon + GRPO"},
    "sft_adamw": {"color": "#1f77b4", "marker": "s", "label": "AdamW + SFT"},
    "sft_muon": {"color": "#d62728", "marker": "s", "label": "Muon + SFT"},
}


def extract_tokens(name: str) -> int:
    """Extract token count from checkpoint name."""
    if "tokens_" in name:
        try:
            return int(name.split("tokens_")[-1].split("_")[0].replace(".0", ""))
        except ValueError:
            pass
    return 0


def load_and_merge_results(exp_dir: str, exp_type: str) -> list[dict]:
    """Load eval and KL results for an experiment, merge them."""
    exp_path = Path(exp_dir)

    eval_file = exp_path / "all_checkpoints_gsm8k_test.json"
    kl_file = exp_path / "all_checkpoints_kl.json"

    if not eval_file.exists():
        print(f"[WARN] Missing eval file: {eval_file}")
        return []
    if not kl_file.exists():
        print(f"[WARN] Missing KL file: {kl_file}")
        return []

    with open(eval_file) as f:
        eval_results = json.load(f)
    with open(kl_file) as f:
        kl_results = json.load(f)

    # Build KL lookup by token count
    kl_by_tokens = {}
    for k, v in kl_results.items():
        tokens = v.get("tokens", 0) or extract_tokens(k)
        if tokens > 0:
            kl_by_tokens[tokens] = v

    # Merge by token count
    merged = []
    for ckpt_name, eval_data in eval_results.items():
        tokens = extract_tokens(ckpt_name)
        if tokens == 0:
            continue

        # Find matching KL result by token count
        kl_data = kl_by_tokens.get(tokens)
        if kl_data:
            merged.append({
                "name": ckpt_name,
                "tokens": tokens,
                "accuracy": eval_data["accuracy"],
                "kl": kl_data["kl_divergence"],
                "exp_type": exp_type,
            })

    return sorted(merged, key=lambda x: x["tokens"])


def plot_variant(variant_num: int, base_dir: str, output_path: str):
    """Create a plot for one variant showing all experiments."""
    experiments = {
        "grpo_adamw": f"{base_dir}/qwen2-1.5b-gsm8k-grpo-adamw_verify_{variant_num}",
        "grpo_muon": f"{base_dir}/qwen2-1.5b-gsm8k-grpo-muon_verify_{variant_num}",
        "sft_adamw": f"{base_dir}/qwen2-1.5b-gsm8k-sft-adamw_verify_{variant_num}",
        "sft_muon": f"{base_dir}/qwen2-1.5b-gsm8k-sft-muon_verify_{variant_num}",
    }

    fig, ax = plt.subplots(figsize=(10, 7))

    all_kl = []
    all_acc = []

    for exp_type, exp_dir in experiments.items():
        data = load_and_merge_results(exp_dir, exp_type)
        if not data:
            continue

        style = EXPERIMENT_STYLES[exp_type]
        kls = [d["kl"] for d in data]
        accs = [d["accuracy"] for d in data]

        all_kl.extend(kls)
        all_acc.extend(accs)

        ax.scatter(
            kls, accs,
            c=style["color"],
            marker=style["marker"],
            s=80,
            label=style["label"],
            alpha=0.8,
            edgecolors="white",
            linewidths=0.5,
        )

    if not all_kl:
        print(f"[WARN] No data found for variant {variant_num}")
        plt.close()
        return

    ax.set_xlabel(r"$D_{\mathrm{KL}}(p_{\mathrm{base}} \| p_{\mathrm{trained}})$", fontsize=13)
    ax.set_ylabel("GSM8K Test Accuracy", fontsize=13)
    ax.set_title(f"Accuracy vs KL Divergence (Seed {variant_num})", fontsize=14)
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(True, alpha=0.3)

    # Format y-axis as percentage
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))

    # Auto-scale with padding
    kl_min, kl_max = min(all_kl), max(all_kl)
    acc_min, acc_max = min(all_acc), max(all_acc)
    kl_pad = (kl_max - kl_min) * 0.1 or 0.01
    acc_pad = (acc_max - acc_min) * 0.1 or 0.05

    ax.set_xlim(max(0, kl_min - kl_pad), kl_max + kl_pad)
    ax.set_ylim(max(0, acc_min - acc_pad), min(1.0, acc_max + acc_pad))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot all checkpoints for each variant")
    parser.add_argument(
        "--base-dir",
        type=str,
        default="/mnt/nvme2n1/checkpoints/verify-exps-variable-seeds",
        help="Base directory containing experiment folders",
    )
    parser.add_argument(
        "--variants",
        type=str,
        default="1,2,3",
        help="Comma-separated list of variant numbers to plot",
    )

    args = parser.parse_args()

    variants = [int(v.strip()) for v in args.variants.split(",")]

    for v in variants:
        output_path = f"{args.base_dir}/variant{v}_all_checkpoints.png"
        plot_variant(v, args.base_dir, output_path)


if __name__ == "__main__":
    main()
