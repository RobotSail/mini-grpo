#!/usr/bin/env python3
"""
Plot accuracy vs KL for checkpoints at a fixed token budget (e.g., ~1.1M tokens).
"""

import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt


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


def load_checkpoint_at_budget(exp_dir: str, exp_type: str, target_tokens: int, tolerance: float = 0.2) -> dict | None:
    """Load the checkpoint closest to target_tokens within tolerance."""
    exp_path = Path(exp_dir)

    eval_file = exp_path / "all_checkpoints_gsm8k_test.json"
    kl_file = exp_path / "all_checkpoints_kl.json"

    if not eval_file.exists() or not kl_file.exists():
        print(f"[WARN] Missing files for {exp_dir}")
        return None

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

    # Find checkpoint closest to target within tolerance
    min_tokens = int(target_tokens * (1 - tolerance))
    max_tokens = int(target_tokens * (1 + tolerance))

    best_match = None
    best_distance = float("inf")

    for ckpt_name, eval_data in eval_results.items():
        tokens = extract_tokens(ckpt_name)
        if tokens == 0:
            continue

        if min_tokens <= tokens <= max_tokens:
            distance = abs(tokens - target_tokens)
            if distance < best_distance and tokens in kl_by_tokens:
                best_distance = distance
                kl_data = kl_by_tokens[tokens]
                best_match = {
                    "name": ckpt_name,
                    "tokens": tokens,
                    "accuracy": eval_data["accuracy"],
                    "kl": kl_data["kl_divergence"],
                    "exp_type": exp_type,
                }

    return best_match


def plot_variant(variant_num: int, base_dir: str, output_path: str, target_tokens: int):
    """Create a plot for one variant at fixed token budget."""
    experiments = {
        "grpo_adamw": f"{base_dir}/qwen2-1.5b-gsm8k-grpo-adamw_verify_{variant_num}",
        "grpo_muon": f"{base_dir}/qwen2-1.5b-gsm8k-grpo-muon_verify_{variant_num}",
        "sft_adamw": f"{base_dir}/qwen2-1.5b-gsm8k-sft-adamw_verify_{variant_num}",
        "sft_muon": f"{base_dir}/qwen2-1.5b-gsm8k-sft-muon_verify_{variant_num}",
    }

    fig, ax = plt.subplots(figsize=(8, 6))

    points = []
    for exp_type, exp_dir in experiments.items():
        data = load_checkpoint_at_budget(exp_dir, exp_type, target_tokens)
        if data:
            points.append(data)
            style = EXPERIMENT_STYLES[exp_type]
            ax.scatter(
                data["kl"], data["accuracy"],
                c=style["color"],
                marker=style["marker"],
                s=150,
                label=f"{style['label']} ({data['tokens']/1e6:.2f}M)",
                alpha=0.9,
                edgecolors="black",
                linewidths=1,
            )

    if not points:
        print(f"[WARN] No data found for variant {variant_num}")
        plt.close()
        return

    ax.set_xlabel(r"$D_{\mathrm{KL}}(p_{\mathrm{base}} \| p_{\mathrm{trained}})$", fontsize=13)
    ax.set_ylabel("GSM8K Test Accuracy", fontsize=13)
    ax.set_title(f"Accuracy vs KL at ~{target_tokens/1e6:.1f}M Tokens (Seed {variant_num})", fontsize=14)
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)

    # Format y-axis as percentage
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))

    # Set reasonable axis limits
    kls = [p["kl"] for p in points]
    accs = [p["accuracy"] for p in points]

    kl_min, kl_max = min(kls), max(kls)
    acc_min, acc_max = min(accs), max(accs)

    kl_pad = (kl_max - kl_min) * 0.3 or 0.01
    acc_pad = (acc_max - acc_min) * 0.3 or 0.05

    ax.set_xlim(max(0, kl_min - kl_pad), kl_max + kl_pad)
    ax.set_ylim(max(0, acc_min - acc_pad), min(1.0, acc_max + acc_pad))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")

    # Print the data
    print(f"\nVariant {variant_num} at ~{target_tokens/1e6:.1f}M tokens:")
    for p in sorted(points, key=lambda x: x["kl"]):
        print(f"  {EXPERIMENT_STYLES[p['exp_type']]['label']:15} | "
              f"tokens={p['tokens']:>8} | acc={p['accuracy']:.1%} | KL={p['kl']:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Plot checkpoints at fixed token budget")
    parser.add_argument(
        "--base-dir",
        type=str,
        default="/mnt/nvme2n1/checkpoints/verify-exps-variable-seeds",
    )
    parser.add_argument(
        "--variants",
        type=str,
        default="1,2,3",
    )
    parser.add_argument(
        "--target-tokens",
        type=int,
        default=1_100_000,
        help="Target token count for comparison",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.2,
        help="Tolerance as fraction of target (e.g., 0.2 = ±20%)",
    )

    args = parser.parse_args()
    variants = [int(v.strip()) for v in args.variants.split(",")]

    for v in variants:
        output_path = f"{args.base_dir}/variant{v}_fixed_{args.target_tokens//1000}k.png"
        plot_variant(v, args.base_dir, output_path, args.target_tokens)


if __name__ == "__main__":
    main()
