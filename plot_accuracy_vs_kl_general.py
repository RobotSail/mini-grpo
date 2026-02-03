#!/usr/bin/env python3
"""
General-purpose plot for GSM8K accuracy vs KL divergence.

Usage:
    python plot_accuracy_vs_kl_general.py \
        --eval-results /path/to/eval_results.json \
        --kl-results /path/to/kl_results.json \
        --output /path/to/plot.png \
        --title "My Plot Title"

The eval-results JSON should have format:
{
    "grpo_adamw_tokens_XXX": {"accuracy": 0.85, ...},
    "grpo_muon_tokens_XXX": {"accuracy": 0.82, ...},
    ...
}

The kl-results JSON should have format:
{
    "grpo_adamw": {"kl_divergence": 0.10, ...},
    "grpo_muon": {"kl_divergence": 0.04, ...},
    ...
}
"""

import argparse
import json
import re
import matplotlib.pyplot as plt
from pathlib import Path


# Color scheme: Muon = red, AdamW = blue
# Marker scheme: GRPO = circle (o), SFT = square (s)
OPTIMIZER_COLORS = {
    "adamw": "#1f77b4",  # blue
    "muon": "#d62728",   # red
}

METHOD_MARKERS = {
    "grpo": "o",  # circle
    "sft": "s",   # square
}

BASELINE_STYLE = {
    "color": "#7f7f7f",  # gray
    "marker": "*",
}


def parse_experiment_name(name: str) -> tuple[str, str, str] | None:
    """
    Parse experiment name to extract method, optimizer, and checkpoint info.

    Handles formats like:
    - grpo_adamw_tokens_1110186
    - sft_muon_tokens_1104739
    - grpo_adamw
    - sft_muon

    Returns (method, optimizer, label) or None if not parseable.
    """
    name_lower = name.lower()

    # Detect method
    if "grpo" in name_lower:
        method = "grpo"
    elif "sft" in name_lower:
        method = "sft"
    else:
        return None

    # Detect optimizer
    if "adamw" in name_lower:
        optimizer = "adamw"
    elif "muon" in name_lower:
        optimizer = "muon"
    else:
        return None

    # Create display label
    opt_display = "AdamW" if optimizer == "adamw" else "Muon"
    method_display = method.upper()
    label = f"{opt_display} + {method_display}"

    return method, optimizer, label


def load_results(eval_path: str, kl_path: str) -> dict:
    """Load and merge eval results with KL results."""
    with open(eval_path) as f:
        eval_results = json.load(f)

    with open(kl_path) as f:
        kl_results = json.load(f)

    experiments = {}

    # Match eval results to KL results
    for eval_name, eval_data in eval_results.items():
        parsed = parse_experiment_name(eval_name)
        if not parsed:
            print(f"[WARN] Could not parse experiment name: {eval_name}")
            continue

        method, optimizer, label = parsed

        # Find matching KL result
        kl_key = f"{method}_{optimizer}"
        if kl_key not in kl_results:
            # Try alternate formats
            for k in kl_results:
                if method in k.lower() and optimizer in k.lower():
                    kl_key = k
                    break
            else:
                print(f"[WARN] No KL result found for {eval_name} (tried {kl_key})")
                continue

        kl_data = kl_results[kl_key]

        experiments[label] = {
            "accuracy": eval_data["accuracy"],
            "kl": kl_data["kl_divergence"],
            "color": OPTIMIZER_COLORS[optimizer],
            "marker": METHOD_MARKERS[method],
            "method": method,
            "optimizer": optimizer,
        }

    return experiments


def plot_accuracy_vs_kl(
    experiments: dict,
    output_path: str,
    title: str = "GSM8K Accuracy vs KL Divergence",
    include_baseline: bool = False,
    baseline_accuracy: float = 0.0493,
):
    """Create the accuracy vs KL plot."""
    fig, ax = plt.subplots(figsize=(10, 7))

    # Plot baseline if requested
    if include_baseline:
        ax.scatter(
            0.0,
            baseline_accuracy,
            c=BASELINE_STYLE["color"],
            marker=BASELINE_STYLE["marker"],
            s=300,
            label="Baseline",
            edgecolors="black",
            linewidths=1,
            zorder=3,
        )
        ax.annotate(
            "Baseline",
            (0.0, baseline_accuracy),
            textcoords="offset points",
            xytext=(10, 5),
            fontsize=10,
            fontweight="bold",
            color=BASELINE_STYLE["color"],
        )

    # Plot experiments
    for name, data in experiments.items():
        ax.scatter(
            data["kl"],
            data["accuracy"],
            c=data["color"],
            marker=data["marker"],
            s=200,
            label=name,
            edgecolors="black",
            linewidths=1,
            zorder=3,
        )
        ax.annotate(
            name,
            (data["kl"], data["accuracy"]),
            textcoords="offset points",
            xytext=(10, 5),
            fontsize=10,
            fontweight="bold",
            color=data["color"],
        )

    ax.set_xlabel(r"$D_{\mathrm{KL}}(p_{\mathrm{base}} \| p_{\mathrm{exp}})$", fontsize=14)
    ax.set_ylabel("GSM8K Accuracy", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3, zorder=1)

    # Format y-axis as percentage
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))

    # Auto-scale axes with some padding
    all_kl = [d["kl"] for d in experiments.values()]
    all_acc = [d["accuracy"] for d in experiments.values()]
    if include_baseline:
        all_kl.append(0.0)
        all_acc.append(baseline_accuracy)

    kl_min, kl_max = min(all_kl), max(all_kl)
    acc_min, acc_max = min(all_acc), max(all_acc)

    kl_pad = (kl_max - kl_min) * 0.15 or 0.01
    acc_pad = (acc_max - acc_min) * 0.15 or 0.05

    ax.set_xlim(max(-0.01, kl_min - kl_pad), kl_max + kl_pad)
    ax.set_ylim(max(0, acc_min - acc_pad), min(1.0, acc_max + acc_pad))

    plt.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot GSM8K accuracy vs KL divergence"
    )
    parser.add_argument(
        "--eval-results",
        type=str,
        required=True,
        help="Path to eval results JSON file",
    )
    parser.add_argument(
        "--kl-results",
        type=str,
        required=True,
        help="Path to KL divergence results JSON file",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output path for the plot (PNG)",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="GSM8K Accuracy vs KL Divergence",
        help="Plot title",
    )
    parser.add_argument(
        "--include-baseline",
        action="store_true",
        help="Include baseline (Qwen2-1.5B-Instruct) point at KL=0",
    )
    parser.add_argument(
        "--baseline-accuracy",
        type=float,
        default=0.0493,
        help="Baseline accuracy (default: 0.0493 from real GSM8K test)",
    )

    args = parser.parse_args()

    # Load and merge results
    experiments = load_results(args.eval_results, args.kl_results)

    if not experiments:
        print("ERROR: No experiments found. Check your input files.")
        return

    print(f"Found {len(experiments)} experiments:")
    for name, data in experiments.items():
        print(f"  {name}: accuracy={data['accuracy']:.2%}, KL={data['kl']:.4f}")

    # Generate plot
    plot_accuracy_vs_kl(
        experiments,
        args.output,
        title=args.title,
        include_baseline=args.include_baseline,
        baseline_accuracy=args.baseline_accuracy,
    )


if __name__ == "__main__":
    main()
