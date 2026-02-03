#!/usr/bin/env python3
"""
Plot accuracy vs training progress for v1 artifacts (training set eval).
Creates separate plots for GRPO and SFT experiments with data point labels.
"""

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt


def extract_tokens_from_path(path: str) -> int:
    """Extract tokens count from checkpoint path."""
    match = re.search(r"tokens_(\d+)", path)
    if match:
        return int(match.group(1))
    return 0


def extract_samples_from_path(path: str) -> float:
    """Extract samples count from checkpoint path."""
    match = re.search(r"samples_([\d.]+)_", path)
    if match:
        return float(match.group(1))
    return 0.0


def load_results(json_path: str, use_samples: bool = False) -> list[tuple[float, float]]:
    """Load results and return list of (x_val, accuracy) tuples."""
    with open(json_path) as f:
        data = json.load(f)

    results = []
    for label, metrics in data.items():
        path = metrics.get("path", "")
        accuracy = metrics.get("accuracy", 0.0)

        if use_samples:
            x_val = extract_samples_from_path(path)
        else:
            x_val = extract_tokens_from_path(path)

        if x_val > 0:
            results.append((x_val, accuracy))

    results.sort(key=lambda x: x[0])
    return results


def plot_grpo(base_dir: Path, output_path: Path):
    """Create plot for GRPO experiments (tokens on x-axis)."""
    experiments = {
        "adamw": {
            "file": base_dir / "grpo_adamw/eval_results_train.json",
            "color": "#1f77b4",
            "marker": "o",
            "label": "AdamW + GRPO",
            "offset": (5, 5),
        },
        "muon": {
            "file": base_dir / "grpo_muon/eval_results_train.json",
            "color": "#d62728",
            "marker": "D",
            "label": "Muon + GRPO",
            "offset": (5, -10),
        },
    }

    fig, ax = plt.subplots(figsize=(14, 8))

    for exp_name, config in experiments.items():
        if not config["file"].exists():
            print(f"Warning: {config['file']} not found, skipping")
            continue

        data = load_results(str(config["file"]), use_samples=False)
        if not data:
            print(f"Warning: No valid data in {config['file']}")
            continue

        x_vals, y_vals = zip(*data)
        ax.plot(
            x_vals, y_vals,
            color=config["color"],
            marker=config["marker"],
            markersize=6,
            linewidth=2,
            label=config["label"],
        )

        # Add labels for each point
        for x, y in data:
            label = f"{int(x)}"
            ax.annotate(
                label,
                (x, y),
                textcoords="offset points",
                xytext=config["offset"],
                fontsize=7,
                color=config["color"],
                alpha=0.8,
            )

    ax.set_xlabel("Training Tokens", fontsize=12)
    ax.set_ylabel("GSM8K Accuracy (Train Set)", fontsize=12)
    ax.set_title("GRPO: AdamW vs Muon - GSM8K Train Accuracy", fontsize=14)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.0)

    ax.xaxis.set_major_formatter(plt.FuncFormatter(
        lambda x, p: f"{x/1e6:.1f}M" if x >= 1e6 else f"{x/1e3:.0f}K"
    ))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_sft(base_dir: Path, output_path: Path):
    """Create plot for SFT experiments (samples on x-axis)."""
    experiments = {
        "adamw": {
            "file": base_dir / "sft_adamw/hf_format/eval_results_train.json",
            "color": "#1f77b4",
            "marker": "o",
            "label": "AdamW + SFT",
            "offset": (5, 5),
        },
        "muon": {
            "file": base_dir / "sft_muon/hf_format/eval_results_train.json",
            "color": "#ff7f0e",
            "marker": "^",
            "label": "Muon + SFT",
            "offset": (5, -10),
        },
    }

    fig, ax = plt.subplots(figsize=(14, 8))

    for exp_name, config in experiments.items():
        if not config["file"].exists():
            print(f"Warning: {config['file']} not found, skipping")
            continue

        data = load_results(str(config["file"]), use_samples=True)
        if not data:
            print(f"Warning: No valid data in {config['file']}")
            continue

        x_vals, y_vals = zip(*data)
        ax.plot(
            x_vals, y_vals,
            color=config["color"],
            marker=config["marker"],
            markersize=6,
            linewidth=2,
            label=config["label"],
        )

        # Add labels for each point
        for x, y in data:
            label = f"{int(x)}"
            ax.annotate(
                label,
                (x, y),
                textcoords="offset points",
                xytext=config["offset"],
                fontsize=7,
                color=config["color"],
                alpha=0.8,
            )

    ax.set_xlabel("Training Samples", fontsize=12)
    ax.set_ylabel("GSM8K Accuracy (Train Set)", fontsize=12)
    ax.set_title("SFT: AdamW vs Muon - GSM8K Train Accuracy", fontsize=14)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.0)

    ax.xaxis.set_major_formatter(plt.FuncFormatter(
        lambda x, p: f"{x/1e3:.1f}K" if x >= 1e3 else f"{x:.0f}"
    ))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def main():
    base_dir = Path("/mnt/nvme3n1/workspace/osilkin/mini-grpo/adamw-vs-muon-grpo-v1-artifacts/checkpoints")
    output_dir = Path("/mnt/nvme3n1/workspace/osilkin/mini-grpo/adamw-vs-muon-grpo-v1-artifacts")

    # Create GRPO plot
    grpo_output = output_dir / "grpo_accuracy_plot_train.png"
    plot_grpo(base_dir, grpo_output)

    # Create SFT plot
    sft_output = output_dir / "sft_accuracy_plot_train.png"
    plot_sft(base_dir, sft_output)

    print("\nPlots generated:")
    print(f"  GRPO: {grpo_output}")
    print(f"  SFT: {sft_output}")


if __name__ == "__main__":
    main()
