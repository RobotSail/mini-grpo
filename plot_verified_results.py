#!/usr/bin/env python3
"""
Plot accuracy vs samples for verified experiment results.
Creates one plot per verification run comparing all 4 experiments.
"""

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def extract_samples_from_path(path: str) -> float:
    """Extract samples count from checkpoint path."""
    # Match patterns like "samples_1088.0_tokens_" or "samples_1088.0_step_"
    match = re.search(r"samples_(\d+\.?\d*)_", path)
    if match:
        return float(match.group(1))

    # For GRPO checkpoints, extract tokens and estimate samples
    # (this is a fallback - tokens are used as x-axis for GRPO)
    match = re.search(r"tokens_(\d+)", path)
    if match:
        return float(match.group(1))

    return 0.0


def load_results(json_path: str) -> list[tuple[float, float]]:
    """Load results and return list of (samples/tokens, accuracy) tuples."""
    with open(json_path) as f:
        data = json.load(f)

    results = []
    for label, metrics in data.items():
        path = metrics.get("path", "")
        accuracy = metrics.get("accuracy", 0.0)

        # Extract x-axis value (samples or tokens)
        x_val = extract_samples_from_path(path)
        if x_val > 0:
            results.append((x_val, accuracy))

    # Sort by x value
    results.sort(key=lambda x: x[0])
    return results


def plot_verification_run(run_dir: Path, output_path: Path, run_name: str):
    """Create a single plot for one verification run."""
    experiments = {
        "adamw_sft": {"color": "#1f77b4", "marker": "o", "label": "AdamW + SFT"},
        "adamw_grpo": {"color": "#2ca02c", "marker": "s", "label": "AdamW + GRPO"},
        "muon_sft": {"color": "#ff7f0e", "marker": "^", "label": "Muon + SFT"},
        "muon_grpo": {"color": "#d62728", "marker": "D", "label": "Muon + GRPO"},
    }

    fig, ax = plt.subplots(figsize=(10, 6))

    for exp_name, style in experiments.items():
        results_file = run_dir / f"{exp_name}_results.json"
        if not results_file.exists():
            print(f"Warning: {results_file} not found, skipping")
            continue

        data = load_results(str(results_file))
        if not data:
            print(f"Warning: No valid data in {results_file}")
            continue

        x_vals, y_vals = zip(*data)
        ax.plot(
            x_vals, y_vals,
            color=style["color"],
            marker=style["marker"],
            markersize=6,
            linewidth=2,
            label=style["label"],
        )

    ax.set_xlabel("Training Tokens", fontsize=12)
    ax.set_ylabel("GSM8K Accuracy", fontsize=12)
    ax.set_title(f"GSM8K Accuracy vs Training Tokens - {run_name}", fontsize=14)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.0)

    # Format x-axis with K/M suffixes
    ax.xaxis.set_major_formatter(plt.FuncFormatter(
        lambda x, p: f"{x/1e6:.1f}M" if x >= 1e6 else f"{x/1e3:.0f}K"
    ))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def main():
    base_dir = Path("/mnt/nvme3n1/workspace/osilkin/mini-grpo/verified-weights")

    for run_num in [1, 2, 3]:
        run_name = f"verify_run_{run_num}"
        run_dir = base_dir / run_name

        if not run_dir.exists():
            print(f"Warning: {run_dir} not found, skipping")
            continue

        output_path = run_dir / f"{run_name}_accuracy_plot.png"
        plot_verification_run(run_dir, output_path, f"Verification Run {run_num}")

    print("\nAll plots generated!")


if __name__ == "__main__":
    main()
