#!/usr/bin/env python3
"""
Condition Number Analysis

Computes and plots condition number (κ = σ_max / σ_min) for each layer,
comparing experiments against a baseline.

Usage:
    python plot_condition_number.py \
        --baseline svd_baseline.json \
        --checkpoints svd_grpo_muon.json svd_grpo_adamw.json svd_sft_muon.json svd_sft_adamw.json \
        --labels "GRPO + Muon" "GRPO + AdamW" "SFT + Muon" "SFT + AdamW" \
        --output-dir ./condition_plots
"""

import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def compute_condition_number(singular_values: list[float], eps: float = 1e-10) -> float:
    """
    Compute condition number κ = σ_max / σ_min.

    Args:
        singular_values: List of singular values (assumed sorted descending)
        eps: Small value to avoid division by zero

    Returns:
        Condition number
    """
    sv = np.array(singular_values)
    sv = sv[sv > eps]  # Remove near-zero values

    if len(sv) == 0:
        return float('inf')

    sigma_max = sv[0]  # Assumes sorted descending
    sigma_min = sv[-1]

    return sigma_max / max(sigma_min, eps)


def compute_stable_rank(singular_values: list[float]) -> float:
    """Compute stable rank = ||A||_F² / σ₁² = Σσᵢ² / σ₁²"""
    sv = np.array(singular_values)
    sv = sv[sv > 0]
    if len(sv) == 0:
        return 0
    frobenius_sq = np.sum(sv ** 2)
    spectral_sq = sv[0] ** 2
    return frobenius_sq / spectral_sq


def compute_effective_rank(singular_values: list[float]) -> float:
    """Compute effective rank = (Σσᵢ)² / Σσᵢ²"""
    sv = np.array(singular_values)
    sv = sv[sv > 0]
    if len(sv) == 0:
        return 0
    nuclear = np.sum(sv)
    frobenius_sq = np.sum(sv ** 2)
    return (nuclear ** 2) / frobenius_sq


def load_svd_data(path: str) -> dict:
    """Load SVD data from JSON file."""
    with open(path) as f:
        return json.load(f)


def get_layer_type(param_name: str) -> str:
    """Extract layer type from parameter name."""
    # Common patterns: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
    for layer_type in ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj', 'lm_head', 'embed']:
        if layer_type in param_name:
            return layer_type
    return 'other'


def plot_condition_number_comparison(
    baseline_data: dict,
    checkpoint_datasets: list[dict],
    labels: list[str],
    output_dir: Path,
):
    """Create comparison plots for condition number."""

    # Get common parameters
    all_param_names = set(baseline_data["parameters"].keys())
    for data in checkpoint_datasets:
        all_param_names &= set(data["parameters"].keys())

    print(f"Found {len(all_param_names)} common parameters")

    # Compute condition numbers for baseline
    baseline_cond = {}
    for param_name in all_param_names:
        sv = baseline_data["parameters"][param_name]
        baseline_cond[param_name] = compute_condition_number(sv)

    # Compute condition numbers for each checkpoint
    checkpoint_conds = []
    for data in checkpoint_datasets:
        cond = {}
        for param_name in all_param_names:
            sv = data["parameters"][param_name]
            cond[param_name] = compute_condition_number(sv)
        checkpoint_conds.append(cond)

    # Compute differences from baseline
    diffs = []
    for cond in checkpoint_conds:
        diff = {}
        for param_name in all_param_names:
            diff[param_name] = cond[param_name] - baseline_cond[param_name]
        diffs.append(diff)

    # Colors by optimizer, markers by technique (same as accuracy plot)
    colors = {
        "GRPO + Muon": "#e74c3c",
        "GRPO + AdamW": "#3498db",
        "SFT + Muon": "#e74c3c",
        "SFT + AdamW": "#3498db",
    }
    markers = {
        "GRPO + Muon": "o",
        "GRPO + AdamW": "o",
        "SFT + Muon": "s",
        "SFT + AdamW": "s",
    }

    # Default colors if labels don't match
    default_colors = plt.cm.tab10(np.linspace(0, 1, len(labels)))

    # =========================================================================
    # Plot 1: Distribution of condition number differences (box plot by layer type)
    # =========================================================================
    layer_types = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']

    fig, ax = plt.subplots(figsize=(14, 8))

    # Group data by layer type for each checkpoint
    positions = np.arange(len(layer_types))
    width = 0.8 / len(labels)

    for idx, (label, diff) in enumerate(zip(labels, diffs)):
        layer_diffs = {lt: [] for lt in layer_types}
        for param_name, d in diff.items():
            lt = get_layer_type(param_name)
            if lt in layer_types:
                layer_diffs[lt].append(d)

        # Get means for bar plot
        means = [np.mean(layer_diffs[lt]) if layer_diffs[lt] else 0 for lt in layer_types]
        stds = [np.std(layer_diffs[lt]) if layer_diffs[lt] else 0 for lt in layer_types]

        color = colors.get(label, default_colors[idx])
        marker = markers.get(label, 'o')

        offset = (idx - len(labels)/2 + 0.5) * width
        bars = ax.bar(positions + offset, means, width,
                      label=label, color=color, alpha=0.7, edgecolor='black')
        ax.errorbar(positions + offset, means, yerr=stds,
                    fmt='none', color='black', capsize=3, alpha=0.5)

    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
    ax.set_xticks(positions)
    ax.set_xticklabels(layer_types, rotation=45, ha='right')
    ax.set_xlabel("Layer Type", fontsize=12, fontweight='bold')
    ax.set_ylabel(r"$\Delta$ Condition Number ($\kappa - \kappa_0$)", fontsize=12, fontweight='bold')
    ax.set_title("Condition Number Change from Baseline by Layer Type\n" +
                 r"$\kappa = \sigma_{max} / \sigma_{min}$", fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_dir / "condition_number_by_layer.png", dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / "condition_number_by_layer.pdf", bbox_inches='tight')
    print(f"Saved: {output_dir / 'condition_number_by_layer.png'}")
    plt.close()

    # =========================================================================
    # Plot 2: Overall condition number change (scatter plot like accuracy vs KL)
    # =========================================================================
    fig, ax = plt.subplots(figsize=(10, 7))

    for idx, (label, diff, cond) in enumerate(zip(labels, diffs, checkpoint_conds)):
        # Mean absolute condition number change
        mean_diff = np.mean(list(diff.values()))
        mean_cond = np.mean(list(cond.values()))

        color = colors.get(label, default_colors[idx])
        marker = markers.get(label, 'o')

        ax.scatter(mean_diff, mean_cond, c=color, marker=marker, s=200,
                   edgecolors='black', linewidths=1.5, zorder=5)
        ax.annotate(label, (mean_diff + 0.5, mean_cond + 0.5), fontsize=10, fontweight='bold')

    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel(r"Mean $\Delta$ Condition Number from Baseline", fontsize=12, fontweight='bold')
    ax.set_ylabel(r"Mean Condition Number ($\kappa$)", fontsize=12, fontweight='bold')
    ax.set_title("Overall Condition Number Analysis\n" +
                 r"$\kappa = \sigma_{max} / \sigma_{min}$ (lower = more stable)",
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.savefig(output_dir / "condition_number_overall.png", dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / "condition_number_overall.pdf", bbox_inches='tight')
    print(f"Saved: {output_dir / 'condition_number_overall.png'}")
    plt.close()

    # =========================================================================
    # Plot 3: Per-layer condition number differences (heatmap style)
    # =========================================================================
    # Get layers in order (by layer number)
    def get_layer_num(name):
        import re
        match = re.search(r'layers\.(\d+)', name)
        return int(match.group(1)) if match else -1

    sorted_params = sorted(all_param_names, key=lambda x: (get_layer_num(x), x))

    # Filter to just attention layers for cleaner visualization
    attn_params = [p for p in sorted_params if any(x in p for x in ['q_proj', 'k_proj', 'v_proj', 'o_proj'])]

    if attn_params:
        fig, ax = plt.subplots(figsize=(14, max(8, len(attn_params) * 0.15)))

        # Create matrix: rows = layers, cols = checkpoints
        matrix = np.zeros((len(attn_params), len(labels)))
        for i, param_name in enumerate(attn_params):
            for j, diff in enumerate(diffs):
                matrix[i, j] = diff[param_name]

        # Clip extreme values for better visualization
        vmax = np.percentile(np.abs(matrix), 95)

        im = ax.imshow(matrix, aspect='auto', cmap='RdBu_r', vmin=-vmax, vmax=vmax)

        ax.set_xticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_yticks(np.arange(0, len(attn_params), max(1, len(attn_params)//20)))
        ax.set_yticklabels([attn_params[i].split('.')[-1] for i in range(0, len(attn_params), max(1, len(attn_params)//20))], fontsize=8)

        ax.set_xlabel("Experiment", fontsize=12, fontweight='bold')
        ax.set_ylabel("Attention Layer", fontsize=12, fontweight='bold')
        ax.set_title(r"Condition Number Change ($\Delta\kappa$) - Attention Layers",
                     fontsize=14, fontweight='bold')

        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label(r"$\Delta$ Condition Number", fontsize=11)

        plt.tight_layout()
        plt.savefig(output_dir / "condition_number_heatmap.png", dpi=150, bbox_inches='tight')
        plt.savefig(output_dir / "condition_number_heatmap.pdf", bbox_inches='tight')
        print(f"Saved: {output_dir / 'condition_number_heatmap.png'}")
        plt.close()

    # =========================================================================
    # Plot 4: Summary statistics table as text
    # =========================================================================
    print("\n" + "="*70)
    print("CONDITION NUMBER SUMMARY")
    print("="*70)
    print(f"\n{'Experiment':<25} {'Mean κ':>12} {'Median κ':>12} {'Mean Δκ':>12} {'Max |Δκ|':>12}")
    print("-"*70)

    baseline_mean = np.mean(list(baseline_cond.values()))
    print(f"{'Baseline':<25} {baseline_mean:>12.2f} {np.median(list(baseline_cond.values())):>12.2f} {'—':>12} {'—':>12}")

    for label, cond, diff in zip(labels, checkpoint_conds, diffs):
        mean_cond = np.mean(list(cond.values()))
        median_cond = np.median(list(cond.values()))
        mean_diff = np.mean(list(diff.values()))
        max_abs_diff = np.max(np.abs(list(diff.values())))
        print(f"{label:<25} {mean_cond:>12.2f} {median_cond:>12.2f} {mean_diff:>+12.2f} {max_abs_diff:>12.2f}")

    print("="*70)


def main():
    parser = argparse.ArgumentParser(description="Condition Number Analysis")
    parser.add_argument("--baseline", "-b", type=str, required=True,
                        help="Baseline SVD JSON file")
    parser.add_argument("--checkpoints", "-c", type=str, nargs="+", required=True,
                        help="Checkpoint SVD JSON files")
    parser.add_argument("--labels", "-l", type=str, nargs="+", required=True,
                        help="Labels for each checkpoint")
    parser.add_argument("--output-dir", "-o", type=str, default="./condition_plots",
                        help="Output directory for plots")

    args = parser.parse_args()

    if len(args.checkpoints) != len(args.labels):
        parser.error("Number of checkpoints must match number of labels")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print(f"Loading baseline: {args.baseline}")
    baseline_data = load_svd_data(args.baseline)

    checkpoint_datasets = []
    for path, label in zip(args.checkpoints, args.labels):
        print(f"Loading: {label} from {path}")
        checkpoint_datasets.append(load_svd_data(path))

    # Generate plots
    plot_condition_number_comparison(
        baseline_data=baseline_data,
        checkpoint_datasets=checkpoint_datasets,
        labels=args.labels,
        output_dir=output_dir,
    )

    print(f"\nPlots saved to: {output_dir}")


if __name__ == "__main__":
    main()
