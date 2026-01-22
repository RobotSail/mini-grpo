#!/usr/bin/env python3
"""
Spectral Summary: Aggregate metrics and clean visualizations for SVD analysis.

Computes summary statistics per layer and generates interpretable visualizations
comparing optimizer and training objective effects on singular value spectra.

Usage:
    python spectral_summary.py --cache-dir ./svd_cache --output-dir ./spectral_summary
"""

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch


# =============================================================================
# CONFIGURATION
# =============================================================================

EXPERIMENTS = {
    "adamw_sft": {"label": "AdamW + SFT", "color": "#1f77b4", "marker": "o"},
    "muon_sft": {"label": "Muon + SFT", "color": "#ff7f0e", "marker": "s"},
    "adamw_grpo": {"label": "AdamW + GRPO", "color": "#2ca02c", "marker": "^"},
    "muon_grpo": {"label": "Muon + GRPO", "color": "#d62728", "marker": "D"},
}

COMPONENT_ORDER = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


# =============================================================================
# DATA LOADING
# =============================================================================

def load_cached_svd(cache_dir: Path) -> dict:
    """Load all cached SVD files."""
    data = {}

    # Load baseline
    baseline_path = cache_dir / "svd_baseline.json"
    if baseline_path.exists():
        with open(baseline_path) as f:
            data["baseline"] = json.load(f)
    else:
        raise FileNotFoundError(f"Baseline SVD not found: {baseline_path}")

    # Load experiments
    for exp_name in EXPERIMENTS:
        exp_path = cache_dir / f"svd_{exp_name}.json"
        if exp_path.exists():
            with open(exp_path) as f:
                data[exp_name] = json.load(f)
        else:
            raise FileNotFoundError(f"Experiment SVD not found: {exp_path}")

    return data


def parse_param_name(name: str) -> dict | None:
    """Parse parameter name into layer index and component type.

    Examples:
        model.layers.5.mlp.gate_proj.weight -> {layer: 5, component: gate_proj, type: mlp}
        model.layers.10.self_attn.q_proj.weight -> {layer: 10, component: q_proj, type: attn}
    """
    # Match layer parameters
    match = re.match(r"model\.layers\.(\d+)\.(self_attn|mlp)\.(\w+)\.weight", name)
    if match:
        layer_idx = int(match.group(1))
        block_type = match.group(2)
        component = match.group(3)
        return {
            "layer": layer_idx,
            "component": component,
            "type": "attn" if block_type == "self_attn" else "mlp",
            "param_name": name,
        }

    # Match embed_tokens
    if "embed_tokens" in name:
        return {"layer": -1, "component": "embed", "type": "embed", "param_name": name}

    # Match lm_head
    if "lm_head" in name:
        return {"layer": 99, "component": "lm_head", "type": "head", "param_name": name}

    return None


# =============================================================================
# METRIC COMPUTATION
# =============================================================================

def compute_metrics(baseline_sv: list, experiment_sv: list) -> dict:
    """Compute spectral metrics comparing experiment to baseline.

    Args:
        baseline_sv: Singular values of baseline
        experiment_sv: Singular values of experiment

    Returns:
        Dictionary of metrics
    """
    base = np.array(baseline_sv)
    exp = np.array(experiment_sv)

    # Ensure same length
    min_len = min(len(base), len(exp))
    base = base[:min_len]
    exp = exp[:min_len]

    # Differences
    diff = exp - base

    # Spectral norm (largest SV)
    spectral_norm_base = base[0]
    spectral_norm_exp = exp[0]
    spectral_norm_change = spectral_norm_exp - spectral_norm_base
    spectral_norm_rel_change = spectral_norm_change / spectral_norm_base if spectral_norm_base > 0 else 0

    # Nuclear norm (sum of SVs)
    nuclear_norm_base = np.sum(base)
    nuclear_norm_exp = np.sum(exp)
    nuclear_norm_change = nuclear_norm_exp - nuclear_norm_base

    # Frobenius norm change: sqrt(sum((exp - base)^2)) = sqrt(sum(diff^2))
    # This equals ||W_new - W_base||_F when SVD is of the difference
    # But we approximate with SV differences
    frobenius_change = np.sqrt(np.sum(diff ** 2))

    # Effective rank: sum(sv) / max(sv)
    effective_rank_base = nuclear_norm_base / spectral_norm_base if spectral_norm_base > 0 else 0
    effective_rank_exp = nuclear_norm_exp / spectral_norm_exp if spectral_norm_exp > 0 else 0
    effective_rank_change = effective_rank_exp - effective_rank_base

    # Top-10 energy ratio: sum(sv[:10]^2) / sum(sv^2)
    energy_base = np.sum(base ** 2)
    energy_exp = np.sum(exp ** 2)
    top10_energy_base = np.sum(base[:10] ** 2) / energy_base if energy_base > 0 else 0
    top10_energy_exp = np.sum(exp[:10] ** 2) / energy_exp if energy_exp > 0 else 0
    top10_energy_change = top10_energy_exp - top10_energy_base

    # Mean absolute change in top-30 SVs (relative)
    top30_rel_change = np.mean(np.abs(diff[:30]) / base[:30]) if len(base) >= 30 else 0

    return {
        "spectral_norm_base": spectral_norm_base,
        "spectral_norm_exp": spectral_norm_exp,
        "spectral_norm_change": spectral_norm_change,
        "spectral_norm_rel_change": spectral_norm_rel_change,
        "nuclear_norm_change": nuclear_norm_change,
        "frobenius_change": frobenius_change,
        "effective_rank_base": effective_rank_base,
        "effective_rank_exp": effective_rank_exp,
        "effective_rank_change": effective_rank_change,
        "top10_energy_change": top10_energy_change,
        "top30_rel_change": top30_rel_change,
    }


def compute_all_metrics(svd_data: dict) -> pd.DataFrame:
    """Compute metrics for all layers and experiments.

    Returns DataFrame with columns:
        experiment, layer, component, type, param_name, + all metrics
    """
    baseline_params = svd_data["baseline"]["parameters"]
    rows = []

    for exp_name in EXPERIMENTS:
        exp_params = svd_data[exp_name]["parameters"]

        for param_name in baseline_params:
            if param_name not in exp_params:
                continue

            parsed = parse_param_name(param_name)
            if parsed is None:
                continue

            metrics = compute_metrics(baseline_params[param_name], exp_params[param_name])

            row = {
                "experiment": exp_name,
                "experiment_label": EXPERIMENTS[exp_name]["label"],
                **parsed,
                **metrics,
            }
            rows.append(row)

    return pd.DataFrame(rows)


# =============================================================================
# VISUALIZATION: BAR CHARTS
# =============================================================================

def plot_metric_by_layer(df: pd.DataFrame, metric: str, title: str, output_path: Path):
    """Plot a metric as grouped bar chart by layer."""
    # Filter to transformer layers only (0-27)
    layer_df = df[(df["layer"] >= 0) & (df["layer"] < 28)].copy()

    # Aggregate by layer (mean across components)
    agg_df = layer_df.groupby(["experiment", "layer"])[metric].mean().reset_index()

    fig, ax = plt.subplots(figsize=(16, 6))

    n_experiments = len(EXPERIMENTS)
    n_layers = 28
    width = 0.8 / n_experiments
    x = np.arange(n_layers)

    for i, (exp_name, exp_config) in enumerate(EXPERIMENTS.items()):
        exp_data = agg_df[agg_df["experiment"] == exp_name].set_index("layer")[metric]
        values = [exp_data.get(l, 0) for l in range(n_layers)]
        offset = (i - n_experiments / 2 + 0.5) * width
        ax.bar(x + offset, values, width, label=exp_config["label"], color=exp_config["color"], alpha=0.8)

    ax.set_xlabel("Layer Index")
    ax.set_ylabel(title)
    ax.set_title(f"{title} by Layer")
    ax.set_xticks(x)
    ax.set_xticklabels(x)
    ax.legend(loc="upper right")
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_metric_by_component(df: pd.DataFrame, metric: str, title: str, output_path: Path):
    """Plot a metric as grouped bar chart by component type."""
    # Filter to transformer layers only
    layer_df = df[(df["layer"] >= 0) & (df["layer"] < 28)].copy()

    # Filter to known components
    layer_df = layer_df[layer_df["component"].isin(COMPONENT_ORDER)]

    # Aggregate by component (mean across layers)
    agg_df = layer_df.groupby(["experiment", "component"])[metric].mean().reset_index()

    fig, ax = plt.subplots(figsize=(12, 6))

    n_experiments = len(EXPERIMENTS)
    n_components = len(COMPONENT_ORDER)
    width = 0.8 / n_experiments
    x = np.arange(n_components)

    for i, (exp_name, exp_config) in enumerate(EXPERIMENTS.items()):
        exp_data = agg_df[agg_df["experiment"] == exp_name].set_index("component")[metric]
        values = [exp_data.get(c, 0) for c in COMPONENT_ORDER]
        offset = (i - n_experiments / 2 + 0.5) * width
        ax.bar(x + offset, values, width, label=exp_config["label"], color=exp_config["color"], alpha=0.8)

    ax.set_xlabel("Component Type")
    ax.set_ylabel(title)
    ax.set_title(f"{title} by Component Type")
    ax.set_xticks(x)
    ax.set_xticklabels(COMPONENT_ORDER, rotation=45, ha="right")
    ax.legend(loc="upper right")
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# VISUALIZATION: HEATMAPS
# =============================================================================

def plot_component_heatmap(df: pd.DataFrame, metric: str, title: str, output_path: Path):
    """Plot heatmaps of metric by layer x component, one per experiment."""
    # Filter to transformer layers and known components
    layer_df = df[(df["layer"] >= 0) & (df["layer"] < 28)].copy()
    layer_df = layer_df[layer_df["component"].isin(COMPONENT_ORDER)]

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()

    # Determine global color scale
    vmin = layer_df[metric].min()
    vmax = layer_df[metric].max()
    # Center at 0 for diverging colormap
    vmax_abs = max(abs(vmin), abs(vmax))
    vmin, vmax = -vmax_abs, vmax_abs

    for idx, (exp_name, exp_config) in enumerate(EXPERIMENTS.items()):
        ax = axes[idx]
        exp_df = layer_df[layer_df["experiment"] == exp_name]

        # Pivot to matrix form
        pivot = exp_df.pivot_table(
            index="layer", columns="component", values=metric, aggfunc="mean"
        )
        # Reorder columns
        pivot = pivot.reindex(columns=[c for c in COMPONENT_ORDER if c in pivot.columns])

        sns.heatmap(
            pivot, ax=ax, cmap="RdBu_r", center=0, vmin=vmin, vmax=vmax,
            cbar_kws={"label": title}, annot=False
        )
        ax.set_title(exp_config["label"])
        ax.set_xlabel("Component")
        ax.set_ylabel("Layer")

    plt.suptitle(f"{title} by Layer and Component", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_optimizer_diff_heatmap(df: pd.DataFrame, metric: str, title: str, output_path: Path):
    """Plot heatmap showing Muon - AdamW difference."""
    # Filter to transformer layers and known components
    layer_df = df[(df["layer"] >= 0) & (df["layer"] < 28)].copy()
    layer_df = layer_df[layer_df["component"].isin(COMPONENT_ORDER)]

    fig, axes = plt.subplots(1, 2, figsize=(14, 8))

    for idx, (obj_suffix, obj_label) in enumerate([("sft", "SFT"), ("grpo", "GRPO")]):
        ax = axes[idx]

        adamw_df = layer_df[layer_df["experiment"] == f"adamw_{obj_suffix}"]
        muon_df = layer_df[layer_df["experiment"] == f"muon_{obj_suffix}"]

        # Merge and compute difference
        adamw_pivot = adamw_df.pivot_table(index="layer", columns="component", values=metric)
        muon_pivot = muon_df.pivot_table(index="layer", columns="component", values=metric)

        diff_pivot = muon_pivot - adamw_pivot
        diff_pivot = diff_pivot.reindex(columns=[c for c in COMPONENT_ORDER if c in diff_pivot.columns])

        vmax_abs = max(abs(diff_pivot.min().min()), abs(diff_pivot.max().max()))

        sns.heatmap(
            diff_pivot, ax=ax, cmap="PiYG", center=0, vmin=-vmax_abs, vmax=vmax_abs,
            cbar_kws={"label": f"Muon - AdamW ({title})"}
        )
        ax.set_title(f"{obj_label}: Muon - AdamW")
        ax.set_xlabel("Component")
        ax.set_ylabel("Layer")

    plt.suptitle(f"Optimizer Difference: {title}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# VISUALIZATION: SCATTER PLOTS
# =============================================================================

def plot_optimizer_scatter(df: pd.DataFrame, metric: str, title: str, output_path: Path):
    """Scatter plot: AdamW metric vs Muon metric, colored by training objective."""
    # Filter to transformer layers
    layer_df = df[(df["layer"] >= 0) & (df["layer"] < 28)].copy()

    fig, ax = plt.subplots(figsize=(10, 10))

    for obj_suffix, obj_label, marker, color in [
        ("sft", "SFT", "o", "#1f77b4"),
        ("grpo", "GRPO", "^", "#d62728"),
    ]:
        adamw_df = layer_df[layer_df["experiment"] == f"adamw_{obj_suffix}"]
        muon_df = layer_df[layer_df["experiment"] == f"muon_{obj_suffix}"]

        # Match by param_name
        merged = adamw_df.merge(
            muon_df[["param_name", metric]],
            on="param_name",
            suffixes=("_adamw", "_muon")
        )

        ax.scatter(
            merged[f"{metric}_adamw"],
            merged[f"{metric}_muon"],
            label=obj_label, marker=marker, alpha=0.6, s=50, c=color
        )

    # Diagonal line (equal change)
    all_vals = layer_df[metric]
    lim_min, lim_max = all_vals.min(), all_vals.max()
    margin = (lim_max - lim_min) * 0.1
    ax.plot([lim_min - margin, lim_max + margin], [lim_min - margin, lim_max + margin],
            'k--', alpha=0.5, label="Equal change")

    ax.set_xlabel(f"AdamW: {title}")
    ax.set_ylabel(f"Muon: {title}")
    ax.set_title(f"Optimizer Comparison: {title}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# VISUALIZATION: TOP-K SPECTRUM PLOTS
# =============================================================================

def plot_top_k_spectrum(svd_data: dict, component: str, top_k: int, output_path: Path):
    """Plot relative change in top-k SVs for a component type, grouped by layer position."""
    baseline_params = svd_data["baseline"]["parameters"]

    # Group layers: early (0-9), mid (10-18), late (19-27)
    layer_groups = {
        "Early (0-9)": list(range(0, 10)),
        "Mid (10-18)": list(range(10, 19)),
        "Late (19-27)": list(range(19, 28)),
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()

    for exp_idx, (exp_name, exp_config) in enumerate(EXPERIMENTS.items()):
        ax = axes[exp_idx]
        exp_params = svd_data[exp_name]["parameters"]

        for group_name, layer_indices in layer_groups.items():
            # Collect all relative changes for this group
            all_rel_changes = []

            for layer_idx in layer_indices:
                param_name = f"model.layers.{layer_idx}.{'mlp' if component in ['gate_proj', 'up_proj', 'down_proj'] else 'self_attn'}.{component}.weight"
                if param_name not in baseline_params or param_name not in exp_params:
                    continue

                base_sv = np.array(baseline_params[param_name][:top_k])
                exp_sv = np.array(exp_params[param_name][:top_k])

                rel_change = (exp_sv - base_sv) / base_sv * 100  # Percentage
                all_rel_changes.append(rel_change)

            if not all_rel_changes:
                continue

            # Average across layers in group
            mean_rel_change = np.mean(all_rel_changes, axis=0)
            std_rel_change = np.std(all_rel_changes, axis=0)

            ranks = np.arange(1, top_k + 1)
            ax.plot(ranks, mean_rel_change, label=group_name, linewidth=1.5)
            ax.fill_between(ranks, mean_rel_change - std_rel_change, mean_rel_change + std_rel_change, alpha=0.2)

        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel("Singular Value Rank")
        ax.set_ylabel("Relative Change (%)")
        ax.set_title(exp_config["label"])
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.suptitle(
        f"Top-{top_k} SV Relative Change: {component}\n"
        r"$(\sigma_i(W_{exp}) - \sigma_i(W_{base})) / \sigma_i(W_{base}) \times 100\%$",
        fontsize=12, fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_cumulative_energy(svd_data: dict, component: str, output_path: Path):
    """Plot cumulative energy curves for a component type."""
    baseline_params = svd_data["baseline"]["parameters"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()

    # Sample layers: early, mid, late
    sample_layers = [2, 13, 25]
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(sample_layers)))

    for exp_idx, (exp_name, exp_config) in enumerate(EXPERIMENTS.items()):
        ax = axes[exp_idx]
        exp_params = svd_data[exp_name]["parameters"]

        # Plot baseline first
        for layer_idx, color in zip(sample_layers, colors):
            param_name = f"model.layers.{layer_idx}.{'mlp' if component in ['gate_proj', 'up_proj', 'down_proj'] else 'self_attn'}.{component}.weight"
            if param_name not in baseline_params:
                continue

            base_sv = np.array(baseline_params[param_name])
            exp_sv = np.array(exp_params.get(param_name, base_sv))

            # Cumulative energy
            base_energy = np.cumsum(base_sv ** 2) / np.sum(base_sv ** 2)
            exp_energy = np.cumsum(exp_sv ** 2) / np.sum(exp_sv ** 2)

            ranks = np.arange(1, len(base_sv) + 1)
            ax.plot(ranks, base_energy, '--', color=color, alpha=0.5, label=f"Layer {layer_idx} (base)")
            ax.plot(ranks, exp_energy, '-', color=color, linewidth=1.5, label=f"Layer {layer_idx}")

        ax.set_xlabel("Singular Value Rank")
        ax.set_ylabel("Cumulative Energy Fraction")
        ax.set_title(exp_config["label"])
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 200)  # Focus on first 200 SVs

    plt.suptitle(
        f"Cumulative Energy Fraction: {component}\n"
        r"$\sum_{j=1}^{k} \sigma_j^2 \;/\; \sum_{j} \sigma_j^2$",
        fontsize=12, fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Spectral Summary Analysis")
    parser.add_argument("--cache-dir", type=str, default="./svd_cache",
                       help="Directory with cached SVD JSON files")
    parser.add_argument("--output-dir", "-o", type=str, default="./spectral_summary",
                       help="Output directory for plots")

    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading cached SVD data...")
    svd_data = load_cached_svd(cache_dir)

    print("Computing metrics...")
    df = compute_all_metrics(svd_data)

    # Save metrics CSV
    metrics_path = output_dir / "metrics.csv"
    df.to_csv(metrics_path, index=False)
    print(f"Saved metrics to: {metrics_path}")

    print("\nGenerating visualizations...")

    # Mathematical notation for titles:
    # - σᵢ(W) = i-th singular value of weight matrix W
    # - σ(W) = vector of all singular values
    # - W_exp = experiment weights, W_base = baseline weights

    # 1. Bar charts by layer
    print("  - Spectral norm by layer...")
    plot_metric_by_layer(
        df, "spectral_norm_change",
        r"Spectral Norm Change: $\sigma_1(W_{exp}) - \sigma_1(W_{base})$",
        output_dir / "spectral_norm_by_layer.png"
    )
    print("  - Frobenius change by layer...")
    plot_metric_by_layer(
        df, "frobenius_change",
        r"SV Difference Norm: $\|\sigma(W_{exp}) - \sigma(W_{base})\|_2$",
        output_dir / "frobenius_by_layer.png"
    )
    print("  - Effective rank change by layer...")
    plot_metric_by_layer(
        df, "effective_rank_change",
        r"Effective Rank Change: $\Delta(\sum_i \sigma_i / \sigma_1)$",
        output_dir / "effective_rank_by_layer.png"
    )

    # 2. Bar charts by component
    print("  - Metrics by component...")
    plot_metric_by_component(
        df, "frobenius_change",
        r"SV Difference Norm: $\|\sigma(W_{exp}) - \sigma(W_{base})\|_2$",
        output_dir / "frobenius_by_component.png"
    )

    # 3. Heatmaps
    print("  - Component heatmaps...")
    plot_component_heatmap(
        df, "frobenius_change",
        r"$\|\sigma(W_{exp}) - \sigma(W_{base})\|_2$",
        output_dir / "heatmap_frobenius.png"
    )
    plot_component_heatmap(
        df, "spectral_norm_rel_change",
        r"$(\sigma_1(W_{exp}) - \sigma_1(W_{base})) / \sigma_1(W_{base})$",
        output_dir / "heatmap_spectral_rel.png"
    )

    # 4. Optimizer difference heatmaps
    print("  - Optimizer difference heatmaps...")
    plot_optimizer_diff_heatmap(
        df, "frobenius_change",
        r"SV Difference Norm: $\|\sigma(W_{exp}) - \sigma(W_{base})\|_2$",
        output_dir / "heatmap_optimizer_diff.png"
    )

    # 5. Scatter plots
    print("  - Optimizer scatter plots...")
    plot_optimizer_scatter(
        df, "frobenius_change",
        r"SV Difference Norm: $\|\sigma(W_{exp}) - \sigma(W_{base})\|_2$",
        output_dir / "scatter_optimizer.png"
    )
    plot_optimizer_scatter(
        df, "spectral_norm_change",
        r"Spectral Norm Change: $\sigma_1(W_{exp}) - \sigma_1(W_{base})$",
        output_dir / "scatter_spectral_norm.png"
    )

    # 6. Top-k spectrum plots
    print("  - Top-k spectrum plots...")
    for component in ["down_proj", "up_proj", "gate_proj", "q_proj", "v_proj"]:
        plot_top_k_spectrum(svd_data, component, top_k=30, output_path=output_dir / f"top_svs_{component}.png")

    # 7. Cumulative energy curves
    print("  - Cumulative energy curves...")
    for component in ["down_proj", "up_proj"]:
        plot_cumulative_energy(svd_data, component, output_path=output_dir / f"cumulative_energy_{component}.png")

    # Save summary
    summary = {
        "n_params": len(df["param_name"].unique()),
        "n_layers": len(df[df["layer"] >= 0]["layer"].unique()),
        "experiments": list(EXPERIMENTS.keys()),
        "metrics_computed": [
            "spectral_norm_change", "frobenius_change", "effective_rank_change",
            "nuclear_norm_change", "top10_energy_change", "top30_rel_change"
        ],
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print("Analysis complete!")
    print(f"  Output directory: {output_dir}")
    print(f"  Metrics CSV: {metrics_path}")
    print(f"  Visualizations: {len(list(output_dir.glob('*.png')))} plots")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
