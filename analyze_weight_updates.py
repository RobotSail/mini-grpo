#!/usr/bin/env python3
"""
Analyze Weight Updates: SVD analysis of ΔW = W_exp - W_base

Computes the SVD of weight differences to understand the spectral structure
of updates produced by different optimizers and training objectives.

Key metrics:
- Frobenius norm: ||ΔW||_F - total magnitude of update
- Spectral norm: σ₁(ΔW) - maximum singular value
- Effective rank: (Σσᵢ)² / Σσᵢ² - how "spread out" the update is
- Stable rank: ||ΔW||_F² / σ₁² - another dimensionality measure

Usage:
    python analyze_weight_updates.py --gpu 0 --output-dir ./weight_updates
"""

import argparse
import json
import re
from pathlib import Path
from typing import Iterator

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from safetensors import safe_open
from tqdm import tqdm
from transformers import AutoModelForCausalLM


# =============================================================================
# CONFIGURATION
# =============================================================================

BASELINE_MODEL = "Qwen/Qwen2-1.5B-Instruct"

EXPERIMENTS = {
    "adamw_sft": {
        "path": "/mnt/nvme3n1/checkpoints/qwen2-1.5b-instruct_gsm8k_sft_adamw/hf_format/samples_11194.0_step_350/",
        "label": "AdamW + SFT",
        "color": "#1f77b4",
    },
    "muon_sft": {
        "path": "/mnt/nvme3n1/checkpoints/qwen2-1.5b-instruct_gsm8k_sft_muon/hf_format/samples_12788.0_step_400/",
        "label": "Muon + SFT",
        "color": "#ff7f0e",
    },
    "adamw_grpo": {
        "path": "/mnt/nvme3n1/checkpoints/qwen2-1.5b-instruct_gsm8k_adamw/step_800/",
        "label": "AdamW + GRPO",
        "color": "#2ca02c",
    },
    "muon_grpo": {
        "path": "/mnt/nvme3n1/checkpoints/qwen2-1.5b-instruct_gsm8k_muon/step_800/",
        "label": "Muon + GRPO",
        "color": "#d62728",
    },
}

COMPONENT_ORDER = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


# =============================================================================
# WEIGHT LOADING
# =============================================================================

def load_weights(checkpoint_path: str, device: torch.device) -> dict[str, torch.Tensor]:
    """Load weight tensors from a checkpoint directory or HuggingFace model."""
    path = Path(checkpoint_path)
    is_hf_model = "/" in checkpoint_path and not path.exists()

    if is_hf_model:
        print(f"  Loading from HuggingFace: {checkpoint_path}")
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint_path,
            torch_dtype=torch.float32,
            device_map=device,
        )
        weights = {name: param.data.clone() for name, param in model.named_parameters()}
        del model
        torch.cuda.empty_cache()
        return weights

    safetensor_files = sorted(path.glob("*.safetensors"))
    if not safetensor_files:
        raise ValueError(f"No safetensors files found in {checkpoint_path}")

    device_str = str(device) if device.type == "cuda" else "cpu"
    weights = {}
    for sf_file in safetensor_files:
        with safe_open(sf_file, framework="pt", device=device_str) as f:
            for key in f.keys():
                weights[key] = f.get_tensor(key)

    return weights


def iter_2d_params(weights: dict[str, torch.Tensor]) -> Iterator[tuple[str, torch.Tensor]]:
    """Yield 2D weight matrices."""
    for name, tensor in sorted(weights.items()):
        if tensor.ndim == 2 and tensor.shape[0] > 1 and tensor.shape[1] > 1:
            yield name, tensor


def parse_param_name(name: str) -> dict | None:
    """Parse parameter name into layer index and component type."""
    match = re.match(r"model\.layers\.(\d+)\.(self_attn|mlp)\.(\w+)\.weight", name)
    if match:
        return {
            "layer": int(match.group(1)),
            "component": match.group(3),
            "type": "attn" if match.group(2) == "self_attn" else "mlp",
            "param_name": name,
        }
    if "embed_tokens" in name:
        return {"layer": -1, "component": "embed", "type": "embed", "param_name": name}
    if "lm_head" in name:
        return {"layer": 99, "component": "lm_head", "type": "head", "param_name": name}
    return None


# =============================================================================
# SVD COMPUTATION
# =============================================================================

def compute_update_svd(delta_w: torch.Tensor) -> list[float]:
    """Compute singular values of the weight update."""
    delta_w_f32 = delta_w.float()
    try:
        sv = torch.linalg.svdvals(delta_w_f32)
        return sv.cpu().tolist()
    except RuntimeError as e:
        print(f"  SVD failed: {e}")
        return []


def compute_update_metrics(sv: list[float]) -> dict:
    """Compute metrics from singular values of the update.

    Args:
        sv: Singular values of ΔW = W_exp - W_base

    Returns:
        Dictionary of metrics
    """
    if not sv:
        return {}

    sv_arr = np.array(sv)

    # Frobenius norm: ||ΔW||_F = sqrt(Σσᵢ²)
    frobenius_norm = np.sqrt(np.sum(sv_arr ** 2))

    # Spectral norm: ||ΔW||_2 = σ₁
    spectral_norm = sv_arr[0]

    # Nuclear norm: ||ΔW||_* = Σσᵢ
    nuclear_norm = np.sum(sv_arr)

    # Effective rank: (Σσᵢ)² / Σσᵢ²
    # Ranges from 1 (rank-1) to n (full-rank uniform)
    effective_rank = (nuclear_norm ** 2) / (frobenius_norm ** 2) if frobenius_norm > 0 else 0

    # Stable rank: ||ΔW||_F² / σ₁²
    stable_rank = (frobenius_norm ** 2) / (spectral_norm ** 2) if spectral_norm > 0 else 0

    # Top-k energy ratios
    total_energy = frobenius_norm ** 2
    top1_energy = sv_arr[0] ** 2 / total_energy if total_energy > 0 else 0
    top5_energy = np.sum(sv_arr[:5] ** 2) / total_energy if total_energy > 0 and len(sv_arr) >= 5 else 0
    top10_energy = np.sum(sv_arr[:10] ** 2) / total_energy if total_energy > 0 and len(sv_arr) >= 10 else 0
    top50_energy = np.sum(sv_arr[:50] ** 2) / total_energy if total_energy > 0 and len(sv_arr) >= 50 else 0

    return {
        "frobenius_norm": frobenius_norm,
        "spectral_norm": spectral_norm,
        "nuclear_norm": nuclear_norm,
        "effective_rank": effective_rank,
        "stable_rank": stable_rank,
        "top1_energy": top1_energy,
        "top5_energy": top5_energy,
        "top10_energy": top10_energy,
        "top50_energy": top50_energy,
        "num_sv": len(sv_arr),
    }


# =============================================================================
# CACHING
# =============================================================================

def get_cache_path(cache_dir: Path, exp_name: str) -> Path:
    """Get cache file path for an experiment."""
    return cache_dir / f"update_svd_{exp_name}.json"


def save_cache(cache_path: Path, data: dict):
    """Save SVD results to cache."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(data, f)


def load_cache(cache_path: Path) -> dict | None:
    """Load SVD results from cache if exists."""
    if cache_path.exists():
        with open(cache_path) as f:
            return json.load(f)
    return None


# =============================================================================
# MAIN COMPUTATION
# =============================================================================

def compute_all_updates(
    baseline_weights: dict[str, torch.Tensor],
    exp_name: str,
    exp_path: str,
    device: torch.device,
    cache_dir: Path | None,
) -> dict:
    """Compute SVD of weight updates for one experiment.

    Returns:
        {
            "experiment": exp_name,
            "parameters": {
                param_name: {
                    "singular_values": [...],
                    "metrics": {...},
                },
                ...
            }
        }
    """
    # Check cache
    if cache_dir:
        cache_path = get_cache_path(cache_dir, exp_name)
        cached = load_cache(cache_path)
        if cached:
            print(f"\nLoaded cached update SVDs: {exp_name}")
            return cached

    print(f"\nComputing update SVDs: {exp_name}")
    exp_weights = load_weights(exp_path, device)

    result = {
        "experiment": exp_name,
        "parameters": {},
    }

    # Get common 2D parameters
    base_params = dict(iter_2d_params(baseline_weights))
    exp_params = dict(iter_2d_params(exp_weights))
    common_params = set(base_params.keys()) & set(exp_params.keys())

    for param_name in tqdm(sorted(common_params), desc="  Computing SVD(ΔW)"):
        base_w = base_params[param_name]
        exp_w = exp_params[param_name]

        # Compute update
        delta_w = exp_w - base_w

        # Compute SVD
        sv = compute_update_svd(delta_w)
        if not sv:
            continue

        # Compute metrics
        metrics = compute_update_metrics(sv)

        result["parameters"][param_name] = {
            "singular_values": sv,
            "metrics": metrics,
        }

    # Save cache
    if cache_dir:
        save_cache(cache_path, result)
        print(f"  Cached to: {cache_path}")

    # Clean up
    del exp_weights
    torch.cuda.empty_cache()

    return result


def build_metrics_dataframe(all_results: dict[str, dict]) -> pd.DataFrame:
    """Build DataFrame with all metrics for visualization."""
    rows = []

    for exp_name, result in all_results.items():
        for param_name, data in result["parameters"].items():
            parsed = parse_param_name(param_name)
            if parsed is None:
                continue

            row = {
                "experiment": exp_name,
                "experiment_label": EXPERIMENTS[exp_name]["label"],
                **parsed,
                **data["metrics"],
            }
            rows.append(row)

    return pd.DataFrame(rows)


# =============================================================================
# VISUALIZATION: BAR CHARTS
# =============================================================================

def plot_metric_by_layer(df: pd.DataFrame, metric: str, title: str, ylabel: str, output_path: Path):
    """Plot a metric as grouped bar chart by layer."""
    layer_df = df[(df["layer"] >= 0) & (df["layer"] < 28)].copy()
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
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(x)
    ax.legend(loc="upper right")
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_metric_by_component(df: pd.DataFrame, metric: str, title: str, ylabel: str, output_path: Path):
    """Plot a metric as grouped bar chart by component type."""
    layer_df = df[(df["layer"] >= 0) & (df["layer"] < 28)].copy()
    layer_df = layer_df[layer_df["component"].isin(COMPONENT_ORDER)]
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
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(COMPONENT_ORDER, rotation=45, ha="right")
    ax.legend(loc="upper right")
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# VISUALIZATION: SPECTRAL DECAY
# =============================================================================

def plot_spectral_decay(all_results: dict, component: str, output_path: Path):
    """Plot singular value decay curves (log scale) for a component type."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()

    # Sample layers
    sample_layers = [2, 13, 25]
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(sample_layers)))

    for exp_idx, (exp_name, exp_config) in enumerate(EXPERIMENTS.items()):
        ax = axes[exp_idx]
        result = all_results[exp_name]

        for layer_idx, color in zip(sample_layers, colors):
            block = "mlp" if component in ["gate_proj", "up_proj", "down_proj"] else "self_attn"
            param_name = f"model.layers.{layer_idx}.{block}.{component}.weight"

            if param_name not in result["parameters"]:
                continue

            sv = np.array(result["parameters"][param_name]["singular_values"])
            ranks = np.arange(1, len(sv) + 1)

            ax.semilogy(ranks, sv, color=color, linewidth=1.5, label=f"Layer {layer_idx}")

        ax.set_xlabel("Singular Value Rank")
        ax.set_ylabel(r"$\sigma_i(\Delta W)$ (log scale)")
        ax.set_title(exp_config["label"])
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.suptitle(
        f"Spectral Decay of Weight Updates: {component}\n"
        r"$\sigma_i(\Delta W)$ where $\Delta W = W_{exp} - W_{base}$",
        fontsize=12, fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_cumulative_energy(all_results: dict, component: str, output_path: Path):
    """Plot cumulative energy curves for a component type."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()

    sample_layers = [2, 13, 25]
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(sample_layers)))

    for exp_idx, (exp_name, exp_config) in enumerate(EXPERIMENTS.items()):
        ax = axes[exp_idx]
        result = all_results[exp_name]

        for layer_idx, color in zip(sample_layers, colors):
            block = "mlp" if component in ["gate_proj", "up_proj", "down_proj"] else "self_attn"
            param_name = f"model.layers.{layer_idx}.{block}.{component}.weight"

            if param_name not in result["parameters"]:
                continue

            sv = np.array(result["parameters"][param_name]["singular_values"])
            total_energy = np.sum(sv ** 2)
            cumulative_energy = np.cumsum(sv ** 2) / total_energy

            ranks = np.arange(1, len(sv) + 1)
            ax.plot(ranks, cumulative_energy, color=color, linewidth=1.5, label=f"Layer {layer_idx}")

        ax.axhline(y=0.9, color='gray', linestyle='--', alpha=0.5, label="90% energy")
        ax.axhline(y=0.99, color='gray', linestyle=':', alpha=0.5, label="99% energy")
        ax.set_xlabel("Singular Value Rank")
        ax.set_ylabel("Cumulative Energy Fraction")
        ax.set_title(exp_config["label"])
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 200)
        ax.set_ylim(0, 1.05)

    plt.suptitle(
        f"Cumulative Energy of Weight Updates: {component}\n"
        r"$\sum_{j=1}^{k} \sigma_j^2(\Delta W) \;/\; \|\Delta W\|_F^2$",
        fontsize=12, fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# VISUALIZATION: HEATMAPS
# =============================================================================

def plot_component_heatmap(df: pd.DataFrame, metric: str, title: str, output_path: Path):
    """Plot heatmaps of metric by layer x component."""
    layer_df = df[(df["layer"] >= 0) & (df["layer"] < 28)].copy()
    layer_df = layer_df[layer_df["component"].isin(COMPONENT_ORDER)]

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()

    vmin = layer_df[metric].min()
    vmax = layer_df[metric].max()

    for idx, (exp_name, exp_config) in enumerate(EXPERIMENTS.items()):
        ax = axes[idx]
        exp_df = layer_df[layer_df["experiment"] == exp_name]

        pivot = exp_df.pivot_table(index="layer", columns="component", values=metric, aggfunc="mean")
        pivot = pivot.reindex(columns=[c for c in COMPONENT_ORDER if c in pivot.columns])

        sns.heatmap(pivot, ax=ax, cmap="YlOrRd", vmin=vmin, vmax=vmax, cbar_kws={"label": metric})
        ax.set_title(exp_config["label"])
        ax.set_xlabel("Component")
        ax.set_ylabel("Layer")

    plt.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# VISUALIZATION: SCATTER
# =============================================================================

def plot_optimizer_scatter(df: pd.DataFrame, metric: str, title: str, output_path: Path):
    """Scatter plot: AdamW metric vs Muon metric."""
    layer_df = df[(df["layer"] >= 0) & (df["layer"] < 28)].copy()

    fig, ax = plt.subplots(figsize=(10, 10))

    for obj_suffix, obj_label, marker, color in [
        ("sft", "SFT", "o", "#1f77b4"),
        ("grpo", "GRPO", "^", "#d62728"),
    ]:
        adamw_df = layer_df[layer_df["experiment"] == f"adamw_{obj_suffix}"]
        muon_df = layer_df[layer_df["experiment"] == f"muon_{obj_suffix}"]

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

    all_vals = layer_df[metric]
    lim_min, lim_max = all_vals.min(), all_vals.max()
    margin = (lim_max - lim_min) * 0.1
    ax.plot([lim_min - margin, lim_max + margin], [lim_min - margin, lim_max + margin],
            'k--', alpha=0.5, label="Equal")

    ax.set_xlabel(f"AdamW: {metric}")
    ax.set_ylabel(f"Muon: {metric}")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Analyze weight update spectral structure")
    parser.add_argument("--baseline", type=str, default=BASELINE_MODEL)
    parser.add_argument("--output-dir", "-o", type=str, default="./weight_updates")
    parser.add_argument("--cache-dir", type=str, default="./update_svd_cache")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--no-cache", action="store_true")

    args = parser.parse_args()

    device = torch.device("cuda", args.gpu)
    print(f"Using device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cache_dir = None if args.no_cache else Path(args.cache_dir)

    # Load baseline weights
    print("\nLoading baseline weights...")
    baseline_weights = load_weights(args.baseline, device)

    # Compute SVD of updates for each experiment
    all_results = {}
    for exp_name, exp_config in EXPERIMENTS.items():
        result = compute_all_updates(
            baseline_weights, exp_name, exp_config["path"], device, cache_dir
        )
        all_results[exp_name] = result

    # Build metrics DataFrame
    print("\nBuilding metrics DataFrame...")
    df = build_metrics_dataframe(all_results)

    metrics_path = output_dir / "metrics.csv"
    df.to_csv(metrics_path, index=False)
    print(f"Saved metrics to: {metrics_path}")

    # Generate visualizations
    print("\nGenerating visualizations...")

    # Bar charts
    print("  - Frobenius norm by layer...")
    plot_metric_by_layer(
        df, "frobenius_norm",
        r"Mean Update Magnitude by Layer: $\langle\|\Delta W\|_F\rangle_{\mathrm{components}}$",
        r"Mean $\|\Delta W\|_F$",
        output_dir / "frobenius_by_layer.png"
    )

    print("  - Effective rank by layer...")
    plot_metric_by_layer(
        df, "effective_rank",
        r"Mean Effective Rank by Layer: $\langle(\sum_i \sigma_i)^2 / \sum_i \sigma_i^2\rangle_{\mathrm{components}}$",
        "Mean Effective Rank",
        output_dir / "effective_rank_by_layer.png"
    )

    print("  - Stable rank by layer...")
    plot_metric_by_layer(
        df, "stable_rank",
        r"Mean Stable Rank by Layer: $\langle\|\Delta W\|_F^2 / \sigma_1^2\rangle_{\mathrm{components}}$",
        "Mean Stable Rank",
        output_dir / "stable_rank_by_layer.png"
    )

    print("  - Metrics by component...")
    plot_metric_by_component(
        df, "frobenius_norm",
        r"Mean Update Magnitude by Component: $\langle\|\Delta W\|_F\rangle_{\mathrm{layers}}$",
        r"Mean $\|\Delta W\|_F$",
        output_dir / "frobenius_by_component.png"
    )

    plot_metric_by_component(
        df, "effective_rank",
        r"Mean Effective Rank by Component (averaged across layers)",
        "Mean Effective Rank",
        output_dir / "effective_rank_by_component.png"
    )

    # Spectral decay plots
    print("  - Spectral decay plots...")
    for component in ["down_proj", "up_proj", "q_proj"]:
        plot_spectral_decay(all_results, component, output_dir / f"spectral_decay_{component}.png")

    # Cumulative energy plots
    print("  - Cumulative energy plots...")
    for component in ["down_proj", "up_proj", "q_proj"]:
        plot_cumulative_energy(all_results, component, output_dir / f"cumulative_energy_{component}.png")

    # Heatmaps
    print("  - Component heatmaps...")
    plot_component_heatmap(
        df, "frobenius_norm",
        r"Update Magnitude: $\|\Delta W\|_F$ by Layer and Component",
        output_dir / "heatmap_frobenius.png"
    )
    plot_component_heatmap(
        df, "effective_rank",
        r"Effective Rank of $\Delta W$ by Layer and Component",
        output_dir / "heatmap_effective_rank.png"
    )

    # Scatter plots
    print("  - Optimizer scatter plots...")
    plot_optimizer_scatter(
        df, "frobenius_norm",
        r"Optimizer Comparison: $\|\Delta W\|_F$",
        output_dir / "scatter_frobenius.png"
    )
    plot_optimizer_scatter(
        df, "effective_rank",
        r"Optimizer Comparison: Effective Rank",
        output_dir / "scatter_effective_rank.png"
    )

    # Summary
    summary = {
        "baseline": args.baseline,
        "experiments": list(EXPERIMENTS.keys()),
        "n_params": len(df["param_name"].unique()),
        "metrics": [
            "frobenius_norm", "spectral_norm", "nuclear_norm",
            "effective_rank", "stable_rank",
            "top1_energy", "top5_energy", "top10_energy", "top50_energy"
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
