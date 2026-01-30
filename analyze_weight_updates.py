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
        "path": "checkpoints/sft_adamw/hf_format/samples_11610.0_tokens_1104739/",
        "label": "AdamW + SFT",
        "color": "#1f77b4",
    },
    "muon_sft": {
        "path": "checkpoints/sft_muon/hf_format/samples_11610.0_tokens_1104739/",
        "label": "Muon + SFT",
        "color": "#ff7f0e",
    },
    "adamw_grpo": {
        "path": "checkpoints/grpo_adamw/tokens_1110186/",
        "label": "AdamW + GRPO",
        "color": "#2ca02c",
    },
    "muon_grpo": {
        "path": "checkpoints/grpo_muon/tokens_1107170/",
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

    # Spectral entropy: H = -1/log(n) * Σ (σ_i²/Σσ_j²) * log(σ_i²/Σσ_j²)
    # Normalized to [0, 1]: 0 = all energy in one SV, 1 = uniform distribution
    n = len(sv_arr)
    if total_energy > 0 and n > 1:
        p = (sv_arr ** 2) / total_energy
        p = p[p > 0]  # Avoid log(0)
        spectral_entropy = (-1.0 / np.log(n)) * np.sum(p * np.log(p))
    else:
        spectral_entropy = 0.0

    # Rank for X% energy: minimum k such that Σσᵢ²(1:k) / ||ΔW||_F² >= threshold
    # More interpretable measure of low-rank structure
    if total_energy > 0:
        cumulative_energy = np.cumsum(sv_arr ** 2) / total_energy
        rank_90 = int(np.searchsorted(cumulative_energy, 0.90) + 1)
        rank_95 = int(np.searchsorted(cumulative_energy, 0.95) + 1)
        rank_99 = int(np.searchsorted(cumulative_energy, 0.99) + 1)
    else:
        rank_90 = rank_95 = rank_99 = 0

    # Gini coefficient of singular values
    # 0 = perfectly uniform (all SVs equal), 1 = maximally concentrated (one SV has all energy)
    # Formula: G = (2 * Σᵢ i*σᵢ) / (n * Σᵢ σᵢ) - (n+1)/n
    # Using sorted SVs (already sorted in descending order, we need ascending for standard Gini)
    if n > 1 and nuclear_norm > 0:
        sv_sorted_asc = sv_arr[::-1]  # Ascending order
        indices = np.arange(1, n + 1)
        gini = (2 * np.sum(indices * sv_sorted_asc)) / (n * nuclear_norm) - (n + 1) / n
    else:
        gini = 0.0

    return {
        "frobenius_norm": frobenius_norm,
        "spectral_norm": spectral_norm,
        "nuclear_norm": nuclear_norm,
        "effective_rank": effective_rank,
        "stable_rank": stable_rank,
        "spectral_entropy": spectral_entropy,
        "gini_coefficient": gini,
        "rank_90": rank_90,
        "rank_95": rank_95,
        "rank_99": rank_99,
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
# WEIGHT SVD (not update SVD)
# =============================================================================

def compute_weight_svd(w: torch.Tensor) -> list[float]:
    """Compute singular values of a weight matrix."""
    w_f32 = w.float()
    try:
        sv = torch.linalg.svdvals(w_f32)
        return sv.cpu().tolist()
    except RuntimeError as e:
        print(f"  SVD failed: {e}")
        return []


def compute_all_weight_svds(
    weights: dict[str, torch.Tensor],
    name: str,
    cache_dir: Path | None,
) -> dict[str, list[float]]:
    """Compute SVD of all 2D weight matrices.

    Returns:
        {param_name: [singular_values], ...}
    """
    cache_path = cache_dir / f"weight_svd_{name}.json" if cache_dir else None

    if cache_path and cache_path.exists():
        print(f"  Loaded cached weight SVDs: {name}")
        with open(cache_path) as f:
            return json.load(f)

    print(f"  Computing weight SVDs: {name}")
    result = {}

    for param_name, w in tqdm(iter_2d_params(weights), desc=f"    SVD({name})"):
        sv = compute_weight_svd(w)
        if sv:
            result[param_name] = sv

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump(result, f)

    return result


# =============================================================================
# CONDITION NUMBER ANALYSIS
# =============================================================================

def compute_condition_number(sv: list[float], eps: float = 1e-10) -> float:
    """Compute condition number κ = σ₁/σₙ from singular values."""
    if not sv or len(sv) < 2:
        return 0.0
    sigma_max = sv[0]
    sigma_min = sv[-1]
    if sigma_min < eps:
        return float('inf')
    return sigma_max / sigma_min


def build_condition_number_dataframe(
    baseline_svd: dict[str, list[float]],
    all_exp_svds: dict[str, dict[str, list[float]]],
) -> pd.DataFrame:
    """Build DataFrame with condition numbers for baseline and all experiments."""
    rows = []

    # Add baseline
    for param_name, sv in baseline_svd.items():
        parsed = parse_param_name(param_name)
        if parsed is None:
            continue
        cond = compute_condition_number(sv)
        if cond == float('inf'):
            continue  # Skip degenerate matrices
        rows.append({
            "experiment": "baseline",
            "experiment_label": "Baseline",
            **parsed,
            "condition_number": cond,
            "log_condition_number": np.log10(cond) if cond > 0 else 0,
        })

    # Add experiments
    for exp_name, exp_svd in all_exp_svds.items():
        for param_name, sv in exp_svd.items():
            parsed = parse_param_name(param_name)
            if parsed is None:
                continue
            cond = compute_condition_number(sv)
            if cond == float('inf'):
                continue
            rows.append({
                "experiment": exp_name,
                "experiment_label": EXPERIMENTS[exp_name]["label"],
                **parsed,
                "condition_number": cond,
                "log_condition_number": np.log10(cond) if cond > 0 else 0,
            })

    return pd.DataFrame(rows)


def plot_condition_number_by_layer(df: pd.DataFrame, output_path: Path):
    """Plot condition number by layer for all experiments + baseline."""
    layer_df = df[(df["layer"] >= 0) & (df["layer"] < 28)].copy()
    agg_df = layer_df.groupby(["experiment", "layer"])["log_condition_number"].mean().reset_index()

    fig, ax = plt.subplots(figsize=(16, 6))

    # Include baseline
    all_experiments = ["baseline"] + list(EXPERIMENTS.keys())
    colors = {"baseline": "black"}
    colors.update({k: v["color"] for k, v in EXPERIMENTS.items()})
    labels = {"baseline": "Baseline"}
    labels.update({k: v["label"] for k, v in EXPERIMENTS.items()})

    n_exp = len(all_experiments)
    n_layers = 28
    width = 0.8 / n_exp
    x = np.arange(n_layers)

    for i, exp_name in enumerate(all_experiments):
        exp_data = agg_df[agg_df["experiment"] == exp_name].set_index("layer")["log_condition_number"]
        values = [exp_data.get(l, 0) for l in range(n_layers)]
        offset = (i - n_exp / 2 + 0.5) * width
        ax.bar(x + offset, values, width, label=labels[exp_name], color=colors[exp_name], alpha=0.8)

    ax.set_xlabel("Layer")
    ax.set_ylabel(r"$\log_{10}(\kappa)$")
    ax.set_title(r"Mean Condition Number by Layer: $\kappa(W) = \sigma_1 / \sigma_n$")
    ax.set_xticks(x[::2])
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_condition_number_by_component(df: pd.DataFrame, output_path: Path):
    """Plot condition number by component for all experiments + baseline."""
    layer_df = df[(df["layer"] >= 0) & (df["layer"] < 28)].copy()
    layer_df = layer_df[layer_df["component"].isin(COMPONENT_ORDER)]
    agg_df = layer_df.groupby(["experiment", "component"])["log_condition_number"].mean().reset_index()

    fig, ax = plt.subplots(figsize=(12, 6))

    all_experiments = ["baseline"] + list(EXPERIMENTS.keys())
    colors = {"baseline": "black"}
    colors.update({k: v["color"] for k, v in EXPERIMENTS.items()})
    labels = {"baseline": "Baseline"}
    labels.update({k: v["label"] for k, v in EXPERIMENTS.items()})

    n_exp = len(all_experiments)
    n_components = len(COMPONENT_ORDER)
    width = 0.8 / n_exp
    x = np.arange(n_components)

    for i, exp_name in enumerate(all_experiments):
        exp_data = agg_df[agg_df["experiment"] == exp_name].set_index("component")["log_condition_number"]
        values = [exp_data.get(c, 0) for c in COMPONENT_ORDER]
        offset = (i - n_exp / 2 + 0.5) * width
        ax.bar(x + offset, values, width, label=labels[exp_name], color=colors[exp_name], alpha=0.8)

    ax.set_xlabel("Component")
    ax.set_ylabel(r"$\log_{10}(\kappa)$")
    ax.set_title(r"Mean Condition Number by Component: $\kappa(W) = \sigma_1 / \sigma_n$")
    ax.set_xticks(x)
    ax.set_xticklabels(COMPONENT_ORDER, rotation=45, ha="right")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_condition_number_heatmap(df: pd.DataFrame, output_path: Path):
    """Plot heatmaps of condition number by layer x component for each experiment."""
    layer_df = df[(df["layer"] >= 0) & (df["layer"] < 28)].copy()
    layer_df = layer_df[layer_df["component"].isin(COMPONENT_ORDER)]

    all_experiments = ["baseline"] + list(EXPERIMENTS.keys())
    n_exp = len(all_experiments)

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    # Use same color scale across all
    vmin = layer_df["log_condition_number"].min()
    vmax = layer_df["log_condition_number"].max()

    labels = {"baseline": "Baseline"}
    labels.update({k: v["label"] for k, v in EXPERIMENTS.items()})

    for idx, exp_name in enumerate(all_experiments):
        if idx >= len(axes):
            break
        ax = axes[idx]
        exp_df = layer_df[layer_df["experiment"] == exp_name]

        pivot = exp_df.pivot_table(
            index="layer", columns="component", values="log_condition_number", aggfunc="mean"
        )
        pivot = pivot.reindex(columns=[c for c in COMPONENT_ORDER if c in pivot.columns])

        sns.heatmap(pivot, ax=ax, cmap="YlOrRd", vmin=vmin, vmax=vmax,
                    cbar_kws={"label": r"$\log_{10}(\kappa)$"})
        ax.set_title(labels[exp_name])
        ax.set_xlabel("Component")
        ax.set_ylabel("Layer")

    # Hide unused axes
    for idx in range(n_exp, len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle(r"Condition Number $\kappa(W) = \sigma_1 / \sigma_n$ by Layer and Component",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def build_condition_number_diff_dataframe(
    baseline_svd: dict[str, list[float]],
    all_exp_svds: dict[str, dict[str, list[float]]],
) -> pd.DataFrame:
    """Build DataFrame with condition number differences: κ(W_exp) - κ(W_base)."""
    rows = []

    for exp_name, exp_svd in all_exp_svds.items():
        for param_name, sv_exp in exp_svd.items():
            if param_name not in baseline_svd:
                continue
            parsed = parse_param_name(param_name)
            if parsed is None:
                continue

            sv_base = baseline_svd[param_name]
            cond_exp = compute_condition_number(sv_exp)
            cond_base = compute_condition_number(sv_base)

            if cond_exp == float('inf') or cond_base == float('inf'):
                continue

            cond_diff = cond_exp - cond_base
            # Also compute log ratio for large differences
            log_ratio = np.log10(cond_exp / cond_base) if cond_base > 0 else 0

            rows.append({
                "experiment": exp_name,
                "experiment_label": EXPERIMENTS[exp_name]["label"],
                **parsed,
                "condition_number_base": cond_base,
                "condition_number_exp": cond_exp,
                "condition_number_diff": cond_diff,
                "condition_number_log_ratio": log_ratio,
            })

    return pd.DataFrame(rows)


def plot_condition_number_diff_by_layer(df: pd.DataFrame, output_path: Path):
    """Plot condition number difference by layer: κ(W_exp) - κ(W_base)."""
    layer_df = df[(df["layer"] >= 0) & (df["layer"] < 28)].copy()
    agg_df = layer_df.groupby(["experiment", "layer"])["condition_number_log_ratio"].mean().reset_index()

    fig, ax = plt.subplots(figsize=(16, 6))

    n_exp = len(EXPERIMENTS)
    n_layers = 28
    width = 0.8 / n_exp
    x = np.arange(n_layers)

    for i, (exp_name, exp_config) in enumerate(EXPERIMENTS.items()):
        exp_data = agg_df[agg_df["experiment"] == exp_name].set_index("layer")["condition_number_log_ratio"]
        values = [exp_data.get(l, 0) for l in range(n_layers)]
        offset = (i - n_exp / 2 + 0.5) * width
        ax.bar(x + offset, values, width, label=exp_config["label"], color=exp_config["color"], alpha=0.8)

    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_xlabel("Layer")
    ax.set_ylabel(r"$\log_{10}(\kappa_{exp} / \kappa_{base})$")
    ax.set_title(r"Condition Number Change vs Baseline: $\log_{10}(\kappa_{exp} / \kappa_{base})$")
    ax.set_xticks(x[::2])
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_condition_number_diff_by_component(df: pd.DataFrame, output_path: Path):
    """Plot condition number difference by component."""
    layer_df = df[(df["layer"] >= 0) & (df["layer"] < 28)].copy()
    layer_df = layer_df[layer_df["component"].isin(COMPONENT_ORDER)]
    agg_df = layer_df.groupby(["experiment", "component"])["condition_number_log_ratio"].mean().reset_index()

    fig, ax = plt.subplots(figsize=(12, 6))

    n_exp = len(EXPERIMENTS)
    n_components = len(COMPONENT_ORDER)
    width = 0.8 / n_exp
    x = np.arange(n_components)

    for i, (exp_name, exp_config) in enumerate(EXPERIMENTS.items()):
        exp_data = agg_df[agg_df["experiment"] == exp_name].set_index("component")["condition_number_log_ratio"]
        values = [exp_data.get(c, 0) for c in COMPONENT_ORDER]
        offset = (i - n_exp / 2 + 0.5) * width
        ax.bar(x + offset, values, width, label=exp_config["label"], color=exp_config["color"], alpha=0.8)

    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_xlabel("Component")
    ax.set_ylabel(r"$\log_{10}(\kappa_{exp} / \kappa_{base})$")
    ax.set_title(r"Condition Number Change vs Baseline: $\log_{10}(\kappa_{exp} / \kappa_{base})$")
    ax.set_xticks(x)
    ax.set_xticklabels(COMPONENT_ORDER, rotation=45, ha="right")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_condition_number_diff_heatmap(df: pd.DataFrame, output_path: Path):
    """Plot heatmaps of condition number change by layer x component."""
    layer_df = df[(df["layer"] >= 0) & (df["layer"] < 28)].copy()
    layer_df = layer_df[layer_df["component"].isin(COMPONENT_ORDER)]

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()

    # Symmetric color scale around 0
    vmax = layer_df["condition_number_log_ratio"].abs().max()
    vmin = -vmax

    for idx, (exp_name, exp_config) in enumerate(EXPERIMENTS.items()):
        ax = axes[idx]
        exp_df = layer_df[layer_df["experiment"] == exp_name]

        pivot = exp_df.pivot_table(
            index="layer", columns="component", values="condition_number_log_ratio", aggfunc="mean"
        )
        pivot = pivot.reindex(columns=[c for c in COMPONENT_ORDER if c in pivot.columns])

        sns.heatmap(pivot, ax=ax, cmap="RdBu_r", vmin=vmin, vmax=vmax, center=0,
                    cbar_kws={"label": r"$\log_{10}(\kappa_{exp} / \kappa_{base})$"})
        ax.set_title(exp_config["label"])
        ax.set_xlabel("Component")
        ax.set_ylabel("Layer")

    plt.suptitle(r"Condition Number Change: $\log_{10}(\kappa_{exp} / \kappa_{base})$ by Layer and Component",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# VISUALIZATION: SPECTRAL COMPARISON (σ(W_exp) vs σ(W_base))
# =============================================================================

def plot_spectral_comparison(
    baseline_svd: dict[str, list[float]],
    exp_svd: dict[str, list[float]],
    exp_name: str,
    exp_label: str,
    component: str,
    output_path: Path,
):
    """Plot σ(W_exp) vs σ(W_base) for a given component type."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    sample_layers = [0, 5, 10, 15, 20, 27]

    for idx, layer_idx in enumerate(sample_layers):
        ax = axes[idx // 3, idx % 3]
        block = "mlp" if component in ["gate_proj", "up_proj", "down_proj"] else "self_attn"
        param_name = f"model.layers.{layer_idx}.{block}.{component}.weight"

        if param_name not in baseline_svd or param_name not in exp_svd:
            ax.text(0.5, 0.5, "No data", ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f"Layer {layer_idx}")
            continue

        sv_base = np.array(baseline_svd[param_name])
        sv_exp = np.array(exp_svd[param_name])
        ranks = np.arange(1, len(sv_base) + 1)

        ax.semilogy(ranks, sv_base, 'b-', linewidth=1.5, label='Baseline', alpha=0.8)
        ax.semilogy(ranks, sv_exp, 'r-', linewidth=1.5, label=exp_label, alpha=0.8)

        ax.set_xlabel("Singular Value Rank")
        ax.set_ylabel(r"$\sigma_i$ (log scale)")
        ax.set_title(f"Layer {layer_idx}")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle(
        f"Spectral Comparison: {component}\n"
        r"$\sigma_i(W_{base})$ vs $\sigma_i(W_{exp})$",
        fontsize=12, fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_spectral_difference(
    baseline_svd: dict[str, list[float]],
    exp_svd: dict[str, list[float]],
    exp_name: str,
    exp_label: str,
    component: str,
    output_path: Path,
):
    """Plot σ(W_exp) - σ(W_base) for a given component type."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    sample_layers = [0, 5, 10, 15, 20, 27]

    for idx, layer_idx in enumerate(sample_layers):
        ax = axes[idx // 3, idx % 3]
        block = "mlp" if component in ["gate_proj", "up_proj", "down_proj"] else "self_attn"
        param_name = f"model.layers.{layer_idx}.{block}.{component}.weight"

        if param_name not in baseline_svd or param_name not in exp_svd:
            ax.text(0.5, 0.5, "No data", ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f"Layer {layer_idx}")
            continue

        sv_base = np.array(baseline_svd[param_name])
        sv_exp = np.array(exp_svd[param_name])
        sv_diff = sv_exp - sv_base
        ranks = np.arange(1, len(sv_base) + 1)

        ax.plot(ranks, sv_diff, 'k-', linewidth=1, alpha=0.8)
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.fill_between(ranks, 0, sv_diff, where=(sv_diff > 0), alpha=0.3, color='green', label='+')
        ax.fill_between(ranks, 0, sv_diff, where=(sv_diff < 0), alpha=0.3, color='red', label='-')

        ax.set_xlabel("Singular Value Rank")
        ax.set_ylabel(r"$\sigma_i(W_{exp}) - \sigma_i(W_{base})$")
        ax.set_title(f"Layer {layer_idx}")
        ax.grid(True, alpha=0.3)

    plt.suptitle(
        f"Spectral Difference: {component} ({exp_label})\n"
        r"$\sigma_i(W_{exp}) - \sigma_i(W_{base})$",
        fontsize=12, fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_spectral_comparison_all(
    baseline_svd: dict[str, list[float]],
    all_exp_svds: dict[str, dict[str, list[float]]],
    component: str,
    output_path: Path,
):
    """Plot σ(W_base) and σ(W_exp) for all experiments on same axes."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    sample_layers = [0, 5, 10, 15, 20, 27]

    for idx, layer_idx in enumerate(sample_layers):
        ax = axes[idx // 3, idx % 3]
        block = "mlp" if component in ["gate_proj", "up_proj", "down_proj"] else "self_attn"
        param_name = f"model.layers.{layer_idx}.{block}.{component}.weight"

        if param_name not in baseline_svd:
            ax.text(0.5, 0.5, "No data", ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f"Layer {layer_idx}")
            continue

        sv_base = np.array(baseline_svd[param_name])
        ranks = np.arange(1, len(sv_base) + 1)

        # Plot baseline
        ax.semilogy(ranks, sv_base, 'k-', linewidth=2, label='Baseline', alpha=0.9)

        # Plot all experiments
        for exp_name, exp_config in EXPERIMENTS.items():
            if exp_name not in all_exp_svds or param_name not in all_exp_svds[exp_name]:
                continue
            sv_exp = np.array(all_exp_svds[exp_name][param_name])
            ax.semilogy(ranks, sv_exp, color=exp_config["color"], linewidth=1.2,
                       label=exp_config["label"], alpha=0.7)

        ax.set_xlabel("Singular Value Rank")
        ax.set_ylabel(r"$\sigma_i$ (log scale)")
        ax.set_title(f"Layer {layer_idx}")
        if idx == 0:
            ax.legend(loc="upper right", fontsize=7)
        ax.grid(True, alpha=0.3)

    plt.suptitle(
        f"Spectral Comparison (All Experiments): {component}\n"
        r"$\sigma_i(W)$ for baseline and all trained models",
        fontsize=12, fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_spectral_diff_summary(
    baseline_svd: dict[str, list[float]],
    all_exp_svds: dict[str, dict[str, list[float]]],
    component: str,
    output_path: Path,
):
    """Plot σ(W_exp) - σ(W_base) for all experiments on same axes."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    sample_layers = [0, 5, 10, 15, 20, 27]

    for idx, layer_idx in enumerate(sample_layers):
        ax = axes[idx // 3, idx % 3]
        block = "mlp" if component in ["gate_proj", "up_proj", "down_proj"] else "self_attn"
        param_name = f"model.layers.{layer_idx}.{block}.{component}.weight"

        if param_name not in baseline_svd:
            ax.text(0.5, 0.5, "No data", ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f"Layer {layer_idx}")
            continue

        sv_base = np.array(baseline_svd[param_name])
        ranks = np.arange(1, len(sv_base) + 1)

        for exp_name, exp_config in EXPERIMENTS.items():
            if exp_name not in all_exp_svds or param_name not in all_exp_svds[exp_name]:
                continue
            sv_exp = np.array(all_exp_svds[exp_name][param_name])
            sv_diff = sv_exp - sv_base
            ax.plot(ranks, sv_diff, color=exp_config["color"], linewidth=1.2,
                   label=exp_config["label"], alpha=0.8)

        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel("Singular Value Rank")
        ax.set_ylabel(r"$\sigma_i(W_{exp}) - \sigma_i(W_{base})$")
        ax.set_title(f"Layer {layer_idx}")
        if idx == 0:
            ax.legend(loc="upper right", fontsize=7)
        ax.grid(True, alpha=0.3)

    plt.suptitle(
        f"Spectral Difference by Experiment: {component}\n"
        r"$\sigma_i(W_{exp}) - \sigma_i(W_{base})$",
        fontsize=12, fontweight="bold"
    )
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

    # Compute SVD of baseline weights
    print("\nComputing baseline weight SVDs...")
    baseline_svd = compute_all_weight_svds(baseline_weights, "baseline", cache_dir)

    # Compute SVD of updates for each experiment
    all_results = {}
    all_exp_svds = {}
    for exp_name, exp_config in EXPERIMENTS.items():
        result = compute_all_updates(
            baseline_weights, exp_name, exp_config["path"], device, cache_dir
        )
        all_results[exp_name] = result

        # Also compute weight SVDs for spectral comparison
        exp_weights = load_weights(exp_config["path"], device)
        all_exp_svds[exp_name] = compute_all_weight_svds(exp_weights, exp_name, cache_dir)
        del exp_weights
        torch.cuda.empty_cache()

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

    print("  - Spectral entropy by layer...")
    plot_metric_by_layer(
        df, "spectral_entropy",
        r"Mean Spectral Entropy by Layer: $\langle H \rangle_{\mathrm{components}}$"
        + "\n" + r"$H = \frac{-1}{\log n} \sum_i \frac{\sigma_i^2}{\sum_j \sigma_j^2} \log\frac{\sigma_i^2}{\sum_j \sigma_j^2}$",
        "Mean Spectral Entropy",
        output_dir / "spectral_entropy_by_layer.png"
    )

    print("  - Gini coefficient by layer...")
    plot_metric_by_layer(
        df, "gini_coefficient",
        r"Mean Gini Coefficient by Layer: $\langle G \rangle_{\mathrm{components}}$"
        + "\n" + r"$G \in [0,1]$: 0 = uniform SVs, 1 = all energy in one SV",
        "Mean Gini Coefficient",
        output_dir / "gini_by_layer.png"
    )

    print("  - Rank for 90%/95%/99% energy by layer...")
    plot_metric_by_layer(
        df, "rank_90",
        r"Mean Rank for 90% Energy by Layer: min $k$ s.t. $\sum_{i=1}^{k} \sigma_i^2 \geq 0.9 \|\Delta W\|_F^2$",
        "Mean Rank (90% energy)",
        output_dir / "rank90_by_layer.png"
    )
    plot_metric_by_layer(
        df, "rank_95",
        r"Mean Rank for 95% Energy by Layer",
        "Mean Rank (95% energy)",
        output_dir / "rank95_by_layer.png"
    )
    plot_metric_by_layer(
        df, "rank_99",
        r"Mean Rank for 99% Energy by Layer",
        "Mean Rank (99% energy)",
        output_dir / "rank99_by_layer.png"
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

    plot_metric_by_component(
        df, "spectral_entropy",
        r"Mean Spectral Entropy by Component: $\langle H \rangle_{\mathrm{layers}}$",
        "Mean Spectral Entropy",
        output_dir / "spectral_entropy_by_component.png"
    )

    plot_metric_by_component(
        df, "stable_rank",
        r"Mean Stable Rank by Component: $\langle\|\Delta W\|_F^2 / \sigma_1^2\rangle_{\mathrm{layers}}$",
        "Mean Stable Rank",
        output_dir / "stable_rank_by_component.png"
    )

    plot_metric_by_component(
        df, "gini_coefficient",
        r"Mean Gini Coefficient by Component: $\langle G \rangle_{\mathrm{layers}}$",
        "Mean Gini Coefficient",
        output_dir / "gini_by_component.png"
    )

    plot_metric_by_component(
        df, "rank_90",
        r"Mean Rank for 90% Energy by Component",
        "Mean Rank (90% energy)",
        output_dir / "rank90_by_component.png"
    )

    plot_metric_by_component(
        df, "rank_95",
        r"Mean Rank for 95% Energy by Component",
        "Mean Rank (95% energy)",
        output_dir / "rank95_by_component.png"
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
    plot_component_heatmap(
        df, "stable_rank",
        r"Stable Rank of $\Delta W$: $\|\Delta W\|_F^2 / \sigma_1^2$ by Layer and Component",
        output_dir / "heatmap_stable_rank.png"
    )
    plot_component_heatmap(
        df, "gini_coefficient",
        r"Gini Coefficient of $\Delta W$ by Layer and Component",
        output_dir / "heatmap_gini.png"
    )
    plot_component_heatmap(
        df, "rank_90",
        r"Rank for 90% Energy of $\Delta W$ by Layer and Component",
        output_dir / "heatmap_rank90.png"
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

    # Spectral comparison plots: σ(W_exp) vs σ(W_base)
    print("  - Spectral comparison plots (σ(W_exp) vs σ(W_base))...")
    for component in ["down_proj", "up_proj", "q_proj"]:
        # Combined plot: baseline + all experiments on same axes
        plot_spectral_comparison_all(
            baseline_svd, all_exp_svds, component,
            output_dir / f"spectral_all_{component}.png"
        )
        # Difference plot with all experiments
        plot_spectral_diff_summary(
            baseline_svd, all_exp_svds, component,
            output_dir / f"spectral_diff_{component}.png"
        )
        # Individual experiment comparison plots
        for exp_name, exp_config in EXPERIMENTS.items():
            plot_spectral_comparison(
                baseline_svd, all_exp_svds[exp_name], exp_name, exp_config["label"],
                component, output_dir / f"spectral_cmp_{exp_name}_{component}.png"
            )

    # Condition number analysis
    print("  - Condition number analysis...")
    cond_df = build_condition_number_dataframe(baseline_svd, all_exp_svds)
    cond_metrics_path = output_dir / "condition_numbers.csv"
    cond_df.to_csv(cond_metrics_path, index=False)
    print(f"    Saved condition numbers to: {cond_metrics_path}")

    plot_condition_number_by_layer(cond_df, output_dir / "condition_number_by_layer.png")
    plot_condition_number_by_component(cond_df, output_dir / "condition_number_by_component.png")
    plot_condition_number_heatmap(cond_df, output_dir / "heatmap_condition_number.png")

    # Condition number difference analysis (κ_exp - κ_base)
    print("  - Condition number difference analysis...")
    cond_diff_df = build_condition_number_diff_dataframe(baseline_svd, all_exp_svds)
    cond_diff_path = output_dir / "condition_number_diff.csv"
    cond_diff_df.to_csv(cond_diff_path, index=False)
    print(f"    Saved condition number differences to: {cond_diff_path}")

    plot_condition_number_diff_by_layer(cond_diff_df, output_dir / "condition_number_diff_by_layer.png")
    plot_condition_number_diff_by_component(cond_diff_df, output_dir / "condition_number_diff_by_component.png")
    plot_condition_number_diff_heatmap(cond_diff_df, output_dir / "heatmap_condition_number_diff.png")

    # Summary
    summary = {
        "baseline": args.baseline,
        "experiments": list(EXPERIMENTS.keys()),
        "n_params": len(df["param_name"].unique()),
        "metrics": [
            "frobenius_norm", "spectral_norm", "nuclear_norm",
            "effective_rank", "stable_rank", "spectral_entropy",
            "gini_coefficient", "rank_90", "rank_95", "rank_99",
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
