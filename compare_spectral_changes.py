#!/usr/bin/env python3
"""
Spectral Analysis: Compare SVD changes across training experiments.

Compares 4 training experiments (AdamW/Muon x SFT/GRPO) against a baseline model,
identifying the parameters with highest variance and focusing on the most changed
singular values within those parameters.

Usage:
    python compare_spectral_changes.py --gpu 0 --output-dir ./spectral_analysis
"""

import argparse
import json
from pathlib import Path
from typing import Iterator

import torch
import matplotlib.pyplot as plt
import numpy as np
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
        "color": "#1f77b4",  # Blue
    },
    "muon_sft": {
        "path": "/mnt/nvme3n1/checkpoints/qwen2-1.5b-instruct_gsm8k_sft_muon/hf_format/samples_12788.0_step_400/",
        "label": "Muon + SFT",
        "color": "#ff7f0e",  # Orange
    },
    "adamw_grpo": {
        "path": "/mnt/nvme3n1/checkpoints/qwen2-1.5b-instruct_gsm8k_adamw/step_800/",
        "label": "AdamW + GRPO",
        "color": "#2ca02c",  # Green
    },
    "muon_grpo": {
        "path": "/mnt/nvme3n1/checkpoints/qwen2-1.5b-instruct_gsm8k_muon/step_800/",
        "label": "Muon + GRPO",
        "color": "#d62728",  # Red
    },
}


# =============================================================================
# SVD EXTRACTION
# =============================================================================

def load_state_dict(checkpoint_path: str, device: torch.device) -> dict[str, torch.Tensor]:
    """Load model state dict from a checkpoint directory or HuggingFace model."""
    path = Path(checkpoint_path)

    # Check if it's a HuggingFace model ID (contains "/" and doesn't exist locally)
    is_hf_model = "/" in checkpoint_path and not path.exists()

    if is_hf_model:
        print(f"  Loading from HuggingFace: {checkpoint_path}")
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint_path,
            torch_dtype=torch.float32,
            device_map=device,
        )
        state_dict = {name: param.data for name, param in model.named_parameters()}
        del model
        torch.cuda.empty_cache()
        return state_dict

    # Load from local safetensors files
    safetensor_files = sorted(path.glob("*.safetensors"))
    if not safetensor_files:
        raise ValueError(f"No safetensors files found in {checkpoint_path}")

    device_str = str(device) if device.type == "cuda" else "cpu"
    state_dict = {}
    for sf_file in safetensor_files:
        with safe_open(sf_file, framework="pt", device=device_str) as f:
            for key in f.keys():
                state_dict[key] = f.get_tensor(key)

    return state_dict


def iter_2d_params(state_dict: dict[str, torch.Tensor]) -> Iterator[tuple[str, torch.Tensor]]:
    """Yield 2D dense matrices from state dict."""
    for name, param in sorted(state_dict.items()):
        if param.ndim == 2 and param.shape[0] > 1 and param.shape[1] > 1:
            yield name, param


def compute_singular_values(matrix: torch.Tensor) -> list[float]:
    """Compute singular values of a matrix."""
    matrix_f32 = matrix.float()
    try:
        sv = torch.linalg.svdvals(matrix_f32)
        return sv.cpu().tolist()
    except RuntimeError as e:
        print(f"  SVD failed: {e}")
        return []


def extract_svd(checkpoint_path: str, device: torch.device, name: str) -> dict:
    """Extract SVD singular values from a checkpoint.

    Returns:
        {
            "name": str,
            "path": str,
            "parameters": {param_name: [sv1, sv2, ...], ...}
        }
    """
    print(f"\nExtracting SVD: {name}")
    state_dict = load_state_dict(checkpoint_path, device)
    params_2d = list(iter_2d_params(state_dict))
    print(f"  Found {len(params_2d)} 2D matrices")

    result = {
        "name": name,
        "path": checkpoint_path,
        "parameters": {}
    }

    for param_name, tensor in tqdm(params_2d, desc="  Computing SVD"):
        sv = compute_singular_values(tensor)
        if sv:
            result["parameters"][param_name] = sv

    return result


def get_cache_path(cache_dir: Path, name: str) -> Path:
    """Get cache file path for a given extraction name."""
    safe_name = name.replace("/", "_").replace(" ", "_")
    return cache_dir / f"svd_{safe_name}.json"


def load_or_extract_svd(checkpoint_path: str, device: torch.device,
                         name: str, cache_dir: Path | None) -> dict:
    """Load SVD from cache or extract it."""
    if cache_dir:
        cache_path = get_cache_path(cache_dir, name)
        if cache_path.exists():
            print(f"\nLoading cached SVD: {name}")
            with open(cache_path) as f:
                return json.load(f)

    result = extract_svd(checkpoint_path, device, name)

    if cache_dir:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = get_cache_path(cache_dir, name)
        with open(cache_path, "w") as f:
            json.dump(result, f)
        print(f"  Cached to: {cache_path}")

    return result


# =============================================================================
# VARIANCE ANALYSIS
# =============================================================================

def compute_parameter_variance(diffs: dict[str, dict[str, list[float]]]) -> dict[str, float]:
    """Compute variance score for each parameter across experiments.

    Args:
        diffs: {experiment_name: {param_name: [diff1, diff2, ...], ...}, ...}

    Returns:
        {param_name: variance_score, ...}
    """
    experiment_names = list(diffs.keys())

    # Get common parameters across all experiments
    common_params = set(diffs[experiment_names[0]].keys())
    for exp_name in experiment_names[1:]:
        common_params &= set(diffs[exp_name].keys())

    variances = {}
    for param_name in common_params:
        # Stack diffs: shape (num_experiments, num_svs)
        all_diffs = []
        for exp_name in experiment_names:
            all_diffs.append(np.array(diffs[exp_name][param_name]))

        # Ensure all have same length
        min_len = min(len(d) for d in all_diffs)
        stacked = np.stack([d[:min_len] for d in all_diffs])

        # Variance across experiments for each SV, then mean
        var_per_sv = np.var(stacked, axis=0)
        variances[param_name] = float(np.mean(var_per_sv))

    return variances


def select_top_k_params(variances: dict[str, float], k: int) -> list[str]:
    """Return top k parameter names by variance."""
    sorted_params = sorted(variances.items(), key=lambda x: x[1], reverse=True)
    return [name for name, _ in sorted_params[:k]]


# =============================================================================
# HIGH-VARIANCE SV IDENTIFICATION
# =============================================================================

def identify_high_variance_svs(diffs: dict[str, dict[str, list[float]]],
                                param_name: str,
                                percentile: float = 95) -> list[int]:
    """Identify SV ranks with variance in the top percentile.

    Returns:
        List of SV indices (0-indexed) with high variance.
    """
    experiment_names = list(diffs.keys())

    # Stack diffs for this parameter
    all_diffs = []
    for exp_name in experiment_names:
        all_diffs.append(np.array(diffs[exp_name][param_name]))

    min_len = min(len(d) for d in all_diffs)
    stacked = np.stack([d[:min_len] for d in all_diffs])

    # Variance across experiments for each SV
    var_per_sv = np.var(stacked, axis=0)

    # Find threshold for top percentile
    threshold = np.percentile(var_per_sv, percentile)

    # Get indices above threshold
    high_var_indices = np.where(var_per_sv >= threshold)[0].tolist()

    return high_var_indices


def find_neighborhoods(indices: list[int], max_gap: int = 5) -> list[tuple[int, int]]:
    """Group indices into contiguous neighborhoods.

    Indices within max_gap of each other are merged into one neighborhood.

    Returns:
        List of (start, end) tuples (inclusive bounds).
    """
    if not indices:
        return []

    sorted_indices = sorted(indices)
    neighborhoods = []

    start = sorted_indices[0]
    end = sorted_indices[0]

    for idx in sorted_indices[1:]:
        if idx - end <= max_gap:
            # Extend current neighborhood
            end = idx
        else:
            # Start new neighborhood
            neighborhoods.append((start, end))
            start = idx
            end = idx

    # Don't forget the last neighborhood
    neighborhoods.append((start, end))

    return neighborhoods


# =============================================================================
# VISUALIZATION
# =============================================================================

def sanitize_filename(name: str) -> str:
    """Convert parameter name to valid filename."""
    return name.replace("/", "_").replace(".", "_")


def plot_focused_comparison(param_name: str,
                            diffs: dict[str, dict[str, list[float]]],
                            neighborhoods: list[tuple[int, int]],
                            experiments: dict,
                            variance_score: float,
                            output_path: Path):
    """Create focused plot(s) for high-variance neighborhoods.

    Args:
        param_name: Name of the parameter
        diffs: {experiment_name: {param_name: [diff1, ...], ...}, ...}
        neighborhoods: List of (start, end) index tuples
        experiments: EXPERIMENTS config dict
        variance_score: Overall variance score for this parameter
        output_path: Path to save the plot
    """
    n_neighborhoods = len(neighborhoods)

    if n_neighborhoods == 0:
        return

    # Determine layout
    if n_neighborhoods == 1:
        fig, axes = plt.subplots(1, 1, figsize=(12, 6))
        axes = [axes]
    elif n_neighborhoods <= 3:
        fig, axes = plt.subplots(1, n_neighborhoods, figsize=(5 * n_neighborhoods, 6))
        axes = list(axes)
    else:
        # Grid layout for many neighborhoods
        n_cols = min(3, n_neighborhoods)
        n_rows = (n_neighborhoods + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
        axes = axes.flatten().tolist()

    for i, (start, end) in enumerate(neighborhoods):
        ax = axes[i]

        # Add padding around the neighborhood
        padding = max(2, (end - start) // 4)
        plot_start = max(0, start - padding)
        plot_end = end + padding

        for exp_name, exp_config in experiments.items():
            diff_values = np.array(diffs[exp_name][param_name])

            # Ensure we don't go out of bounds
            plot_end_actual = min(plot_end, len(diff_values) - 1)

            # Extract the neighborhood slice
            ranks = np.arange(plot_start + 1, plot_end_actual + 2)  # 1-indexed for display
            values = diff_values[plot_start:plot_end_actual + 1]

            ax.plot(ranks, values,
                   color=exp_config["color"],
                   linewidth=1.5,
                   label=exp_config["label"],
                   alpha=0.8)

        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel("Singular Value Rank")
        ax.set_ylabel("Difference from Baseline")

        if n_neighborhoods > 1:
            ax.set_title(f"Ranks {start + 1}-{end + 1}")

        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)

    # Hide unused axes
    for j in range(n_neighborhoods, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(f"{param_name}\n(variance score: {variance_score:.2e})",
                fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_full_spectrum(param_name: str,
                       diffs: dict[str, dict[str, list[float]]],
                       high_var_indices: list[int],
                       experiments: dict,
                       variance_score: float,
                       output_path: Path):
    """Create full spectrum plot with high-variance regions highlighted.

    Args:
        param_name: Name of the parameter
        diffs: {experiment_name: {param_name: [diff1, ...], ...}, ...}
        high_var_indices: Indices of high-variance SVs (for highlighting)
        experiments: EXPERIMENTS config dict
        variance_score: Overall variance score for this parameter
        output_path: Path to save the plot
    """
    fig, ax = plt.subplots(figsize=(14, 6))

    # Get max length
    max_len = max(len(diffs[exp_name][param_name]) for exp_name in experiments)

    # Highlight high-variance regions
    if high_var_indices:
        for idx in high_var_indices:
            ax.axvspan(idx + 0.5, idx + 1.5, alpha=0.1, color='yellow')

    for exp_name, exp_config in experiments.items():
        diff_values = np.array(diffs[exp_name][param_name])
        ranks = np.arange(1, len(diff_values) + 1)

        ax.plot(ranks, diff_values,
               color=exp_config["color"],
               linewidth=0.8,
               label=exp_config["label"],
               alpha=0.7)

    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel("Singular Value Rank")
    ax.set_ylabel("Difference from Baseline")
    ax.set_title(f"{param_name}\n(variance score: {variance_score:.2e})")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# MAIN WORKFLOW
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Compare spectral changes across training experiments"
    )
    parser.add_argument("--baseline", type=str, default=BASELINE_MODEL,
                       help=f"Baseline model (default: {BASELINE_MODEL})")
    parser.add_argument("--output-dir", "-o", type=str, default="./spectral_analysis",
                       help="Output directory for plots")
    parser.add_argument("--cache-dir", type=str, default="./svd_cache",
                       help="Directory to cache SVD extractions")
    parser.add_argument("--top-k-params", type=int, default=5,
                       help="Number of top parameters to analyze")
    parser.add_argument("--percentile", type=float, default=95,
                       help="Percentile threshold for SV selection")
    parser.add_argument("--neighborhood-gap", type=int, default=5,
                       help="Max gap to merge neighborhoods")
    parser.add_argument("--gpu", type=int, default=0,
                       help="GPU device ID")
    parser.add_argument("--no-cache", action="store_true",
                       help="Disable caching")

    args = parser.parse_args()

    device = torch.device("cuda", args.gpu)
    print(f"Using device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cache_dir = None if args.no_cache else Path(args.cache_dir)

    # Step 1: Extract SVD from baseline
    baseline_svd = load_or_extract_svd(
        args.baseline, device, "baseline", cache_dir
    )
    baseline_params = baseline_svd["parameters"]

    # Step 2: Extract SVD from each experiment and compute diffs
    experiment_svds = {}
    diffs = {}

    for exp_name, exp_config in EXPERIMENTS.items():
        svd_data = load_or_extract_svd(
            exp_config["path"], device, exp_name, cache_dir
        )
        experiment_svds[exp_name] = svd_data

        # Compute diff from baseline
        diffs[exp_name] = {}
        for param_name in svd_data["parameters"]:
            if param_name in baseline_params:
                exp_sv = np.array(svd_data["parameters"][param_name])
                base_sv = np.array(baseline_params[param_name])
                min_len = min(len(exp_sv), len(base_sv))
                diff = (exp_sv[:min_len] - base_sv[:min_len]).tolist()
                diffs[exp_name][param_name] = diff

    # Step 3: Compute variance and select top-k parameters
    print("\nComputing parameter variances...")
    variances = compute_parameter_variance(diffs)
    top_params = select_top_k_params(variances, args.top_k_params)

    print(f"\nTop {args.top_k_params} parameters by variance:")
    for i, param_name in enumerate(top_params, 1):
        print(f"  {i}. {param_name} (variance: {variances[param_name]:.2e})")

    # Step 4: For each top parameter, identify high-variance SVs and plot
    print("\nGenerating focused plots...")

    full_spectrum_dir = output_dir / "full_spectrum"
    full_spectrum_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "baseline": args.baseline,
        "experiments": {name: cfg["path"] for name, cfg in EXPERIMENTS.items()},
        "top_params": [],
    }

    for param_name in tqdm(top_params, desc="Plotting"):
        # Identify high-variance SVs
        high_var_indices = identify_high_variance_svs(
            diffs, param_name, args.percentile
        )

        # Group into neighborhoods
        neighborhoods = find_neighborhoods(high_var_indices, args.neighborhood_gap)

        variance_score = variances[param_name]

        # Record in summary
        summary["top_params"].append({
            "name": param_name,
            "variance_score": variance_score,
            "high_variance_indices": high_var_indices,
            "neighborhoods": neighborhoods,
        })

        # Generate focused plot
        safe_name = sanitize_filename(param_name)
        plot_focused_comparison(
            param_name, diffs, neighborhoods, EXPERIMENTS,
            variance_score, output_dir / f"{safe_name}.png"
        )

        # Also generate full spectrum plot for reference
        plot_full_spectrum(
            param_name, diffs, high_var_indices, EXPERIMENTS,
            variance_score, full_spectrum_dir / f"{safe_name}.png"
        )

    # Save summary
    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Analysis complete!")
    print(f"  Output directory: {output_dir}")
    print(f"  Focused plots: {len(top_params)} parameters")
    print(f"  Full spectrum plots: {full_spectrum_dir}")
    print(f"  Summary: {summary_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
