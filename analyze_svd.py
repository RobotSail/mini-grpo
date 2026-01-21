#!/usr/bin/env python3
"""
SVD Spectral Analysis Script

Two-stage analysis:
1. `extract` - Computes SVD and saves singular values to JSON
2. `plot` - Generates visualizations from one or more extraction results

Usage:
    python analyze_svd.py extract --checkpoint-dir <path> --output <output.json>
    python analyze_svd.py plot --input file1.json file2.json --output-dir ./plots
"""

import argparse
import json
from pathlib import Path
from typing import Iterator

import torch
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from safetensors import safe_open
from tqdm import tqdm
from transformers import AutoModelForCausalLM


def get_checkpoint_dirs(base_path: str) -> list[Path | str]:
    """Get all checkpoint directories sorted by step number."""
    base = Path(base_path)

    if "/" in base_path and not base.exists():
        return [base_path]

    if not base.exists():
        raise ValueError(f"Checkpoint path does not exist: {base_path}")

    if (base / "config.json").exists():
        return [base]

    checkpoint_dirs = sorted(
        [d for d in base.iterdir() if d.is_dir() and d.name.startswith("step_")],
        key=lambda x: int(x.name.split("_")[1])
    )

    if not checkpoint_dirs:
        raise ValueError(f"No step_* checkpoint directories found in {base_path}")

    return checkpoint_dirs


def load_state_dict_from_checkpoint(checkpoint_dir: Path | str, device: torch.device) -> dict[str, torch.Tensor]:
    """Load model state dict from checkpoint directory or HuggingFace model."""
    checkpoint_path = Path(checkpoint_dir) if isinstance(checkpoint_dir, str) else checkpoint_dir
    is_hf_model = isinstance(checkpoint_dir, str) and "/" in checkpoint_dir and not checkpoint_path.exists()

    if is_hf_model:
        print(f"Loading model from HuggingFace: {checkpoint_dir}")
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint_dir,
            torch_dtype=torch.float32,
            device_map=device,
        )
        state_dict = {name: param.data for name, param in model.named_parameters()}
        del model
        torch.cuda.empty_cache()
        return state_dict

    safetensor_files = sorted(checkpoint_path.glob("*.safetensors"))

    if not safetensor_files:
        raise ValueError(f"No safetensors files found in {checkpoint_dir}")

    device_str = str(device) if device.type == "cuda" else "cpu"
    state_dict = {}
    for sf_file in safetensor_files:
        with safe_open(sf_file, framework="pt", device=device_str) as f:
            for key in f.keys():
                state_dict[key] = f.get_tensor(key)

    return state_dict


def iter_2d_dense_params(state_dict: dict[str, torch.Tensor]) -> Iterator[tuple[str, torch.Tensor]]:
    """Iterate over 2D dense matrices in the state dict."""
    for name, param in sorted(state_dict.items()):  # sorted for stable order
        if param.ndim == 2 and param.shape[0] > 1 and param.shape[1] > 1:
            yield name, param


def compute_svd_singular_values(matrix: torch.Tensor) -> list[float]:
    """Compute singular values of a matrix using SVD."""
    matrix_f32 = matrix.float()
    try:
        singular_values = torch.linalg.svdvals(matrix_f32)
        return singular_values.cpu().tolist()
    except RuntimeError as e:
        print(f"SVD computation failed: {e}")
        return []


def sanitize_filename(name: str) -> str:
    """Convert parameter name to valid filename."""
    return name.replace("/", "_").replace(".", "_")


# =============================================================================
# EXTRACT COMMAND
# =============================================================================

def extract_checkpoint(checkpoint_dir: Path | str, device: torch.device, name: str) -> dict:
    """Extract SVD data from a single checkpoint.

    Returns:
    {
        "model_path": str,
        "name": str,
        "parameters": {
            "param_name": [float, ...],  # singular values
            ...
        }
    }
    """
    print(f"\nExtracting SVD from: {checkpoint_dir}")

    state_dict = load_state_dict_from_checkpoint(checkpoint_dir, device)
    params_2d = list(iter_2d_dense_params(state_dict))
    print(f"Found {len(params_2d)} 2D dense matrices")

    result = {
        "model_path": str(checkpoint_dir),
        "name": name,
        "parameters": {}
    }

    for param_name, param_tensor in tqdm(params_2d, desc="Computing SVD"):
        singular_values = compute_svd_singular_values(param_tensor)
        if singular_values:
            result["parameters"][param_name] = singular_values

    return result


def cmd_extract(args):
    """Handle the extract subcommand."""
    device = torch.device("cuda", args.gpu)
    print(f"Using device: {device}")

    checkpoint_dirs = get_checkpoint_dirs(args.checkpoint_dir)

    if args.steps:
        requested_steps = set(int(s.strip()) for s in args.steps.split(","))
        checkpoint_dirs = [
            d for d in checkpoint_dirs
            if isinstance(d, str) or not d.name.startswith("step_") or int(d.name.split("_")[1]) in requested_steps
        ]

    print(f"Found {len(checkpoint_dirs)} checkpoint(s) to extract")

    for checkpoint_dir in checkpoint_dirs:
        # Determine name
        if args.name:
            name = args.name
        elif isinstance(checkpoint_dir, str):
            name = checkpoint_dir.replace("/", "_")
        elif checkpoint_dir.name.startswith("step_"):
            name = checkpoint_dir.name
        else:
            name = checkpoint_dir.name

        result = extract_checkpoint(checkpoint_dir, device, name=name)

        # Determine output path
        if args.output:
            if len(checkpoint_dirs) == 1:
                output_path = Path(args.output)
            else:
                base = Path(args.output)
                output_path = base.parent / f"{base.stem}_{name}{base.suffix}"
        else:
            output_path = Path(f"svd_{name}.json")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(result, f)

        print(f"Saved: {output_path} ({len(result['parameters'])} parameters)")

    print("\nExtraction complete!")


# =============================================================================
# PLOT COMMAND
# =============================================================================

def cmd_plot(args):
    """Handle the plot subcommand."""
    # Load all SVD data files
    svd_datasets = []
    for input_path in args.input:
        with open(input_path) as f:
            data = json.load(f)
        svd_datasets.append(data)
        print(f"Loaded: {data['name']} ({len(data['parameters'])} parameters)")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load baseline if specified
    baseline_data = None
    if args.baseline:
        with open(args.baseline) as f:
            baseline_data = json.load(f)
        print(f"Baseline: {baseline_data['name']}")

    # Get common parameters
    all_param_names = set(svd_datasets[0]["parameters"].keys())
    for data in svd_datasets[1:]:
        all_param_names &= set(data["parameters"].keys())
    if baseline_data:
        all_param_names &= set(baseline_data["parameters"].keys())

    if args.params:
        requested = set(args.params.split(","))
        all_param_names = {p for p in all_param_names if any(r in p for r in requested)}

    print(f"Plotting {len(all_param_names)} common parameters")

    colors = plt.cm.tab10(np.linspace(0, 1, len(svd_datasets)))
    top_k = args.top_k if args.top_k else None

    # Track variance for each parameter (for high-variance subdirectory)
    param_variances = {}

    for param_name in tqdm(sorted(all_param_names), desc="Generating plots"):
        fig, ax = plt.subplots(figsize=(12, 7))

        # Collect all SVs for variance calculation
        all_svs = []

        for idx, data in enumerate(svd_datasets):
            sv = np.array(data["parameters"][param_name])

            # Apply difference if baseline specified
            if baseline_data:
                baseline_sv = np.array(baseline_data["parameters"][param_name])
                sv = sv - baseline_sv

            # Apply top-k
            if top_k:
                sv = sv[:top_k]

            all_svs.append(sv)
            ranks = np.arange(1, len(sv) + 1)
            color = colors[idx]
            label = data["name"]

            if args.mode == "scatter":
                ax.scatter(ranks, sv, c=[color], s=10, alpha=0.7, label=label)
            else:
                ax.plot(ranks, sv, color=color, linewidth=1.0, label=label)

        # Calculate variance across checkpoints for this parameter
        # Uses top-k SVs if specified, so variance reflects changes in top singular values
        if len(all_svs) > 1:
            min_len = min(len(s) for s in all_svs)
            stacked = np.stack([s[:min_len] for s in all_svs])
            param_variances[param_name] = np.mean(np.var(stacked, axis=0))

        # Set scale based on whether we're doing difference plots
        if baseline_data:
            ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            ax.set_ylabel("Singular Value Difference")
        else:
            ax.set_yscale("log")
            ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
            ax.yaxis.get_major_formatter().set_scientific(False)
            ax.set_ylabel("Singular Value Magnitude")

        ax.set_xlabel("Singular Value Rank")
        ax.set_title(param_name)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)

        plt.tight_layout()
        safe_name = sanitize_filename(param_name)
        plt.savefig(output_dir / f"{safe_name}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    # Create high-variance subdirectory with top K most variable parameters
    if args.top_k_variance and len(param_variances) > 0:
        high_var_dir = output_dir / "high_variance"
        high_var_dir.mkdir(parents=True, exist_ok=True)

        sorted_params = sorted(param_variances.items(), key=lambda x: x[1], reverse=True)
        top_params = sorted_params[:args.top_k_variance]

        print(f"\nRe-rendering top {len(top_params)} high-variance parameters...")

        for param_name, variance in tqdm(top_params, desc="High variance plots"):
            fig, ax = plt.subplots(figsize=(12, 7))

            for idx, data in enumerate(svd_datasets):
                sv = np.array(data["parameters"][param_name])

                if baseline_data:
                    baseline_sv = np.array(baseline_data["parameters"][param_name])
                    sv = sv - baseline_sv

                if top_k:
                    sv = sv[:top_k]

                ranks = np.arange(1, len(sv) + 1)
                color = colors[idx]
                label = data["name"]

                if args.mode == "scatter":
                    ax.scatter(ranks, sv, c=[color], s=10, alpha=0.7, label=label)
                else:
                    ax.plot(ranks, sv, color=color, linewidth=1.0, label=label)

            if baseline_data:
                ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
                ax.set_ylabel("Singular Value Difference")
            else:
                ax.set_yscale("log")
                ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
                ax.yaxis.get_major_formatter().set_scientific(False)
                ax.set_ylabel("Singular Value Magnitude")

            ax.set_xlabel("Singular Value Rank")
            ax.set_title(f"{param_name}\n(variance: {variance:.2e})")
            ax.grid(True, alpha=0.3)
            ax.legend(loc="upper right", fontsize=8)

            plt.tight_layout()
            safe_name = sanitize_filename(param_name)
            plt.savefig(high_var_dir / f"{safe_name}.png", dpi=150, bbox_inches="tight")
            plt.close(fig)

        print(f"High-variance plots saved to: {high_var_dir}")

    print(f"\nPlots saved to: {output_dir}")


# =============================================================================
# DIFF COMMAND
# =============================================================================

def cmd_diff(args):
    """Plot singular value differences from a baseline."""
    # Load baseline
    with open(args.baseline) as f:
        baseline_data = json.load(f)
    print(f"Baseline: {baseline_data['name']} ({len(baseline_data['parameters'])} parameters)")

    # Load all checkpoint files
    svd_datasets = []
    for input_path in args.input:
        with open(input_path) as f:
            data = json.load(f)
        svd_datasets.append(data)
        print(f"Loaded: {data['name']}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get common parameters
    all_param_names = set(baseline_data["parameters"].keys())
    for data in svd_datasets:
        all_param_names &= set(data["parameters"].keys())

    if args.params:
        requested = set(args.params.split(","))
        all_param_names = {p for p in all_param_names if any(r in p for r in requested)}

    print(f"Plotting {len(all_param_names)} common parameters")

    colors = plt.cm.tab10(np.linspace(0, 1, len(svd_datasets)))
    top_k = args.top_k

    # Track variance for high-variance subdirectory
    param_variances = {}

    for param_name in tqdm(sorted(all_param_names), desc="Generating diff plots"):
        fig, ax = plt.subplots(figsize=(12, 7))

        baseline_sv = np.array(baseline_data["parameters"][param_name])
        all_diffs = []

        for idx, data in enumerate(svd_datasets):
            sv = np.array(data["parameters"][param_name])
            diff = sv - baseline_sv

            if top_k:
                diff = diff[:top_k]

            all_diffs.append(diff)
            ranks = np.arange(1, len(diff) + 1)
            color = colors[idx]
            label = data["name"]

            if args.mode == "scatter":
                ax.scatter(ranks, diff, c=[color], s=10, alpha=0.7, label=label)
            else:
                ax.plot(ranks, diff, color=color, linewidth=1.0, label=label)

        # Calculate variance for high-variance selection
        if len(all_diffs) > 1:
            min_len = min(len(d) for d in all_diffs)
            stacked = np.stack([d[:min_len] for d in all_diffs])
            param_variances[param_name] = np.mean(np.var(stacked, axis=0))

        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel("Singular Value Rank")
        ax.set_ylabel(f"Difference from {baseline_data['name']}")
        ax.set_title(param_name)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)

        plt.tight_layout()
        safe_name = sanitize_filename(param_name)
        plt.savefig(output_dir / f"{safe_name}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    # High-variance subdirectory
    if args.top_k_variance and len(param_variances) > 0:
        high_var_dir = output_dir / "high_variance"
        high_var_dir.mkdir(parents=True, exist_ok=True)

        sorted_params = sorted(param_variances.items(), key=lambda x: x[1], reverse=True)
        top_params = sorted_params[:args.top_k_variance]

        print(f"\nRe-rendering top {len(top_params)} high-variance parameters...")

        for param_name, variance in tqdm(top_params, desc="High variance plots"):
            fig, ax = plt.subplots(figsize=(12, 7))

            baseline_sv = np.array(baseline_data["parameters"][param_name])

            for idx, data in enumerate(svd_datasets):
                sv = np.array(data["parameters"][param_name])
                diff = sv - baseline_sv

                if top_k:
                    diff = diff[:top_k]

                ranks = np.arange(1, len(diff) + 1)
                color = colors[idx]
                label = data["name"]

                if args.mode == "scatter":
                    ax.scatter(ranks, diff, c=[color], s=10, alpha=0.7, label=label)
                else:
                    ax.plot(ranks, diff, color=color, linewidth=1.0, label=label)

            ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            ax.set_xlabel("Singular Value Rank")
            ax.set_ylabel(f"Difference from {baseline_data['name']}")
            ax.set_title(f"{param_name}\n(variance: {variance:.2e})")
            ax.grid(True, alpha=0.3)
            ax.legend(loc="upper right", fontsize=8)

            plt.tight_layout()
            safe_name = sanitize_filename(param_name)
            plt.savefig(high_var_dir / f"{safe_name}.png", dpi=150, bbox_inches="tight")
            plt.close(fig)

        print(f"High-variance plots saved to: {high_var_dir}")

    print(f"\nDiff plots saved to: {output_dir}")


# =============================================================================
# COMPARE COMMAND
# =============================================================================

def extract_step_from_name(name: str) -> int | None:
    """Extract step number from a name like 'step_200', 'Step 200', or 'Muon (Step 1000)'."""
    import re
    # Match patterns like: step_200, step-200, Step 200, (Step 1000)
    match = re.search(r'step[_\-\s]?(\d+)', name, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return None


def cmd_compare(args):
    """Compare two optimizers (e.g., Muon vs AdamW) against a baseline."""
    # Load baseline
    with open(args.baseline) as f:
        baseline_data = json.load(f)
    print(f"Baseline: {baseline_data['name']} ({len(baseline_data['parameters'])} parameters)")

    # Load optimizer A files
    opt_a_datasets = []
    for input_path in args.optimizer_a:
        with open(input_path) as f:
            data = json.load(f)
        step = extract_step_from_name(data['name'])
        opt_a_datasets.append((step, data))
        print(f"Loaded {args.label_a}: {data['name']} (step {step})")

    # Load optimizer B files
    opt_b_datasets = []
    for input_path in args.optimizer_b:
        with open(input_path) as f:
            data = json.load(f)
        step = extract_step_from_name(data['name'])
        opt_b_datasets.append((step, data))
        print(f"Loaded {args.label_b}: {data['name']} (step {step})")

    # Match by step number
    opt_a_by_step = {step: data for step, data in opt_a_datasets if step is not None}
    opt_b_by_step = {step: data for step, data in opt_b_datasets if step is not None}
    common_steps = sorted(set(opt_a_by_step.keys()) & set(opt_b_by_step.keys()))

    if not common_steps:
        print("ERROR: No common steps found between the two optimizers!")
        print(f"  {args.label_a} steps: {sorted(opt_a_by_step.keys())}")
        print(f"  {args.label_b} steps: {sorted(opt_b_by_step.keys())}")
        return

    print(f"\nComparing at steps: {common_steps}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get common parameters across all datasets
    all_param_names = set(baseline_data["parameters"].keys())
    for _, data in opt_a_datasets:
        all_param_names &= set(data["parameters"].keys())
    for _, data in opt_b_datasets:
        all_param_names &= set(data["parameters"].keys())

    if args.params:
        requested = set(args.params.split(","))
        all_param_names = {p for p in all_param_names if any(r in p for r in requested)}

    print(f"Plotting {len(all_param_names)} common parameters")

    top_k = args.top_k

    # Colors for the two optimizers
    color_a = '#1f77b4'  # Blue
    color_b = '#ff7f0e'  # Orange

    # Track variance for high-variance subdirectory
    param_variances = {}

    for param_name in tqdm(sorted(all_param_names), desc="Generating comparison plots"):
        # Create a figure with subplots for each step
        n_steps = len(common_steps)
        fig, axes = plt.subplots(1, n_steps, figsize=(6 * n_steps, 6), squeeze=False)
        axes = axes[0]  # Flatten to 1D

        baseline_sv = np.array(baseline_data["parameters"][param_name])

        # Track all diffs for variance calculation
        all_diffs_a = []
        all_diffs_b = []

        for idx, step in enumerate(common_steps):
            ax = axes[idx]

            # Get data for this step
            data_a = opt_a_by_step[step]
            data_b = opt_b_by_step[step]

            sv_a = np.array(data_a["parameters"][param_name])
            sv_b = np.array(data_b["parameters"][param_name])

            diff_a = sv_a - baseline_sv
            diff_b = sv_b - baseline_sv

            if top_k:
                diff_a = diff_a[:top_k]
                diff_b = diff_b[:top_k]

            all_diffs_a.append(diff_a)
            all_diffs_b.append(diff_b)

            ranks = np.arange(1, len(diff_a) + 1)

            if args.mode == "scatter":
                ax.scatter(ranks, diff_a, c=color_a, s=10, alpha=0.7, label=args.label_a)
                ax.scatter(ranks, diff_b, c=color_b, s=10, alpha=0.7, label=args.label_b)
            else:
                ax.plot(ranks, diff_a, color=color_a, linewidth=1.0, label=args.label_a)
                ax.plot(ranks, diff_b, color=color_b, linewidth=1.0, label=args.label_b)

            ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            ax.set_xlabel("Singular Value Rank")
            ax.set_ylabel("Difference from Baseline")
            ax.set_title(f"Step {step}")
            ax.grid(True, alpha=0.3)
            ax.legend(loc="upper right", fontsize=8)

        # Calculate variance across steps for this parameter
        if len(common_steps) > 1:
            min_len = min(min(len(d) for d in all_diffs_a), min(len(d) for d in all_diffs_b))
            stacked_a = np.stack([d[:min_len] for d in all_diffs_a])
            stacked_b = np.stack([d[:min_len] for d in all_diffs_b])
            # Variance is the mean of variance across steps for both optimizers
            var_a = np.mean(np.var(stacked_a, axis=0))
            var_b = np.mean(np.var(stacked_b, axis=0))
            # Also consider the difference between optimizers
            diff_between = np.mean(np.abs(stacked_a - stacked_b))
            param_variances[param_name] = var_a + var_b + diff_between

        fig.suptitle(param_name, fontsize=12, fontweight='bold')
        plt.tight_layout()
        safe_name = sanitize_filename(param_name)
        plt.savefig(output_dir / f"{safe_name}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    # High-variance subdirectory
    if args.top_k_variance and len(param_variances) > 0:
        high_var_dir = output_dir / "high_variance"
        high_var_dir.mkdir(parents=True, exist_ok=True)

        sorted_params = sorted(param_variances.items(), key=lambda x: x[1], reverse=True)
        top_params = sorted_params[:args.top_k_variance]

        print(f"\nRe-rendering top {len(top_params)} high-variance parameters...")

        for param_name, variance in tqdm(top_params, desc="High variance plots"):
            n_steps = len(common_steps)
            fig, axes = plt.subplots(1, n_steps, figsize=(6 * n_steps, 6), squeeze=False)
            axes = axes[0]

            baseline_sv = np.array(baseline_data["parameters"][param_name])

            for idx, step in enumerate(common_steps):
                ax = axes[idx]

                data_a = opt_a_by_step[step]
                data_b = opt_b_by_step[step]

                sv_a = np.array(data_a["parameters"][param_name])
                sv_b = np.array(data_b["parameters"][param_name])

                diff_a = sv_a - baseline_sv
                diff_b = sv_b - baseline_sv

                if top_k:
                    diff_a = diff_a[:top_k]
                    diff_b = diff_b[:top_k]

                ranks = np.arange(1, len(diff_a) + 1)

                if args.mode == "scatter":
                    ax.scatter(ranks, diff_a, c=color_a, s=10, alpha=0.7, label=args.label_a)
                    ax.scatter(ranks, diff_b, c=color_b, s=10, alpha=0.7, label=args.label_b)
                else:
                    ax.plot(ranks, diff_a, color=color_a, linewidth=1.0, label=args.label_a)
                    ax.plot(ranks, diff_b, color=color_b, linewidth=1.0, label=args.label_b)

                ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
                ax.set_xlabel("Singular Value Rank")
                ax.set_ylabel("Difference from Baseline")
                ax.set_title(f"Step {step}")
                ax.grid(True, alpha=0.3)
                ax.legend(loc="upper right", fontsize=8)

            fig.suptitle(f"{param_name}\n(variance score: {variance:.2e})", fontsize=12, fontweight='bold')
            plt.tight_layout()
            safe_name = sanitize_filename(param_name)
            plt.savefig(high_var_dir / f"{safe_name}.png", dpi=150, bbox_inches="tight")
            plt.close(fig)

        print(f"High-variance plots saved to: {high_var_dir}")

    print(f"\nComparison plots saved to: {output_dir}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="SVD Spectral Analysis")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Extract
    extract_parser = subparsers.add_parser("extract", help="Extract SVD from checkpoints")
    extract_parser.add_argument("--checkpoint-dir", type=str, required=True)
    extract_parser.add_argument("--output", "-o", type=str, default=None)
    extract_parser.add_argument("--name", type=str, default=None, help="Human-readable name")
    extract_parser.add_argument("--steps", type=str, default=None)
    extract_parser.add_argument("--gpu", type=int, default=0)

    # Plot
    plot_parser = subparsers.add_parser("plot", help="Generate plots from extractions")
    plot_parser.add_argument("--input", "-i", type=str, nargs="+", required=True)
    plot_parser.add_argument("--output-dir", "-o", type=str, default="./svd_plots")
    plot_parser.add_argument("--mode", choices=["plot", "scatter"], default="plot")
    plot_parser.add_argument("--baseline", "-b", type=str, default=None,
                            help="Baseline JSON file for difference plots")
    plot_parser.add_argument("--top-k", type=int, default=None,
                            help="Only plot top-k singular values")
    plot_parser.add_argument("--top-k-variance", type=int, default=None,
                            help="Create subdirectory with K highest-variance parameters")
    plot_parser.add_argument("--params", type=str, default=None, help="Filter params (e.g., 'q_proj,k_proj')")

    # Diff
    diff_parser = subparsers.add_parser("diff", help="Plot differences from a baseline")
    diff_parser.add_argument("--baseline", "-b", type=str, required=True,
                            help="Baseline JSON file to compare against")
    diff_parser.add_argument("--input", "-i", type=str, nargs="+", required=True,
                            help="Checkpoint JSON files to compare")
    diff_parser.add_argument("--output-dir", "-o", type=str, default="./svd_diff")
    diff_parser.add_argument("--mode", choices=["plot", "scatter"], default="plot")
    diff_parser.add_argument("--top-k", type=int, default=None,
                            help="Only plot top-k singular values")
    diff_parser.add_argument("--top-k-variance", type=int, default=None,
                            help="Create subdirectory with K highest-variance parameters")
    diff_parser.add_argument("--params", type=str, default=None, help="Filter params")

    # Compare
    compare_parser = subparsers.add_parser("compare", help="Compare two optimizers side-by-side")
    compare_parser.add_argument("--baseline", "-b", type=str, required=True,
                                help="Baseline JSON file to compare against")
    compare_parser.add_argument("--optimizer-a", "-a", type=str, nargs="+", required=True,
                                help="First optimizer's checkpoint JSON files (e.g., Muon)")
    compare_parser.add_argument("--optimizer-b", "-B", type=str, nargs="+", required=True,
                                help="Second optimizer's checkpoint JSON files (e.g., AdamW)")
    compare_parser.add_argument("--label-a", type=str, default="Optimizer A",
                                help="Label for first optimizer (default: 'Optimizer A')")
    compare_parser.add_argument("--label-b", type=str, default="Optimizer B",
                                help="Label for second optimizer (default: 'Optimizer B')")
    compare_parser.add_argument("--output-dir", "-o", type=str, default="./svd_compare")
    compare_parser.add_argument("--mode", choices=["plot", "scatter"], default="plot")
    compare_parser.add_argument("--top-k", type=int, default=None,
                                help="Only plot top-k singular values")
    compare_parser.add_argument("--top-k-variance", type=int, default=None,
                                help="Create subdirectory with K highest-variance parameters")
    compare_parser.add_argument("--params", type=str, default=None, help="Filter params")

    args = parser.parse_args()

    if args.command == "extract":
        cmd_extract(args)
    elif args.command == "plot":
        cmd_plot(args)
    elif args.command == "diff":
        cmd_diff(args)
    elif args.command == "compare":
        cmd_compare(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
