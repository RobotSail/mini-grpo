#!/usr/bin/env python3
"""
Compute geometry metrics for the best validation checkpoint in a run.

Reads validation_accuracy_kl.json to find the best checkpoint,
then computes ΔW = W_selected - W_initial and reports:
  1. Frobenius norm: ||ΔW||_F = sqrt(sum_l ||ΔW_l||_F^2)
  2. k_90: average rank at 90% SV energy concentration
  3. L0 sparsity: fraction of |ΔW| >= 1e-5
"""

import argparse
import json
import os

import torch
import numpy as np
from safetensors.torch import load_file


def load_state_dict(checkpoint_dir: str) -> dict[str, torch.Tensor]:
    """Load state dict from safetensors files."""
    index_path = os.path.join(checkpoint_dir, "model.safetensors.index.json")
    if os.path.exists(index_path):
        with open(index_path) as f:
            index = json.load(f)
        shard_files = set(index["weight_map"].values())
        state_dict = {}
        for shard in sorted(shard_files):
            shard_path = os.path.join(checkpoint_dir, shard)
            state_dict.update(load_file(shard_path))
        return state_dict
    else:
        # Single file
        sf_path = os.path.join(checkpoint_dir, "model.safetensors")
        return load_file(sf_path)


def compute_k90(singular_values: np.ndarray) -> int:
    """Find rank k where top-k SVs capture 90% of total energy."""
    energy = singular_values ** 2
    total = energy.sum()
    if total == 0:
        return 0
    cumulative = np.cumsum(energy)
    k = int(np.searchsorted(cumulative, 0.9 * total) + 1)
    return k


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, help="Path to run directory")
    args = parser.parse_args()

    run_dir = args.run_dir

    # Find best validation checkpoint
    val_path = os.path.join(run_dir, "validation_accuracy_kl.json")
    if not os.path.exists(val_path):
        print(f"ERROR: {val_path} not found. Run validation eval first.")
        return

    with open(val_path) as f:
        val_results = json.load(f)

    if not val_results:
        print("ERROR: No validation results found.")
        return

    best_label = max(val_results, key=lambda k: val_results[k].get("accuracy", 0))
    best_info = val_results[best_label]
    best_path = best_info["path"]
    print(f"Best validation checkpoint: {best_label}")
    print(f"  Accuracy: {best_info['accuracy']:.2%}")
    print(f"  Path: {best_path}")

    # Load initial and best checkpoints
    initial_path = os.path.join(run_dir, "checkpoint-initial")
    if not os.path.exists(initial_path):
        print(f"ERROR: {initial_path} not found.")
        return

    print("Loading initial checkpoint...")
    w_initial = load_state_dict(initial_path)
    print("Loading best checkpoint...")
    w_best = load_state_dict(best_path)

    # Compute ΔW metrics
    total_frob_sq = 0.0
    total_elements = 0
    sparse_elements = 0
    k90_values = []

    for name in sorted(w_initial.keys()):
        if name not in w_best:
            continue

        w0 = w_initial[name].float()
        w1 = w_best[name].float()
        delta = w1 - w0

        # Frobenius norm contribution
        frob_sq = delta.norm(2).square().item()
        total_frob_sq += frob_sq

        # L0 sparsity
        n_elements = delta.numel()
        n_nonzero = (delta.abs() >= 1e-5).sum().item()
        total_elements += n_elements
        sparse_elements += n_nonzero

        # k_90 for 2D+ weight matrices only
        if delta.ndim >= 2 and min(delta.shape) > 1:
            # Reshape to 2D if needed (e.g. conv weights)
            mat = delta.reshape(delta.shape[0], -1)
            try:
                svs = torch.linalg.svdvals(mat).numpy()
                k90 = compute_k90(svs)
                k90_values.append(k90)
            except Exception as e:
                print(f"  SVD failed for {name}: {e}")

    total_frob = total_frob_sq ** 0.5
    l0_sparsity = sparse_elements / total_elements if total_elements > 0 else 0.0
    avg_k90 = float(np.mean(k90_values)) if k90_values else 0.0

    metrics = {
        "best_checkpoint": best_label,
        "best_checkpoint_path": best_path,
        "best_validation_accuracy": best_info["accuracy"],
        "frobenius_norm": total_frob,
        "avg_k90": avg_k90,
        "l0_sparsity_1e5": l0_sparsity,
        "num_layers_svd": len(k90_values),
        "total_parameters": total_elements,
    }

    out_path = os.path.join(run_dir, "geometry_metrics.json")
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nGeometry metrics saved to: {out_path}")
    print(f"  ||ΔW||_F = {total_frob:.6f}")
    print(f"  avg k_90 = {avg_k90:.1f}")
    print(f"  L0 sparsity (>=1e-5) = {l0_sparsity:.4%}")


if __name__ == "__main__":
    main()
