#!/usr/bin/env python3
"""
Plot average Frobenius norm of 2D matrix updates over training steps.

Reads update_norms.jsonl from a run directory and plots the average
||ΔW_t||_F across all 2D weight matrices at each optimizer step.
"""

import argparse
import json
import os
import re

import matplotlib.pyplot as plt
import numpy as np


def is_2d_weight(param_name: str) -> bool:
    """Check if a parameter name corresponds to a 2D weight matrix.

    Excludes embeddings, biases, layernorms, and output heads.
    """
    # Skip 1D params (biases, layernorms)
    if any(s in param_name for s in [".bias", "layernorm", "layer_norm", "ln_"]):
        return False
    # Skip embeddings
    if "embed" in param_name:
        return False
    # Skip output head
    if "lm_head" in param_name:
        return False
    # Keep attention/MLP weight matrices
    if ".weight" in param_name or "self_attn" in param_name or "mlp" in param_name:
        return True
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()

    norms_path = os.path.join(args.run_dir, "update_norms.jsonl")
    if not os.path.exists(norms_path):
        print(f"No update_norms.jsonl found in {args.run_dir}, skipping plot")
        return

    steps = []
    tokens = []
    avg_norms = []

    with open(norms_path) as f:
        for line in f:
            record = json.loads(line)
            norms = record["norms"]

            # Filter to 2D weight matrices
            matrix_norms = [v for k, v in norms.items() if is_2d_weight(k)]

            if matrix_norms:
                steps.append(record["step"])
                tokens.append(record["tokens"])
                avg_norms.append(np.mean(matrix_norms))

    if not steps:
        print(f"No 2D matrix norms found in {norms_path}")
        return

    run_name = os.path.basename(args.run_dir)

    fig, ax = plt.subplots(figsize=(10, 5))
    tokens_m = [t / 1e6 for t in tokens]
    ax.plot(tokens_m, avg_norms, linewidth=1.5, color="#1f77b4")
    ax.set_xlabel("Tokens Trained (M)", fontsize=12)
    ax.set_ylabel(r"Average $\|\Delta W_t\|_F$ (2D matrices)", fontsize=12)
    ax.set_title(f"Update Norm Trajectory: {run_name}", fontsize=13)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    out_path = os.path.join(args.run_dir, "update_norm_trajectory.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
