#!/usr/bin/env python3
"""Compute weight-space metrics for all checkpoints: L0, thresholded L0, Frobenius norm, spectral."""

import json, math, sys, os
import torch
import numpy as np
from pathlib import Path
from safetensors.torch import load_file
from tqdm import tqdm

def load_checkpoint(path):
    """Load checkpoint weights as flat dict of fp32 tensors."""
    p = Path(path)
    if (p / "model.safetensors").exists():
        return {k: v.float() for k, v in load_file(str(p / "model.safetensors")).items()}
    # Multi-shard
    idx_file = p / "model.safetensors.index.json"
    if idx_file.exists():
        with open(idx_file) as f:
            idx = json.load(f)
        sd = {}
        for shard in set(idx["weight_map"].values()):
            sd.update({k: v.float() for k, v in load_file(str(p / shard)).items()})
        return sd
    raise FileNotFoundError(f"No model files in {path}")

def compute_metrics(base_sd, ckpt_sd):
    total = changed_exact = changed_thresh = 0
    frob_sq = 0.0
    for k in base_sd:
        if k not in ckpt_sd:
            continue
        b, c = base_sd[k], ckpt_sd[k]
        delta = c - b
        n = b.numel()
        total += n
        changed_exact += (delta != 0).sum().item()
        changed_thresh += (delta.abs() > 1e-5).sum().item()
        frob_sq += (delta ** 2).sum().item()
    return {
        "l0_exact": changed_exact / total if total > 0 else 0,
        "l0_thresh_1e5": changed_thresh / total if total > 0 else 0,
        "frob_norm": math.sqrt(frob_sq),
        "total_params": total,
    }

def compute_spectral(base_sd, ckpt_sd):
    """Compute SVD-based spectral metrics for 2D weight matrices."""
    all_rank90 = []
    all_eff_rank = []
    for k in base_sd:
        if k not in ckpt_sd or base_sd[k].dim() != 2:
            continue
        delta = ckpt_sd[k] - base_sd[k]
        if delta.abs().max() == 0:
            continue
        sv = torch.linalg.svdvals(delta.float()).numpy()
        total_energy = (sv**2).sum()
        if total_energy <= 0:
            continue
        # Effective rank (Shannon)
        p = sv**2 / total_energy
        p = p[p > 0]
        entropy = -(p * np.log(p)).sum()
        eff_rank = np.exp(entropy)
        all_eff_rank.append(eff_rank)
        # K @ 90% energy
        cum_energy = np.cumsum(sv**2) / total_energy
        rank90 = int(np.searchsorted(cum_energy, 0.90) + 1)
        all_rank90.append(rank90)
    return {
        "avg_effective_rank": float(np.mean(all_eff_rank)) if all_eff_rank else 0,
        "avg_k_at_90pct_energy": float(np.mean(all_rank90)) if all_rank90 else 0,
        "n_layers_analyzed": len(all_eff_rank),
    }

def main():
    exp_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("experiments/grpo-adamw-mixed-m10lattice")
    base_path = exp_dir / "checkpoint-initial"

    print(f"Loading base checkpoint: {base_path}")
    base_sd = load_checkpoint(base_path)
    print(f"  {len(base_sd)} tensors, {sum(v.numel() for v in base_sd.values()):,} parameters")

    checkpoints = sorted([d for d in exp_dir.iterdir()
                          if d.is_dir() and d.name.startswith("checkpoint-") and d.name != "checkpoint-initial"],
                         key=lambda x: x.name)

    results = {}
    for ckpt_path in tqdm(checkpoints, desc="Evaluating checkpoints"):
        name = ckpt_path.name
        ckpt_sd = load_checkpoint(ckpt_path)
        metrics = compute_metrics(base_sd, ckpt_sd)

        # Only compute spectral for a subset (expensive)
        if name == checkpoints[-1].name:  # Always compute for final
            spectral = compute_spectral(base_sd, ckpt_sd)
            metrics.update(spectral)

        results[name] = metrics
        tqdm.write(f"  {name}: L0={metrics['l0_exact']:.4f}, L0(1e-5)={metrics['l0_thresh_1e5']:.4f}, "
                   f"||dW||={metrics['frob_norm']:.4f}")
        del ckpt_sd

    # Save results
    out_path = exp_dir / "eval_weight_metrics.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")

    # Print summary table
    print(f"\n{'Checkpoint':<40} {'L0 exact':>10} {'L0(1e-5)':>10} {'||dW||':>10}")
    print("-" * 75)
    for name, m in results.items():
        spectral_str = ""
        if "avg_effective_rank" in m:
            spectral_str = f"  EffRank={m['avg_effective_rank']:.1f}, K@90%={m['avg_k_at_90pct_energy']:.1f}"
        print(f"  {name:<38} {m['l0_exact']:>10.4f} {m['l0_thresh_1e5']:>10.4f} {m['frob_norm']:>10.4f}{spectral_str}")

if __name__ == "__main__":
    main()
