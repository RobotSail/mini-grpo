#!/usr/bin/env python3
"""
Extract a lottery ticket mask from a trained checkpoint.

Computes ΔW = W_trained - W_base for each parameter and creates a boolean mask
where True = parameter changed (|ΔW| > 0). Also generates random masks with
the same per-layer sparsity budget for control experiments.

Usage:
    # Extract lottery mask from bf16-master checkpoint
    python extract_lottery_mask.py \
        --checkpoint /path/to/checkpoint-best \
        --initial /path/to/checkpoint-initial \
        --output masks/lottery_mask.pt

    # Also generate random control masks
    python extract_lottery_mask.py \
        --checkpoint /path/to/checkpoint-best \
        --initial /path/to/checkpoint-initial \
        --output masks/lottery_mask.pt \
        --random-seeds 1,2,3 \
        --random-output-prefix masks/random_mask
"""

import argparse
import os

import torch
from safetensors.torch import load_file


def load_checkpoint(path: str) -> dict[str, torch.Tensor]:
    """Load all safetensors from a checkpoint directory."""
    state = {}
    for f in sorted(os.listdir(path)):
        if f.startswith("model") and f.endswith(".safetensors"):
            state.update(load_file(os.path.join(path, f), device="cpu"))
    return state


def extract_mask(checkpoint_path: str, initial_path: str) -> dict:
    """Extract lottery ticket mask by comparing trained checkpoint to initial."""
    print(f"Loading initial checkpoint: {initial_path}")
    initial = load_checkpoint(initial_path)

    print(f"Loading trained checkpoint: {checkpoint_path}")
    trained = load_checkpoint(checkpoint_path)

    masks = {}
    total_params = 0
    total_changed = 0
    per_layer = {}

    for name in initial:
        if name not in trained:
            continue

        delta = trained[name].float() - initial[name].float()
        changed = (delta.abs() > 0)
        masks[name] = changed

        n = delta.numel()
        n_changed = changed.sum().item()
        total_params += n
        total_changed += n_changed
        density = n_changed / n if n > 0 else 0
        per_layer[name] = {"total": n, "changed": n_changed, "density": density}

    sparsity = 1.0 - total_changed / total_params if total_params > 0 else 0
    density = total_changed / total_params if total_params > 0 else 0

    print(f"\nMask statistics:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Changed parameters: {total_changed:,}")
    print(f"  Density (fraction trainable): {density:.6f} ({density*100:.4f}%)")
    print(f"  Sparsity: {sparsity:.6f} ({sparsity*100:.4f}%)")

    return {
        "masks": masks,
        "sparsity": sparsity,
        "density": density,
        "total_params": total_params,
        "total_changed": total_changed,
        "per_layer": per_layer,
        "source_checkpoint": checkpoint_path,
        "initial_checkpoint": initial_path,
        "mask_type": "lottery",
    }


def generate_random_mask(lottery_data: dict, seed: int) -> dict:
    """Generate a random mask with the same per-layer density as the lottery mask."""
    rng = torch.Generator()
    rng.manual_seed(seed)

    masks = {}
    total_changed = 0
    total_params = 0

    for name, layer_info in lottery_data["per_layer"].items():
        n = layer_info["total"]
        n_changed = layer_info["changed"]
        shape = lottery_data["masks"][name].shape

        # Random permutation: keep exactly n_changed positions as True
        flat_mask = torch.zeros(n, dtype=torch.bool)
        if n_changed > 0:
            perm = torch.randperm(n, generator=rng)
            flat_mask[perm[:n_changed]] = True
        masks[name] = flat_mask.reshape(shape)

        total_changed += n_changed
        total_params += n

    density = total_changed / total_params if total_params > 0 else 0

    return {
        "masks": masks,
        "sparsity": 1.0 - density,
        "density": density,
        "total_params": total_params,
        "total_changed": total_changed,
        "source_checkpoint": lottery_data["source_checkpoint"],
        "initial_checkpoint": lottery_data["initial_checkpoint"],
        "mask_type": f"random_seed{seed}",
        "random_seed": seed,
    }


def main():
    parser = argparse.ArgumentParser(description="Extract lottery ticket mask from checkpoint")
    parser.add_argument("--checkpoint", required=True, help="Path to trained checkpoint directory")
    parser.add_argument("--initial", required=True, help="Path to initial (base) checkpoint directory")
    parser.add_argument("--output", required=True, help="Output path for lottery mask .pt file")
    parser.add_argument("--random-seeds", default="", help="Comma-separated seeds for random control masks")
    parser.add_argument("--random-output-prefix", default="", help="Output prefix for random masks (e.g. masks/random_mask)")
    args = parser.parse_args()

    # Extract lottery mask
    lottery_data = extract_mask(args.checkpoint, args.initial)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    # Save with masks as float (True->1.0, False->0.0) for direct gradient multiplication
    save_data = {**lottery_data}
    save_data["masks"] = {name: mask.float() for name, mask in lottery_data["masks"].items()}
    torch.save(save_data, args.output)
    print(f"\nSaved lottery mask to {args.output}")

    # Generate random masks
    if args.random_seeds:
        seeds = [int(s) for s in args.random_seeds.split(",")]
        for seed in seeds:
            random_data = generate_random_mask(lottery_data, seed)
            random_path = f"{args.random_output_prefix}_seed{seed}.pt"

            save_data = {**random_data}
            save_data["masks"] = {name: mask.float() for name, mask in random_data["masks"].items()}
            save_data.pop("per_layer", None)
            torch.save(save_data, random_path)
            print(f"Saved random mask (seed={seed}) to {random_path}")

    # Print per-layer summary for top layers by density
    print(f"\nTop 10 layers by density:")
    sorted_layers = sorted(lottery_data["per_layer"].items(),
                           key=lambda x: x[1]["density"], reverse=True)
    for name, info in sorted_layers[:10]:
        print(f"  {name}: {info['changed']:,}/{info['total']:,} ({info['density']*100:.4f}%)")


if __name__ == "__main__":
    main()
