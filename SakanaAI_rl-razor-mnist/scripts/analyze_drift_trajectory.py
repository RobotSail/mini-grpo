#!/usr/bin/env python3
"""Analyze representational drift trajectories using CKNNA.

For each fine-tuning run (identified by a results.json), loads every saved
checkpoint and computes CKNNA between the base-model representations and the
checkpoint representations.  The resulting trajectories are plotted and
colored by how much catastrophic forgetting occurred.

Usage
-----
python scripts/analyze_drift_trajectory.py \\
    --results-dir experiments/sweep_pretrain_epoch2 \\
    --pretrained-model experiments/pretrain_*/pretrained_model.pt \\
    --output-dir plots/drift_trajectory

The probe dataset (for extracting representations) defaults to FashionMNIST
(old task) to mirror the paper's use of task-neutral probe data; switch to
ParityMNIST with --probe-task new.
"""

import argparse
import os
import sys
import json
import glob

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from typing import Dict, List, Optional, Tuple

from rl_razor.model import MLP
from rl_razor.data import get_parity_mnist, get_fashion_mnist, create_dataloader
from rl_razor.metrics import compute_cknna_from_models
from rl_razor.utils import get_device, set_seed, load_results


# ── Method styling ────────────────────────────────────────────────────────────

METHOD_COLORS = {
    "sft1": "#1f77b4",
    "sft2": "#ff7f0e",
    "oracle": "#2ca02c",
    "grpo": "#d62728",
    "grpo_kl": "#9467bd",
}
METHOD_LABELS = {
    "sft1": "SFT-1",
    "sft2": "SFT-2",
    "oracle": "SFT-Oracle",
    "grpo": "GRPO",
    "grpo_kl": "GRPO+KL",
}


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Representational drift trajectory analysis via CKNNA")

    p.add_argument("--results-dir", required=True,
                   help="Root directory containing results.json files (searched recursively)")
    p.add_argument("--pretrained-model", required=True,
                   help="Path to pretrained base model checkpoint")
    p.add_argument("--pretrain-fashion-acc", type=float, default=None,
                   help="Pretrained FashionMNIST accuracy (used to compute forgetting). "
                        "Computed automatically from --pretrained-model if omitted.")
    p.add_argument("--output-dir", default="plots/drift_trajectory",
                   help="Directory to save plots")
    p.add_argument("--data-dir", default="./data")
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=42)

    # CKNNA options
    p.add_argument("--k", type=int, default=10,
                   help="k-NN neighborhood size for CKNNA")
    p.add_argument("--n-samples", type=int, default=2000,
                   help="Number of probe samples for CKNNA (subsampled for tractability)")
    p.add_argument("--probe-task", choices=["old", "new", "both"], default="old",
                   help="Which task dataset to use as probe data: "
                        "'old' = FashionMNIST (default, most neutral), "
                        "'new' = ParityMNIST, 'both' = concatenate both")
    p.add_argument("--layer-idx", type=int, default=-1,
                   help="Hidden layer index to compare (−1 = last, 0 = first ReLU)")

    # Filtering
    p.add_argument("--methods", nargs="*", default=None,
                   help="Only include these methods (default: all)")
    p.add_argument("--max-runs", type=int, default=None,
                   help="Max runs per method (useful for quick testing)")

    return p.parse_args()


# ── Data helpers ─────────────────────────────────────────────────────────────

def build_probe_loader(probe_task: str, data_dir: str, batch_size: int = 256):
    """Build a DataLoader for the probe dataset."""
    if probe_task == "old":
        ds = get_fashion_mnist(train=False, data_dir=data_dir)
    elif probe_task == "new":
        ds = get_parity_mnist(train=False, data_dir=data_dir)
    else:  # both
        from torch.utils.data import ConcatDataset
        ds = ConcatDataset([
            get_fashion_mnist(train=False, data_dir=data_dir),
            get_parity_mnist(train=False, data_dir=data_dir),
        ])
    return create_dataloader(ds, batch_size=batch_size, shuffle=False)


def find_results_files(root: str) -> List[str]:
    found = []
    for dirpath, _, filenames in os.walk(root):
        if "results.json" in filenames:
            found.append(os.path.join(dirpath, "results.json"))
    return sorted(found)


# ── Trajectory computation ────────────────────────────────────────────────────

def _eval_parity(model: MLP, loader, device: str) -> float:
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            preds = model(x).argmax(dim=-1)
            correct += ((preds % 2) == (y % 2)).sum().item()
            total += x.size(0)
    return correct / total


def _eval_fashion(model: MLP, loader, device: str) -> float:
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            preds = model(x).argmax(dim=-1)
            correct += (preds == y).sum().item()
            total += x.size(0)
    return correct / total


def compute_trajectory(
    results_path: str,
    base_model: MLP,
    probe_loader,
    device: str,
    k: int,
    n_samples: int,
    layer_idx: int,
    pretrain_fashion_acc: Optional[float],
    parity_loader=None,
    fashion_loader=None,
) -> Optional[dict]:
    """Compute CKNNA trajectory for one run.

    Returns a dict with keys:
        method, forgetting, epochs, cknna_scores, parity_accs, fashion_accs
    or None if the run cannot be processed.
    """
    exp_dir = os.path.dirname(results_path)

    try:
        results = load_results(results_path)
    except Exception as e:
        print(f"  [skip] Cannot read {results_path}: {e}")
        return None

    method = results.get("method")
    if method is None:
        return None

    final_fashion_acc = results.get("final_fashion_acc")
    final_parity_acc  = results.get("final_parity_acc")
    if final_fashion_acc is None or final_parity_acc is None:
        return None

    forgetting = (
        (pretrain_fashion_acc - final_fashion_acc)
        if pretrain_fashion_acc is not None
        else (1.0 - final_fashion_acc)
    )

    # Build checkpoint list: [(epoch, path, parity_acc, fashion_acc), ...]
    checkpoints = results.get("checkpoints", [])
    # Add the final model as the last entry
    final_model_path = os.path.join(exp_dir, "finetuned_model.pt")
    total_epochs = results.get("config", {}).get("epochs", None)

    # Determine total epochs from config or infer from checkpoints
    if total_epochs is None and checkpoints:
        total_epochs = max(c.get("epoch", 1) for c in checkpoints)

    entries = []
    for ckpt in checkpoints:
        path = ckpt.get("path")
        epoch = ckpt.get("epoch")
        if path and os.path.exists(path) and epoch is not None:
            entries.append({
                "epoch": epoch,
                "path": path,
                "parity_acc": ckpt.get("parity_acc"),
                "fashion_acc": ckpt.get("fashion_acc"),
            })

    # Add final model if it has a separate path (or same as last checkpoint)
    if os.path.exists(final_model_path):
        last_epoch = total_epochs if total_epochs else (entries[-1]["epoch"] if entries else 1)
        # Avoid duplicating the final epoch checkpoint
        if not entries or entries[-1]["epoch"] != last_epoch:
            entries.append({
                "epoch": last_epoch,
                "path": final_model_path,
                "parity_acc": final_parity_acc,
                "fashion_acc": final_fashion_acc,
            })

    if not entries:
        print(f"  [skip] No checkpoints found for {results_path}")
        return None

    # Compute CKNNA at each checkpoint
    base_parity  = _eval_parity(base_model, parity_loader, device)  if parity_loader  else None
    base_fashion = _eval_fashion(base_model, fashion_loader, device) if fashion_loader else None

    epochs       = [0]    # epoch 0 = base model
    cknna_scores = [1.0]  # CKNNA(base, base) = 1 by definition
    parity_accs  = [base_parity]
    fashion_accs = [base_fashion]

    for entry in sorted(entries, key=lambda e: e["epoch"]):
        try:
            ft_model = MLP.from_checkpoint(entry["path"], device=device)
            score = compute_cknna_from_models(
                base_model, ft_model, probe_loader,
                device=device, k=k, layer_idx=layer_idx, n_samples=n_samples,
            )
            epochs.append(entry["epoch"])
            cknna_scores.append(score)
            parity_accs.append(entry["parity_acc"])
            fashion_accs.append(entry["fashion_acc"])
        except Exception as e:
            print(f"    Warning: could not process checkpoint {entry['path']}: {e}")

    return {
        "method": method,
        "forgetting": forgetting,
        "total_epochs": total_epochs or max(epochs),
        "epochs": epochs,
        "cknna_scores": cknna_scores,
        "parity_accs": parity_accs,
        "fashion_accs": fashion_accs,
        "run_dir": exp_dir,
    }


# ── Plotting ─────────────────────────────────────────────────────────────────

def plot_run_trajectory(
    traj: dict,
    output_path: str,
    pretrain_fashion_acc: Optional[float] = None,
) -> None:
    """Three-panel trajectory figure for a single run.

    Panels (left to right):
      1. CKNNA similarity to base model
      2. New task accuracy (ParityMNIST)
      3. Old task accuracy (FashionMNIST)

    X-axis is the absolute epoch value (float), e.g. 0, 0.5, 1.0, …

    Args:
        traj: Trajectory dict from compute_trajectory.
        output_path: Path to save the figure.
        pretrain_fashion_acc: Baseline fashion accuracy drawn as a reference
            line in the old-task panel.
    """
    epochs       = traj["epochs"]        # list of floats, epoch 0 = base model
    cknna_scores = traj["cknna_scores"]
    parity_accs  = traj["parity_accs"]
    fashion_accs = traj["fashion_accs"]
    method       = traj["method"]
    forgetting   = traj["forgetting"]

    # Filter out None entries for accuracy panels (epoch 0 has no acc values)
    parity_xy  = [(e, a) for e, a in zip(epochs, parity_accs)  if a is not None]
    fashion_xy = [(e, a) for e, a in zip(epochs, fashion_accs) if a is not None]

    fig, ax = plt.subplots(figsize=(6, 4))

    ax.plot(epochs, cknna_scores, linewidth=2, marker="o", markersize=5,
            label="CKNNA", color="#1f77b4")

    if parity_xy:
        xs, ys = zip(*parity_xy)
        ax.plot(xs, ys, linewidth=2, marker="s", markersize=5,
                label="New task acc (parity)", color="#2ca02c")

    if fashion_xy:
        xs, ys = zip(*fashion_xy)
        ax.plot(xs, ys, linewidth=2, marker="^", markersize=5,
                label="Old task acc (fashion)", color="#d62728")

    if pretrain_fashion_acc is not None:
        ax.axhline(pretrain_fashion_acc, color="#d62728", linestyle="--",
                   linewidth=0.8, alpha=0.5, label=f"Pretrained fashion ({pretrain_fashion_acc:.3f})")

    ax.set_xlabel("Epoch")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, loc="best")

    label = METHOD_LABELS.get(method, method)
    ax.set_title(
        f"{label}  |  forgetting={forgetting:.3f}  |  {os.path.basename(traj['run_dir'])}",
        fontsize=9,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    set_seed(args.seed)
    device = args.device or get_device()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load base model
    print(f"Loading pretrained model: {args.pretrained_model}")
    base_model = MLP.from_checkpoint(args.pretrained_model, device=device)
    base_model.eval()

    # Compute baseline FashionMNIST accuracy if not provided
    pretrain_fashion_acc = args.pretrain_fashion_acc
    if pretrain_fashion_acc is None:
        fashion_val = get_fashion_mnist(train=False, data_dir=args.data_dir)
        fashion_loader_eval = create_dataloader(fashion_val, batch_size=256, shuffle=False)
        correct = total = 0
        with torch.no_grad():
            for x, y in fashion_loader_eval:
                x, y = x.to(device), y.to(device)
                preds = base_model(x).argmax(dim=-1)
                correct += (preds == y).sum().item()
                total += x.size(0)
        pretrain_fashion_acc = correct / total
        print(f"Pretrained FashionMNIST accuracy: {pretrain_fashion_acc:.4f}")

    # Build eval loaders (for base-model accuracy at epoch 0)
    parity_val_ds  = get_parity_mnist(train=False, data_dir=args.data_dir)
    fashion_val_ds = get_fashion_mnist(train=False, data_dir=args.data_dir)
    eval_parity_loader  = create_dataloader(parity_val_ds,  batch_size=256, shuffle=False)
    eval_fashion_loader = create_dataloader(fashion_val_ds, batch_size=256, shuffle=False)

    # Build probe loader (fixed probe data for all CKNNA computations)
    print(f"Probe task: {args.probe_task}, k={args.k}, n_samples={args.n_samples}")
    probe_loader = build_probe_loader(args.probe_task, args.data_dir, batch_size=256)

    # Find all results.json files
    results_files = find_results_files(args.results_dir)
    print(f"\nFound {len(results_files)} results.json files under {args.results_dir}")

    # Filter by method if requested
    if args.methods:
        valid_methods = set(args.methods)

    # Compute trajectories
    trajectories = []
    method_counts: Dict[str, int] = {}

    for results_path in results_files:
        # Peek at method before loading checkpoints
        try:
            with open(results_path) as f:
                peek = json.load(f)
            method = peek.get("method")
        except Exception:
            continue

        if method is None:
            continue
        if args.methods and method not in valid_methods:
            continue
        if args.max_runs is not None:
            if method_counts.get(method, 0) >= args.max_runs:
                continue

        print(f"Processing [{method}] {results_path}")
        traj = compute_trajectory(
            results_path=results_path,
            base_model=base_model,
            probe_loader=probe_loader,
            device=device,
            k=args.k,
            n_samples=args.n_samples,
            layer_idx=args.layer_idx,
            pretrain_fashion_acc=pretrain_fashion_acc,
            parity_loader=eval_parity_loader,
            fashion_loader=eval_fashion_loader,
        )
        if traj is not None:
            trajectories.append(traj)
            method_counts[method] = method_counts.get(method, 0) + 1

    if not trajectories:
        print("No trajectories computed. Check --results-dir and checkpoint paths.")
        return

    print(f"\nComputed {len(trajectories)} trajectories across {len(method_counts)} methods.")
    for method, count in sorted(method_counts.items()):
        ft = [t["forgetting"] for t in trajectories if t["method"] == method]
        ck = [t["cknna_scores"][-1] for t in trajectories if t["method"] == method]
        print(f"  {METHOD_LABELS.get(method, method):15s}: {count:3d} runs  "
              f"forgetting={np.mean(ft):.3f}±{np.std(ft):.3f}  "
              f"final_CKNNA={np.mean(ck):.3f}±{np.std(ck):.3f}")

    # ── Generate plots (one figure per run) ───────────────────────────────────
    print("\nGenerating plots...")
    for traj in trajectories:
        run_name = os.path.basename(traj["run_dir"])
        out_path = os.path.join(args.output_dir, f"{traj['method']}_{run_name}.png")
        plot_run_trajectory(traj, out_path, pretrain_fashion_acc)
    print(f"  Saved {len(trajectories)} per-run figures.")

    # Save trajectory data as JSON for further analysis
    traj_data = []
    for t in trajectories:
        traj_data.append({
            "method":       t["method"],
            "run_dir":      t["run_dir"],
            "forgetting":   t["forgetting"],
            "total_epochs": t["total_epochs"],
            "epochs":       t["epochs"],
            "cknna_scores": t["cknna_scores"],
            "parity_accs":  t["parity_accs"],
            "fashion_accs": t["fashion_accs"],
        })
    out_json = os.path.join(args.output_dir, "trajectories.json")
    with open(out_json, "w") as f:
        json.dump(traj_data, f, indent=2)
    print(f"Saved trajectory data: {out_json}")

    print(f"\nAll outputs saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
