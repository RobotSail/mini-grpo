#!/usr/bin/env python3
"""Plot results from local experiment results.json files."""

import argparse
import glob
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

from rl_razor.utils import load_results, save_results


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

ALTERNATIVE_METRICS: List[Tuple[str, str]] = [
    ("forward_kl_new", "KL, forward (new task)"),
    ("forward_kl_old", "KL, forward (old task)"),
    ("reverse_kl_new", "KL, reverse (new task)"),
    ("reverse_kl_old", "KL, reverse (old task)"),
    ("total_variation_new", "TV (new task)"),
    ("total_variation_old", "TV (old task)"),
    ("distribution_l2_new", "Distribution L2 (new task)"),
    ("distribution_l2_old", "Distribution L2 (old task)"),
    ("weight_l1", "Weight change, L1"),
    ("weight_fisher_l2_new", "Weight Fisher L2 (new task)"),
    ("weight_fisher_l2_old", "Weight Fisher L2 (old task)"),
    ("weight_spectral_norm", "Weight change, spectral norm"),
    ("activation_l1_new", "Activation L1 (new task)"),
    ("activation_l1_old", "Activation L1 (old task)"),
    ("activation_l2_new", "Activation L2 (new task)"),
    ("activation_l2_old", "Activation L2 (old task)"),
]


def parse_args():
    parser = argparse.ArgumentParser(description="Plot results from local results.json files")
    parser.add_argument("--results-dir", type=str, required=True, help="Directory tree containing results.json files")
    parser.add_argument(
        "--output-dir", type=str, default=None, help="Output directory for plots (default: <results-dir>/plots)"
    )
    parser.add_argument(
        "--pretrained-results",
        type=str,
        default=None,
        help="Path to the pretrain results.json; its final_val_fashion_acc is used as the forgetting baseline",
    )
    return parser.parse_args()


def load_pretrain_fashion_acc(results_path: str) -> Optional[float]:
    """Read final_val_fashion_acc from a pretrain results.json."""
    with open(results_path) as f:
        data = json.load(f)
    acc = data.get("final_val_fashion_acc")
    if acc is None:
        print(f"Warning: 'final_val_fashion_acc' not found in {results_path}")
    return acc


def load_all_results(results_dir: str) -> Dict[str, List[dict]]:
    """Collect every data point (final + checkpoints) from all results.json files."""
    all_data: Dict[str, List[dict]] = {}

    for result_file in glob.glob(os.path.join(results_dir, "**/results.json"), recursive=True):
        try:
            results = load_results(result_file)
        except Exception as e:
            print(f"Warning: could not load {result_file}: {e}")
            continue

        method = results.get("method", "unknown")
        if method not in all_data:
            all_data[method] = []

        # Checkpoints
        for ckpt in results.get("checkpoints", []):
            if "parity_acc" in ckpt and "kl_divergence" in ckpt:
                point = {
                    "parity_accuracy": ckpt["parity_acc"],
                    "fashion_accuracy": ckpt.get("fashion_acc", 0.0),
                    "forward_kl": ckpt["kl_divergence"],
                }
                for key, _ in ALTERNATIVE_METRICS:
                    if key in ckpt:
                        point[key] = ckpt[key]
                all_data[method].append(point)

    return all_data


def extract_pareto_frontier(points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """Return Pareto-optimal (parity_acc, fashion_acc) points (maximize both)."""
    frontier, best_y = [], float("-inf")
    for x, y in sorted(points, key=lambda p: p[0], reverse=True):
        if y >= best_y:
            frontier.append((x, y))
            best_y = y
    return frontier


def _poly_fit(X: np.ndarray, y: np.ndarray, degree: int = 2) -> Pipeline:
    pipe = Pipeline(
        [
            ("poly", PolynomialFeatures(degree=degree, include_bias=True)),
            ("lin", LinearRegression()),
        ]
    )
    pipe.fit(X.reshape(-1, 1), y)
    return pipe


def plot_combined(
    all_data: Dict[str, List[dict]],
    output_path: str,
    pretrain_fashion_acc: Optional[float],
) -> float:
    """Three-panel figure reproducing Figure 3 of the paper. Returns R²."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # ── Panel 1: Learning vs Forgetting ─────────────────────────────────────
    ax = axes[0]
    for method, data in all_data.items():
        color = METHOD_COLORS.get(method, "gray")
        label = METHOD_LABELS.get(method, method)
        xs = [d["parity_accuracy"] for d in data]
        ys = [d["fashion_accuracy"] for d in data]
        ax.scatter(xs, ys, c=color, alpha=0.3, s=20)
        frontier = extract_pareto_frontier(list(zip(xs, ys)))
        if frontier:
            fx, fy = zip(*sorted(frontier))
            ax.plot(fx, fy, c=color, linewidth=2, label=label, marker="o", markersize=4)
    if pretrain_fashion_acc is not None:
        ax.axhline(pretrain_fashion_acc, color="gray", linestyle="--", alpha=0.7)
    ax.set_xlabel("ParityMNIST Accuracy")
    ax.set_ylabel("FashionMNIST Accuracy")
    ax.set_title("Learning vs Forgetting")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── Panel 2: KL vs Forgetting ────────────────────────────────────────────
    ax = axes[1]
    all_kl, all_forg = [], []
    for method, data in all_data.items():
        kls = [d["forward_kl"] for d in data]
        forg = [
            (pretrain_fashion_acc - d["fashion_accuracy"])
            if pretrain_fashion_acc is not None
            else (1.0 - d["fashion_accuracy"])
            for d in data
        ]
        all_kl.extend(kls)
        all_forg.extend(forg)
        ax.scatter(kls, forg, c=METHOD_COLORS.get(method, "gray"), alpha=0.5, s=30)

    r2 = 0.0
    if len(all_kl) > 2:
        pipe = _poly_fit(np.array(all_kl), np.array(all_forg))
        r2 = pipe.score(np.array(all_kl).reshape(-1, 1), np.array(all_forg))
        x_line = np.linspace(min(all_kl), max(all_kl), 100)
        ax.plot(x_line, pipe.predict(x_line.reshape(-1, 1)), "k--", linewidth=2, label=f"R²={r2:.2f}")
        ax.legend()
    ax.set_xlabel("KL Divergence")
    ax.set_ylabel("Forgetting")
    ax.set_title(f"KL Predicts Forgetting (R²={r2:.2f})")
    ax.grid(True, alpha=0.3)

    # ── Panel 3: KL vs Learning ──────────────────────────────────────────────
    ax = axes[2]
    for method, data in all_data.items():
        ax.scatter(
            [d["forward_kl"] for d in data],
            [d["parity_accuracy"] for d in data],
            c=METHOD_COLORS.get(method, "gray"),
            alpha=0.5,
            s=30,
            label=METHOD_LABELS.get(method, method),
        )
    ax.set_xlabel("KL Divergence")
    ax.set_ylabel("ParityMNIST Accuracy")
    ax.set_title("KL vs Learning")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")
    return r2


def compute_and_print_table1(
    all_data: Dict[str, List[dict]],
    pretrain_fashion_acc: Optional[float],
    output_path: str,
) -> List[dict]:
    """Fit degree-2 polynomial of each metric vs forgetting; print and plot Table 1."""
    forgetting_list, metric_lists = [], {k: [] for k, _ in ALTERNATIVE_METRICS}

    for data in all_data.values():
        for d in data:
            if "fashion_accuracy" not in d:
                continue
            forg = (
                pretrain_fashion_acc - d["fashion_accuracy"]
                if pretrain_fashion_acc is not None
                else 1.0 - d["fashion_accuracy"]
            )
            forgetting_list.append(forg)
            for key, _ in ALTERNATIVE_METRICS:
                metric_lists[key].append(d.get(key, np.nan))

    forgetting = np.array(forgetting_list)
    rows = []
    for key, label in ALTERNATIVE_METRICS:
        x = np.array(metric_lists[key], dtype=float)
        valid = ~np.isnan(x)
        n = int(valid.sum())
        if n < 3:
            rows.append({"metric": label, "key": key, "r2": None, "n": n})
            continue
        pipe = _poly_fit(x[valid], forgetting[valid])
        rows.append({"metric": label, "key": key, "r2": pipe.score(x[valid].reshape(-1, 1), forgetting[valid]), "n": n})

    rows.sort(key=lambda r: (r["r2"] is None, -(r["r2"] or 0.0)))

    # Print
    W = 34
    print("\n" + "─" * (W + 22))
    print(f"{'Variable':<{W}}  R² (2nd deg. polynomial)")
    print("─" * (W + 22))
    for r in rows:
        r2_str = f"{r['r2']:.2f}  (n={r['n']})" if r["r2"] is not None else f"N/A  (n={r['n']})"
        print(f"{r['metric']:<{W}}  {r2_str}")
    print("─" * (W + 22))

    # Bar chart
    labels = [r["metric"] for r in rows]
    r2_vals = [r["r2"] if r["r2"] is not None else 0.0 for r in rows]
    colors = ["#2ca02c" if r["key"].startswith("forward_kl") else "#1f77b4" for r in rows]

    fig, ax = plt.subplots(figsize=(8, max(3, len(rows) * 0.55 + 1.5)))
    bars = ax.barh(labels[::-1], r2_vals[::-1], color=colors[::-1])
    for bar, val in zip(bars, r2_vals[::-1]):
        if val > 0:
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2, f"{val:.2f}", va="center", fontsize=9)
    ax.set_xlabel("R² (2nd degree polynomial fit vs forgetting)")
    ax.set_title("Predictive Power of Alternative Variables (Table 1)")
    ax.set_xlim(0, 1.05)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")

    return rows


def main():
    args = parse_args()
    output_dir = args.output_dir or os.path.join(args.results_dir, "plots")
    os.makedirs(output_dir, exist_ok=True)

    pretrain_fashion_acc: Optional[float] = None
    if args.pretrained_results:
        pretrain_fashion_acc = load_pretrain_fashion_acc(args.pretrained_results)
        if pretrain_fashion_acc is not None:
            print(f"Pretrain FashionMNIST accuracy: {pretrain_fashion_acc:.4f}")
    else:
        print("No --pretrained-results provided; forgetting computed as 1 - fashion_acc")

    print(f"Loading results from: {args.results_dir}")
    all_data = load_all_results(args.results_dir)

    if not all_data:
        print("No fine-tuning results found.")
        return

    # Summary
    print("\nResults summary:")
    for method, data in all_data.items():
        parity = [d["parity_accuracy"] for d in data]
        fashion = [d["fashion_accuracy"] for d in data]
        kl = [d["forward_kl"] for d in data]
        print(
            f"  {METHOD_LABELS.get(method, method)}: {len(data)} points  "
            f"parity={np.mean(parity):.3f}  fashion={np.mean(fashion):.3f}  kl={np.mean(kl):.3f}"
        )

    r2 = plot_combined(
        all_data,
        os.path.join(output_dir, "figure3.png"),
        pretrain_fashion_acc,
    )
    print(f"\nKL-Forgetting R² = {r2:.4f}")

    rows = compute_and_print_table1(
        all_data,
        pretrain_fashion_acc,
        os.path.join(output_dir, "table1.png"),
    )

    save_results(
        {
            "pretrain_fashion_acc": pretrain_fashion_acc,
            "kl_forgetting_r2": r2,
            "predictive_power": [{"metric": r["metric"], "r2": r["r2"], "n": r["n"]} for r in rows],
        },
        os.path.join(output_dir, "summary.json"),
    )
    print(f"\nPlots saved to: {output_dir}")


if __name__ == "__main__":
    main()
