#!/usr/bin/env python3
"""
Paper figures plotted directly from eval data.

Figure 1: Accuracy vs KL trajectory + Accuracy vs Tokens
Figure 2: Path efficiency + L0 sparsity (from cached data)

Usage:
    python plot_paper_figures.py
"""

import json
import os
import re
from pathlib import Path

import matplotlib
matplotlib.rcParams['text.usetex'] = False
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.image as mpimg
from matplotlib.lines import Line2D


DATA_DIR = Path("tmp-scripts/eval-data")
OUTPUT_DIR = "plots"

BASELINE_TEST_ACC = 0.0432

RUNS = {
    "AdamW + BF16":  {"dir": "grpo-adamw-bf16",      "color": "#6baed6", "linestyle": "-",  "marker": "o"},
    "AdamW + Mixed": {"dir": "grpo-adamw-mixed-600k", "color": "#08519c", "linestyle": "--", "marker": "o"},
    "Muon + BF16":   {"dir": "grpo-muon-bf16",        "color": "#e53935", "linestyle": "-",  "marker": "o"},
    "Muon + Mixed":  {"dir": "grpo-muon-mixed",       "color": "#7f0000", "linestyle": "--", "marker": "o"},
}


def _load_accuracy_kl(path):
    with open(path) as f:
        data = json.load(f)
    rows = []
    for key, entry in data.items():
        if "initial" in key:
            continue
        m = re.search(r"checkpoint-(\d+)", key)
        tokens = int(entry.get("tokens", m.group(1) if m else 0))
        rows.append({"tokens": tokens, "accuracy": entry["accuracy"],
                      "forward_kl": entry["forward_kl"]})
    rows.sort(key=lambda r: r["tokens"])
    return rows


def _truncate(rows, threshold=0.10):
    if len(rows) <= 2:
        return rows
    peak = max(r["accuracy"] for r in rows)
    cutoff = peak * (1 - threshold)
    peak_seen = False
    for i, r in enumerate(rows):
        if r["accuracy"] >= peak - 1e-9:
            peak_seen = True
        if peak_seen and r["accuracy"] < cutoff:
            return rows[:max(i, 2)]
    return rows


def compose_figure1():
    """Plot accuracy vs KL and accuracy vs tokens from eval data."""
    val_data, test_data = {}, {}
    for label, cfg in RUNS.items():
        d = DATA_DIR / cfg["dir"]
        val_data[label] = _load_accuracy_kl(d / "validation_accuracy_kl_auto.json")
        test_path = d / "test_accuracy_kl_auto.json"
        if test_path.exists():
            test_data[label] = _load_accuracy_kl(test_path)

    # Best validation checkpoint per run
    best_tokens = {}
    for label in RUNS:
        best_val = max(val_data[label], key=lambda r: r["accuracy"])
        best_tokens[label] = best_val["tokens"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

    # Title and subtitle with explicit vertical positioning
    fig.text(0.5, 0.97,
             "For both optimizers, BF16 achieves far higher accuracy per unit of KL,\n"
             "but requires 10x more training steps.",
             ha="center", va="top", fontsize=16, fontweight="bold",
             transform=fig.transFigure)
    fig.text(0.5, 0.89,
             "GSM8K test set  \u00b7  bfloat16 inference  \u00b7  greedy decoding",
             ha="center", va="top", fontsize=11, color="#555555",
             transform=fig.transFigure)

    # ── Left: Accuracy vs Forward KL ──
    ax1.scatter(0.0, BASELINE_TEST_ACC, c="#7f7f7f", marker="o", s=120,
                label="Baseline", edgecolors="black", linewidths=1, zorder=3)

    for label, cfg in RUNS.items():
        if label not in test_data:
            continue
        rows = _truncate(test_data[label])
        kls = [r["forward_kl"] for r in rows]
        accs = [r["accuracy"] for r in rows]
        ax1.plot(kls, accs, color=cfg["color"], linestyle=cfg["linestyle"],
                 linewidth=1.5, alpha=0.8, label=label, zorder=2)
        ax1.scatter(kls, accs, color=cfg["color"], marker=cfg["marker"],
                    s=20, alpha=0.6, edgecolors="none", zorder=3)
        best_row = min(rows, key=lambda r: abs(r["tokens"] - best_tokens[label]))
        ax1.scatter(best_row["forward_kl"], best_row["accuracy"],
                    color=cfg["color"], marker="*", s=300,
                    edgecolors="black", linewidths=1, zorder=5)

    ax1.set_xlabel(r"Forward KL: $D_{\mathrm{KL}}(\pi_0 \| \pi)$", fontsize=12)
    ax1.set_ylabel("GSM8K Test Accuracy", fontsize=12)
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y:.0%}"))
    handles, leg = ax1.get_legend_handles_labels()
    handles.append(Line2D([0], [0], marker="*", color="w", markerfacecolor="black",
                   markersize=12, markeredgecolor="black", markeredgewidth=0.8))
    leg.append("Checkpoint selected by val.")
    ax1.legend(handles=handles, labels=leg, loc="lower right", fontsize=9)
    ax1.grid(True, alpha=0.3, zorder=1)

    # ── Right: Accuracy vs Tokens ──
    ax2.axhline(BASELINE_TEST_ACC, color="#7f7f7f", linestyle=":", linewidth=1,
                alpha=0.7, label="Baseline")

    for label, cfg in RUNS.items():
        if label not in test_data:
            continue
        rows = test_data[label]
        tokens_m = [r["tokens"] / 1e6 for r in rows]
        accs = [r["accuracy"] for r in rows]
        ax2.plot(tokens_m, accs, color=cfg["color"], linestyle=cfg["linestyle"],
                 linewidth=1.5, alpha=0.85, label=label)
        ax2.scatter(tokens_m, accs, color=cfg["color"], marker=cfg["marker"],
                    s=15, alpha=0.5, edgecolors="none", zorder=3)
        best_row = min(rows, key=lambda r: abs(r["tokens"] - best_tokens[label]))
        ax2.scatter(best_row["tokens"] / 1e6, best_row["accuracy"],
                    color=cfg["color"], marker="*", s=300,
                    edgecolors="black", linewidths=1, zorder=5)

    ax2.set_xlabel("Tokens Backpropagated (M)", fontsize=12)
    ax2.set_ylabel("GSM8K Test Accuracy", fontsize=12)
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y:.0%}"))
    handles, leg = ax2.get_legend_handles_labels()
    handles.append(Line2D([0], [0], marker="*", color="w", markerfacecolor="black",
                   markersize=12, markeredgecolor="black", markeredgewidth=0.8))
    leg.append("Checkpoint selected by val.")
    ax2.legend(handles=handles, labels=leg, loc="lower right", fontsize=9)
    ax2.grid(True, alpha=0.3, zorder=1)

    plt.tight_layout(rect=[0, 0, 1, 0.84])

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUTPUT_DIR, f"paper_figure1_kl_tokens.{ext}"),
                    dpi=150, bbox_inches="tight")
    print("Saved Figure 1")
    plt.close()


def compose_figure2():
    """Use path efficiency + sparsity plot (already generated)."""
    img = mpimg.imread(os.path.join(OUTPUT_DIR, "path_efficiency_and_sparsity.png"))

    fig, ax = plt.subplots(figsize=(18, 6))

    fig.suptitle(
        "BF16's near-zero path efficiency and 99%+ sparse updates are signatures\n"
        "of precision-induced truncation, not selective optimization.",
        fontsize=17, fontweight="bold", y=1.02,
    )

    ax.imshow(img)
    ax.axis("off")

    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUTPUT_DIR, f"paper_figure2_path_sparsity.{ext}"),
                    dpi=200, bbox_inches="tight")
    print("Saved Figure 2")
    plt.close()


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    compose_figure1()
    compose_figure2()


if __name__ == "__main__":
    main()
