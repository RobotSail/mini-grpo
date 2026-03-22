#!/usr/bin/env python3
"""
Plot comparison of GRPO runs using dtype=auto evaluation results.

Reads *_auto.json files. Excludes runs that don't have auto-dtype evals.

Style convention:
  - AdamW = blue, Muon = red  (optimizer determines color)
  - BF16 = solid line, Mixed = dashed line  (precision determines linestyle)

Produces:
  1a. Scatter: Best-checkpoint test accuracy vs forward KL
  2a. Bar charts: Frobenius norm, L0 sparsity, k90
  3a. Validation accuracy vs forward KL trajectory (all checkpoints, with markers)
  4a. Test accuracy vs forward KL trajectory (all checkpoints, with markers)
  5a. Frobenius norm trajectory over training (from update_norms.jsonl)
"""

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ── Paths ────────────────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).parent / "eval-data"
OUT_DIR = Path(__file__).parent / "plots-auto"
OUT_DIR.mkdir(exist_ok=True)

# Style: color encodes both optimizer AND precision
# AdamW = blue family, Muon = red family
# BF16 = lighter shade, Mixed = darker shade
RUNS = {
    "AdamW + BF16": {
        "dir": "grpo-adamw-bf16",
        "color": "#6baed6",  # light blue (BF16)
        "marker": "o",
        "linestyle": "-",
    },
    "AdamW + Mixed": {
        "dir": "grpo-adamw-mixed-600k",
        "color": "#08519c",  # dark blue (Mixed)
        "marker": "o",
        "linestyle": "--",
    },
    "Muon + BF16": {
        "dir": "grpo-muon-bf16",
        "color": "#e53935",  # light red (BF16)
        "marker": "o",
        "linestyle": "-",
    },
    "Muon + Mixed": {
        "dir": "grpo-muon-mixed",
        "color": "#7f0000",  # dark red (Mixed)
        "marker": "o",
        "linestyle": "--",
    },
}

TITLE_SUFFIX = ""
EVAL_NOTE = "Eval: bfloat16, greedy decoding"


# ── Helpers ──────────────────────────────────────────────────────────────────
def extract_tokens(key: str, entry: dict) -> int:
    if "tokens" in entry:
        return int(entry["tokens"])
    m = re.search(r"checkpoint-(\d+)", key)
    if m:
        return int(m.group(1))
    raise ValueError(f"Cannot determine token count for key: {key}")


def load_accuracy_kl(path: str) -> list[dict]:
    with open(path) as f:
        data = json.load(f)
    rows = []
    for key, entry in data.items():
        if "initial" in key:
            continue
        tokens = extract_tokens(key, entry)
        rows.append({
            "key": key,
            "tokens": tokens,
            "accuracy": entry["accuracy"],
            "forward_kl": entry["forward_kl"],
        })
    rows.sort(key=lambda r: r["tokens"])
    return rows


def load_geometry(path: str) -> dict:
    with open(path) as f:
        data = json.load(f)
    return {
        "frobenius_norm": data["frobenius_norm"],
        "k90": data.get("avg_k90", data.get("k90_average")),
        "l0_sparsity": data.get("l0_sparsity", data.get("l0_sparsity_1e5")),
    }


def is_2d_weight(param_name: str) -> bool:
    if any(s in param_name for s in [".bias", "layernorm", "layer_norm", "ln_"]):
        return False
    if "embed" in param_name:
        return False
    if "lm_head" in param_name:
        return False
    return ".weight" in param_name


def truncate_after_divergence(rows: list[dict], threshold: float = 0.10) -> list[dict]:
    """Truncate trajectory after accuracy drops more than `threshold` below the peak.
    Always keeps at least 2 points so the line is visible."""
    if len(rows) <= 2:
        return rows
    peak_acc = max(r["accuracy"] for r in rows)
    cutoff = peak_acc * (1 - threshold)
    peak_seen = False
    for i, r in enumerate(rows):
        if r["accuracy"] >= peak_acc - 1e-9:
            peak_seen = True
        if peak_seen and r["accuracy"] < cutoff:
            return rows[:max(i, 2)]
    return rows


def load_update_norms(path: str) -> tuple[list[int], list[float]]:
    tokens_list, norms_list = [], []
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            matrix_norms = [v for k, v in rec["norms"].items() if is_2d_weight(k)]
            if matrix_norms:
                tokens_list.append(rec["tokens"])
                norms_list.append(float(np.mean(matrix_norms)))
    return tokens_list, norms_list


def plot_trajectory(ax, rows, cfg, label, best_tokens=None):
    """Plot a trajectory line with checkpoint markers and a highlighted best checkpoint."""
    kls = [r["forward_kl"] for r in rows]
    accs = [r["accuracy"] for r in rows]

    # Line
    ax.plot(
        kls, accs,
        color=cfg["color"], linestyle=cfg["linestyle"],
        linewidth=1.5, alpha=0.8, label=label, zorder=2,
    )
    # Small markers at every checkpoint
    ax.scatter(
        kls, accs,
        color=cfg["color"], marker=cfg["marker"],
        s=20, alpha=0.6, edgecolors="none", zorder=3,
    )
    # Best checkpoint: large star
    if best_tokens is not None:
        best_row = min(rows, key=lambda r: abs(r["tokens"] - best_tokens))
        ax.scatter(
            best_row["forward_kl"], best_row["accuracy"],
            color=cfg["color"], marker="*", s=300,
            edgecolors="black", linewidths=1, zorder=5,
        )


# ── Load all data (using _auto.json files) ───────────────────────────────────
val_data = {}
test_data = {}
geo_data = {}
norm_data = {}

for label, cfg in RUNS.items():
    d = DATA_DIR / cfg["dir"]
    val_data[label] = load_accuracy_kl(d / "validation_accuracy_kl_auto.json")
    test_path = d / "test_accuracy_kl_auto.json"
    if test_path.exists():
        test_data[label] = load_accuracy_kl(test_path)
    geo_path = d / "geometry_metrics.json"
    if geo_path.exists():
        geo_data[label] = load_geometry(geo_path)
    norms_path = d / "update_norms.jsonl"
    if norms_path.exists():
        norm_data[label] = load_update_norms(norms_path)

# Find best validation checkpoint per run -> get its test scores
best_test = {}
for label in RUNS:
    best_val_row = max(val_data[label], key=lambda r: r["accuracy"])
    best_tokens = best_val_row["tokens"]
    entry = {
        "val_accuracy": best_val_row["accuracy"],
        "val_kl": best_val_row["forward_kl"],
        "tokens": best_tokens,
    }
    if label in test_data:
        match = min(test_data[label], key=lambda r: abs(r["tokens"] - best_tokens))
        entry["test_accuracy"] = match["accuracy"]
        entry["test_kl"] = match["forward_kl"]
        print(f"{label}: best val ckpt @ {best_tokens:,} tokens -> "
              f"val acc={best_val_row['accuracy']:.4f}, "
              f"test acc={match['accuracy']:.4f}, test KL={match['forward_kl']:.6f}")
    else:
        print(f"{label}: best val ckpt @ {best_tokens:,} tokens -> "
              f"val acc={best_val_row['accuracy']:.4f}, (test data pending)")
    best_test[label] = entry


# ── Plot 1a: Best-checkpoint scatter ─────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 6))

# Baseline: Qwen2-1.5B-Instruct (no finetuning)
ax.scatter(
    0.0, 0.0432,
    c="#7f7f7f", marker="*", s=250,
    label="Baseline (Qwen2-1.5B-Instruct)",
    edgecolors="black", linewidths=1, zorder=3,
)
ax.annotate(
    "Baseline",
    (0.0, 0.0432),
    textcoords="offset points", xytext=(12, 6),
    fontsize=9, fontweight="bold", color="#7f7f7f",
)

for label, cfg in RUNS.items():
    bt = best_test[label]
    if "test_kl" not in bt:
        continue
    ax.scatter(
        bt["test_kl"], bt["test_accuracy"],
        c=cfg["color"], marker=cfg["marker"],
        s=200, label=label,
        edgecolors="black", linewidths=1, zorder=3,
    )
    ax.annotate(
        label,
        (bt["test_kl"], bt["test_accuracy"]),
        textcoords="offset points", xytext=(12, 6),
        fontsize=9, fontweight="bold", color=cfg["color"],
    )

ax.set_xlabel(r"Forward KL: $D_{\mathrm{KL}}(\pi_0 \| \pi)$", fontsize=12)
ax.set_ylabel("GSM8K Test Accuracy", fontsize=12)
ax.set_title("Best Checkpoint: Test Accuracy vs Forward KL" + TITLE_SUFFIX, fontsize=13)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y:.0%}"))
ax.set_ylim(0, None)
ax.legend(loc="best", fontsize=10)
ax.grid(True, alpha=0.3, zorder=1)
plt.tight_layout()
fig.savefig(OUT_DIR / "1a_best_checkpoint_accuracy_vs_kl.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {OUT_DIR / '1a_best_checkpoint_accuracy_vs_kl.png'}")


# ── Plot 2a: Geometry bar charts ─────────────────────────────────────────────
labels_with_geo = [l for l in RUNS if l in geo_data]
if labels_with_geo:
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    colors = [RUNS[l]["color"] for l in labels_with_geo]

    metrics = [
        ("frobenius_norm", r"$\|\Delta W\|_F$", "Frobenius Norm"),
        ("l0_sparsity", r"$1 - \|\Delta W\|_0 / n$", r"$L_0$ Sparsity"),
        ("k90", r"$k_{90}$ (90% SV energy)", "Effective Rank $k_{90}$"),
    ]

    for ax, (key, ylabel, title) in zip(axes, metrics):
        values = [geo_data[l][key] for l in labels_with_geo]
        # L0 sparsity: stored as fraction of non-zero elements, invert to get sparsity
        if key == "l0_sparsity":
            values = [1.0 - v for v in values]
        bars = ax.bar(labels_with_geo, values, color=colors, edgecolor="black", linewidth=0.8)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=12)
        ax.grid(axis="y", alpha=0.3)
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{val:.4f}" if val < 1 else f"{val:.1f}",
                ha="center", va="bottom", fontsize=9, fontweight="bold",
            )
        ax.tick_params(axis="x", rotation=15)

    plt.suptitle("Geometry Metrics of Best Checkpoints" + TITLE_SUFFIX, fontsize=14, y=1.02)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "2a_geometry_bar_charts.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {OUT_DIR / '2a_geometry_bar_charts.png'}")


# ── Plot 3a: Validation trajectory ───────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 7))

ax.scatter(0.0, 0.0742, c="#7f7f7f", marker="*", s=250,
           label="Baseline", edgecolors="black", linewidths=1, zorder=3)

for label, cfg in RUNS.items():
    rows = truncate_after_divergence(val_data[label])
    plot_trajectory(ax, rows, cfg, label, best_tokens=best_test[label]["tokens"])

ax.set_xlabel(r"Forward KL: $D_{\mathrm{KL}}(\pi_0 \| \pi)$", fontsize=12)
ax.set_ylabel("GSM8K Validation Accuracy", fontsize=12)
ax.set_title("Training Trajectory: Validation Accuracy vs Forward KL" + TITLE_SUFFIX, fontsize=13)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y:.0%}"))
ax.legend(loc="best", fontsize=10)
ax.grid(True, alpha=0.3, zorder=1)
plt.tight_layout()
fig.savefig(OUT_DIR / "3a_val_accuracy_vs_kl_trajectory.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {OUT_DIR / '3a_val_accuracy_vs_kl_trajectory.png'}")


# ── Plot 4a: Test trajectory ─────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 7))

ax.scatter(0.0, 0.0432, c="#7f7f7f", marker="o", s=120,
           label="Baseline", edgecolors="black", linewidths=1, zorder=3)

for label, cfg in RUNS.items():
    if label not in test_data:
        continue
    rows = truncate_after_divergence(test_data[label])
    plot_trajectory(ax, rows, cfg, label, best_tokens=best_test[label]["tokens"])

ax.set_xlabel(r"Forward KL: $D_{\mathrm{KL}}(\pi_0 \| \pi)$", fontsize=12)
ax.set_ylabel("GSM8K Test Accuracy", fontsize=12)
ax.set_title("Training Trajectory: Test Accuracy vs Forward KL", fontsize=13)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y:.0%}"))
from matplotlib.lines import Line2D
handles, labels = ax.get_legend_handles_labels()
handles.append(Line2D([0], [0], marker="*", color="w", markerfacecolor="black",
               markersize=12, markeredgecolor="black", markeredgewidth=0.8))
labels.append("Best val. accuracy")
ax.legend(handles=handles, labels=labels, loc="lower right", fontsize=10)
ax.grid(True, alpha=0.3, zorder=1)
plt.tight_layout()
fig.savefig(OUT_DIR / "4a_test_accuracy_vs_kl_trajectory.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {OUT_DIR / '4a_test_accuracy_vs_kl_trajectory.png'}")


# ── Plot 5a: Frobenius norm trajectory ───────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))

for label, cfg in RUNS.items():
    if label not in norm_data:
        continue
    tokens_arr, norms_arr = norm_data[label]
    tokens_m = [t / 1e6 for t in tokens_arr]
    ax.plot(
        tokens_m, norms_arr,
        color=cfg["color"], linestyle=cfg["linestyle"],
        linewidth=1.2, alpha=0.8, label=label,
    )
    # Mark best checkpoint
    best_tok = best_test[label]["tokens"]
    closest_idx = min(range(len(tokens_arr)), key=lambda i: abs(tokens_arr[i] - best_tok))
    ax.scatter(
        tokens_m[closest_idx], norms_arr[closest_idx],
        color=cfg["color"], marker="*", s=200,
        edgecolors="black", linewidths=0.8, zorder=5,
    )

ax.set_xlabel("Tokens Trained (M)", fontsize=12)
ax.set_ylabel(r"Average $\|\Delta W_t\|_F$ (2D matrices)", fontsize=12)
ax.set_title("Update Norm Trajectory Over Training" + TITLE_SUFFIX, fontsize=13)
ax.legend(loc="best", fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
fig.savefig(OUT_DIR / "5a_update_norm_trajectory.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {OUT_DIR / '5a_update_norm_trajectory.png'}")

# ── Plot 6a: Validation accuracy over tokens ─────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))

ax.axhline(0.0742, color="#7f7f7f", linestyle=":", linewidth=1, alpha=0.7, label="Baseline")

for label, cfg in RUNS.items():
    rows = val_data[label]
    tokens_m = [r["tokens"] / 1e6 for r in rows]
    accs = [r["accuracy"] for r in rows]
    ax.plot(tokens_m, accs, color=cfg["color"], linestyle=cfg["linestyle"],
            linewidth=1.5, alpha=0.85, label=label)
    ax.scatter(tokens_m, accs, color=cfg["color"], marker=cfg["marker"],
               s=15, alpha=0.5, edgecolors="none", zorder=3)
    # Mark best checkpoint
    best_tok = best_test[label]["tokens"]
    best_row = min(rows, key=lambda r: abs(r["tokens"] - best_tok))
    ax.scatter(best_tok / 1e6, best_row["accuracy"], color=cfg["color"],
               marker="*", s=300, edgecolors="black", linewidths=1, zorder=5)

ax.set_xlabel("Tokens Backpropagated (M)", fontsize=12)
ax.set_ylabel("GSM8K Validation Accuracy", fontsize=12)
ax.set_title("Validation Accuracy Over Training" + TITLE_SUFFIX, fontsize=13)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y:.0%}"))
ax.legend(loc="lower right", fontsize=10)
ax.grid(True, alpha=0.3, zorder=1)
plt.tight_layout()
fig.savefig(OUT_DIR / "6a_val_accuracy_over_tokens.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {OUT_DIR / '6a_val_accuracy_over_tokens.png'}")


# ── Plot 7a: Test accuracy over tokens ───────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))

ax.axhline(0.0432, color="#7f7f7f", linestyle=":", linewidth=1, alpha=0.7, label="Baseline")

for label, cfg in RUNS.items():
    if label not in test_data:
        continue
    rows = test_data[label]
    tokens_m = [r["tokens"] / 1e6 for r in rows]
    accs = [r["accuracy"] for r in rows]
    ax.plot(tokens_m, accs, color=cfg["color"], linestyle=cfg["linestyle"],
            linewidth=1.5, alpha=0.85, label=label)
    ax.scatter(tokens_m, accs, color=cfg["color"], marker=cfg["marker"],
               s=15, alpha=0.5, edgecolors="none", zorder=3)
    # Mark best checkpoint
    best_tok = best_test[label]["tokens"]
    best_row = min(rows, key=lambda r: abs(r["tokens"] - best_tok))
    ax.scatter(best_tok / 1e6, best_row["accuracy"], color=cfg["color"],
               marker="*", s=300, edgecolors="black", linewidths=1, zorder=5)

ax.set_xlabel("Tokens Backpropagated (M)", fontsize=12)
ax.set_ylabel("GSM8K Test Accuracy", fontsize=12)
ax.set_title("Test Accuracy Over Training", fontsize=13)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y:.0%}"))
handles, labels = ax.get_legend_handles_labels()
handles.append(Line2D([0], [0], marker="*", color="w", markerfacecolor="black",
               markersize=12, markeredgecolor="black", markeredgewidth=0.8))
labels.append("Best val. accuracy")
ax.legend(handles=handles, labels=labels, loc="lower right", fontsize=10)
ax.grid(True, alpha=0.3, zorder=1)
plt.tight_layout()
fig.savefig(OUT_DIR / "7a_test_accuracy_over_tokens.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {OUT_DIR / '7a_test_accuracy_over_tokens.png'}")

print("\nAll plots saved to:", OUT_DIR)
