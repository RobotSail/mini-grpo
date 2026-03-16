#!/usr/bin/env python3
"""
LR sweep plots: AdamW vs Muon across 5 learning rates.
Grouped by optimizer color (blue=AdamW, red=Muon), shade varies by LR.
"""

import json
import re
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

matplotlib.rcParams.update({
    "font.family": "serif",
    "font.serif": ["CMU Serif", "Computer Modern Roman", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 9,
    "figure.dpi": 150,
})

DATA_DIR = Path(__file__).parent / "eval-data" / "lr-sweep"
OUT_DIR = Path(__file__).parent / "plots-auto" / "lr-sweep"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# LRs from low to high
LRS = ["1e-7", "2.5e-7", "5e-7", "7.5e-7", "1e-6"]

# Blue shades (light to dark) for AdamW, red shades for Muon
ADAMW_SHADES = ["#c6dbef", "#9ecae1", "#6baed6", "#3182bd", "#08519c"]
MUON_SHADES = ["#fcbba1", "#fc9272", "#fb6a4a", "#de2d26", "#a50f15"]

RUNS = {}
for i, lr in enumerate(LRS):
    RUNS[f"AdamW lr={lr}"] = {
        "dir": f"grpo-adamw-{lr}",
        "color": ADAMW_SHADES[i],
        "linestyle": "-",
        "marker": "o",
    }
    RUNS[f"Muon lr={lr}"] = {
        "dir": f"grpo-muon-{lr}",
        "color": MUON_SHADES[i],
        "linestyle": "-",
        "marker": "o",
    }


# ── Helpers ──────────────────────────────────────────────────────────────────
def extract_tokens(key, entry):
    if "tokens" in entry:
        return int(entry["tokens"])
    m = re.search(r"checkpoint-(\d+)", key)
    if m:
        return int(m.group(1))
    raise ValueError(f"Cannot determine token count for key: {key}")


def load_accuracy_kl(path):
    with open(path) as f:
        data = json.load(f)
    rows = []
    for key, entry in data.items():
        if "initial" in key:
            continue
        tokens = extract_tokens(key, entry)
        rows.append({
            "key": key, "tokens": tokens,
            "accuracy": entry["accuracy"],
            "forward_kl": entry["forward_kl"],
        })
    rows.sort(key=lambda r: r["tokens"])
    return rows


def load_geometry(path):
    with open(path) as f:
        data = json.load(f)
    return {
        "frobenius_norm": data.get("frobenius_norm", data.get("frobenius_norm_total")),
        "k90": data.get("avg_k90", data.get("k90_average", data.get("rank_k90_avg"))),
        "l0_sparsity": data.get("l0_sparsity", data.get("l0_sparsity_1e5")),
    }


def is_2d_weight(name):
    if any(s in name for s in [".bias", "layernorm", "layer_norm", "ln_"]):
        return False
    if "embed" in name:
        return False
    if "lm_head" in name:
        return False
    return ".weight" in name


def truncate_after_divergence(rows, threshold=0.10):
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


def load_update_norms(path):
    tokens_list, norms_list = [], []
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            matrix_norms = [v for k, v in rec["norms"].items() if is_2d_weight(k)]
            if matrix_norms:
                tokens_list.append(rec["tokens"])
                norms_list.append(float(np.mean(matrix_norms)))
    return tokens_list, norms_list


# ── Load data ────────────────────────────────────────────────────────────────
val_data, test_data, geo_data, norm_data = {}, {}, {}, {}

for label, cfg in RUNS.items():
    d = DATA_DIR / cfg["dir"]
    val_path = d / "validation_accuracy_kl_auto.json"
    if val_path.exists():
        val_data[label] = load_accuracy_kl(val_path)
    test_path = d / "test_accuracy_kl_auto.json"
    if test_path.exists():
        test_data[label] = load_accuracy_kl(test_path)
    geo_path = d / "geometry_metrics_auto.json"
    if geo_path.exists():
        geo_data[label] = load_geometry(geo_path)
    norms_path = d / "update_norms.jsonl"
    if norms_path.exists():
        norm_data[label] = load_update_norms(norms_path)

best_test = {}
for label in RUNS:
    if label not in val_data:
        continue
    best_val_row = max(val_data[label], key=lambda r: r["accuracy"])
    best_tokens = best_val_row["tokens"]
    entry = {"val_accuracy": best_val_row["accuracy"],
             "val_kl": best_val_row["forward_kl"], "tokens": best_tokens}
    if label in test_data:
        match = min(test_data[label], key=lambda r: abs(r["tokens"] - best_tokens))
        entry["test_accuracy"] = match["accuracy"]
        entry["test_kl"] = match["forward_kl"]
    best_test[label] = entry
    print(f"{label}: best val={best_val_row['accuracy']:.4f} @ {best_tokens:,} tok, KL={best_val_row['forward_kl']:.4f}")


# ── Plot 1: Best checkpoint scatter (test) ───────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 7))

ax.scatter(0.0, 0.0432, c="#7f7f7f", marker="*", s=250,
           label="Baseline", edgecolors="black", linewidths=1, zorder=3)

for label, cfg in RUNS.items():
    if label not in best_test or "test_kl" not in best_test[label]:
        continue
    bt = best_test[label]
    ax.scatter(bt["test_kl"], bt["test_accuracy"], c=cfg["color"],
               marker="o", s=180, label=label,
               edgecolors="black", linewidths=0.8, zorder=3)

ax.set_xlabel(r"Forward KL: $D_{\mathrm{KL}}(\pi_0 \| \pi)$")
ax.set_ylabel("GSM8K Test Accuracy")
ax.set_title("LR Sweep: Best Checkpoint Test Accuracy vs Forward KL", fontsize=14, fontweight="bold")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y:.0%}"))
ax.set_ylim(0, None)
ax.legend(loc="center right", fontsize=8, ncol=2)
ax.grid(True, alpha=0.25)
plt.tight_layout()
fig.savefig(OUT_DIR / "1_best_checkpoint_accuracy_vs_kl.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {OUT_DIR / '1_best_checkpoint_accuracy_vs_kl.png'}")


# ── Plot 2: Geometry bars ────────────────────────────────────────────────────
labels_with_geo = [l for l in RUNS if l in geo_data]
if labels_with_geo:
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    colors = [RUNS[l]["color"] for l in labels_with_geo]

    metrics_list = [
        ("frobenius_norm", r"$\|\Delta W\|_F$", "Frobenius Norm"),
        ("l0_sparsity", r"$1 - \|\Delta W\|_0 / n$", r"$L_0$ Sparsity"),
        ("k90", r"$k_{90}$", r"Effective Rank $k_{90}$"),
    ]

    for ax, (key, ylabel, title) in zip(axes, metrics_list):
        values = [geo_data[l][key] for l in labels_with_geo]
        if key == "l0_sparsity":
            values = [1.0 - v for v in values]
        bars = ax.bar(range(len(labels_with_geo)), values, color=colors,
                       edgecolor="black", linewidth=0.6, width=0.8)
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontweight="bold")
        ax.set_xticks(range(len(labels_with_geo)))
        ax.set_xticklabels(labels_with_geo, rotation=45, ha="right", fontsize=8)
        ax.grid(axis="y", alpha=0.25)

    plt.suptitle("LR Sweep: Geometry Metrics of Best Checkpoints", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "2_geometry_bar_charts.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {OUT_DIR / '2_geometry_bar_charts.png'}")


# ── Plot 3: Validation trajectory ────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 7))

ax.scatter(0.0, 0.0742, c="#7f7f7f", marker="*", s=250,
           label="Baseline", edgecolors="black", linewidths=1, zorder=3)

for label, cfg in RUNS.items():
    if label not in val_data:
        continue
    rows = truncate_after_divergence(val_data[label])
    kls = [r["forward_kl"] for r in rows]
    accs = [r["accuracy"] for r in rows]
    ax.plot(kls, accs, color=cfg["color"], linewidth=1.5, alpha=0.85, label=label)
    ax.scatter(kls, accs, color=cfg["color"], s=12, alpha=0.4, edgecolors="none", zorder=3)
    if label in best_test:
        best_row = min(rows, key=lambda r: abs(r["tokens"] - best_test[label]["tokens"]))
        ax.scatter(best_row["forward_kl"], best_row["accuracy"], color=cfg["color"],
                   marker="*", s=300, edgecolors="black", linewidths=1, zorder=5)

ax.set_xlabel(r"Forward KL: $D_{\mathrm{KL}}(\pi_0 \| \pi)$")
ax.set_ylabel("GSM8K Validation Accuracy")
ax.set_title("LR Sweep: Validation Accuracy vs Forward KL", fontsize=14, fontweight="bold")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y:.0%}"))
ax.legend(loc="lower right", fontsize=8, ncol=2)
ax.grid(True, alpha=0.25)
plt.tight_layout()
fig.savefig(OUT_DIR / "3_val_accuracy_vs_kl_trajectory.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {OUT_DIR / '3_val_accuracy_vs_kl_trajectory.png'}")


# ── Plot 4: Test trajectory ──────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 7))

ax.scatter(0.0, 0.0432, c="#7f7f7f", marker="*", s=250,
           label="Baseline", edgecolors="black", linewidths=1, zorder=3)

for label, cfg in RUNS.items():
    if label not in test_data:
        continue
    rows = truncate_after_divergence(test_data[label])
    kls = [r["forward_kl"] for r in rows]
    accs = [r["accuracy"] for r in rows]
    ax.plot(kls, accs, color=cfg["color"], linewidth=1.5, alpha=0.85, label=label)
    ax.scatter(kls, accs, color=cfg["color"], s=12, alpha=0.4, edgecolors="none", zorder=3)
    if label in best_test:
        best_row = min(rows, key=lambda r: abs(r["tokens"] - best_test[label]["tokens"]))
        ax.scatter(best_row["forward_kl"], best_row["accuracy"], color=cfg["color"],
                   marker="*", s=300, edgecolors="black", linewidths=1, zorder=5)

ax.set_xlabel(r"Forward KL: $D_{\mathrm{KL}}(\pi_0 \| \pi)$")
ax.set_ylabel("GSM8K Test Accuracy")
ax.set_title("LR Sweep: Test Accuracy vs Forward KL", fontsize=14, fontweight="bold")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y:.0%}"))
ax.legend(loc="lower right", fontsize=8, ncol=2)
ax.grid(True, alpha=0.25)
plt.tight_layout()
fig.savefig(OUT_DIR / "4_test_accuracy_vs_kl_trajectory.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {OUT_DIR / '4_test_accuracy_vs_kl_trajectory.png'}")


# ── Plot 5: Update norm trajectory ───────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 5))

for label, cfg in RUNS.items():
    if label not in norm_data:
        continue
    tokens_arr, norms_arr = norm_data[label]
    tokens_m = [t / 1e6 for t in tokens_arr]
    ax.plot(tokens_m, norms_arr, color=cfg["color"], linewidth=1.0, alpha=0.8, label=label)
    if label in best_test:
        best_tok = best_test[label]["tokens"]
        ci = min(range(len(tokens_arr)), key=lambda i: abs(tokens_arr[i] - best_tok))
        ax.scatter(tokens_m[ci], norms_arr[ci], color=cfg["color"], marker="*",
                   s=200, edgecolors="black", linewidths=0.8, zorder=5)

ax.set_xlabel("Tokens Trained (M)")
ax.set_ylabel(r"Avg. $\|\Delta W_t\|_F$ (2D matrices)")
ax.set_title("LR Sweep: Update Norm Trajectory", fontsize=14, fontweight="bold")
ax.legend(loc="upper right", fontsize=8, ncol=2)
ax.grid(True, alpha=0.25)
plt.tight_layout()
fig.savefig(OUT_DIR / "5_update_norm_trajectory.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {OUT_DIR / '5_update_norm_trajectory.png'}")


# ── Plot 6: Validation accuracy over tokens ──────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 6))

ax.axhline(0.0742, color="#7f7f7f", linestyle=":", linewidth=1, alpha=0.7, label="Baseline")

for label, cfg in RUNS.items():
    if label not in val_data:
        continue
    rows = val_data[label]
    tokens_m = [r["tokens"] / 1e6 for r in rows]
    accs = [r["accuracy"] for r in rows]
    ax.plot(tokens_m, accs, color=cfg["color"], linewidth=1.5, alpha=0.85, label=label)
    if label in best_test:
        best_tok = best_test[label]["tokens"]
        best_row = min(rows, key=lambda r: abs(r["tokens"] - best_tok))
        ax.scatter(best_tok / 1e6, best_row["accuracy"], color=cfg["color"],
                   marker="*", s=300, edgecolors="black", linewidths=1, zorder=5)

ax.set_xlabel("Tokens Backpropagated (M)")
ax.set_ylabel("GSM8K Validation Accuracy")
ax.set_title("LR Sweep: Validation Accuracy Over Training", fontsize=14, fontweight="bold")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y:.0%}"))
ax.legend(loc="lower right", fontsize=8, ncol=2)
ax.grid(True, alpha=0.25)
plt.tight_layout()
fig.savefig(OUT_DIR / "6_val_accuracy_over_tokens.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {OUT_DIR / '6_val_accuracy_over_tokens.png'}")


# ── Plot 7: Test accuracy over tokens ────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 6))

ax.axhline(0.0432, color="#7f7f7f", linestyle=":", linewidth=1, alpha=0.7, label="Baseline")

for label, cfg in RUNS.items():
    if label not in test_data:
        continue
    rows = test_data[label]
    tokens_m = [r["tokens"] / 1e6 for r in rows]
    accs = [r["accuracy"] for r in rows]
    ax.plot(tokens_m, accs, color=cfg["color"], linewidth=1.5, alpha=0.85, label=label)
    if label in best_test:
        best_tok = best_test[label]["tokens"]
        best_row = min(rows, key=lambda r: abs(r["tokens"] - best_tok))
        ax.scatter(best_tok / 1e6, best_row["accuracy"], color=cfg["color"],
                   marker="*", s=300, edgecolors="black", linewidths=1, zorder=5)

ax.set_xlabel("Tokens Backpropagated (M)")
ax.set_ylabel("GSM8K Test Accuracy")
ax.set_title("LR Sweep: Test Accuracy Over Training", fontsize=14, fontweight="bold")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y:.0%}"))
ax.legend(loc="lower right", fontsize=8, ncol=2)
ax.grid(True, alpha=0.25)
plt.tight_layout()
fig.savefig(OUT_DIR / "7_test_accuracy_over_tokens.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {OUT_DIR / '7_test_accuracy_over_tokens.png'}")

print(f"\nAll LR sweep plots saved to: {OUT_DIR}")
