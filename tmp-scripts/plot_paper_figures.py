#!/usr/bin/env python3
"""
Paper-quality composite figures for Muon vs AdamW comparison.

Figure 1: Accuracy vs KL (validation trajectory + best checkpoint scatter)
Figure 2: Update dynamics (per-step norm trajectory + cumulative Frobenius norm)
Figure 3: Sparsity & spectral structure (L0 sparsity + K90)
"""

import json
import re
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ── Global style ─────────────────────────────────────────────────────────────
matplotlib.rcParams.update({
    "font.family": "serif",
    "font.serif": ["CMU Serif", "Computer Modern Roman", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 10,
    "figure.dpi": 150,
})

# ── Paths ────────────────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).parent / "eval-data"
OUT_DIR = Path(__file__).parent / "plots-auto"
OUT_DIR.mkdir(exist_ok=True)

# Style: color encodes both optimizer AND precision
RUNS = {
    "AdamW + BF16": {
        "dir": "grpo-adamw-bf16",
        "color": "#6baed6",
        "marker": "o",
        "linestyle": "-",
    },
    "AdamW + Mixed": {
        "dir": "grpo-adamw-mixed-600k",
        "color": "#08519c",
        "marker": "o",
        "linestyle": "--",
    },
    "Muon + BF16": {
        "dir": "grpo-muon-bf16",
        "color": "#e53935",
        "marker": "o",
        "linestyle": "-",
    },
    "Muon + Mixed": {
        "dir": "grpo-muon-mixed",
        "color": "#7f0000",
        "marker": "o",
        "linestyle": "--",
    },
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
        "frobenius_norm": data["frobenius_norm"],
        "k90": data.get("avg_k90", data.get("k90_average")),
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


def plot_trajectory(ax, rows, cfg, label, best_tokens=None):
    kls = [r["forward_kl"] for r in rows]
    accs = [r["accuracy"] for r in rows]
    ax.plot(kls, accs, color=cfg["color"], linestyle=cfg["linestyle"],
            linewidth=1.8, alpha=0.85, label=label, zorder=2)
    ax.scatter(kls, accs, color=cfg["color"], marker=cfg["marker"],
               s=18, alpha=0.5, edgecolors="none", zorder=3)
    if best_tokens is not None:
        best_row = min(rows, key=lambda r: abs(r["tokens"] - best_tokens))
        ax.scatter(best_row["forward_kl"], best_row["accuracy"],
                   color=cfg["color"], marker="*", s=350,
                   edgecolors="black", linewidths=1, zorder=5)


# ── Load data ────────────────────────────────────────────────────────────────
val_data, test_data, geo_data, norm_data = {}, {}, {}, {}

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

best_test = {}
for label in RUNS:
    best_val_row = max(val_data[label], key=lambda r: r["accuracy"])
    best_tokens = best_val_row["tokens"]
    entry = {"val_accuracy": best_val_row["accuracy"],
             "val_kl": best_val_row["forward_kl"], "tokens": best_tokens}
    if label in test_data:
        match = min(test_data[label], key=lambda r: abs(r["tokens"] - best_tokens))
        entry["test_accuracy"] = match["accuracy"]
        entry["test_kl"] = match["forward_kl"]
    best_test[label] = entry


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 1: Accuracy vs Forward KL
# ═════════════════════════════════════════════════════════════════════════════
fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(16, 6.5))

# Left: Validation trajectory
ax_left.scatter(0.0, 0.0742, c="#7f7f7f", marker="*", s=280,
                label="Baseline", edgecolors="black", linewidths=1, zorder=3)
ax_left.annotate("Baseline", (0.0, 0.0742), textcoords="offset points",
                 xytext=(10, 6), fontsize=10, fontweight="bold", color="#7f7f7f")

for label, cfg in RUNS.items():
    rows = truncate_after_divergence(val_data[label])
    plot_trajectory(ax_left, rows, cfg, label, best_tokens=best_test[label]["tokens"])

ax_left.set_xlabel(r"Forward KL: $D_{\mathrm{KL}}(\pi_0 \| \pi)$")
ax_left.set_ylabel("GSM8K Validation Accuracy")
ax_left.set_title("(a) Validation Trajectory", fontsize=14, fontweight="bold")
ax_left.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y:.0%}"))
ax_left.legend(loc="lower right", fontsize=10, framealpha=0.9)
ax_left.grid(True, alpha=0.25, zorder=1)

# Right: Best checkpoint scatter (test)
ax_right.scatter(0.0, 0.0432, c="#7f7f7f", marker="*", s=280,
                 label="Baseline", edgecolors="black", linewidths=1, zorder=3)
ax_right.annotate("Baseline", (0.0, 0.0432), textcoords="offset points",
                  xytext=(10, 6), fontsize=10, fontweight="bold", color="#7f7f7f")

for label, cfg in RUNS.items():
    bt = best_test[label]
    if "test_kl" not in bt:
        continue
    ax_right.scatter(bt["test_kl"], bt["test_accuracy"], c=cfg["color"],
                     marker=cfg["marker"], s=220, label=label,
                     edgecolors="black", linewidths=1, zorder=3)
    ax_right.annotate(label, (bt["test_kl"], bt["test_accuracy"]),
                      textcoords="offset points", xytext=(10, 6),
                      fontsize=10, fontweight="bold", color=cfg["color"])

ax_right.set_xlabel(r"Forward KL: $D_{\mathrm{KL}}(\pi_0 \| \pi)$")
ax_right.set_ylabel("GSM8K Test Accuracy")
ax_right.set_title("(b) Best Checkpoint (Test Set)", fontsize=14, fontweight="bold")
ax_right.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y:.0%}"))
ax_right.set_ylim(0, None)
ax_right.legend(loc="center right", fontsize=10, framealpha=0.9)
ax_right.grid(True, alpha=0.25, zorder=1)

fig.suptitle(
    "Muon achieves higher accuracy per unit of forward KL\n"
    "across both precision modes and evaluation splits",
    fontsize=18, fontweight="bold", y=1.05,
)
plt.tight_layout()
fig.savefig(OUT_DIR / "figure1_accuracy_vs_kl.png", dpi=200, bbox_inches="tight")
fig.savefig(OUT_DIR / "figure1_accuracy_vs_kl.pdf", bbox_inches="tight")
plt.close(fig)
print(f"Saved: {OUT_DIR / 'figure1_accuracy_vs_kl.png'}")


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 2: Update dynamics
# ═════════════════════════════════════════════════════════════════════════════
fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(16, 6))

# Left: Per-step update norm trajectory
for label, cfg in RUNS.items():
    if label not in norm_data:
        continue
    tokens_arr, norms_arr = norm_data[label]
    tokens_m = [t / 1e6 for t in tokens_arr]
    ax_left.plot(tokens_m, norms_arr, color=cfg["color"], linestyle=cfg["linestyle"],
                 linewidth=1.4, alpha=0.85, label=label)
    # Mark best checkpoint
    best_tok = best_test[label]["tokens"]
    ci = min(range(len(tokens_arr)), key=lambda i: abs(tokens_arr[i] - best_tok))
    ax_left.scatter(tokens_m[ci], norms_arr[ci], color=cfg["color"], marker="*",
                    s=250, edgecolors="black", linewidths=0.8, zorder=5)

ax_left.set_xlabel("Tokens Trained (M)")
ax_left.set_ylabel(r"Avg. $\|\Delta W_t\|_F$ (2D matrices)")
ax_left.set_title(r"(a) Per-Step Update Norm $\|\Delta W_t\|_F$", fontsize=14, fontweight="bold")
ax_left.legend(loc="upper right", fontsize=10, framealpha=0.9)
ax_left.grid(True, alpha=0.25)

# Right: Cumulative Frobenius norm bars
labels_with_geo = [l for l in RUNS if l in geo_data]
colors = [RUNS[l]["color"] for l in labels_with_geo]
values = [geo_data[l]["frobenius_norm"] for l in labels_with_geo]
bars = ax_right.bar(labels_with_geo, values, color=colors, edgecolor="black", linewidth=0.8, width=0.65)
for bar, val in zip(bars, values):
    ax_right.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                  f"{val:.4f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
ax_right.set_ylabel(r"$\|\Delta W\|_F$")
ax_right.set_title(r"(b) Cumulative $\|\Delta W\|_F$ of Best Checkpoint", fontsize=14, fontweight="bold")
ax_right.grid(axis="y", alpha=0.25)
ax_right.tick_params(axis="x", rotation=12)

fig.suptitle(
    r"AdamW's per-step updates are $\sim$1.75$\times$ larger than Muon's, yet accumulate"
    "\n"
    r"$\sim$40% less net displacement $\|\Delta W\|_F$, suggesting inter-step cancellation",
    fontsize=18, fontweight="bold", y=1.06,
)
plt.tight_layout()
fig.savefig(OUT_DIR / "figure2_update_dynamics.png", dpi=200, bbox_inches="tight")
fig.savefig(OUT_DIR / "figure2_update_dynamics.pdf", bbox_inches="tight")
plt.close(fig)
print(f"Saved: {OUT_DIR / 'figure2_update_dynamics.png'}")


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 3: Sparsity & spectral structure
# ═════════════════════════════════════════════════════════════════════════════
fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(15, 6))

labels_with_geo = [l for l in RUNS if l in geo_data]
colors = [RUNS[l]["color"] for l in labels_with_geo]

# Left: L0 sparsity (inverted: 1 - fraction_nonzero)
l0_values = [1.0 - geo_data[l]["l0_sparsity"] for l in labels_with_geo]
bars = ax_left.bar(labels_with_geo, l0_values, color=colors, edgecolor="black", linewidth=0.8, width=0.65)
for bar, val in zip(bars, l0_values):
    ax_left.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                 f"{val:.4f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
ax_left.set_ylabel(r"$1 - \|\Delta W\|_0 \,/\, n$")
ax_left.set_title(r"(a) $L_0$ Sparsity", fontsize=14, fontweight="bold")
ax_left.grid(axis="y", alpha=0.25)
ax_left.tick_params(axis="x", rotation=12)

# Right: K90
k90_values = [geo_data[l]["k90"] for l in labels_with_geo]
bars = ax_right.bar(labels_with_geo, k90_values, color=colors, edgecolor="black", linewidth=0.8, width=0.65)
for bar, val in zip(bars, k90_values):
    ax_right.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                  f"{val:.1f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
ax_right.set_ylabel(r"$k_{90}$ (90\% SV energy)")
ax_right.set_title(r"(b) Effective Rank $k_{90}$", fontsize=14, fontweight="bold")
ax_right.grid(axis="y", alpha=0.25)
ax_right.tick_params(axis="x", rotation=12)

fig.suptitle(
    r"$L_0$ sparsity and $k_{90}$ are both sensitive to precision —"
    "\n"
    r"under mixed precision, where these metrics are uncontaminated by truncation effects,"
    "\n"
    r"AdamW's updates are $\sim$3$\times$ more spectrally concentrated than Muon's",
    fontsize=18, fontweight="bold", y=1.10,
)
plt.tight_layout()
fig.savefig(OUT_DIR / "figure3_sparsity_spectral.png", dpi=200, bbox_inches="tight")
fig.savefig(OUT_DIR / "figure3_sparsity_spectral.pdf", bbox_inches="tight")
plt.close(fig)
print(f"Saved: {OUT_DIR / 'figure3_sparsity_spectral.png'}")

print("\nAll paper figures saved to:", OUT_DIR)
