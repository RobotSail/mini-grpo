#!/usr/bin/env python3
"""
Plot comparison of GRPO runs: AdamW vs Muon, BF16 vs Mixed Precision.

Checkpoint selection: earliest checkpoint within 1 percentage point of
the best validation accuracy for each run.

Produces:
  6.  Scatter: Selected-checkpoint test accuracy vs forward KL
  7.  Bar charts: Frobenius norm, L0 sparsity, k90 (from original geometry files)
  8.  Validation accuracy vs forward KL trajectory, selected checkpoint marked
  9.  Test accuracy vs forward KL trajectory, selected checkpoint marked
  10. Frobenius norm trajectory over training, selected checkpoint marked
"""

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ── Paths ────────────────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).parent / "eval-data"
OUT_DIR = Path(__file__).parent / "plots"
OUT_DIR.mkdir(exist_ok=True)

RUNS = {
    "AdamW + BF16": {
        "dir": "grpo-adamw-bf16",
        "color": "#1f77b4",
        "marker": "o",
        "linestyle": "-",
    },
    "AdamW + Mixed": {
        "dir": "grpo-adamw-mixed",
        "color": "#17becf",
        "marker": "s",
        "linestyle": "--",
    },
    "Muon + Mixed": {
        "dir": "grpo-muon-mixed",
        "color": "#d62728",
        "marker": "D",
        "linestyle": "-.",
    },
    "Muon + BF16": {
        "dir": "grpo-muon-bf16",
        "color": "#ff7f0e",
        "marker": "^",
        "linestyle": ":",
    },
    "AdamW + Mixed (200K)": {
        "dir": "grpo-adamw-mixed-200k",
        "color": "#2ca02c",
        "marker": "P",
        "linestyle": (0, (3, 1, 1, 1)),
    },
}


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


# ── Load all data ────────────────────────────────────────────────────────────
val_data = {}
test_data = {}
geo_data = {}
norm_data = {}

for label, cfg in RUNS.items():
    d = DATA_DIR / cfg["dir"]
    val_data[label] = load_accuracy_kl(d / "validation_accuracy_kl.json")
    test_data[label] = load_accuracy_kl(d / "test_accuracy_kl.json")
    geo_data[label] = load_geometry(d / "geometry_metrics.json")
    norms_path = d / "update_norms.jsonl"
    if norms_path.exists():
        norm_data[label] = load_update_norms(norms_path)


# ── Checkpoint selection: earliest within 1pp of best val accuracy ───────────
selected = {}
for label in RUNS:
    rows = val_data[label]
    best_acc = max(r["accuracy"] for r in rows)
    threshold = best_acc - 0.01  # 1 percentage point

    # Rows are already sorted by tokens (ascending)
    earliest = None
    for r in rows:
        if r["accuracy"] >= threshold:
            earliest = r
            break

    # Find matching test entry
    test_rows = test_data[label]
    test_match = min(test_rows, key=lambda r: abs(r["tokens"] - earliest["tokens"]))

    selected[label] = {
        "val_accuracy": earliest["accuracy"],
        "val_kl": earliest["forward_kl"],
        "test_accuracy": test_match["accuracy"],
        "test_kl": test_match["forward_kl"],
        "tokens": earliest["tokens"],
        "best_val_acc": best_acc,
    }
    print(f"{label}: best val acc={best_acc:.4f}, threshold={threshold:.4f}")
    print(f"  Selected: {earliest['key']} @ {earliest['tokens']:,} tokens")
    print(f"  Val acc={earliest['accuracy']:.4f}, "
          f"Test acc={test_match['accuracy']:.4f}, Test KL={test_match['forward_kl']:.6f}")


# ── Plot 6: Selected-checkpoint scatter (test accuracy vs forward KL) ────────
fig, ax = plt.subplots(figsize=(8, 6))

for label, cfg in RUNS.items():
    s = selected[label]
    ax.scatter(
        s["test_kl"], s["test_accuracy"],
        c=cfg["color"], marker=cfg["marker"],
        s=200, label=f"{label}\n({s['tokens']/1e6:.1f}M tok)",
        edgecolors="black", linewidths=1, zorder=3,
    )
    ax.annotate(
        label,
        (s["test_kl"], s["test_accuracy"]),
        textcoords="offset points", xytext=(12, 6),
        fontsize=9, fontweight="bold", color=cfg["color"],
    )

ax.set_xlabel(r"Forward KL: $D_{\mathrm{KL}}(\pi_0 \| \pi)$", fontsize=12)
ax.set_ylabel("GSM8K Test Accuracy", fontsize=12)
ax.set_title(
    "Earliest Checkpoint Within 1pp of Best Val Acc:\n"
    "Test Accuracy vs Forward KL",
    fontsize=13,
)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y:.0%}"))
ax.legend(loc="best", fontsize=10)
ax.grid(True, alpha=0.3, zorder=1)
plt.tight_layout()
fig.savefig(OUT_DIR / "6_earliest_checkpoint_accuracy_vs_kl.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {OUT_DIR / '6_earliest_checkpoint_accuracy_vs_kl.png'}")


# ── Plot 7: Geometry bar charts ──────────────────────────────────────────────
# NOTE: geometry_metrics.json was computed for the overall-best validation
# checkpoints, not the "earliest within 1pp" ones. We plot what we have and
# annotate which checkpoint it corresponds to.
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
labels_list = list(RUNS.keys())
colors = [RUNS[l]["color"] for l in labels_list]

metrics = [
    ("frobenius_norm", r"$\|\Delta W\|_F$", "Frobenius Norm"),
    ("l0_sparsity", r"$L_0$ Sparsity ($|\Delta W| \geq 10^{-5}$)", "L0 Sparsity"),
    ("k90", r"$k_{90}$ (90% SV energy)", "Effective Rank $k_{90}$"),
]

for ax, (key, ylabel, title) in zip(axes, metrics):
    values = [geo_data[l][key] for l in labels_list]
    bars = ax.bar(labels_list, values, color=colors, edgecolor="black", linewidth=0.8)
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

plt.suptitle(
    "Geometry Metrics (computed on best-val-acc checkpoints,\n"
    "not the earliest-within-1pp selection)",
    fontsize=13, y=1.04,
)
plt.tight_layout()
fig.savefig(OUT_DIR / "7_geometry_bar_charts_note.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {OUT_DIR / '7_geometry_bar_charts_note.png'}")


# ── Plot 8: Validation trajectory with selected checkpoint marked ────────────
fig, ax = plt.subplots(figsize=(10, 7))

for label, cfg in RUNS.items():
    rows = truncate_after_divergence(val_data[label])
    kls = [r["forward_kl"] for r in rows]
    accs = [r["accuracy"] for r in rows]
    ax.plot(
        kls, accs,
        color=cfg["color"], linestyle=cfg["linestyle"],
        linewidth=1.5, alpha=0.7, label=label, zorder=2,
    )
    # Mark start and end
    ax.scatter(kls[0], accs[0], color=cfg["color"], marker="^",
               s=60, edgecolors="black", linewidths=0.5, zorder=3)
    ax.scatter(kls[-1], accs[-1], color=cfg["color"], marker="v",
               s=60, edgecolors="black", linewidths=0.5, zorder=3)
    # Mark the selected checkpoint with a large star
    s = selected[label]
    ax.scatter(
        s["val_kl"], s["val_accuracy"],
        color=cfg["color"], marker="*", s=350,
        edgecolors="black", linewidths=1, zorder=4,
    )

ax.set_xlabel(r"Forward KL: $D_{\mathrm{KL}}(\pi_0 \| \pi)$", fontsize=12)
ax.set_ylabel("GSM8K Validation Accuracy", fontsize=12)
ax.set_title(
    "Training Trajectory: Validation Accuracy vs Forward KL\n"
    r"($\bigstar$ = earliest checkpoint within 1pp of best)",
    fontsize=13,
)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y:.0%}"))
ax.legend(loc="best", fontsize=10)
ax.grid(True, alpha=0.3, zorder=1)
plt.tight_layout()
fig.savefig(OUT_DIR / "8_val_trajectory_marked.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {OUT_DIR / '8_val_trajectory_marked.png'}")


# ── Plot 9: Test trajectory with selected checkpoint marked ──────────────────
fig, ax = plt.subplots(figsize=(10, 7))

for label, cfg in RUNS.items():
    rows = truncate_after_divergence(test_data[label])
    kls = [r["forward_kl"] for r in rows]
    accs = [r["accuracy"] for r in rows]
    ax.plot(
        kls, accs,
        color=cfg["color"], linestyle=cfg["linestyle"],
        linewidth=1.5, alpha=0.7, label=label, zorder=2,
    )
    ax.scatter(kls[0], accs[0], color=cfg["color"], marker="^",
               s=60, edgecolors="black", linewidths=0.5, zorder=3)
    ax.scatter(kls[-1], accs[-1], color=cfg["color"], marker="v",
               s=60, edgecolors="black", linewidths=0.5, zorder=3)
    # Mark the selected checkpoint
    s = selected[label]
    ax.scatter(
        s["test_kl"], s["test_accuracy"],
        color=cfg["color"], marker="*", s=350,
        edgecolors="black", linewidths=1, zorder=4,
    )

ax.set_xlabel(r"Forward KL: $D_{\mathrm{KL}}(\pi_0 \| \pi)$", fontsize=12)
ax.set_ylabel("GSM8K Test Accuracy", fontsize=12)
ax.set_title(
    "Training Trajectory: Test Accuracy vs Forward KL\n"
    r"($\bigstar$ = earliest checkpoint within 1pp of best)",
    fontsize=13,
)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y:.0%}"))
ax.legend(loc="best", fontsize=10)
ax.grid(True, alpha=0.3, zorder=1)
plt.tight_layout()
fig.savefig(OUT_DIR / "9_test_trajectory_marked.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {OUT_DIR / '9_test_trajectory_marked.png'}")


# ── Plot 10: Update norm trajectory with selected checkpoint marked ──────────
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
    # Mark the selected checkpoint token count with a vertical line
    s = selected[label]
    ax.axvline(
        s["tokens"] / 1e6,
        color=cfg["color"], linestyle=":", linewidth=1.5, alpha=0.7,
    )
    ax.scatter(
        [s["tokens"] / 1e6],
        # Find the closest norm value at the selected token count
        [norms_arr[min(range(len(tokens_arr)),
                       key=lambda i: abs(tokens_arr[i] - s["tokens"]))]],
        color=cfg["color"], marker="*", s=200,
        edgecolors="black", linewidths=0.8, zorder=4,
    )

ax.set_xlabel("Tokens Trained (M)", fontsize=12)
ax.set_ylabel(r"Average $\|\Delta W_t\|_F$ (2D matrices)", fontsize=12)
ax.set_title(
    "Update Norm Trajectory Over Training\n"
    r"($\bigstar$ = earliest checkpoint within 1pp of best val acc)",
    fontsize=13,
)
ax.legend(loc="best", fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
fig.savefig(OUT_DIR / "10_update_norm_trajectory_marked.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {OUT_DIR / '10_update_norm_trajectory_marked.png'}")

print("\nAll plots saved to:", OUT_DIR)
