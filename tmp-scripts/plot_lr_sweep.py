#!/usr/bin/env python3
"""Plot GSM8K accuracy vs KL divergence trajectories for the LR sweep."""

import json
import re
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

EVAL_PATH = "/mnt/nvme2n1/checkpoints/lr-sweep-3m/eval_results_all.json"
OUTPUT_PATH = "/mnt/nvme2n1/checkpoints/lr-sweep-3m/accuracy_vs_kl_trajectories.png"

with open(EVAL_PATH) as f:
    results = json.load(f)

# Group checkpoints by run, extract token count from checkpoint name
runs = defaultdict(list)
for name, data in results.items():
    m_tokens = re.search(r"checkpoint-(\d+)", name)
    if not m_tokens:
        continue
    tokens = int(m_tokens.group(1))

    if "muon" in name:
        run_key = "Muon lr=1e-6"
    else:
        m_lr = re.search(r"adamw-(\d+\.?\d*e-\d+)", name)
        lr = m_lr.group(1) if m_lr else "?"
        run_key = f"AdamW lr={lr}"

    runs[run_key].append({
        "tokens": tokens,
        "accuracy": data["accuracy"],
        "kl": data["kl_divergence"],
    })

# Sort each run by tokens
for key in runs:
    runs[key].sort(key=lambda x: x["tokens"])

# Color/style config
# Muon: bold red, visually distinct
# AdamW: perceptually separated colors from viridis colormap
import matplotlib.cm as cm
_adamw_cmap = cm.get_cmap("viridis", 6)
STYLES = {
    "Muon lr=1e-6":   {"color": "#d62728", "linestyle": "-",  "linewidth": 3.0, "marker": "D", "markersize": 50, "best_size": 250, "zorder": 6, "alpha": 0.85},
    "AdamW lr=1e-7":  {"color": _adamw_cmap(0.05), "linestyle": "-",  "linewidth": 1.8, "marker": "o", "markersize": 20, "best_size": 160, "zorder": 4, "alpha": 0.7},
    "AdamW lr=2e-7":  {"color": _adamw_cmap(0.3),  "linestyle": "-",  "linewidth": 1.8, "marker": "o", "markersize": 20, "best_size": 160, "zorder": 4, "alpha": 0.7},
    "AdamW lr=3e-7":  {"color": _adamw_cmap(0.55), "linestyle": "-",  "linewidth": 1.8, "marker": "o", "markersize": 20, "best_size": 160, "zorder": 4, "alpha": 0.7},
    "AdamW lr=5e-7":  {"color": _adamw_cmap(0.85), "linestyle": "-",  "linewidth": 1.8, "marker": "o", "markersize": 20, "best_size": 160, "zorder": 4, "alpha": 0.7},
}

fig, ax = plt.subplots(figsize=(10, 7))

for run_key, points in runs.items():
    style = STYLES.get(run_key, {"color": "gray", "linestyle": "-", "linewidth": 1.5, "marker": "o", "zorder": 3})
    kls = [p["kl"] for p in points]
    accs = [p["accuracy"] for p in points]

    # Find best accuracy checkpoint
    best_idx = max(range(len(accs)), key=lambda i: accs[i])

    # Draw trajectory line
    ax.plot(kls, accs, color=style["color"], linestyle=style["linestyle"],
            linewidth=style["linewidth"], alpha=style["alpha"], zorder=style["zorder"])

    # Draw all points small
    ax.scatter(kls, accs, c=style["color"], marker=style["marker"],
               s=style["markersize"], alpha=style["alpha"], zorder=style["zorder"] + 1)

    # Highlight best accuracy checkpoint
    ax.scatter(kls[best_idx], accs[best_idx], c=style["color"], marker=style["marker"],
               s=style["best_size"], edgecolors="black", linewidths=1.5, zorder=style["zorder"] + 2,
               label=run_key)

# Baseline
ax.scatter(0.0, 0.0493, c="#7f7f7f", marker="*", s=300,
           edgecolors="black", linewidths=1, zorder=10, label="Baseline")

ax.set_xlabel(r"$D_{\mathrm{KL}}(\pi_{\theta} \| \pi_{\mathrm{base}})$", fontsize=13)
ax.set_ylabel("GSM8K Accuracy", fontsize=13)
ax.set_title("LR Sweep: GSM8K Accuracy vs Forward KL (3M tokens, seed=2020, mixed precision)", fontsize=12)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
ax.grid(True, alpha=0.3, zorder=0)
ax.legend(loc="lower right", fontsize=9)

plt.tight_layout()
Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {OUTPUT_PATH}")
