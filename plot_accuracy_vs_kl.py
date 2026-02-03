#!/usr/bin/env python3
"""
Plot GSM8K accuracy vs KL divergence for best checkpoints.
Results from real GSM8K test set (1319 samples).
"""

import matplotlib.pyplot as plt

# Data from evaluations on real GSM8K test set (old ~1.1M token checkpoints)
#
# Color scheme: Muon = red, AdamW = blue, Baseline = gray
# Marker scheme: GRPO = circle (o), SFT = square (s), Baseline = star (*)
experiments = {
    "Baseline": {
        "accuracy": 0.0493,  # 4.93% (Qwen2-1.5B-Instruct)
        "kl": 0.0,  # KL = 0 by definition
        "color": "#7f7f7f",  # gray
        "marker": "*",
    },
    "AdamW + GRPO": {
        "accuracy": 0.6080,  # 60.80% (tokens_1110186)
        "kl": 0.0965,
        "color": "#1f77b4",  # blue
        "marker": "o",       # GRPO = circle
    },
    "Muon + GRPO": {
        "accuracy": 0.5777,  # 57.77% (tokens_1107170)
        "kl": 0.0407,
        "color": "#d62728",  # red
        "marker": "o",       # GRPO = circle
    },
    "AdamW + SFT": {
        "accuracy": 0.4238,  # 42.38% (tokens_1104739)
        "kl": 0.1301,
        "color": "#1f77b4",  # blue
        "marker": "s",       # SFT = square
    },
    "Muon + SFT": {
        "accuracy": 0.4503,  # 45.03% (tokens_1104739)
        "kl": 0.0803,
        "color": "#d62728",  # red
        "marker": "s",       # SFT = square
    },
}

fig, ax = plt.subplots(figsize=(10, 7))

for name, data in experiments.items():
    ax.scatter(
        data["kl"],
        data["accuracy"],
        c=data["color"],
        marker=data["marker"],
        s=200,
        label=name,
        edgecolors="black",
        linewidths=1,
        zorder=3,
    )
    # Add label next to point
    ax.annotate(
        name,
        (data["kl"], data["accuracy"]),
        textcoords="offset points",
        xytext=(10, 5),
        fontsize=10,
        fontweight="bold",
        color=data["color"],
    )

ax.set_xlabel(r"$D_{\mathrm{KL}}(p_{\mathrm{base}} \| p_{\mathrm{exp}})$", fontsize=14)
ax.set_ylabel("GSM8K Test Accuracy", fontsize=12)
ax.set_title(r"GSM8K Test Accuracy vs $D_{\mathrm{KL}}(p_{\mathrm{base}} \| p_{\mathrm{exp}})$", fontsize=14)
ax.legend(loc="lower right", fontsize=10)
ax.grid(True, alpha=0.3, zorder=1)

# Format y-axis as percentage
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))

ax.set_xlim(-0.01, 0.15)
ax.set_ylim(0.0, 0.70)

plt.tight_layout()

output_path = "/mnt/nvme3n1/workspace/osilkin/mini-grpo/adamw-vs-muon-grpo-v1-artifacts/accuracy_vs_kl.png"
plt.savefig(output_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {output_path}")
