#!/usr/bin/env python3
"""
Plot GSM8K accuracy vs KL divergence for best checkpoints.
Results from validation set (gsm8k_sft_test.jsonl) with fixed eval.
"""

import matplotlib.pyplot as plt

# Data from evaluations on validation set (old ~1.1M token checkpoints)
# KL values from old_checkpoints_kl_results.json
#
# Color scheme: Muon = red, AdamW = blue
# Marker scheme: GRPO = circle (o), SFT = square (s)
experiments = {
    "AdamW + GRPO": {
        "accuracy": 0.8923,  # 89.23% (tokens_1110186)
        "kl": 0.0965,
        "color": "#1f77b4",  # blue
        "marker": "o",       # GRPO = circle
    },
    "Muon + GRPO": {
        "accuracy": 0.8803,  # 88.03% (tokens_1107170)
        "kl": 0.0407,
        "color": "#d62728",  # red
        "marker": "o",       # GRPO = circle
    },
    "AdamW + SFT": {
        "accuracy": 0.7171,  # 71.71% (tokens_1104739)
        "kl": 0.1301,
        "color": "#1f77b4",  # blue
        "marker": "s",       # SFT = square
    },
    "Muon + SFT": {
        "accuracy": 0.7224,  # 72.24% (tokens_1104739)
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
ax.set_ylabel("GSM8K Validation Accuracy", fontsize=12)
ax.set_title(r"GSM8K Validation Accuracy vs $D_{\mathrm{KL}}(p_{\mathrm{base}} \| p_{\mathrm{exp}})$ (~1.1M tokens)", fontsize=14)
ax.legend(loc="lower right", fontsize=10)
ax.grid(True, alpha=0.3, zorder=1)

# Format y-axis as percentage
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))

ax.set_xlim(-0.01, 0.15)
ax.set_ylim(0.65, 0.95)

plt.tight_layout()

output_path = "/mnt/nvme3n1/workspace/osilkin/mini-grpo/adamw-vs-muon-grpo-v1-artifacts/accuracy_vs_kl_validation.png"
plt.savefig(output_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {output_path}")
