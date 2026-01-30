import matplotlib.pyplot as plt
import numpy as np

# Data from experiments
data = {
    "GRPO + Muon": {"accuracy": 89.23, "kl_div": 0.0237, "optimizer": "Muon", "technique": "GRPO"},
    "GRPO + AdamW": {"accuracy": 89.63, "kl_div": 0.0470, "optimizer": "AdamW", "technique": "GRPO"},
    "SFT + AdamW": {"accuracy": 70.97, "kl_div": 0.1622, "optimizer": "AdamW", "technique": "SFT"},
    "SFT + Muon": {"accuracy": 73.04, "kl_div": 0.1355, "optimizer": "Muon", "technique": "SFT"},
}

# Colors by optimizer, markers by technique
optimizer_colors = {
    "AdamW": "#3498db",  # Blue
    "Muon": "#e74c3c",   # Red
}

technique_markers = {
    "GRPO": "o",  # Circle
    "SFT": "s",   # Square
}

fig, ax = plt.subplots(figsize=(10, 7))

# Plot each point
for name, values in data.items():
    ax.scatter(
        values["kl_div"],
        values["accuracy"],
        c=optimizer_colors[values["optimizer"]],
        marker=technique_markers[values["technique"]],
        s=200,
        edgecolors="black",
        linewidths=1.5,
        zorder=5,
    )

# Add labels next to points
for name, values in data.items():
    offset_x = 0.003
    offset_y = 1.0
    ax.annotate(
        name,
        (values["kl_div"] + offset_x, values["accuracy"] + offset_y),
        fontsize=10,
        fontweight="bold",
    )

# Create custom legend
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=12,
           markeredgecolor='black', label='GRPO'),
    Line2D([0], [0], marker='s', color='w', markerfacecolor='gray', markersize=12,
           markeredgecolor='black', label='SFT'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#3498db', markersize=12,
           markeredgecolor='black', label='AdamW'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#e74c3c', markersize=12,
           markeredgecolor='black', label='Muon'),
]

# Styling
ax.set_xlabel(r"KL Divergence from Base Model: $D_{KL}(\pi_0 \| \pi)$", fontsize=12, fontweight="bold")
ax.set_ylabel("Accuracy (%)", fontsize=12, fontweight="bold")
ax.set_title("GSM8K Accuracy vs. KL Divergence\n(Higher accuracy + Lower KL = Better)", fontsize=14, fontweight="bold")

# Add grid
ax.grid(True, alpha=0.3, linestyle="--")

# Set axis limits with padding
ax.set_xlim(0, 0.18)
ax.set_ylim(65, 95)

# Add legend
ax.legend(handles=legend_elements, loc="lower left", fontsize=11, ncol=2)

plt.tight_layout()
plt.savefig("accuracy_vs_kl.png", dpi=150, bbox_inches="tight")
plt.savefig("accuracy_vs_kl.pdf", bbox_inches="tight")
print("Saved: accuracy_vs_kl.png and accuracy_vs_kl.pdf")
plt.show()
