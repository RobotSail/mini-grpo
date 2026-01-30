"""Plot AdamW + SFT and Muon + SFT training metrics on GSM8K."""

import matplotlib.pyplot as plt
import numpy as np

# AdamW + SFT data
adamw_steps = np.array(
    [50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000]
)

adamw_accuracy = np.array(
    [
        53.38,
        66.56,
        68.36,
        69.63,
        70.77,
        71.84,
        73.24,
        67.56,
        58.33,
        69.16,
        71.97,
        69.03,
        71.04,
        61.61,
        65.62,
        53.38,
        63.41,
        54.98,
        61.67,
        57.19,
    ]
)

adamw_parsable = np.array(
    [
        62.94,
        77.06,
        81.00,
        82.21,
        84.62,
        87.56,
        87.69,
        84.41,
        81.74,
        86.35,
        88.23,
        87.36,
        89.50,
        87.42,
        87.69,
        86.42,
        89.30,
        86.35,
        87.09,
        88.09,
    ]
)

# Muon + SFT data
muon_steps = np.array(
    [10, 20, 30, 40, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000]
)

muon_accuracy = np.array(
    [
        19.33,
        19.00,
        21.47,
        21.47,
        23.21,
        27.16,
        27.89,
        34.92,
        43.68,
        52.31,
        51.64,
        53.85,
        48.76,
        51.97,
        46.35,
        53.51,
        48.16,
        42.81,
        40.13,
        36.86,
        42.61,
        39.60,
        36.19,
        41.20,
    ]
)

muon_parsable = np.array(
    [
        19.67,
        19.13,
        21.87,
        21.81,
        23.88,
        28.83,
        30.97,
        38.53,
        49.83,
        60.94,
        62.61,
        65.35,
        62.81,
        65.02,
        63.34,
        67.96,
        64.55,
        65.15,
        62.34,
        62.34,
        66.69,
        64.68,
        66.69,
        71.64,
    ]
)

# set up the plot with a dark theme
plt.style.use("dark_background")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# color palette - warm tones for AdamW, cool tones for Muon
color_adamw_acc = "#FF9F1C"  # amber
color_adamw_parse = "#FFDC5E"  # light gold
color_muon_acc = "#2EC4B6"  # teal
color_muon_parse = "#7FDBDA"  # light teal

# === LEFT PLOT: Accuracy ===
ax1.plot(
    adamw_steps, adamw_accuracy, color=color_adamw_acc, linewidth=2.5, marker="o", markersize=5, label="AdamW", zorder=3
)
ax1.plot(
    muon_steps, muon_accuracy, color=color_muon_acc, linewidth=2.5, marker="s", markersize=5, label="Muon", zorder=3
)

# highlight peak accuracy for both
adamw_peak_idx = np.argmax(adamw_accuracy)
muon_peak_idx = np.argmax(muon_accuracy)

ax1.scatter(
    [adamw_steps[adamw_peak_idx]],
    [adamw_accuracy[adamw_peak_idx]],
    color=color_adamw_acc,
    s=150,
    zorder=4,
    edgecolors="white",
    linewidths=2,
)
ax1.annotate(
    f"Peak: {adamw_accuracy[adamw_peak_idx]:.1f}%",
    xy=(adamw_steps[adamw_peak_idx], adamw_accuracy[adamw_peak_idx]),
    xytext=(adamw_steps[adamw_peak_idx] + 80, adamw_accuracy[adamw_peak_idx] + 2),
    fontsize=10,
    color=color_adamw_acc,
    arrowprops=dict(arrowstyle="->", color=color_adamw_acc, lw=1.5),
)

ax1.scatter(
    [muon_steps[muon_peak_idx]],
    [muon_accuracy[muon_peak_idx]],
    color=color_muon_acc,
    s=150,
    zorder=4,
    edgecolors="white",
    linewidths=2,
)
ax1.annotate(
    f"Peak: {muon_accuracy[muon_peak_idx]:.1f}%",
    xy=(muon_steps[muon_peak_idx], muon_accuracy[muon_peak_idx]),
    xytext=(muon_steps[muon_peak_idx] + 80, muon_accuracy[muon_peak_idx] - 5),
    fontsize=10,
    color=color_muon_acc,
    arrowprops=dict(arrowstyle="->", color=color_muon_acc, lw=1.5),
)

ax1.set_xlabel("Training Step", fontsize=13, fontweight="medium")
ax1.set_ylabel("Accuracy (%)", fontsize=13, fontweight="medium")
ax1.set_title("Accuracy Comparison", fontsize=14, fontweight="bold", pad=10)
ax1.grid(True, alpha=0.2, linestyle="-", linewidth=0.5)
ax1.set_xlim(0, 1050)
ax1.set_ylim(10, 80)
ax1.legend(loc="lower right", fontsize=11, framealpha=0.8)
ax1.tick_params(axis="both", labelsize=11)

# === RIGHT PLOT: Parsable ===
ax2.plot(
    adamw_steps,
    adamw_parsable,
    color=color_adamw_parse,
    linewidth=2.5,
    marker="o",
    markersize=5,
    label="AdamW",
    zorder=3,
)
ax2.plot(
    muon_steps, muon_parsable, color=color_muon_parse, linewidth=2.5, marker="s", markersize=5, label="Muon", zorder=3
)

ax2.set_xlabel("Training Step", fontsize=13, fontweight="medium")
ax2.set_ylabel("Parsable (%)", fontsize=13, fontweight="medium")
ax2.set_title("Parsable Output Comparison", fontsize=14, fontweight="bold", pad=10)
ax2.grid(True, alpha=0.2, linestyle="-", linewidth=0.5)
ax2.set_xlim(0, 1050)
ax2.set_ylim(10, 95)
ax2.legend(loc="lower right", fontsize=11, framealpha=0.8)
ax2.tick_params(axis="both", labelsize=11)

# style spines
for ax in [ax1, ax2]:
    for spine in ax.spines.values():
        spine.set_color("#444444")
        spine.set_linewidth(0.5)

# main title
fig.suptitle("SFT on GSM8K: AdamW vs Muon", fontsize=18, fontweight="bold", y=0.98)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(
    "/mnt/nvme3n1/workspace/osilkin/mini-grpo/adamw_vs_muon_sft_gsm8k.png",
    dpi=120,
    facecolor="#1a1a1a",
    edgecolor="none",
    bbox_inches="tight",
)
print("Plot saved to adamw_vs_muon_sft_gsm8k.png")
