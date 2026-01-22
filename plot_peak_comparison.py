"""Bar plot comparing peak test accuracy for AdamW/Muon × SFT/GRPO on GSM8K."""

import matplotlib.pyplot as plt
import numpy as np

# peak accuracy scores from each configuration
peak_scores = {
    'AdamW + SFT': 73.24,   # step 350
    'Muon + SFT': 53.85,    # step 400
    'AdamW + GRPO': 95.18,  # step 800
    'Muon + GRPO': 86.22,   # step 800
}

labels = list(peak_scores.keys())
values = list(peak_scores.values())

# set up the plot
plt.style.use('dark_background')
fig, ax = plt.subplots(figsize=(10, 7))

# colors: warm for AdamW, cool for Muon, saturated for GRPO, muted for SFT
colors = ['#FF9F1C', '#2EC4B6', '#FF6B35', '#00B4D8']

# create bars
bars = ax.bar(labels, values, color=colors, edgecolor='white', linewidth=1.5, width=0.65)

# add value labels on top of bars
for bar, val in zip(bars, values):
    height = bar.get_height()
    ax.annotate(f'{val:.2f}%',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 8),
                textcoords="offset points",
                ha='center', va='bottom',
                fontsize=14, fontweight='bold', color='white')

# styling
ax.set_ylabel('Peak Test Accuracy (%)', fontsize=14, fontweight='medium')
ax.set_title('GSM8K Peak Test Accuracy: Optimizer × Training Method', 
             fontsize=16, fontweight='bold', pad=15)

ax.set_ylim(0, 105)
ax.set_xlim(-0.6, 3.6)

# add horizontal grid lines
ax.yaxis.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
ax.set_axisbelow(True)

# style the axes
ax.tick_params(axis='x', labelsize=12, rotation=0)
ax.tick_params(axis='y', labelsize=11)

for spine in ax.spines.values():
    spine.set_color('#444444')
    spine.set_linewidth(0.5)

# add a reference line at random baseline (~8.5% for GSM8K with 8 choices approx)
# actually GSM8K is open-ended, so no meaningful random baseline

# add separator between SFT and GRPO groups
ax.axvline(x=1.5, color='#666666', linestyle='--', linewidth=1, alpha=0.7)
ax.text(0.5, 98, 'SFT', ha='center', fontsize=12, color='#888888', style='italic')
ax.text(2.5, 98, 'GRPO', ha='center', fontsize=12, color='#888888', style='italic')

plt.tight_layout()
plt.savefig('/mnt/nvme3n1/workspace/osilkin/mini-grpo/peak_accuracy_comparison.png', 
            dpi=120, facecolor='#1a1a1a', edgecolor='none', bbox_inches='tight')
print("Plot saved to peak_accuracy_comparison.png")
