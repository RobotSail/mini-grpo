import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 12,
    'axes.linewidth': 0.8,
    'axes.labelsize': 14,
    'axes.titlesize': 15,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    'xtick.minor.width': 0.5,
    'ytick.minor.width': 0.5,
    'figure.dpi': 200,
})

# All points: (label, pct_changed, frob_norm, fwd_kl, accuracy, group)
all_points = [
    # AdamW master weights
    ("BF16 trained",                  0.88,   0.221,   0.002306,  58.8, "adamw"),
    ("FP32 trained",                 78.38,   0.466,   0.129439,  62.4, "adamw"),
    # AdamW downcast
    ("FP32 downcast to BF16",        11.63,   0.320,   0.026841,  61.6, "adamw"),
    ("FP32 values at BF16 positions", 0.88,   0.051,   0.000466,  17.7, "adamw"),
    # Muon
    ("BF16 trained",                  0.55,   0.074,   0.000610,  29.4, "muon"),
    ("FP32 trained",                 91.93,   0.652,   0.051991,  62.1, "muon"),
    ("FP32 downcast to BF16",        15.33,   0.543,   0.021518,  61.8, "muon"),
    ("BF16 shadow of FP32",           0.55,   0.051,   0.000477,   7.5, "muon"),
    # Custom lattice
    ("Mantissa-10 trained",           6.14,   0.198,   0.004900,  60.7, "lattice"),
]

# Annotation positions for each point
ann_pos = [
    # AdamW master
    (0.22,   0.004,   'left'),   # BF16 trained
    (25.0,   0.18,    'left'),   # FP32 trained
    # AdamW downcast
    (2.5,    0.022,   'left'),   # downcast
    (2.5,    0.00028, 'left'),   # mask
    # Muon
    (0.18,   0.00085, 'left'),   # BF16 trained
    (25.0,   0.04,    'left'),   # FP32 trained
    (5.0,    0.012,   'left'),   # downcast
    (0.18,   0.00028, 'left'),   # shadow
    # Lattice
    (15.0,   0.004,   'left'),   # Mantissa-10 trained
]

# Slide definitions: each slide shows all previous + new points (by index)
slides = [
    {"name": "slide1_master_weights",   "indices": [0, 1]},            # BF16 + FP32 trained (AdamW)
    {"name": "slide2_downcast",         "indices": [0, 1, 2, 3]},      # + downcast + mask
    {"name": "slide3_muon",             "indices": [0, 1, 2, 3, 4, 5, 6, 7]},  # + all Muon
    {"name": "slide4_lattice",          "indices": list(range(9))},     # + Mantissa-10
]

cmap = plt.cm.coolwarm
norm_cmap = mpl.colors.Normalize(vmin=0.0, vmax=0.7)
marker_map = {"adamw": "o", "muon": "D", "lattice": "*"}
size_map = {"adamw": 200, "muon": 200, "lattice": 350}


def make_slide(visible_indices, filename):
    fig, ax = plt.subplots(figsize=(11, 7))

    for i in visible_indices:
        label, pct, norm, kl, acc, group = all_points[i]
        color = cmap(norm_cmap(norm))
        ax.scatter(pct, kl, c=[color], s=size_map[group], zorder=5,
                   edgecolors='k', linewidth=0.7, alpha=0.9, marker=marker_map[group])

    for i in visible_indices:
        pt = all_points[i]
        label, pct, norm, kl, acc, group = pt
        lx, ly, ha = ann_pos[i]
        prefix = "(Muon) " if group == "muon" else ""
        text = (f"{prefix}{label}\n"
                r"$\|\Delta W\|_F$ = " + f"{norm:.3f}\n"
                f"acc = {acc:.1f}%")
        ax.annotate(
            text, (pct, kl),
            xytext=(lx, ly),
            textcoords='data',
            fontsize=8, ha=ha, va='center',
            arrowprops=dict(arrowstyle='->', color='#888888', lw=0.6,
                            connectionstyle='arc3,rad=0.15'),
        )

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Parameters Changed (%)')
    ax.set_ylabel(r'$D_{\mathrm{KL}}(p_{\mathrm{base}} \,\|\, p_{\mathrm{fine}})$' + ',  GSM8K prompts',
                  fontsize=13)

    fig.suptitle('Forward KL scales with fraction of parameters changed\nand can be controlled via weight precision',
                 fontsize=14, fontweight='bold', y=0.99)
    fig.text(0.5, 0.92, 'Qwen2-1.5B-Instruct, GRPO on GSM8K (0-shot format learning, not capability acquisition)',
             ha='center', fontsize=10, color='#666666')

    ax.grid(True, which='major', alpha=0.15, linewidth=0.5)
    ax.grid(True, which='minor', alpha=0.07, linewidth=0.3)
    ax.set_xlim(0.12, 250)
    ax.set_ylim(1.5e-4, 0.35)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Marker legend — show all groups that appear in this slide
    groups_shown = set(all_points[i][5] for i in visible_indices)
    if "adamw" in groups_shown:
        ax.scatter([], [], marker='o', c='#bbbbbb', edgecolors='k', linewidth=0.6, s=150, label='AdamW')
    if "muon" in groups_shown:
        ax.scatter([], [], marker='D', c='#bbbbbb', edgecolors='k', linewidth=0.6, s=150, label='Muon')
    if "lattice" in groups_shown:
        ax.scatter([], [], marker='*', c='#bbbbbb', edgecolors='k', linewidth=0.6, s=250, label='Custom lattice')
    ax.legend(loc='lower right', framealpha=0.9, edgecolor='#cccccc', fontsize=10)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=mpl.colors.Normalize(vmin=0.0, vmax=0.7))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.45, aspect=18, pad=0.02)
    cbar.set_label(r'$\|\Delta W\|_F$', fontsize=12)
    cbar.ax.tick_params(labelsize=10)
    cbar.outline.set_linewidth(0.5)

    plt.tight_layout(rect=[0, 0, 1, 0.9])
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {filename}")


for slide in slides:
    make_slide(slide["indices"], f"parity_mnist_results/{slide['name']}.png")

print("Done — all slides generated")
