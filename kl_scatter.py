import marimo

__generated_with = "0.21.1"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    import numpy as np

    return mo, mpl, plt


@app.cell
def _(plt):
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 11,
        'axes.linewidth': 0.8,
        'axes.labelsize': 13,
        'axes.titlesize': 14,
        'xtick.major.width': 0.8,
        'ytick.major.width': 0.8,
        'xtick.minor.width': 0.5,
        'ytick.minor.width': 0.5,
        'figure.dpi': 150,
    })
    return


@app.cell
def _():
    # Full dataset: (label, pct_changed, frob_norm, fwd_kl, accuracy, group)
    points = [
        # --- AdamW ---
        ("BF16 trained",                  0.88,   0.221,   0.002306,  58.8, "adamw"),     # 0
        ("FP32 trained",                 78.38,   0.466,   0.129439,  62.4, "adamw"),     # 1
        ("FP32 downcast to BF16",        11.63,   0.320,   0.026841,  61.6, "adamw"),     # 2
        # --- Muon ---
        ("BF16 trained",                  0.55,   0.074,   0.000610,  29.4, "muon"),      # 3
        ("FP32 trained",                 91.93,   0.652,   0.051991,  62.1, "muon"),      # 4
        ("FP32 downcast to BF16",        15.33,   0.543,   0.021518,  61.8, "muon"),      # 5
        # --- Custom lattice ---
        ("Mantissa-10 trained",           6.14,   0.198,   0.004900,  60.7, "lattice"),   # 6
    ]

    is_muon = [False, False, False, True, True, True, False]

    short_labels = [
        "BF16", "FP32", "downcast",
        "BF16", "FP32", "downcast",
        "Mantissa-10",
    ]

    # Annotation offsets in display points (x, y) from each data point
    offsets = [
        (-15, 40),     # 0: AdamW BF16
        (15, 25),      # 1: AdamW FP32
        (-20, 30),     # 2: AdamW downcast
        (-65, 25),     # 3: Muon BF16
        (15, -30),     # 4: Muon FP32
        (20, -30),     # 5: Muon downcast
        (25, 20),      # 6: Mantissa-10
    ]

    # Each slide: list of point indices visible at that step
    slide_indices = [
        [0, 1],                              # 0: BF16 + FP32 master (AdamW)
        [0, 1, 2],                           # 1: + downcast
        [0, 1, 2, 3, 4, 5],                  # 2: + Muon points
        list(range(7)),                       # 3: + Mantissa-10
    ]

    slide_titles = [
        "The Precision Gap",
        "Downcasting Reveals Redundancy",
        "Muon Tells the Same Story",
        "Custom Lattice as a Tunable Knob",
    ]

    slide_narratives = [
        "BF16 vs FP32 master weights trained with AdamW.",
        "Downcasting the FP32 solution to BF16.",
        "Same experiments repeated with Muon.",
        "Adding a custom 10-bit mantissa lattice between BF16 and FP32.",
    ]
    return (
        is_muon,
        offsets,
        points,
        short_labels,
        slide_indices,
        slide_narratives,
        slide_titles,
    )


@app.cell
def _(mo):
    slide = mo.ui.slider(0, 3, step=1, value=0, label="")
    return (slide,)


@app.cell(hide_code=True)
def _(mo, slide, slide_narratives, slide_titles):
    _step = slide.value
    mo.vstack([
        mo.hstack([slide, mo.md(f"**Step {_step + 1} / 4**")], justify="start", gap=1),
        mo.md(f"### {slide_titles[_step]}"),
        mo.md(slide_narratives[_step]),
    ])
    return


@app.cell(hide_code=True)
def _(is_muon, mpl, offsets, plt, points, short_labels, slide, slide_indices):
    _step = slide.value
    _visible = set(slide_indices[_step])
    _prev = set(slide_indices[_step - 1]) if _step > 0 else set()
    _new = _visible - _prev

    fig, ax = plt.subplots(figsize=(12, 7))
    cmap = plt.cm.coolwarm
    norm_cmap = mpl.colors.Normalize(vmin=0.0, vmax=0.7)

    fig.suptitle('Forward KL scales with fraction of parameters changed\nand can be controlled via weight precision',
                 fontsize=14, fontweight='bold', y=0.99)
    fig.text(0.5, 0.92, 'Qwen2-1.5B-Instruct, GRPO on GSM8K (0-shot format learning, not capability acquisition)',
             ha='center', fontsize=10, color='#666666')

    _marker_map = {"adamw": "o", "muon": "D", "lattice": "*"}
    _size_map = {"adamw": 180, "muon": 180, "lattice": 300}

    # --- scatter points ---
    for i, (_, pct, frob, kl, _acc, group) in enumerate(points):
        if i not in _visible:
            continue
        _is_new = i in _new
        ax.scatter(
            pct, kl,
            c=[cmap(norm_cmap(frob))],
            s=(_size_map[group] if _is_new else _size_map[group] * 0.67),
            zorder=5 if _is_new else 4,
            edgecolors='k' if _is_new else '#999999',
            linewidth=0.7 if _is_new else 0.4,
            alpha=0.92 if _is_new else 0.28,
            marker=_marker_map[group],
        )

    # --- annotations ---
    for i in sorted(_visible):
        _, pct, frob, kl, acc, group = points[i]
        _is_new = i in _new
        _opt = {"adamw": "AdamW", "muon": "Muon", "lattice": "AdamW"}[group]
        dx, dy = offsets[i]
        ax.annotate(
            f"{_opt} {short_labels[i]} (GSM8K {acc:.1f}%)",
            (pct, kl),
            xytext=(dx, dy), textcoords='offset points',
            fontsize=8 if _is_new else 6.5,
            fontweight='bold' if _is_new else 'normal',
            ha='center', va='center',
            color='#222222' if _is_new else '#aaaaaa',
            arrowprops=dict(
                arrowstyle='->',
                lw=0.6 if _is_new else 0.35,
                color='#555555' if _is_new else '#cccccc',
            ),
        )

    # --- axes ---
    ax.set_xscale('log')
    ax.set_yscale('log')
    _xticks = [0.2, 0.5, 1, 2, 5, 10, 20, 50, 100]
    ax.set_xticks(_xticks)
    ax.set_xticklabels([f'{v:g}%' for v in _xticks])
    ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    ax.set_xlabel('Parameters Changed')
    ax.set_ylabel(
        r'$D_{\mathrm{KL}}\!(p_{\mathrm{base}} \,\|\, p_{\mathrm{ft}})$'
        ',  GSM8K prompts',
        fontsize=13,
    )
    ax.grid(True, which='major', alpha=0.18, linewidth=0.5)
    ax.grid(True, which='minor', alpha=0.08, linewidth=0.3)
    ax.set_xlim(0.15, 250)
    ax.set_ylim(1.5e-4, 0.35)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # --- legend ---
    _groups_shown = set(points[i][5] for i in _visible)
    if "adamw" in _groups_shown:
        ax.scatter([], [], marker='o', c='#bbbbbb', edgecolors='k',
                   linewidth=0.6, s=120, label='AdamW')
    if "muon" in _groups_shown:
        ax.scatter([], [], marker='D', c='#bbbbbb', edgecolors='k',
                   linewidth=0.6, s=120, label='Muon')
    if "lattice" in _groups_shown:
        ax.scatter([], [], marker='*', c='#bbbbbb', edgecolors='k',
                   linewidth=0.6, s=200, label='Custom lattice')
    ax.legend(loc='lower right', framealpha=0.9, edgecolor='#cccccc', fontsize=10)

    # --- colorbar ---
    _sm = plt.cm.ScalarMappable(cmap=cmap, norm=mpl.colors.Normalize(vmin=0.0, vmax=0.7))
    _sm.set_array([])
    _cbar = plt.colorbar(_sm, ax=ax, shrink=0.5, aspect=20, pad=0.02)
    _cbar.set_label(r'$\|\Delta W\|_F$', fontsize=11)
    _cbar.ax.tick_params(labelsize=9)
    _cbar.outline.set_linewidth(0.5)

    plt.tight_layout(rect=[0, 0, 1, 0.9])
    fig
    return


if __name__ == "__main__":
    app.run()
