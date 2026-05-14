#!/usr/bin/env python3
"""Fig 2a: Pareto plots — New vs Old task accuracy, and Old task accuracy vs KL.

Generates three outputs:
  - fig2a_pareto_size_labels.png/pdf  (combined 2-panel)
  - fig2a_forgetting.png/pdf          (left panel: parity vs fashion)
  - fig2a_kl_divergence.png/pdf       (right panel: fashion vs KL)

Methods as colors, mantissa widths as marker sizes (small=m1, large=m23).
Labels at m1, m7, m10, m23 landmarks with arrow connectors.
"""
import json, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path
from collections import defaultdict

# ── Data loading ─────────────────────────────────────────────────────────
MANTISSA_MAP = {1:"m1",2:"m2",3:"m3",4:"m4",5:"m5",6:"m6",
                7:"m7",8:"m8",9:"m9",10:"m10",0:"m23",23:"m23"}

def parse_dir(d):
    results = {}
    for rfile in d.glob("*/results.json"):
        with open(rfile) as f: r = json.load(f)
        c = r.get("config", {})
        method = r.get("method", c.get("method",""))
        mbits = c.get("mantissa_bits", 0)
        mlabel = MANTISSA_MAP.get(mbits, f"m{mbits}")
        seed = c.get("seed", 0)
        results[(method, mlabel, mbits, seed)] = {
            "parity": r.get("final_parity_acc", 0),
            "fashion": r.get("final_fashion_acc", 0),
            "kl": r.get("final_kl_divergence", 0),
        }
    return results

SCRIPT_DIR = Path(__file__).resolve().parent
EXP_DIR = SCRIPT_DIR.parent / "experiments"

combined = {}
for k, v in parse_dir(EXP_DIR / "sakana_sub_bf16_grpo30").items(): combined[k] = v
for k, v in parse_dir(EXP_DIR / "sakana_sub_bf16_all5").items(): combined[k] = v
for k, v in parse_dir(EXP_DIR / "sakana_grpo30_bestckpt").items():
    if k[0] == "grpo": combined[k] = v
for k, v in parse_dir(EXP_DIR / "sakana_all5_bestckpt").items():
    if k[0] != "grpo": combined[k] = v

agg = defaultdict(list)
for (method, mlabel, mbits, seed), v in combined.items():
    agg[(method, mlabel, mbits)].append(v)

# ── Constants ────────────────────────────────────────────────────────────
MANTISSAS_ALL = [(1,"m1"),(2,"m2"),(3,"m3"),(4,"m4"),(5,"m5"),(6,"m6"),
                 (7,"m7"),(8,"m8"),(9,"m9"),(10,"m10"),(23,"m23")]
BASE_P, BASE_F = 93.11, 78.18
TR = (10, 8)  # default top-right offset

SIZE_BY_INDEX = {ml: 2.5 + 6.5 * i / (len(MANTISSAS_ALL) - 1)
                 for i, (_, ml) in enumerate(MANTISSAS_ALL)}

METHOD_COLORS = {'sft1': '#d62728', 'sft2': '#ff7f0e', 'oracle': '#2ca02c',
                 'grpo': '#1f77b4', 'grpo_kl': '#9467bd'}
METHOD_LABELS = {'sft1': 'SFT-1', 'sft2': 'SFT-2', 'oracle': 'Oracle',
                 'grpo': 'GRPO', 'grpo_kl': 'GRPO+KL'}
METHODS = ["sft1", "sft2", "oracle", "grpo", "grpo_kl"]

# ── Precompute means ─────────────────────────────────────────────────────
pts = {}
for method in METHODS:
    for mbits, ml in MANTISSAS_ALL:
        vals = agg.get((method, ml, mbits), [])
        if not vals: continue
        pts[(method, ml)] = {
            'p': np.mean([v["parity"]*100 for v in vals]),
            'ps': np.std([v["parity"]*100 for v in vals]),
            'f': np.mean([v["fashion"]*100 for v in vals]),
            'fs': np.std([v["fashion"]*100 for v in vals]),
            'k': np.mean([v["kl"] for v in vals]),
            'ks': np.std([v["kl"] for v in vals]),
        }

def should_share(ml, xkey):
    coords = []
    for m in METHODS:
        p = pts.get((m, ml))
        if not p: continue
        coords.append((p['p'] if xkey=='parity' else p['k'], p['f']))
    if len(coords) < 3: return False
    xs, ys = [c[0] for c in coords], [c[1] for c in coords]
    return (max(xs)-min(xs) < 0.15) and (max(ys)-min(ys) < 0.5)

def make_arrow(color):
    return dict(arrowstyle='->', color=color, lw=0.7, alpha=0.8,
                shrinkA=0, shrinkB=5)

# ── Label config ─────────────────────────────────────────────────────────
LEFT_LABELS = {"m1", "m7", "m10", "m23"}
LEFT_OVERRIDES = {
    ('shared', 'm1'):    (-10, -14),
    ('oracle', 'm7'):    (10, 12),
    ('oracle', 'm10'):   (-14, 12),
    ('oracle', 'm23'):   (10, 12),
    ('grpo', 'm10'):     (12, -12),
    ('grpo', 'm23'):     (12, 10),
    ('grpo_kl', 'm10'):  (-18, -14),
    ('grpo_kl', 'm23'):  (10, -10),
}

RIGHT_LABELS_PER_METHOD = {
    'sft1': {"m7","m10","m23"}, 'sft2': {"m7","m10","m23"},
    'oracle': {"m23"}, 'grpo': {"m7","m10","m23"}, 'grpo_kl': {"m23"},
}
RIGHT_OVERRIDES = {
    ('grpo_kl', 'm23'):  (10, -10),
    ('grpo', 'm7'):      (-18, -10),
    ('grpo', 'm10'):     (-18, -10),
    ('grpo', 'm23'):     (12, 10),
    ('sft2', 'm7'):      (-16, -10),
    ('sft2', 'm10'):     (10, 10),
    ('sft2', 'm23'):     (10, 10),
}

# ── Style ────────────────────────────────────────────────────────────────
RC_PARAMS = {
    'font.family': 'serif', 'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 10, 'mathtext.fontset': 'cm',
    'axes.labelsize': 11, 'xtick.labelsize': 9, 'ytick.labelsize': 9,
    'legend.fontsize': 8,
    'axes.linewidth': 0.6, 'axes.facecolor': 'white', 'figure.facecolor': 'white',
    'axes.grid': True, 'grid.alpha': 0.2, 'grid.linewidth': 0.4,
    'xtick.major.width': 0.5, 'ytick.major.width': 0.5,
    'axes.spines.top': False, 'axes.spines.right': False,
}

def make_legend():
    handles = [Line2D([0],[0], marker='*', color='#555555', linestyle='None',
                      markersize=10, label='Base model')]
    for m in METHODS:
        handles.append(Line2D([0],[0], color=METHOD_COLORS[m], linewidth=2,
                              label=METHOD_LABELS[m]))
    return handles

def draw_panel(ax, xkey, xlabel, lbl_fn, ovr):
    ax.axhline(y=BASE_F, color='gray', linestyle='--', linewidth=0.8, alpha=0.6, zorder=2)
    if xkey == 'parity':
        ax.axvline(x=BASE_P, color='gray', linestyle='--', linewidth=0.8, alpha=0.6, zorder=2)
        ax.plot(BASE_P, BASE_F, '*', color='#555555', markersize=14, zorder=10)
    else:
        ax.plot(0, BASE_F, '*', color='#555555', markersize=14, zorder=10)

    is_left = (xkey == 'parity')
    shared_done = set()
    for method in METHODS:
        col = METHOD_COLORS[method]
        xs, ys = [], []
        for mbits, ml in MANTISSAS_ALL:
            p = pts.get((method, ml))
            if not p: continue
            xm = p['p'] if xkey=='parity' else p['k']
            xe = p['ps'] if xkey=='parity' else p['ks']
            fm, fe = p['f'], p['fs']
            ms = SIZE_BY_INDEX[ml]
            ax.errorbar(xm, fm, xerr=xe, yerr=fe, fmt='none', ecolor=col,
                        capsize=2, capthick=0.8, elinewidth=0.8, alpha=0.5, zorder=4)
            ax.plot(xm, fm, 'o', color=col, markersize=ms,
                    markeredgecolor='k', markeredgewidth=0.3, alpha=0.8, zorder=5)
            label_set = lbl_fn(method)
            if ml in label_set:
                if is_left and should_share(ml, xkey):
                    if ml not in shared_done:
                        shared_done.add(ml)
                        off = ovr.get(('shared', ml), TR)
                        ax.annotate(ml, (xm, fm), fontsize=6, color='#555555',
                                    alpha=0.9, textcoords='offset points', xytext=off,
                                    arrowprops=make_arrow('#555555'), zorder=8)
                else:
                    off = ovr.get((method, ml), TR)
                    ax.annotate(ml, (xm, fm), fontsize=5.5, color=col, alpha=0.9,
                                textcoords='offset points', xytext=off,
                                arrowprops=make_arrow(col), zorder=8)
            xs.append(xm); ys.append(fm)
        ax.plot(xs, ys, '-', color=col, alpha=0.3, linewidth=1.2, zorder=2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Fashion Accuracy (%) \u2014 Old Task')

# ── Output directory ─────────────────────────────────────────────────────
OUT = Path("/mnt/nvme3n1/workspace/osilkin/mini-grpo/parity_mnist_v3")

matplotlib.rcParams.update(RC_PARAMS)

# ── Combined 2-panel ─────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.5))
draw_panel(ax1, 'parity', 'Parity Accuracy (%) \u2014 New Task',
           lambda m: LEFT_LABELS, LEFT_OVERRIDES)
draw_panel(ax2, 'kl', 'Forward KL Divergence',
           lambda m: RIGHT_LABELS_PER_METHOD[m], RIGHT_OVERRIDES)
ax1.legend(handles=make_legend(), loc='lower left', frameon=True, fancybox=False,
           edgecolor='black', framealpha=0.95, ncol=1, handletextpad=0.4)
fig.tight_layout()
fig.savefig(OUT / 'fig2a_pareto_size_labels.png', dpi=300, bbox_inches='tight')
fig.savefig(OUT / 'fig2a_pareto_size_labels.pdf', bbox_inches='tight')
plt.close()
print("Saved: fig2a_pareto_size_labels")

# ── Standalone: Forgetting (parity vs fashion) ───────────────────────────
fig, ax = plt.subplots(figsize=(6, 5.5))
draw_panel(ax, 'parity', 'Parity Accuracy (%) \u2014 New Task',
           lambda m: LEFT_LABELS, LEFT_OVERRIDES)
ax.legend(handles=make_legend(), loc='lower left', frameon=True, fancybox=False,
          edgecolor='black', framealpha=0.95, ncol=1, handletextpad=0.4)
fig.tight_layout()
fig.savefig(OUT / 'fig2a_forgetting.png', dpi=300, bbox_inches='tight')
fig.savefig(OUT / 'fig2a_forgetting.pdf', bbox_inches='tight')
plt.close()
print("Saved: fig2a_forgetting")

# ── Standalone: KL divergence (fashion vs KL) ────────────────────────────
fig, ax = plt.subplots(figsize=(6, 5.5))
draw_panel(ax, 'kl', 'Forward KL Divergence',
           lambda m: RIGHT_LABELS_PER_METHOD[m], RIGHT_OVERRIDES)
ax.legend(handles=make_legend(), loc='lower left', frameon=True, fancybox=False,
          edgecolor='black', framealpha=0.95, ncol=1, handletextpad=0.4)
fig.tight_layout()
fig.savefig(OUT / 'fig2a_kl_divergence.png', dpi=300, bbox_inches='tight')
fig.savefig(OUT / 'fig2a_kl_divergence.pdf', bbox_inches='tight')
plt.close()
print("Saved: fig2a_kl_divergence")
