#!/usr/bin/env python3
"""Fig 2a: Pareto plot — Parity Accuracy vs Fashion Accuracy.

Methods as colors, mantissa widths as marker shapes.
Matches the ICML formatting from fig2a_pareto_icml.png.
"""
import json, sys, os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import torch
from rl_razor.model import MLP
from rl_razor.data import get_parity_mnist, get_fashion_mnist, create_dataloader

# ── Collect results ──────────────────────────────────────────────────────
data = defaultdict(lambda: defaultdict(lambda: {'parity': [], 'kl': [], 'fashion': []}))

exp_dirs = [
    'experiments/sakana_sub_bf16_grpo30',
    'experiments/sakana_sub_bf16_all5',
    'experiments/sakana_grpo30_bestckpt',
    'experiments/sakana_all5_bestckpt',
]

for exp_dir in exp_dirs:
    p = Path(exp_dir)
    if not p.exists():
        continue
    for d in sorted(p.iterdir()):
        rfile = d / 'results.json'
        if not rfile.exists():
            continue
        r = json.loads(rfile.read_text())
        name = d.name.rsplit('_', 1)[0]
        parts = name.split('_')
        method = parts[0]
        mlabel = parts[1]
        if method == 'grpo' and mlabel == 'kl':
            method = 'grpo_kl'
            mlabel = parts[2]
        data[method][mlabel]['parity'].append(r['final_parity_acc'])
        data[method][mlabel]['kl'].append(r['final_kl_divergence'])
        data[method][mlabel]['fashion'].append(r['final_fashion_acc'])

# ── Baseline ─────────────────────────────────────────────────────────────
device = 'cuda' if torch.cuda.is_available() else 'cpu'
base = MLP.from_checkpoint('experiments/mantissa_sweep/pretrain/pretrained_model.pt', device=device)
base.eval()
parity_loader = create_dataloader(get_parity_mnist(train=False), batch_size=256, shuffle=False)
fashion_loader = create_dataloader(get_fashion_mnist(train=False), batch_size=256, shuffle=False)

with torch.no_grad():
    correct = total = 0
    for x, y in parity_loader:
        x, y = x.to(device), y.to(device)
        correct += ((base(x).argmax(-1) % 2) == (y % 2)).sum().item()
        total += x.size(0)
    base_parity = correct / total * 100

    correct = total = 0
    for x, y in fashion_loader:
        x, y = x.to(device), y.to(device)
        correct += (base(x).argmax(-1) == y).sum().item()
        total += x.size(0)
    base_fashion = correct / total * 100

# ── Style config ─────────────────────────────────────────────────────────
method_order = ['sft1', 'sft2', 'oracle', 'grpo', 'grpo_kl']
method_style = {
    'sft1':    {'color': '#d62728', 'label': 'SFT-1',   'zorder': 3},
    'sft2':    {'color': '#ff7f0e', 'label': 'SFT-2',   'zorder': 3},
    'oracle':  {'color': '#2ca02c', 'label': 'Oracle',  'zorder': 3},
    'grpo':    {'color': '#1f77b4', 'label': 'GRPO',    'zorder': 4},
    'grpo_kl': {'color': '#9467bd', 'label': 'GRPO+KL', 'zorder': 3},
}

mantissa_order = ['m1', 'm2', 'm3', 'm4', 'm5', 'm6', 'm7', 'm8', 'm9', 'm10', 'fp32']
mantissa_marker = {
    'm1':  'P',   # plus (filled)
    'm2':  'X',   # x (filled)
    'm3':  'h',   # hexagon
    'm4':  'p',   # pentagon
    'm5':  'd',   # thin diamond
    'm6':  '8',   # octagon
    'm7':  'o',   # circle
    'm8':  'D',   # diamond
    'm9':  '^',   # triangle up
    'm10': 'v',   # triangle down
    'fp32': 's',  # square
}
mantissa_label = {
    'm1': 'm1', 'm2': 'm2', 'm3': 'm3', 'm4': 'm4', 'm5': 'm5', 'm6': 'm6',
    'm7': 'm7 (bf16)', 'm8': 'm8', 'm9': 'm9', 'm10': 'm10', 'fp32': 'm23 (fp32)',
}

# ── Plot ─────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 7))

for method in method_order:
    if method not in data:
        continue
    mdata = data[method]
    style = method_style[method]
    present = [m for m in mantissa_order if m in mdata and len(mdata[m]['parity']) > 0]
    if not present:
        continue

    par_means = [np.mean(mdata[m]['parity']) * 100 for m in present]
    par_stds = [np.std(mdata[m]['parity']) * 100 for m in present]
    fash_means = [np.mean(mdata[m]['fashion']) * 100 for m in present]
    fash_stds = [np.std(mdata[m]['fashion']) * 100 for m in present]

    # Connect with line (sorted by parity acc)
    sort_idx = np.argsort(par_means)
    ax.plot([par_means[i] for i in sort_idx], [fash_means[i] for i in sort_idx],
            '-', color=style['color'], alpha=0.35, lw=2, zorder=1)

    for j, m in enumerate(present):
        filled = m not in ['fp32']
        marker = mantissa_marker[m]
        fc = style['color'] if filled else 'none'
        ec = style['color']
        ax.errorbar(par_means[j], fash_means[j],
                    xerr=par_stds[j], yerr=fash_stds[j],
                    fmt='none', ecolor=style['color'], capsize=2,
                    capthick=1, alpha=0.6, zorder=style['zorder'])
        ax.scatter(par_means[j], fash_means[j],
                   marker=marker, s=80, facecolors=fc, edgecolors=ec,
                   linewidths=1.5, zorder=style['zorder'] + 1)

# Baseline
ax.axhline(base_fashion, color='gray', ls='--', lw=1, alpha=0.5)
ax.axvline(base_parity, color='gray', ls='--', lw=1, alpha=0.5)
ax.plot(base_parity, base_fashion, '*', color='#555555', ms=18, zorder=10)

# Legend — methods (colors)
method_handles = []
for method in method_order:
    if method not in data:
        continue
    style = method_style[method]
    h = plt.Line2D([0], [0], color=style['color'], lw=2, label=style['label'])
    method_handles.append(h)
h_base = plt.Line2D([0], [0], marker='*', color='#555555', ms=12,
                     linestyle='', label='Base model')
method_handles.insert(0, h_base)

# Legend — mantissa shapes
shape_handles = []
for m in mantissa_order:
    marker = mantissa_marker[m]
    fc = 'gray' if m != 'fp32' else 'none'
    h = plt.Line2D([0], [0], marker=marker, color='gray', ms=8,
                   linestyle='', markerfacecolor=fc, markeredgecolor='gray',
                   markeredgewidth=1.5, label=mantissa_label[m])
    shape_handles.append(h)

leg1 = ax.legend(handles=method_handles, loc='lower left', fontsize=9,
                 framealpha=0.95, edgecolor='black', fancybox=False)
ax.add_artist(leg1)
leg2 = ax.legend(handles=shape_handles, loc='lower left', fontsize=8,
                 framealpha=0.95, edgecolor='black', fancybox=False,
                 bbox_to_anchor=(0.26, 0.0), ncol=2)

ax.set_xlabel('Parity Accuracy (%) \u2014 New Task', fontsize=12)
ax.set_ylabel('Fashion Accuracy (%) \u2014 Old Task', fontsize=12)
ax.tick_params(labelsize=10)

fig.tight_layout()

out_dir = '/mnt/nvme3n1/workspace/osilkin/mini-grpo/parity_mnist_v3'
out = os.path.join(out_dir, 'fig2a_pareto_full_icml_v2.png')
fig.savefig(out, dpi=200, bbox_inches='tight')
print(f'Saved: {out}')
plt.close()
