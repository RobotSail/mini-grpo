#!/usr/bin/env python3
"""Full mantissa tradeoff plot: all methods × all mantissa widths (m1-fp32).

Two panels:
  Left:  New task accuracy (Parity) vs Forward KL
  Right: Old task accuracy (Fashion) vs Forward KL
Each method is a separate series with its own marker shape.
"""
import json, sys, os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import torch
from rl_razor.model import MLP
from rl_razor.data import get_parity_mnist, get_fashion_mnist, create_dataloader

# ── Collect all results ──────────────────────────────────────────────────
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
        name = d.name.rsplit('_', 1)[0]  # strip timestamp
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
    base_parity = correct / total

    correct = total = 0
    for x, y in fashion_loader:
        x, y = x.to(device), y.to(device)
        correct += (base(x).argmax(-1) == y).sum().item()
        total += x.size(0)
    base_fashion = correct / total

# ── Plot config ──────────────────────────────────────────────────────────
method_style = {
    'grpo':    {'marker': 'o', 'label': 'GRPO',    'color': '#1f77b4'},
    'grpo_kl': {'marker': 's', 'label': 'GRPO+KL', 'color': '#ff7f0e'},
    'sft1':    {'marker': '^', 'label': 'SFT-1',   'color': '#2ca02c'},
    'sft2':    {'marker': 'D', 'label': 'SFT-2',   'color': '#d62728'},
    'oracle':  {'marker': 'v', 'label': 'Oracle',  'color': '#9467bd'},
}

mantissa_order = ['m1', 'm2', 'm3', 'm4', 'm5', 'm6', 'm7', 'm8', 'm9', 'm10', 'fp32']
mbits_map = {'m1': 1, 'm2': 2, 'm3': 3, 'm4': 4, 'm5': 5, 'm6': 6,
             'm7': 7, 'm8': 8, 'm9': 9, 'm10': 10, 'fp32': 23}

norm = plt.Normalize(vmin=1, vmax=23)
cmap = cm.viridis

# ── Create figure ────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

for method, style in method_style.items():
    if method not in data:
        continue
    mdata = data[method]
    present = [m for m in mantissa_order if m in mdata and len(mdata[m]['parity']) > 0]
    if not present:
        continue

    kl_means = [np.mean(mdata[m]['kl']) for m in present]
    par_means = [np.mean(mdata[m]['parity']) for m in present]
    par_stds = [np.std(mdata[m]['parity']) for m in present]
    fash_means = [np.mean(mdata[m]['fashion']) for m in present]
    fash_stds = [np.std(mdata[m]['fashion']) for m in present]
    kl_stds = [np.std(mdata[m]['kl']) for m in present]
    colors = [cmap(norm(mbits_map[m])) for m in present]

    for j, m in enumerate(present):
        ax1.errorbar(kl_means[j], par_means[j],
                     xerr=kl_stds[j], yerr=par_stds[j],
                     fmt=style['marker'], color=colors[j], ms=7,
                     capsize=2, capthick=1, markeredgecolor='white',
                     markeredgewidth=0.4, zorder=3, alpha=0.85)

        ax2.errorbar(kl_means[j], fash_means[j],
                     xerr=kl_stds[j], yerr=fash_stds[j],
                     fmt=style['marker'], color=colors[j], ms=7,
                     capsize=2, capthick=1, markeredgecolor='white',
                     markeredgewidth=0.4, zorder=3, alpha=0.85)

    # Connect points with a line for this method
    sort_idx = np.argsort(kl_means)
    ax1.plot([kl_means[i] for i in sort_idx], [par_means[i] for i in sort_idx],
             '-', color=style['color'], alpha=0.3, lw=1.5, zorder=1)
    ax2.plot([kl_means[i] for i in sort_idx], [fash_means[i] for i in sort_idx],
             '-', color=style['color'], alpha=0.3, lw=1.5, zorder=1)

# Baseline
for ax, base_val in [(ax1, base_parity), (ax2, base_fashion)]:
    ax.axhline(base_val, color='gray', ls='--', lw=1, alpha=0.6)
    ax.plot(0, base_val, '*', color='gray', ms=14, zorder=5)
    ax.annotate('base', (0, base_val), fontsize=7,
                textcoords='offset points', xytext=(8, -12))

# Legend for methods (marker shapes)
legend_handles = []
for method, style in method_style.items():
    if method in data:
        h = plt.Line2D([0], [0], marker=style['marker'], color='gray',
                       label=style['label'], ms=8, linestyle='',
                       markeredgecolor='white', markeredgewidth=0.4)
        legend_handles.append(h)

ax1.legend(handles=legend_handles, loc='lower right', fontsize=8, framealpha=0.9)

ax1.set_xlabel('Forward KL Divergence')
ax1.set_ylabel('New Task Accuracy (Parity)')
ax2.set_xlabel('Forward KL Divergence')
ax2.set_ylabel('Old Task Accuracy (Fashion)')

# Colorbar
sm = cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=[ax1, ax2], pad=0.04, aspect=30, shrink=0.85)
cbar.set_label('Mantissa Bits')
cbar.set_ticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 23])
cbar.set_ticklabels(['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'fp32'])

fig.subplots_adjust(left=0.06, right=0.86, wspace=0.28)

out_dir = os.path.join(os.path.dirname(__file__), '..', 'parity_mnist_v3')
out = os.path.join(out_dir, 'full_mantissa_tradeoff.png')
fig.savefig(out, dpi=200, bbox_inches='tight')
print(f'Saved: {out}')

# Also copy to mini-grpo server dir
import shutil
server_dir = '/mnt/nvme3n1/workspace/osilkin/mini-grpo/parity_mnist_v3'
if os.path.isdir(server_dir):
    shutil.copy2(out, os.path.join(server_dir, 'full_mantissa_tradeoff.png'))
    print(f'Copied to: {server_dir}/full_mantissa_tradeoff.png')

plt.close()
