#!/usr/bin/env python3
"""Plot sub-bf16 mantissa tradeoff: parity accuracy vs KL divergence."""
import json, sys, os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import torch
from rl_razor.model import MLP
from rl_razor.data import get_parity_mnist, get_fashion_mnist, create_dataloader

# ── Collect results ──────────────────────────────────────────────────────
results = {}

for exp_dir in [
    'experiments/sakana_sub_bf16_grpo30',
    'experiments/sakana_grpo30_bestckpt',
]:
    for d in sorted(Path(exp_dir).iterdir()):
        rfile = d / 'results.json'
        if not rfile.exists():
            continue
        r = json.loads(rfile.read_text())
        name = d.name.rsplit('_', 1)[0]
        parts = name.split('_')
        if parts[0] != 'grpo':
            continue
        mlabel = parts[1]
        if mlabel not in results:
            results[mlabel] = {'parity': [], 'kl': [], 'fashion': []}
        results[mlabel]['parity'].append(r['final_parity_acc'])
        results[mlabel]['kl'].append(r['final_kl_divergence'])
        results[mlabel]['fashion'].append(r['final_fashion_acc'])

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
        preds = base(x).argmax(dim=-1)
        correct += ((preds % 2) == (y % 2)).sum().item()
        total += x.size(0)
    base_parity = correct / total

    correct = total = 0
    for x, y in fashion_loader:
        x, y = x.to(device), y.to(device)
        preds = base(x).argmax(dim=-1)
        correct += (preds == y).sum().item()
        total += x.size(0)
    base_fashion = correct / total

# ── Mantissa → numeric bits for colormap ─────────────────────────────────
order = ['m1', 'm2', 'm3', 'm4', 'm5', 'm6', 'm7', 'm8', 'm9', 'm10', 'fp32']
mbits_map = {'m1': 1, 'm2': 2, 'm3': 3, 'm4': 4, 'm5': 5, 'm6': 6,
             'm7': 7, 'm8': 8, 'm9': 9, 'm10': 10, 'fp32': 23}

present = [m for m in order if m in results]
norm = plt.Normalize(vmin=1, vmax=23)
cmap = cm.viridis

# ── Plot ─────────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

offsets_left = {
    'm1': (6, -10), 'm2': (6, 2), 'm3': (6, 4), 'm4': (-8, 8),
    'm5': (6, 4), 'm6': (6, 4), 'm7': (6, 4), 'm8': (6, 4),
    'm9': (6, -10), 'm10': (6, 4), 'fp32': (6, -12),
}
offsets_right = {
    'm1': (6, 6), 'm2': (6, -10), 'm3': (-20, 10), 'm4': (6, -10),
    'm5': (6, 4), 'm6': (-20, -10), 'm7': (6, 4), 'm8': (6, 4),
    'm9': (6, 4), 'm10': (6, -12), 'fp32': (6, 6),
}

for mlabel in present:
    r = results[mlabel]
    p = np.array(r['parity'])
    k = np.array(r['kl'])
    f = np.array(r['fashion'])
    bits = mbits_map[mlabel]
    color = cmap(norm(bits))

    ax1.errorbar(k.mean(), p.mean(), xerr=k.std(), yerr=p.std(),
                 fmt='o', color=color, ms=8, capsize=3, capthick=1.5,
                 markeredgecolor='white', markeredgewidth=0.5, zorder=3)
    ax1.annotate(mlabel, (k.mean(), p.mean()), fontsize=7,
                 textcoords='offset points', xytext=offsets_left.get(mlabel, (6, 4)))

    ax2.errorbar(k.mean(), f.mean(), xerr=k.std(), yerr=f.std(),
                 fmt='o', color=color, ms=8, capsize=3, capthick=1.5,
                 markeredgecolor='white', markeredgewidth=0.5, zorder=3)
    ax2.annotate(mlabel, (k.mean(), f.mean()), fontsize=7,
                 textcoords='offset points', xytext=offsets_right.get(mlabel, (6, 4)))

# Baseline
ax1.axhline(base_parity, color='gray', ls='--', lw=1, alpha=0.7)
ax1.plot(0, base_parity, '*', color='gray', ms=12, zorder=5)
ax1.annotate('base', (0, base_parity), fontsize=7,
             textcoords='offset points', xytext=(8, -12))

ax2.axhline(base_fashion, color='gray', ls='--', lw=1, alpha=0.7)
ax2.plot(0, base_fashion, '*', color='gray', ms=12, zorder=5)
ax2.annotate('base', (0, base_fashion), fontsize=7,
             textcoords='offset points', xytext=(8, -12))

ax1.set_xlabel('Forward KL Divergence')
ax1.set_ylabel('New Task Accuracy (Parity)')
ax2.set_xlabel('Forward KL Divergence')
ax2.set_ylabel('Old Task Accuracy (Fashion)')

sm = cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=[ax1, ax2], pad=0.04, aspect=30, shrink=0.85)
cbar.set_label('Mantissa Bits')
cbar.set_ticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 23])
cbar.set_ticklabels(['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'fp32'])

fig.subplots_adjust(left=0.06, right=0.86, wspace=0.28)

out = os.path.join(os.path.dirname(__file__), '..', 'parity_mnist_v3', 'sub_bf16_tradeoff.png')
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out, dpi=200, bbox_inches='tight')
print(f'Saved: {out}')
plt.close()
