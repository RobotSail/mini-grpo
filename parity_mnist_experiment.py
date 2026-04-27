#!/usr/bin/env python3
"""
Replicate Shenfeld et al. ParityMNIST forgetting experiment with custom precision lattices.

Tests whether all precision conditions collapse to the same forgetting-vs-KL curve,
while showing that lower precision (BF16) accumulates KL more slowly due to sparse updates.
"""

import os, sys, json, copy, math, time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# ──────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────

OUTPUT_DIR = Path("parity_mnist_experiment")
SEED = 42

# Pretraining
PRETRAIN_EPOCHS = 30
PRETRAIN_LR = 1e-3
PRETRAIN_BS = 256
MNIST_SUBSET = 5000  # subset of MNIST used during pretraining

# Fine-tuning
FINETUNE_EPOCHS = 100
FINETUNE_LR = 1e-4
FINETUNE_BS = 128
CKPT_EVERY = 2  # record metrics every N epochs

# Conditions: (display_name, mantissa_bits)
# FP32 has 23 mantissa bits; BF16 has 7; custom lattices in between
CONDITIONS = [
    ("FP32",        23),
    ("BF16",         7),
    ("Mantissa-10", 10),
    ("Mantissa-12", 12),
]


# ──────────────────────────────────────────────────────────────────────
# Model
# ──────────────────────────────────────────────────────────────────────

class MLP(nn.Module):
    """3-layer MLP: 784 -> 256 -> 256 -> 10"""
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 10),
        )
    def forward(self, x):
        return self.layers(x)


# ──────────────────────────────────────────────────────────────────────
# Precision lattice snapping
# ──────────────────────────────────────────────────────────────────────

def snap_to_lattice(model, mantissa_bits):
    """Snap all parameters to a reduced-precision lattice after each optimizer step.

    For BF16 (7 bits): use native bfloat16 conversion.
    For custom lattice: view float32 as int32, add rounding bias, mask lower bits.
    For FP32 (23 bits): no-op.
    """
    if mantissa_bits >= 23:
        return
    with torch.no_grad():
        if mantissa_bits == 7:
            for p in model.parameters():
                p.data.copy_(p.data.bfloat16().float())
        else:
            shift = 23 - mantissa_bits
            bias = 1 << (shift - 1)
            mask = ~((1 << shift) - 1)
            for p in model.parameters():
                as_int = p.data.view(torch.int32)
                snapped = (as_int + bias) & mask
                p.data.copy_(snapped.view(torch.float32))


# ──────────────────────────────────────────────────────────────────────
# Evaluation helpers
# ──────────────────────────────────────────────────────────────────────

@torch.no_grad()
def parity_accuracy(model, loader, device):
    """Fraction of samples where predicted class has correct parity (even/odd)."""
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x).argmax(1)
        correct += ((pred % 2) == (y % 2)).sum().item()
        total += y.size(0)
    return correct / total


@torch.no_grad()
def classification_accuracy(model, loader, device):
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        correct += (model(x).argmax(1) == y).sum().item()
        total += y.size(0)
    return correct / total


@torch.no_grad()
def compute_forward_kl(base_model, ft_model, loader, device):
    """KL(base || finetuned) averaged per sample, summed over 10 output classes."""
    base_model.eval()
    ft_model.eval()
    kl_sum = n = 0
    for x, _ in loader:
        x = x.to(device)
        log_p = torch.log_softmax(base_model(x), dim=1)
        log_q = torch.log_softmax(ft_model(x), dim=1)
        # KL(p || q) = sum_i p_i * (log p_i - log q_i)
        kl = (log_p.exp() * (log_p - log_q)).sum(dim=1)
        kl_sum += kl.sum().item()
        n += x.size(0)
    return kl_sum / n


def compute_delta_stats(base_sd, ft_sd):
    """L0 fraction (params changed) and Frobenius norm of weight delta."""
    total = changed = 0
    frob_sq = 0.0
    for k in base_sd:
        b = base_sd[k].float()
        f = ft_sd[k].float()
        total += b.numel()
        changed += (b != f).sum().item()
        frob_sq += ((f - b) ** 2).sum().item()
    return changed / total, math.sqrt(frob_sq)


# ──────────────────────────────────────────────────────────────────────
# Pretraining: joint MNIST (digit classification) + FashionMNIST
# ──────────────────────────────────────────────────────────────────────

def pretrain(device):
    torch.manual_seed(SEED)

    tfm = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    mnist_full  = datasets.MNIST('data', train=True,  download=True, transform=tfm)
    mnist_test  = datasets.MNIST('data', train=False, download=True, transform=tfm)
    fmnist_train = datasets.FashionMNIST('data', train=True,  download=True, transform=tfm)
    fmnist_test  = datasets.FashionMNIST('data', train=False, download=True, transform=tfm)

    # Subset MNIST for pretraining
    idx = torch.randperm(len(mnist_full), generator=torch.Generator().manual_seed(SEED))[:MNIST_SUBSET]
    mnist_sub = Subset(mnist_full, idx.tolist())

    ml = DataLoader(mnist_sub,   PRETRAIN_BS, shuffle=True,  num_workers=4, pin_memory=True)
    fl = DataLoader(fmnist_train, PRETRAIN_BS, shuffle=True,  num_workers=4, pin_memory=True)
    mt = DataLoader(mnist_test,  512, num_workers=4)
    ft = DataLoader(fmnist_test, 512, num_workers=4)

    model = MLP().to(device)
    opt = optim.Adam(model.parameters(), lr=PRETRAIN_LR)
    ce = nn.CrossEntropyLoss()

    nparams = sum(p.numel() for p in model.parameters())
    print(f"Model: {nparams:,} parameters")
    print("=" * 70)
    print("PRETRAINING: Joint MNIST subset + FashionMNIST")
    print("=" * 70)

    for ep in range(1, PRETRAIN_EPOCHS + 1):
        model.train()
        mi = iter(ml)
        for fx, fy in fl:
            # Recycle MNIST subset (smaller dataset)
            try:
                mx, my = next(mi)
            except StopIteration:
                mi = iter(ml)
                mx, my = next(mi)
            # MNIST step
            mx, my = mx.to(device), my.to(device)
            opt.zero_grad(); ce(model(mx), my).backward(); opt.step()
            # FashionMNIST step
            fx, fy = fx.to(device), fy.to(device)
            opt.zero_grad(); ce(model(fx), fy).backward(); opt.step()

        if ep % 5 == 0 or ep == 1:
            pa = parity_accuracy(model, mt, device)
            da = classification_accuracy(model, mt, device)
            fa = classification_accuracy(model, ft, device)
            print(f"  Epoch {ep:2d} | Parity: {pa:.4f} | Digit: {da:.4f} | Fashion: {fa:.4f}")

    pa = parity_accuracy(model, mt, device)
    da = classification_accuracy(model, mt, device)
    fa = classification_accuracy(model, ft, device)
    print(f"\nBase model final: Parity={pa:.4f}, Digit={da:.4f}, Fashion={fa:.4f}")

    return model.state_dict()


# ──────────────────────────────────────────────────────────────────────
# Fine-tuning worker (one per GPU per condition)
# ──────────────────────────────────────────────────────────────────────

def finetune_worker(gpu_id, cond_name, mantissa_bits, base_path, out_path):
    device = f'cuda:{gpu_id}'
    torch.manual_seed(SEED)

    base_sd = torch.load(base_path, map_location=device, weights_only=True)

    tfm = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    mnist_train = datasets.MNIST('data', train=True,  download=False, transform=tfm)
    mnist_test  = datasets.MNIST('data', train=False, download=False, transform=tfm)
    fmnist_test = datasets.FashionMNIST('data', train=False, download=False, transform=tfm)

    tl = DataLoader(mnist_train, FINETUNE_BS, shuffle=True,  num_workers=2, pin_memory=True)
    mt = DataLoader(mnist_test,  512, num_workers=2)
    ft = DataLoader(fmnist_test, 512, num_workers=2)

    # Fine-tuned model (starts from fp32 base)
    model = MLP().to(device)
    model.load_state_dict(base_sd)

    # Frozen base model for KL computation
    base_model = MLP().to(device)
    base_model.load_state_dict(copy.deepcopy(base_sd))
    base_model.eval()

    # CPU copy of base weights for L0 / Frobenius comparison
    base_cpu = {k: v.cpu().clone() for k, v in base_sd.items()}

    opt = optim.AdamW(model.parameters(), lr=FINETUNE_LR)
    ce = nn.CrossEntropyLoss()

    metrics = []

    def record(epoch, epoch_l0=0.0):
        pa = parity_accuracy(model, mt, device)
        fa = classification_accuracy(model, ft, device)
        kl = compute_forward_kl(base_model, model, mt, device)
        sd_cpu = {k: v.cpu() for k, v in model.state_dict().items()}
        fc, fn = compute_delta_stats(base_cpu, sd_cpu)
        metrics.append({
            'epoch': epoch, 'parity_acc': pa, 'fashion_acc': fa,
            'forward_kl': kl, 'frac_changed': fc, 'frob_norm': fn,
            'epoch_l0': epoch_l0,
        })
        print(f"  [{cond_name:>12s}] Ep {epoch:3d} | Par {pa:.4f} | Fash {fa:.4f} | "
              f"KL {kl:.6f} | L0 {fc:.4f} | eL0 {epoch_l0:.4f} | ||dW|| {fn:.4f}", flush=True)

    print(f"\n[{cond_name}] Fine-tuning on GPU {gpu_id} "
          f"(mantissa_bits={mantissa_bits}, lr={FINETUNE_LR}, epochs={FINETUNE_EPOCHS})",
          flush=True)
    record(0)

    prev_sd = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    for ep in range(1, FINETUNE_EPOCHS + 1):
        model.train()
        step_l0_sum = 0
        step_l0_count = 0
        for x, y in tl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            ce(model(x), y).backward()
            opt.step()
            snap_to_lattice(model, mantissa_bits)

        # Measure per-epoch L0: fraction of params changed since last checkpoint
        if ep % CKPT_EVERY == 0:
            cur_sd = {k: v.cpu() for k, v in model.state_dict().items()}
            epoch_l0, _ = compute_delta_stats(prev_sd, cur_sd)
            prev_sd = {k: v.clone() for k, v in cur_sd.items()}
            record(ep, epoch_l0)

    # Save results
    with open(out_path, 'w') as f:
        json.dump({'condition': cond_name, 'mantissa_bits': mantissa_bits, 'metrics': metrics}, f, indent=2)
    print(f"[{cond_name}] Done -> {out_path}", flush=True)


# ──────────────────────────────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────────────────────────────

def make_plots(results):
    colors  = {'FP32': '#1f77b4', 'BF16': '#ff7f0e', 'Mantissa-10': '#2ca02c', 'Mantissa-12': '#d62728'}
    markers = {'FP32': 'o', 'BF16': 's', 'Mantissa-10': '^', 'Mantissa-12': 'D'}

    plot_specs = [
        ('parity_acc',  'fashion_acc',   'ParityMNIST Accuracy',         'FashionMNIST Accuracy',
         'Plot 1: Learning vs Forgetting (Pareto Frontier)', 'plot1_pareto.png'),
        ('forward_kl',  'fashion_acc',   'Forward KL from Base Model',   'FashionMNIST Accuracy',
         'Plot 2: KL vs Forgetting (Shenfeld prediction: all collapse)', 'plot2_kl_forgetting.png'),
        ('epoch',       'epoch_l0',      'Training Epoch',               'Frac Params Changed (per checkpoint interval)',
         'Plot 3: Per-Interval Update Sparsity',                         'plot3_sparsity.png'),
    ]

    # ── Combined 4-panel figure ──
    fig, axes = plt.subplots(1, 4, figsize=(24, 5.5))
    for ax, (xk, yk, xl, yl, title, _) in zip(axes[:3], plot_specs):
        for r in results:
            n = r['condition']; m = r['metrics']
            # Skip epoch 0 for epoch_l0 (it's 0 by definition)
            pts = m[1:] if yk == 'epoch_l0' else m
            ax.plot([d[xk] for d in pts], [d[yk] for d in pts],
                    f'{markers[n]}-', color=colors[n], label=n, ms=3, alpha=0.8, lw=1.5)
        ax.set_xlabel(xl); ax.set_ylabel(yl); ax.set_title(title, fontsize=10)
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    # Plot 4: Frobenius norm vs epoch
    ax = axes[3]
    for r in results:
        n = r['condition']; m = r['metrics']
        ax.plot([d['epoch'] for d in m], [d['frob_norm'] for d in m],
                f'{markers[n]}-', color=colors[n], label=n, ms=3, alpha=0.8, lw=1.5)
    ax.set_xlabel('Training Epoch'); ax.set_ylabel('||W - W_base||_F')
    ax.set_title('Plot 4: Frobenius Norm of Weight Delta', fontsize=10)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / 'all_plots.png', dpi=150, bbox_inches='tight')

    # ── Individual high-res plots ──
    for xk, yk, xl, yl, title, fname in plot_specs:
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        for r in results:
            n = r['condition']; m = r['metrics']
            pts = m[1:] if yk == 'epoch_l0' else m
            ax2.plot([d[xk] for d in pts], [d[yk] for d in pts],
                     f'{markers[n]}-', color=colors[n], label=n, ms=5, alpha=0.8, lw=2)
        ax2.set_xlabel(xl, fontsize=12); ax2.set_ylabel(yl, fontsize=12)
        ax2.set_title(title, fontsize=13)
        ax2.legend(fontsize=11); ax2.grid(True, alpha=0.3)
        fig2.tight_layout()
        fig2.savefig(OUTPUT_DIR / fname, dpi=150, bbox_inches='tight')

    # Frobenius norm individual plot
    fig3, ax3 = plt.subplots(figsize=(8, 6))
    for r in results:
        n = r['condition']; m = r['metrics']
        ax3.plot([d['epoch'] for d in m], [d['frob_norm'] for d in m],
                 f'{markers[n]}-', color=colors[n], label=n, ms=5, alpha=0.8, lw=2)
    ax3.set_xlabel('Training Epoch', fontsize=12); ax3.set_ylabel('||W - W_base||_F', fontsize=12)
    ax3.set_title('Plot 4: Frobenius Norm of Weight Delta', fontsize=13)
    ax3.legend(fontsize=11); ax3.grid(True, alpha=0.3)
    fig3.tight_layout()
    fig3.savefig(OUTPUT_DIR / 'plot4_frobenius.png', dpi=150, bbox_inches='tight')

    plt.close('all')
    print(f"\nPlots saved to {OUTPUT_DIR}/")


# ──────────────────────────────────────────────────────────────────────
# Summary
# ──────────────────────────────────────────────────────────────────────

def print_summary(results):
    W = 100
    print("\n" + "=" * W)
    print(f"{'SUMMARY TABLE':^{W}}")
    print("=" * W)

    def row(label, m):
        print(f"  {label:<15} {m['parity_acc']:>10.4f} {m['fashion_acc']:>12.4f} "
              f"{m['forward_kl']:>13.6f} {m['frac_changed']:>13.4f} {m['frob_norm']:>10.4f}")

    hdr = f"  {'Condition':<15} {'ParityAcc':>10} {'FashionAcc':>12} {'ForwardKL':>13} {'FracChanged':>13} {'FrobNorm':>10}"

    print(f"\n  Final (Epoch {FINETUNE_EPOCHS}):")
    print(hdr)
    print("  " + "-" * (W - 2))
    for r in results:
        row(r['condition'], r['metrics'][-1])

    print(f"\n  Initial (Epoch 0):")
    print(hdr)
    print("  " + "-" * (W - 2))
    for r in results:
        row(r['condition'], r['metrics'][0])

    print("=" * W)


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    t0 = time.time()

    # Step 1: Pretrain base model on GPU 0
    base_sd = pretrain('cuda:0')
    base_path = str(OUTPUT_DIR / 'base_model.pt')
    torch.save(base_sd, base_path)
    print(f"Base model saved to {base_path}")

    # Step 2: Fine-tune under each precision condition (parallel, 1 GPU each)
    print(f"\nLaunching {len(CONDITIONS)} fine-tuning conditions on GPUs 0-{len(CONDITIONS)-1}...")
    mp.set_start_method('spawn', force=True)
    procs = []
    for i, (name, bits) in enumerate(CONDITIONS):
        out = str(OUTPUT_DIR / f'results_{name}.json')
        p = mp.Process(target=finetune_worker, args=(i, name, bits, base_path, out))
        p.start()
        procs.append(p)

    for p in procs:
        p.join()
        if p.exitcode != 0:
            print(f"WARNING: {p.name} exited with code {p.exitcode}", file=sys.stderr)

    # Step 3: Collect results, plot, summarize
    results = []
    for name, _ in CONDITIONS:
        with open(OUTPUT_DIR / f'results_{name}.json') as f:
            results.append(json.load(f))

    make_plots(results)
    print_summary(results)

    elapsed = time.time() - t0
    print(f"\nTotal experiment time: {elapsed:.0f}s ({elapsed/60:.1f}m)")


if __name__ == '__main__':
    main()
