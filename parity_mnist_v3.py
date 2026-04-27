#!/usr/bin/env python3
"""
ParityMNIST v3: Correct per-checkpoint L0 (vs lattice-snapped base),
no EMA, matched 100-epoch duration for both SFT and RL.
"""

import os, sys, json, copy, math, time, shutil
import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from torch.distributions import Categorical
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUTPUT_DIR = Path("parity_mnist_v3")
SEED = 42
N_EPOCHS = 100  # same for SFT and RL
LR = 1e-4
BS = 128
CKPT_EVERY = 2
CONDITIONS = [("FP32", 23), ("BF16", 7), ("Mantissa-10", 10), ("Mantissa-12", 12)]


class MLP(nn.Module):
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


def snap_tensor(x, mantissa_bits):
    """Snap a single tensor to a precision lattice."""
    if mantissa_bits >= 23:
        return x.clone()
    if mantissa_bits == 7:
        return x.bfloat16().float()
    scale = float(1 << (mantissa_bits + 1))
    m, e = torch.frexp(x)
    return torch.ldexp((m * scale).round() / scale, e)


def snap_to_lattice(model, mantissa_bits):
    if mantissa_bits >= 23:
        return
    with torch.no_grad():
        if mantissa_bits == 7:
            for p in model.parameters():
                p.data.copy_(p.data.bfloat16().float())
        else:
            scale = float(1 << (mantissa_bits + 1))
            for p in model.parameters():
                m, e = torch.frexp(p.data)
                p.data.copy_(torch.ldexp((m * scale).round() / scale, e))


@torch.no_grad()
def parity_accuracy(model, loader, device):
    model.eval()
    c = t = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        c += ((model(x).argmax(1) % 2) == (y % 2)).sum().item(); t += y.size(0)
    return c / t

@torch.no_grad()
def classification_accuracy(model, loader, device):
    model.eval()
    c = t = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        c += (model(x).argmax(1) == y).sum().item(); t += y.size(0)
    return c / t

@torch.no_grad()
def compute_forward_kl(base_model, ft_model, loader, device):
    base_model.eval(); ft_model.eval()
    kl_sum = n = 0
    for x, _ in loader:
        x = x.to(device)
        lp = torch.log_softmax(base_model(x), 1)
        lq = torch.log_softmax(ft_model(x), 1)
        kl_sum += (lp.exp() * (lp - lq)).sum().item(); n += x.size(0)
    return kl_sum / n


def frac_changed(sd_a, sd_b):
    """Fraction of parameters where sd_a != sd_b."""
    total = changed = 0
    for k in sd_a:
        a, b = sd_a[k].float(), sd_b[k].float()
        total += a.numel(); changed += (a != b).sum().item()
    return changed / total

def frob_norm(sd_a, sd_b):
    s = 0.0
    for k in sd_a:
        s += ((sd_a[k].float() - sd_b[k].float()) ** 2).sum().item()
    return math.sqrt(s)


# ── Fine-tuning worker ──────────────────────────────────────────────

def finetune_worker(gpu_id, cond_name, mantissa_bits, method, base_path, out_path):
    device = f'cuda:{gpu_id}'
    torch.manual_seed(SEED)

    base_sd = torch.load(base_path, map_location=device, weights_only=True)

    tfm = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    mnist_train = datasets.MNIST('data', True,  download=False, transform=tfm)
    mnist_test  = datasets.MNIST('data', False, download=False, transform=tfm)
    fmnist_test = datasets.FashionMNIST('data', False, download=False, transform=tfm)

    tl = DataLoader(mnist_train, BS, shuffle=True,  num_workers=2, pin_memory=True)
    mt = DataLoader(mnist_test,  512, num_workers=2)
    ft = DataLoader(fmnist_test, 512, num_workers=2)

    model = MLP().to(device)
    model.load_state_dict(base_sd)
    # Snap model to target lattice at init (so training starts ON the lattice)
    snap_to_lattice(model, mantissa_bits)

    # KL reference: original fp32 base (measures total distributional shift)
    base_model = MLP().to(device)
    base_model.load_state_dict(copy.deepcopy(base_sd))
    base_model.eval()

    # L0 reference: base snapped to target lattice (same lattice as model)
    # At epoch 0, model == base_snapped → L0 = 0
    base_snapped_cpu = {k: snap_tensor(v.cpu(), mantissa_bits) for k, v in base_sd.items()}
    # Frobenius reference: original fp32 base
    base_fp32_cpu = {k: v.cpu().clone() for k, v in base_sd.items()}

    opt = optim.AdamW(model.parameters(), lr=LR)
    ce = nn.CrossEntropyLoss()
    metrics = []
    tag = f"{method}-{cond_name}"

    def record(epoch):
        # Evaluate with actual model weights (no EMA)
        pa = parity_accuracy(model, mt, device)
        fa = classification_accuracy(model, ft, device)
        kl = compute_forward_kl(base_model, model, mt, device)
        sd_cpu = {k: v.cpu() for k, v in model.state_dict().items()}
        fc = frac_changed(base_snapped_cpu, sd_cpu)  # vs lattice-snapped base
        fn = frob_norm(base_fp32_cpu, sd_cpu)          # vs fp32 base
        metrics.append(dict(epoch=epoch, parity_acc=pa, fashion_acc=fa,
                            forward_kl=kl, frac_changed=fc, frob_norm=fn))
        if epoch % 20 == 0 or epoch <= 4:
            print(f"  [{tag:>16s}] Ep {epoch:3d} | Par {pa:.4f} | Fash {fa:.4f} | "
                  f"KL {kl:.4f} | L0 {fc:.4f} | ||dW|| {fn:.2f}", flush=True)

    print(f"[{tag}] GPU {gpu_id}", flush=True)
    record(0)

    for ep in range(1, N_EPOCHS + 1):
        model.train()
        for x, y in tl:
            x, y = x.to(device), y.to(device)
            if method == 'SFT':
                opt.zero_grad(); ce(model(x), y).backward(); opt.step()
            else:
                logits = model(x)
                dist = Categorical(logits=logits)
                actions = dist.sample()
                log_probs = dist.log_prob(actions)
                rewards = ((actions % 2) == (y % 2)).float()
                loss = -(( rewards - rewards.mean()) * log_probs).mean()
                opt.zero_grad(); loss.backward(); opt.step()
            snap_to_lattice(model, mantissa_bits)

        if ep % CKPT_EVERY == 0:
            record(ep)

    with open(out_path, 'w') as f:
        json.dump(dict(condition=cond_name, method=method,
                       mantissa_bits=mantissa_bits, metrics=metrics), f, indent=2)
    print(f"[{tag}] Done -> {out_path}", flush=True)


# ── Publication plotting ────────────────────────────────────────────

def make_plots(sft_results, rl_results):
    plt.rcParams.update({'font.family': 'serif', 'font.size': 12,
                         'axes.spines.top': False, 'axes.spines.right': False})
    C = {'FP32': '#1f77b4', 'BF16': '#ff7f0e', 'Mantissa-10': '#2ca02c', 'Mantissa-12': '#d62728'}
    M = {'FP32': 'o', 'BF16': 's', 'Mantissa-10': '^', 'Mantissa-12': 'D'}
    sub = dict(fontsize=9, color='0.45', style='italic')

    all_r = []
    for r in sft_results:
        all_r.append(dict(m=r['metrics'], label=f"SFT {r['condition']}",
                          prec=r['condition'], ls='-'))
    for r in rl_results:
        all_r.append(dict(m=r['metrics'], label=f"RL {r['condition']}",
                          prec=r['condition'], ls='--'))

    def _draw(ax, xk, yk):
        for r in all_r:
            ax.plot([d[xk] for d in r['m']], [d[yk] for d in r['m']],
                    f"{M[r['prec']]}{r['ls']}", color=C[r['prec']],
                    label=r['label'], ms=3.5, alpha=0.85, lw=1.8)

    # Plot 1: KL vs Forgetting
    fig, ax = plt.subplots(figsize=(7, 5.5))
    _draw(ax, 'forward_kl', 'fashion_acc')
    ax.set_xlabel('Forward KL from Base Model'); ax.set_ylabel('FashionMNIST Accuracy')
    ax.set_title('Forward KL predicts forgetting across training methods\n'
                 'and precision conditions', fontweight='bold', fontsize=13, pad=14)
    ax.text(0.5, 1.01, 'SFT (solid) vs REINFORCE (dashed), 100 epochs, AdamW lr=1e-4',
            transform=ax.transAxes, ha='center', **sub)
    ax.legend(fontsize=8.5, ncol=2, framealpha=0.9); ax.grid(True, alpha=0.15)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / 'plot_kl_forgetting.png', dpi=300, bbox_inches='tight')

    # Plot 2: Pareto
    fig, ax = plt.subplots(figsize=(7, 5.5))
    _draw(ax, 'parity_acc', 'fashion_acc')
    ax.set_xlabel('ParityMNIST Accuracy'); ax.set_ylabel('FashionMNIST Accuracy')
    ax.set_title('BF16 lattice achieves the best\nlearning-forgetting tradeoff',
                 fontweight='bold', fontsize=13, pad=14)
    ax.text(0.5, 1.01, 'Pareto frontier: higher = less forgetting, right = better task performance',
            transform=ax.transAxes, ha='center', **sub)
    ax.legend(fontsize=8.5, ncol=2, framealpha=0.9); ax.grid(True, alpha=0.15)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / 'plot_pareto.png', dpi=300, bbox_inches='tight')

    # Plot 3: Per-checkpoint L0 (vs lattice-snapped base)
    fig, ax = plt.subplots(figsize=(7, 5.5))
    _draw(ax, 'epoch', 'frac_changed')
    ax.set_xlabel('Training Epoch')
    ax.set_ylabel('Fraction of Parameters at Different Lattice Point')
    ax.set_title('Lattice precision controls update sparsity',
                 fontweight='bold', fontsize=13, pad=14)
    ax.text(0.5, 1.01, 'Per-checkpoint L0: current weights vs base weights snapped to same lattice',
            transform=ax.transAxes, ha='center', **sub)
    ax.legend(fontsize=8.5, ncol=2, framealpha=0.9); ax.grid(True, alpha=0.15)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / 'plot_sparsity.png', dpi=300, bbox_inches='tight')

    # Plot 4: Frobenius norm
    fig, ax = plt.subplots(figsize=(7, 5.5))
    _draw(ax, 'epoch', 'frob_norm')
    ax.set_xlabel('Training Epoch')
    ax.set_ylabel(r'$\|W - W_\mathrm{base}\|_F$')
    ax.set_title('RL produces smaller weight displacement than SFT\nacross all precision conditions',
                 fontweight='bold', fontsize=13, pad=14)
    ax.text(0.5, 1.01, 'Frobenius norm of weight delta from fp32 base model',
            transform=ax.transAxes, ha='center', **sub)
    ax.legend(fontsize=8.5, ncol=2, framealpha=0.9); ax.grid(True, alpha=0.15)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / 'plot_frobenius.png', dpi=300, bbox_inches='tight')

    plt.close('all')
    print(f"\nPlots saved to {OUTPUT_DIR}/ (300 DPI)")


def print_summary(sft_results, rl_results):
    W = 105
    print("\n" + "=" * W)
    print(f"{'FINAL SUMMARY':^{W}}")
    print("=" * W)
    hdr = (f"  {'Condition':<15} {'Method':>6} {'ParityAcc':>10} {'FashionAcc':>11} "
           f"{'ForwardKL':>10} {'FrobNorm':>9} {'FracChanged':>12}")
    print(hdr)
    print("  " + "-" * (W - 2))
    for r in sft_results + rl_results:
        m = r['metrics'][-1]
        print(f"  {r['condition']:<15} {r['method']:>6} {m['parity_acc']:>10.4f} "
              f"{m['fashion_acc']:>11.4f} {m['forward_kl']:>10.4f} "
              f"{m['frob_norm']:>9.2f} {m['frac_changed']:>12.4f}")
    print("=" * W)


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    t0 = time.time()

    base_path = str(OUTPUT_DIR / 'base_model.pt')
    old_base = Path('parity_mnist_experiment/base_model.pt')
    if not Path(base_path).exists():
        if old_base.exists():
            shutil.copy(old_base, base_path)
            print(f"Reused base model from {old_base}")
        else:
            print("ERROR: no base model found", file=sys.stderr); sys.exit(1)

    mp.set_start_method('spawn', force=True)

    for method in ['SFT', 'RL']:
        print(f"\n{'=' * 70}\n{method} ({N_EPOCHS} epochs)\n{'=' * 70}")
        procs = []
        for i, (name, bits) in enumerate(CONDITIONS):
            out = str(OUTPUT_DIR / f'{method.lower()}_{name}.json')
            p = mp.Process(target=finetune_worker,
                           args=(i, name, bits, method, base_path, out))
            p.start(); procs.append(p)
        for p in procs:
            p.join()
            if p.exitcode != 0:
                print(f"WARNING: {p.name} exit {p.exitcode}", file=sys.stderr)

    sft_results, rl_results = [], []
    for name, _ in CONDITIONS:
        with open(OUTPUT_DIR / f'sft_{name}.json') as f: sft_results.append(json.load(f))
        with open(OUTPUT_DIR / f'rl_{name}.json') as f: rl_results.append(json.load(f))

    make_plots(sft_results, rl_results)
    print_summary(sft_results, rl_results)
    print(f"\nTotal time: {time.time() - t0:.0f}s ({(time.time() - t0)/60:.1f}m)")


if __name__ == '__main__':
    main()
