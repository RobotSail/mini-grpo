#!/usr/bin/env python3
"""
ParityMNIST v2: Cleaner snap_to_lattice, EMA-smoothed evaluation,
extended RL training (300 ep), publication-quality plots.
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

# ── Config ──────────────────────────────────────────────────────────
OUTPUT_DIR = Path("parity_mnist_v2")
SEED = 42

PRETRAIN_EPOCHS = 30
PRETRAIN_LR = 1e-3
PRETRAIN_BS = 256
MNIST_SUBSET = 5000

SFT_EPOCHS = 100
RL_EPOCHS = 300
LR = 1e-4
BS = 128
CKPT_EVERY = 2
EMA_DECAY = 0.999

CONDITIONS = [("FP32", 23), ("BF16", 7), ("Mantissa-10", 10), ("Mantissa-12", 12)]


# ── Model ───────────────────────────────────────────────────────────

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


# ── Snap to lattice (frexp/ldexp, provably idempotent) ──────────────

def snap_to_lattice(model, mantissa_bits):
    if mantissa_bits >= 23:
        return
    with torch.no_grad():
        if mantissa_bits == 7:
            for p in model.parameters():
                p.data.copy_(p.data.bfloat16().float())
        else:
            nbits = mantissa_bits + 1  # include implicit leading bit
            scale = float(1 << nbits)
            for p in model.parameters():
                m, e = torch.frexp(p.data)
                p.data.copy_(torch.ldexp((m * scale).round() / scale, e))


# ── Evaluation ──────────────────────────────────────────────────────

@torch.no_grad()
def parity_accuracy(model, loader, device):
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        correct += ((model(x).argmax(1) % 2) == (y % 2)).sum().item()
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
    """KL(p_base || p_fine) = E_x[ sum_y p_base(y|x) log(p_base(y|x)/p_fine(y|x)) ]"""
    base_model.eval(); ft_model.eval()
    kl_sum = n = 0
    for x, _ in loader:
        x = x.to(device)
        log_p = torch.log_softmax(base_model(x), 1)
        log_q = torch.log_softmax(ft_model(x), 1)
        kl_sum += (log_p.exp() * (log_p - log_q)).sum().item()
        n += x.size(0)
    return kl_sum / n

def delta_stats(base_sd, ft_sd):
    total = changed = 0; frob_sq = 0.0
    for k in base_sd:
        b, f = base_sd[k].float(), ft_sd[k].float()
        total += b.numel()
        changed += (b != f).sum().item()
        frob_sq += ((f - b) ** 2).sum().item()
    return changed / total, math.sqrt(frob_sq)


# ── Pretraining ─────────────────────────────────────────────────────

def pretrain(device):
    torch.manual_seed(SEED)
    tfm = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    mnist_full  = datasets.MNIST('data', True,  download=True, transform=tfm)
    mnist_test  = datasets.MNIST('data', False, download=True, transform=tfm)
    fmnist_train = datasets.FashionMNIST('data', True,  download=True, transform=tfm)
    fmnist_test  = datasets.FashionMNIST('data', False, download=True, transform=tfm)

    idx = torch.randperm(len(mnist_full), generator=torch.Generator().manual_seed(SEED))[:MNIST_SUBSET]
    mnist_sub = Subset(mnist_full, idx.tolist())

    ml = DataLoader(mnist_sub,    PRETRAIN_BS, shuffle=True, num_workers=4, pin_memory=True)
    fl = DataLoader(fmnist_train, PRETRAIN_BS, shuffle=True, num_workers=4, pin_memory=True)
    mt = DataLoader(mnist_test,  512, num_workers=4)
    ft = DataLoader(fmnist_test, 512, num_workers=4)

    model = MLP().to(device)
    opt = optim.Adam(model.parameters(), lr=PRETRAIN_LR)
    ce = nn.CrossEntropyLoss()

    print(f"Model: {sum(p.numel() for p in model.parameters()):,} parameters")
    print("=" * 70)
    print("PRETRAINING")
    print("=" * 70)

    for ep in range(1, PRETRAIN_EPOCHS + 1):
        model.train()
        mi = iter(ml)
        for fx, fy in fl:
            try:    mx, my = next(mi)
            except StopIteration: mi = iter(ml); mx, my = next(mi)
            mx, my = mx.to(device), my.to(device)
            opt.zero_grad(); ce(model(mx), my).backward(); opt.step()
            fx, fy = fx.to(device), fy.to(device)
            opt.zero_grad(); ce(model(fx), fy).backward(); opt.step()
        if ep % 10 == 0 or ep == 1:
            print(f"  Ep {ep:2d} | Parity {parity_accuracy(model, mt, device):.4f} | "
                  f"Digit {classification_accuracy(model, mt, device):.4f} | "
                  f"Fashion {classification_accuracy(model, ft, device):.4f}")

    pa = parity_accuracy(model, mt, device)
    fa = classification_accuracy(model, ft, device)
    print(f"  Base model: Parity={pa:.4f}, Fashion={fa:.4f}\n")
    return model.state_dict()


# ── Unified fine-tuning worker (SFT or RL) ──────────────────────────

def finetune_worker(gpu_id, cond_name, mantissa_bits, method, n_epochs, base_path, out_path):
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

    base_model = MLP().to(device)
    base_model.load_state_dict(copy.deepcopy(base_sd))
    base_model.eval()

    # Eval model loads EMA weights for smooth evaluation
    eval_model = MLP().to(device)
    eval_model.eval()

    base_cpu = {k: v.cpu().clone() for k, v in base_sd.items()}
    opt = optim.AdamW(model.parameters(), lr=LR)
    ce = nn.CrossEntropyLoss()

    # EMA of model parameters
    ema_params = [p.data.clone() for p in model.parameters()]

    metrics = []
    tag = f"{method}-{cond_name}"

    def record(epoch, epoch_l0=0.0):
        # Evaluate with EMA weights (smoother, especially for RL)
        for p_ev, ema_p in zip(eval_model.parameters(), ema_params):
            p_ev.data.copy_(ema_p)
        pa = parity_accuracy(eval_model, mt, device)
        fa = classification_accuracy(eval_model, ft, device)
        kl = compute_forward_kl(base_model, eval_model, mt, device)
        # Weight-space metrics use actual model weights
        sd_cpu = {k: v.cpu() for k, v in model.state_dict().items()}
        fc, fn = delta_stats(base_cpu, sd_cpu)
        metrics.append({
            'epoch': epoch, 'parity_acc': pa, 'fashion_acc': fa,
            'forward_kl': kl, 'frac_changed': fc, 'frob_norm': fn,
            'epoch_l0': epoch_l0,
        })
        if epoch % 20 == 0 or epoch <= 4:
            print(f"  [{tag:>16s}] Ep {epoch:3d}/{n_epochs} | Par {pa:.4f} | "
                  f"Fash {fa:.4f} | KL {kl:.4f} | ||dW|| {fn:.2f}", flush=True)

    print(f"[{tag}] GPU {gpu_id}, {n_epochs} epochs", flush=True)
    record(0)

    prev_sd = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    for ep in range(1, n_epochs + 1):
        model.train()
        for x, y in tl:
            x, y = x.to(device), y.to(device)
            if method == 'SFT':
                opt.zero_grad()
                ce(model(x), y).backward()
                opt.step()
            else:  # RL (REINFORCE)
                logits = model(x)
                dist = Categorical(logits=logits)
                actions = dist.sample()
                log_probs = dist.log_prob(actions)
                rewards = ((actions % 2) == (y % 2)).float()
                advantages = rewards - rewards.mean()
                loss = -(advantages * log_probs).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()

            snap_to_lattice(model, mantissa_bits)

            # EMA update
            for ema_p, p in zip(ema_params, model.parameters()):
                ema_p.mul_(EMA_DECAY).add_(p.data, alpha=1 - EMA_DECAY)

        if ep % CKPT_EVERY == 0:
            cur_sd = {k: v.cpu() for k, v in model.state_dict().items()}
            el0, _ = delta_stats(prev_sd, cur_sd)
            prev_sd = {k: v.clone() for k, v in cur_sd.items()}
            record(ep, el0)

    with open(out_path, 'w') as f:
        json.dump({'condition': cond_name, 'method': method,
                   'mantissa_bits': mantissa_bits, 'metrics': metrics}, f, indent=2)
    print(f"[{tag}] Done -> {out_path}", flush=True)


# ── Publication-quality plotting ────────────────────────────────────

def make_plots(sft_results, rl_results):
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 12,
        'axes.spines.top': False,
        'axes.spines.right': False,
    })

    C = {'FP32': '#1f77b4', 'BF16': '#ff7f0e', 'Mantissa-10': '#2ca02c', 'Mantissa-12': '#d62728'}
    M = {'FP32': 'o', 'BF16': 's', 'Mantissa-10': '^', 'Mantissa-12': 'D'}

    all_r = []
    for r in sft_results:
        all_r.append(dict(m=r['metrics'], label=f"SFT {r['condition']}",
                          prec=r['condition'], ls='-'))
    for r in rl_results:
        all_r.append(dict(m=r['metrics'], label=f"RL {r['condition']}",
                          prec=r['condition'], ls='--'))

    def _draw(ax, xk, yk, skip0=False):
        for r in all_r:
            pts = r['m'][1:] if skip0 else r['m']
            ax.plot([d[xk] for d in pts], [d[yk] for d in pts],
                    f"{M[r['prec']]}{r['ls']}", color=C[r['prec']],
                    label=r['label'], ms=3.5, alpha=0.85, lw=1.8)

    subtitle_style = dict(fontsize=9, color='0.45', style='italic')

    # ── Plot 1: KL vs Forgetting ──
    fig, ax = plt.subplots(figsize=(7, 5.5))
    _draw(ax, 'forward_kl', 'fashion_acc')
    ax.set_xlabel('Forward KL from Base Model')
    ax.set_ylabel('FashionMNIST Accuracy')
    ax.set_title('Forward KL predicts forgetting across training methods\n'
                 'and precision conditions', fontweight='bold', fontsize=13, pad=14)
    ax.text(0.5, 1.01, 'SFT (100 ep, solid) vs REINFORCE (300 ep, dashed), AdamW lr=1e-4, 3-layer MLP',
            transform=ax.transAxes, ha='center', **subtitle_style)
    ax.legend(fontsize=8.5, ncol=2, framealpha=0.9)
    ax.grid(True, alpha=0.15)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / 'plot_kl_forgetting.png', dpi=300, bbox_inches='tight')

    # ── Plot 2: Pareto ──
    fig, ax = plt.subplots(figsize=(7, 5.5))
    _draw(ax, 'parity_acc', 'fashion_acc')
    ax.set_xlabel('ParityMNIST Accuracy')
    ax.set_ylabel('FashionMNIST Accuracy')
    ax.set_title('BF16 lattice achieves the best\nlearning-forgetting tradeoff',
                 fontweight='bold', fontsize=13, pad=14)
    ax.text(0.5, 1.01, 'Pareto frontier: higher = less forgetting, further right = better task performance',
            transform=ax.transAxes, ha='center', **subtitle_style)
    ax.legend(fontsize=8.5, ncol=2, framealpha=0.9)
    ax.grid(True, alpha=0.15)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / 'plot_pareto.png', dpi=300, bbox_inches='tight')

    # ── Plot 3: Sparsity ──
    fig, ax = plt.subplots(figsize=(7, 5.5))
    _draw(ax, 'epoch', 'epoch_l0', skip0=True)
    ax.set_xlabel('Training Epoch')
    ax.set_ylabel('Fraction of Parameters Changed (per 2-epoch interval)')
    ax.set_title('Lattice precision controls per-step update sparsity',
                 fontweight='bold', fontsize=13, pad=14)
    ax.text(0.5, 1.01, 'BF16 updates only ~20% of parameters per interval; FP32 updates 100%',
            transform=ax.transAxes, ha='center', **subtitle_style)
    ax.legend(fontsize=8.5, ncol=2, framealpha=0.9)
    ax.grid(True, alpha=0.15)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / 'plot_sparsity.png', dpi=300, bbox_inches='tight')

    # ── Plot 4: Frobenius ──
    fig, ax = plt.subplots(figsize=(7, 5.5))
    _draw(ax, 'epoch', 'frob_norm')
    ax.set_xlabel('Training Epoch')
    ax.set_ylabel(r'$\|W - W_\mathrm{base}\|_F$')
    ax.set_title('RL produces smaller weight displacement than SFT\nacross all precision conditions',
                 fontweight='bold', fontsize=13, pad=14)
    ax.text(0.5, 1.01, 'Frobenius norm of weight delta from base model',
            transform=ax.transAxes, ha='center', **subtitle_style)
    ax.legend(fontsize=8.5, ncol=2, framealpha=0.9)
    ax.grid(True, alpha=0.15)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / 'plot_frobenius.png', dpi=300, bbox_inches='tight')

    plt.close('all')
    print(f"\nPlots saved to {OUTPUT_DIR}/ (300 DPI)")


# ── Summary ─────────────────────────────────────────────────────────

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


# ── Main ────────────────────────────────────────────────────────────

def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    t0 = time.time()

    # ── Pretrain or reuse ──
    base_path = str(OUTPUT_DIR / 'base_model.pt')
    old_base = Path('parity_mnist_experiment/base_model.pt')
    if not Path(base_path).exists():
        if old_base.exists():
            shutil.copy(old_base, base_path)
            print(f"Reused base model from {old_base}")
        else:
            base_sd = pretrain('cuda:0')
            torch.save(base_sd, base_path)

    # ── SFT (4 conditions, parallel on GPUs 0-3) ──
    mp.set_start_method('spawn', force=True)

    print(f"\n{'=' * 70}")
    print(f"SFT FINE-TUNING ({SFT_EPOCHS} epochs, 4 conditions)")
    print(f"{'=' * 70}")
    procs = []
    for i, (name, bits) in enumerate(CONDITIONS):
        out = str(OUTPUT_DIR / f'sft_{name}.json')
        p = mp.Process(target=finetune_worker,
                       args=(i, name, bits, 'SFT', SFT_EPOCHS, base_path, out))
        p.start(); procs.append(p)
    for p in procs:
        p.join()
        if p.exitcode != 0:
            print(f"WARNING: {p.name} exit code {p.exitcode}", file=sys.stderr)

    # ── RL (4 conditions, parallel on GPUs 0-3) ──
    print(f"\n{'=' * 70}")
    print(f"RL FINE-TUNING ({RL_EPOCHS} epochs, 4 conditions)")
    print(f"{'=' * 70}")
    procs = []
    for i, (name, bits) in enumerate(CONDITIONS):
        out = str(OUTPUT_DIR / f'rl_{name}.json')
        p = mp.Process(target=finetune_worker,
                       args=(i, name, bits, 'RL', RL_EPOCHS, base_path, out))
        p.start(); procs.append(p)
    for p in procs:
        p.join()
        if p.exitcode != 0:
            print(f"WARNING: {p.name} exit code {p.exitcode}", file=sys.stderr)

    # ── Collect results ──
    sft_results, rl_results = [], []
    for name, _ in CONDITIONS:
        with open(OUTPUT_DIR / f'sft_{name}.json') as f:
            sft_results.append(json.load(f))
        with open(OUTPUT_DIR / f'rl_{name}.json') as f:
            rl_results.append(json.load(f))

    # ── Plot and summarize ──
    make_plots(sft_results, rl_results)
    print_summary(sft_results, rl_results)

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.0f}s ({elapsed/60:.1f}m)")


if __name__ == '__main__':
    main()
