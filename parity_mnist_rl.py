#!/usr/bin/env python3
"""
Add REINFORCE (RL) training to the ParityMNIST forgetting experiment.
Uses the same base model from parity_mnist_experiment.py, runs RL fine-tuning
under 4 precision conditions, then creates combined SFT+RL plots.
"""

import os, sys, json, copy, math, time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from torch.distributions import Categorical
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUTPUT_DIR = Path("parity_mnist_experiment")
SEED = 42

FINETUNE_EPOCHS = 100
FINETUNE_LR = 1e-4
FINETUNE_BS = 128
CKPT_EVERY = 2

CONDITIONS = [
    ("FP32",        23),
    ("BF16",         7),
    ("Mantissa-10", 10),
    ("Mantissa-12", 12),
]


# ──────────────────────────────────────────────────────────────────────
# Model (identical to SFT experiment)
# ──────────────────────────────────────────────────────────────────────

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


# ──────────────────────────────────────────────────────────────────────
# Precision lattice (identical to SFT experiment)
# ──────────────────────────────────────────────────────────────────────

def snap_to_lattice(model, mantissa_bits):
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
# Evaluation helpers (identical to SFT experiment)
# ──────────────────────────────────────────────────────────────────────

@torch.no_grad()
def parity_accuracy(model, loader, device):
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
    base_model.eval(); ft_model.eval()
    kl_sum = n = 0
    for x, _ in loader:
        x = x.to(device)
        log_p = torch.log_softmax(base_model(x), dim=1)
        log_q = torch.log_softmax(ft_model(x), dim=1)
        kl = (log_p.exp() * (log_p - log_q)).sum(dim=1)
        kl_sum += kl.sum().item()
        n += x.size(0)
    return kl_sum / n

def compute_delta_stats(base_sd, ft_sd):
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
# REINFORCE fine-tuning worker
# ──────────────────────────────────────────────────────────────────────

def rl_finetune_worker(gpu_id, cond_name, mantissa_bits, base_path, out_path):
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

    model = MLP().to(device)
    model.load_state_dict(base_sd)

    base_model = MLP().to(device)
    base_model.load_state_dict(copy.deepcopy(base_sd))
    base_model.eval()

    base_cpu = {k: v.cpu().clone() for k, v in base_sd.items()}

    opt = optim.AdamW(model.parameters(), lr=FINETUNE_LR)

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
        print(f"  [RL-{cond_name:>10s}] Ep {epoch:3d} | Par {pa:.4f} | Fash {fa:.4f} | "
              f"KL {kl:.6f} | eL0 {epoch_l0:.4f} | ||dW|| {fn:.4f}", flush=True)

    label = f"RL-{cond_name}"
    print(f"\n[{label}] REINFORCE on GPU {gpu_id} "
          f"(mantissa_bits={mantissa_bits}, lr={FINETUNE_LR}, epochs={FINETUNE_EPOCHS})",
          flush=True)
    record(0)

    prev_sd = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    for ep in range(1, FINETUNE_EPOCHS + 1):
        model.train()
        for x, y in tl:
            x, y = x.to(device), y.to(device)

            logits = model(x)
            dist = Categorical(logits=logits)
            actions = dist.sample()
            log_probs = dist.log_prob(actions)

            # Binary parity reward: 1 if sampled action has correct parity, 0 otherwise
            rewards = ((actions % 2) == (y % 2)).float()

            # Baseline: batch mean reward
            advantages = rewards - rewards.mean()

            # REINFORCE policy gradient loss
            loss = -(advantages * log_probs).mean()

            opt.zero_grad()
            loss.backward()
            opt.step()
            snap_to_lattice(model, mantissa_bits)

        if ep % CKPT_EVERY == 0:
            cur_sd = {k: v.cpu() for k, v in model.state_dict().items()}
            epoch_l0, _ = compute_delta_stats(prev_sd, cur_sd)
            prev_sd = {k: v.clone() for k, v in cur_sd.items()}
            record(ep, epoch_l0)

    with open(out_path, 'w') as f:
        json.dump({
            'condition': label, 'mantissa_bits': mantissa_bits,
            'method': 'RL', 'metrics': metrics,
        }, f, indent=2)
    print(f"[{label}] Done -> {out_path}", flush=True)


# ──────────────────────────────────────────────────────────────────────
# Combined plotting (8 conditions: 4 SFT + 4 RL)
# ──────────────────────────────────────────────────────────────────────

def make_combined_plots(sft_results, rl_results):
    prec_colors  = {'FP32': '#1f77b4', 'BF16': '#ff7f0e', 'Mantissa-10': '#2ca02c', 'Mantissa-12': '#d62728'}
    prec_markers = {'FP32': 'o', 'BF16': 's', 'Mantissa-10': '^', 'Mantissa-12': 'D'}

    # Build unified result list with display metadata
    all_results = []
    for r in sft_results:
        prec = r['condition']
        all_results.append({
            'metrics': r['metrics'], 'display': f'SFT {prec}',
            'prec': prec, 'ls': '-', 'method': 'SFT',
        })
    for r in rl_results:
        prec = r['condition'].replace('RL-', '')
        all_results.append({
            'metrics': r['metrics'], 'display': f'RL {prec}',
            'prec': prec, 'ls': '--', 'method': 'RL',
        })

    def _plot(ax, xk, yk, xl, yl, title, skip_ep0=False):
        for r in all_results:
            m = r['metrics']
            pts = m[1:] if skip_ep0 else m
            ax.plot([d[xk] for d in pts], [d[yk] for d in pts],
                    f'{prec_markers[r["prec"]]}{r["ls"]}',
                    color=prec_colors[r['prec']], label=r['display'],
                    ms=4, alpha=0.8, lw=2)
        ax.set_xlabel(xl, fontsize=12); ax.set_ylabel(yl, fontsize=12)
        ax.set_title(title, fontsize=13)
        ax.legend(fontsize=8, ncol=2); ax.grid(True, alpha=0.3)

    # ── Main figure: KL vs Forgetting (the key Shenfeld plot) ──
    fig, ax = plt.subplots(figsize=(10, 7))
    _plot(ax, 'forward_kl', 'fashion_acc',
          'Forward KL from Base Model', 'FashionMNIST Accuracy',
          'KL vs Forgetting: SFT (solid) vs REINFORCE (dashed) x Precision')
    ax.legend(fontsize=11, ncol=2)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / 'combined_kl_forgetting.png', dpi=150, bbox_inches='tight')

    # ── 4-panel combined figure ──
    fig2, axes = plt.subplots(2, 2, figsize=(16, 12))

    _plot(axes[0, 0], 'parity_acc', 'fashion_acc',
          'ParityMNIST Accuracy', 'FashionMNIST Accuracy',
          'Learning vs Forgetting')

    _plot(axes[0, 1], 'forward_kl', 'fashion_acc',
          'Forward KL from Base Model', 'FashionMNIST Accuracy',
          'KL vs Forgetting')

    _plot(axes[1, 0], 'epoch', 'epoch_l0',
          'Training Epoch', 'Frac Params Changed (per interval)',
          'Per-Interval Update Sparsity', skip_ep0=True)

    _plot(axes[1, 1], 'epoch', 'frob_norm',
          'Training Epoch', '||W - W_base||_F',
          'Frobenius Norm of Weight Delta')

    plt.tight_layout()
    fig2.savefig(OUTPUT_DIR / 'combined_all_plots.png', dpi=150, bbox_inches='tight')

    # ── Individual high-res plots ──
    for xk, yk, xl, yl, title, fname, skip0 in [
        ('parity_acc', 'fashion_acc', 'ParityMNIST Accuracy', 'FashionMNIST Accuracy',
         'Learning vs Forgetting: SFT vs REINFORCE', 'combined_pareto.png', False),
        ('epoch', 'epoch_l0', 'Training Epoch', 'Frac Params Changed (per interval)',
         'Per-Interval Sparsity: SFT vs REINFORCE', 'combined_sparsity.png', True),
        ('epoch', 'frob_norm', 'Training Epoch', '||W - W_base||_F',
         'Frobenius Norm: SFT vs REINFORCE', 'combined_frobenius.png', False),
    ]:
        fig3, ax3 = plt.subplots(figsize=(10, 7))
        _plot(ax3, xk, yk, xl, yl, title, skip_ep0=skip0)
        ax3.legend(fontsize=11, ncol=2)
        fig3.tight_layout()
        fig3.savefig(OUTPUT_DIR / fname, dpi=150, bbox_inches='tight')

    plt.close('all')
    print(f"\nCombined plots saved to {OUTPUT_DIR}/")


# ──────────────────────────────────────────────────────────────────────
# Summary
# ──────────────────────────────────────────────────────────────────────

def print_summary(sft_results, rl_results):
    W = 115
    print("\n" + "=" * W)
    print(f"{'COMBINED SUMMARY (Epoch ' + str(FINETUNE_EPOCHS) + ')':^{W}}")
    print("=" * W)
    hdr = (f"  {'Condition':<20} {'Method':>6} {'ParityAcc':>10} {'FashionAcc':>12} "
           f"{'ForwardKL':>13} {'eL0':>8} {'FrobNorm':>10}")
    print(hdr)
    print("  " + "-" * (W - 2))
    for r in sft_results:
        m = r['metrics'][-1]
        print(f"  {r['condition']:<20} {'SFT':>6} {m['parity_acc']:>10.4f} {m['fashion_acc']:>12.4f} "
              f"{m['forward_kl']:>13.6f} {m.get('epoch_l0', 0):>8.4f} {m['frob_norm']:>10.4f}")
    print("  " + "-" * (W - 2))
    for r in rl_results:
        m = r['metrics'][-1]
        print(f"  {r['condition']:<20} {'RL':>6} {m['parity_acc']:>10.4f} {m['fashion_acc']:>12.4f} "
              f"{m['forward_kl']:>13.6f} {m.get('epoch_l0', 0):>8.4f} {m['frob_norm']:>10.4f}")
    print("=" * W)


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main():
    base_path = str(OUTPUT_DIR / 'base_model.pt')
    if not Path(base_path).exists():
        print(f"ERROR: Base model not found at {base_path}. Run parity_mnist_experiment.py first.", file=sys.stderr)
        sys.exit(1)

    t0 = time.time()

    # Run RL fine-tuning under 4 precision conditions (parallel on GPUs 0-3)
    print(f"Launching {len(CONDITIONS)} RL fine-tuning conditions on GPUs 0-{len(CONDITIONS)-1}...")
    mp.set_start_method('spawn', force=True)
    procs = []
    for i, (name, bits) in enumerate(CONDITIONS):
        out = str(OUTPUT_DIR / f'results_RL-{name}.json')
        p = mp.Process(target=rl_finetune_worker, args=(i, name, bits, base_path, out))
        p.start()
        procs.append(p)
    for p in procs:
        p.join()
        if p.exitcode != 0:
            print(f"WARNING: {p.name} exited with code {p.exitcode}", file=sys.stderr)

    # Load SFT results (from previous experiment)
    sft_results = []
    for name, _ in CONDITIONS:
        fpath = OUTPUT_DIR / f'results_{name}.json'
        with open(fpath) as f:
            sft_results.append(json.load(f))

    # Load RL results
    rl_results = []
    for name, _ in CONDITIONS:
        fpath = OUTPUT_DIR / f'results_RL-{name}.json'
        with open(fpath) as f:
            rl_results.append(json.load(f))

    # Combined plots and summary
    make_combined_plots(sft_results, rl_results)
    print_summary(sft_results, rl_results)

    print(f"\nRL experiment time: {time.time() - t0:.0f}s ({(time.time() - t0)/60:.1f}m)")


if __name__ == '__main__':
    main()
