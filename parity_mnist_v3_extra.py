#!/usr/bin/env python3
"""Run Mantissa-8 and Mantissa-9 conditions, then replot all 12 conditions."""

import sys, json, math, copy, time
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

OUTPUT_DIR = Path("parity_mnist_v3")
SEED = 42; N_EPOCHS = 100; LR = 1e-4; BS = 128; CKPT_EVERY = 2

NEW_CONDITIONS = [("Mantissa-8", 8), ("Mantissa-9", 9)]
ALL_CONDITIONS = [("FP32", 23), ("BF16", 7), ("Mantissa-8", 8), ("Mantissa-9", 9),
                  ("Mantissa-10", 10), ("Mantissa-12", 12)]

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.Flatten(), nn.Linear(784,256), nn.ReLU(),
                                     nn.Linear(256,256), nn.ReLU(), nn.Linear(256,10))
    def forward(self, x): return self.layers(x)

def snap_tensor(x, mb):
    if mb >= 23: return x.clone()
    if mb == 7: return x.bfloat16().float()
    s = float(1 << (mb + 1)); m, e = torch.frexp(x)
    return torch.ldexp((m * s).round() / s, e)

def snap_to_lattice(model, mb):
    if mb >= 23: return
    with torch.no_grad():
        if mb == 7:
            for p in model.parameters(): p.data.copy_(p.data.bfloat16().float())
        else:
            s = float(1 << (mb + 1))
            for p in model.parameters():
                m, e = torch.frexp(p.data); p.data.copy_(torch.ldexp((m*s).round()/s, e))

@torch.no_grad()
def parity_acc(model, loader, dev):
    model.eval(); c=t=0
    for x,y in loader: x,y=x.to(dev),y.to(dev); c+=((model(x).argmax(1)%2)==(y%2)).sum().item(); t+=y.size(0)
    return c/t

@torch.no_grad()
def class_acc(model, loader, dev):
    model.eval(); c=t=0
    for x,y in loader: x,y=x.to(dev),y.to(dev); c+=(model(x).argmax(1)==y).sum().item(); t+=y.size(0)
    return c/t

@torch.no_grad()
def fwd_kl(bm, fm, loader, dev):
    bm.eval(); fm.eval(); s=n=0
    for x,_ in loader:
        x=x.to(dev); lp=torch.log_softmax(bm(x),1); lq=torch.log_softmax(fm(x),1)
        s+=(lp.exp()*(lp-lq)).sum().item(); n+=x.size(0)
    return s/n

def frac_changed(a, b):
    t=c=0
    for k in a: t+=a[k].numel(); c+=(a[k].float()!=b[k].float()).sum().item()
    return c/t

def frob(a, b):
    s=0.0
    for k in a: s+=((a[k].float()-b[k].float())**2).sum().item()
    return math.sqrt(s)

def worker(gpu_id, cname, mb, method, base_path, out_path):
    dev = f'cuda:{gpu_id}'; torch.manual_seed(SEED)
    bsd = torch.load(base_path, map_location=dev, weights_only=True)
    tfm = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,),(0.5,))])
    tl = DataLoader(datasets.MNIST('data',True,download=False,transform=tfm), BS, shuffle=True, num_workers=2, pin_memory=True)
    mt = DataLoader(datasets.MNIST('data',False,download=False,transform=tfm), 512, num_workers=2)
    ft = DataLoader(datasets.FashionMNIST('data',False,download=False,transform=tfm), 512, num_workers=2)

    model = MLP().to(dev); model.load_state_dict(bsd); snap_to_lattice(model, mb)
    bm = MLP().to(dev); bm.load_state_dict(copy.deepcopy(bsd)); bm.eval()
    bs_cpu = {k: snap_tensor(v.cpu(), mb) for k,v in bsd.items()}
    bf_cpu = {k: v.cpu().clone() for k,v in bsd.items()}
    opt = optim.AdamW(model.parameters(), lr=LR); ce = nn.CrossEntropyLoss()
    metrics = []; tag = f"{method}-{cname}"

    def rec(ep):
        pa=parity_acc(model,mt,dev); fa=class_acc(model,ft,dev); kl=fwd_kl(bm,model,mt,dev)
        sd={k:v.cpu() for k,v in model.state_dict().items()}
        fc=frac_changed(bs_cpu,sd); fn=frob(bf_cpu,sd)
        metrics.append(dict(epoch=ep,parity_acc=pa,fashion_acc=fa,forward_kl=kl,frac_changed=fc,frob_norm=fn))
        if ep%20==0 or ep<=4:
            print(f"  [{tag:>16s}] Ep {ep:3d} | Par {pa:.4f} | Fash {fa:.4f} | KL {kl:.4f} | L0 {fc:.4f} | ||dW|| {fn:.2f}", flush=True)

    print(f"[{tag}] GPU {gpu_id}", flush=True); rec(0)
    for ep in range(1, N_EPOCHS+1):
        model.train()
        for x,y in tl:
            x,y=x.to(dev),y.to(dev)
            if method=='SFT': opt.zero_grad(); ce(model(x),y).backward(); opt.step()
            else:
                logits=model(x); d=Categorical(logits=logits); a=d.sample(); lp=d.log_prob(a)
                r=((a%2)==(y%2)).float(); loss=-((r-r.mean())*lp).mean()
                opt.zero_grad(); loss.backward(); opt.step()
            snap_to_lattice(model, mb)
        if ep%CKPT_EVERY==0: rec(ep)
    with open(out_path,'w') as f:
        json.dump(dict(condition=cname, method=method, mantissa_bits=mb, metrics=metrics), f, indent=2)
    print(f"[{tag}] Done -> {out_path}", flush=True)


def make_plots(all_results):
    plt.rcParams.update({'font.family':'serif','font.size':12,'axes.spines.top':False,'axes.spines.right':False})
    C = {'FP32':'#1f77b4','BF16':'#ff7f0e','Mantissa-8':'#9467bd','Mantissa-9':'#8c564b',
         'Mantissa-10':'#2ca02c','Mantissa-12':'#d62728'}
    M = {'FP32':'o','BF16':'s','Mantissa-8':'p','Mantissa-9':'h',
         'Mantissa-10':'^','Mantissa-12':'D'}
    sub = dict(fontsize=9, color='0.45', style='italic')

    drawn = []
    for r in all_results:
        drawn.append(dict(m=r['metrics'], label=f"{r['method']} {r['condition']}",
                          prec=r['condition'], ls='-' if r['method']=='SFT' else '--'))

    def _draw(ax, xk, yk):
        for r in drawn:
            ax.plot([d[xk] for d in r['m']], [d[yk] for d in r['m']],
                    f"{M[r['prec']]}{r['ls']}", color=C[r['prec']],
                    label=r['label'], ms=3.5, alpha=0.85, lw=1.8)

    # Plot 1: KL vs Forgetting
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    _draw(ax, 'forward_kl', 'fashion_acc')
    ax.set_xlabel('Forward KL from Base Model'); ax.set_ylabel('FashionMNIST Accuracy')
    ax.set_title('Forward KL predicts forgetting across training methods\nand precision conditions',
                 fontweight='bold', fontsize=13, pad=14)
    ax.text(0.5, 1.01, 'SFT (solid) vs REINFORCE (dashed), 100 epochs, 6 precision lattices',
            transform=ax.transAxes, ha='center', **sub)
    ax.legend(fontsize=7, ncol=3, framealpha=0.9); ax.grid(True, alpha=0.15)
    fig.tight_layout(); fig.savefig(OUTPUT_DIR/'plot_kl_forgetting.png', dpi=300, bbox_inches='tight')

    # Plot 2: Pareto
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    _draw(ax, 'parity_acc', 'fashion_acc')
    ax.set_xlabel('ParityMNIST Accuracy'); ax.set_ylabel('FashionMNIST Accuracy')
    ax.set_title('BF16 lattice achieves the best\nlearning-forgetting tradeoff',
                 fontweight='bold', fontsize=13, pad=14)
    ax.text(0.5, 1.01, 'Pareto frontier: higher = less forgetting, right = better task performance',
            transform=ax.transAxes, ha='center', **sub)
    ax.legend(fontsize=7, ncol=3, framealpha=0.9); ax.grid(True, alpha=0.15)
    fig.tight_layout(); fig.savefig(OUTPUT_DIR/'plot_pareto.png', dpi=300, bbox_inches='tight')

    # Plot 3: Sparsity
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    _draw(ax, 'epoch', 'frac_changed')
    ax.set_xlabel('Training Epoch'); ax.set_ylabel('Fraction of Parameters at Different Lattice Point')
    ax.set_title('Lattice precision controls update sparsity',
                 fontweight='bold', fontsize=13, pad=14)
    ax.text(0.5, 1.01, 'Per-checkpoint L0: current weights vs base weights snapped to same lattice',
            transform=ax.transAxes, ha='center', **sub)
    ax.legend(fontsize=7, ncol=3, framealpha=0.9); ax.grid(True, alpha=0.15)
    fig.tight_layout(); fig.savefig(OUTPUT_DIR/'plot_sparsity.png', dpi=300, bbox_inches='tight')

    # Plot 4: Frobenius
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    _draw(ax, 'epoch', 'frob_norm')
    ax.set_xlabel('Training Epoch'); ax.set_ylabel(r'$\|W - W_\mathrm{base}\|_F$')
    ax.set_title('RL produces smaller weight displacement than SFT\nacross all precision conditions',
                 fontweight='bold', fontsize=13, pad=14)
    ax.text(0.5, 1.01, 'Frobenius norm of weight delta from fp32 base model',
            transform=ax.transAxes, ha='center', **sub)
    ax.legend(fontsize=7, ncol=3, framealpha=0.9); ax.grid(True, alpha=0.15)
    fig.tight_layout(); fig.savefig(OUTPUT_DIR/'plot_frobenius.png', dpi=300, bbox_inches='tight')
    plt.close('all')
    print(f"\nPlots saved to {OUTPUT_DIR}/ (300 DPI)")


def main():
    t0 = time.time()
    base_path = str(OUTPUT_DIR / 'base_model.pt')
    mp.set_start_method('spawn', force=True)

    # Run 4 new conditions: SFT-M8, SFT-M9, RL-M8, RL-M9 on GPUs 0-3
    print(f"Running Mantissa-8 and Mantissa-9 (SFT + RL, 4 parallel jobs)")
    procs = []
    jobs = [(0, "Mantissa-8", 8, "SFT"), (1, "Mantissa-9", 9, "SFT"),
            (2, "Mantissa-8", 8, "RL"),  (3, "Mantissa-9", 9, "RL")]
    for gpu, name, bits, method in jobs:
        out = str(OUTPUT_DIR / f'{method.lower()}_{name}.json')
        p = mp.Process(target=worker, args=(gpu, name, bits, method, base_path, out))
        p.start(); procs.append(p)
    for p in procs:
        p.join()
        if p.exitcode != 0:
            print(f"WARNING: {p.name} exited with code {p.exitcode}", file=sys.stderr)

    # Load all 12 results (6 precisions × 2 methods)
    all_results = []
    for name, _ in ALL_CONDITIONS:
        for method in ['SFT', 'RL']:
            fpath = OUTPUT_DIR / f'{method.lower()}_{name}.json'
            with open(fpath) as f:
                all_results.append(json.load(f))

    make_plots(all_results)

    # Summary table
    W = 105
    print("\n" + "=" * W)
    print(f"{'FINAL SUMMARY (12 conditions)':^{W}}")
    print("=" * W)
    print(f"  {'Condition':<15} {'Method':>6} {'ParityAcc':>10} {'FashionAcc':>11} "
          f"{'ForwardKL':>10} {'FrobNorm':>9} {'FracChanged':>12}")
    print("  " + "-" * (W - 2))
    for r in all_results:
        m = r['metrics'][-1]
        print(f"  {r['condition']:<15} {r['method']:>6} {m['parity_acc']:>10.4f} "
              f"{m['fashion_acc']:>11.4f} {m['forward_kl']:>10.4f} "
              f"{m['frob_norm']:>9.2f} {m['frac_changed']:>12.4f}")
    print("=" * W)
    print(f"\nTime: {time.time()-t0:.0f}s")


if __name__ == '__main__':
    main()
