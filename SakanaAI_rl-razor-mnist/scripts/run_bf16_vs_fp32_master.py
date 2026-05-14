#!/usr/bin/env python3
"""ParityMNIST: bf16 vs fp32 master weights across all 5 methods. CPU only."""
import os, sys, json, copy, torch, torch.nn as nn
from torch.distributions import Categorical
from torch.utils.data import DataLoader
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from rl_razor.model import MLP
from rl_razor.data import get_finetuning_data, get_parity_mnist, get_fashion_mnist, create_dataloader
from rl_razor.metrics import forward_kl, parity_accuracy
from rl_razor.utils import set_seed

SEED = 42; BS = 64; LR = 1e-4; N_EPOCHS = 2; GROUP_SIZE = 8
EVAL_EVERY = 0.2  # epochs
PRETRAIN = "experiments/mantissa_sweep/pretrain/pretrained_model.pt"
OUTPUT = Path("experiments/bf16_vs_fp32_master")

def snap_rne(t, b):
    shift = 23 - b; halfway = 1 << (shift-1); mask = ~((1<<shift)-1); lsb = 1<<shift
    raw = t.view(torch.int32); trunc = raw & mask; rem = raw & ((1<<shift)-1)
    up = (rem > halfway) | ((rem == halfway) & (((trunc >> shift) & 1) == 1))
    t.copy_((trunc + up.int() * lsb).view(torch.float32))

def snap_model(model, mbits):
    if mbits >= 23: return
    with torch.no_grad():
        for p in model.parameters(): snap_rne(p.data, mbits)

@torch.no_grad()
def evaluate(model, base, pl, fl, device):
    model.eval()
    c=t=0
    for x,y in pl: x,y=x.to(device),y.to(device); c+=((model(x).argmax(1)%2)==(y%2)).sum().item(); t+=y.size(0)
    par = c/t
    c=t=0
    for x,y in fl: x,y=x.to(device),y.to(device); c+=(model(x).argmax(1)==y).sum().item(); t+=y.size(0)
    fash = c/t
    kl = forward_kl(base, model, pl, device)
    model.train()
    return par, fash, kl

def run_method(method, mbits, base_model, train_loader, parity_loader, fashion_loader, device):
    label = "bf16" if mbits == 7 else "fp32"
    model = MLP().to(device)
    model.load_state_dict(copy.deepcopy(base_model.state_dict()))
    snap_model(model, mbits)

    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    steps_per_epoch = len(train_loader)
    ckpt_interval = max(1, int(steps_per_epoch * EVAL_EVERY))

    metrics = []
    par, fash, kl = evaluate(model, base_model, parity_loader, fashion_loader, device)
    metrics.append({"epoch": 0, "parity_acc": par, "fashion_acc": fash, "forward_kl": kl})

    global_step = 0
    for ep in range(1, N_EPOCHS + 1):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            n = x.size(0); opt.zero_grad()

            if method in ("sft1", "sft2", "oracle"):
                if method == "sft1":
                    targets = (y % 2).long()
                elif method == "sft2":
                    targets = y.clone()
                    even = y%2==0; odd = ~even
                    targets[even] = torch.where(torch.rand(even.sum())<0.5, torch.zeros(even.sum(),dtype=torch.long), torch.full((even.sum(),),4,dtype=torch.long))
                    targets[odd] = torch.where(torch.rand(odd.sum())<0.5, torch.ones(odd.sum(),dtype=torch.long), torch.full((odd.sum(),),5,dtype=torch.long))
                else:
                    with torch.no_grad():
                        bp = torch.softmax(base_model(x), 1)
                    parity = y % 2
                    mask = torch.zeros(n, 10)
                    for i in range(10): mask[:,i] = ((i%2)==parity).float()
                    op = bp * mask; op = op / op.sum(1, keepdim=True)
                    targets = torch.multinomial(op, 1).squeeze(1)
                nn.CrossEntropyLoss()(model(x), targets).backward()
            else:
                x_g = x.unsqueeze(1).expand(-1,GROUP_SIZE,-1).reshape(n*GROUP_SIZE,-1)
                y_g = y.unsqueeze(1).expand(-1,GROUP_SIZE).reshape(n*GROUP_SIZE)
                logits = model(x_g); probs = torch.softmax(logits,1); lp = torch.log_softmax(logits,1)
                actions = torch.multinomial(probs,1).squeeze(1)
                rewards = ((actions%2)==(y_g%2)).float().reshape(n,GROUP_SIZE)
                adv = (rewards - rewards.mean(1,keepdim=True))/(rewards.std(1,keepdim=True)+1e-8)
                sel = lp.gather(1,actions.unsqueeze(1)).squeeze(1)
                loss = -(adv.reshape(n*GROUP_SIZE)*sel).mean()
                if method == "grpo_kl":
                    with torch.no_grad(): blp = torch.log_softmax(base_model(x_g),1)
                    loss = loss + 0.1*(probs*(lp-blp)).sum(1).mean()
                loss.backward()

            opt.step()
            snap_model(model, mbits)
            global_step += 1

            if global_step % ckpt_interval == 0:
                ep_frac = global_step / steps_per_epoch
                par, fash, kl = evaluate(model, base_model, parity_loader, fashion_loader, device)
                metrics.append({"epoch": round(ep_frac,2), "parity_acc": par, "fashion_acc": fash, "forward_kl": kl})

    return metrics

if __name__ == "__main__":
    OUTPUT.mkdir(parents=True, exist_ok=True)
    device = "cpu"
    set_seed(SEED)

    base = MLP.from_checkpoint(PRETRAIN, device=device)
    base.eval()

    train_data, _ = get_finetuning_data("rl")
    train_loader = create_dataloader(train_data, batch_size=BS, shuffle=True)
    pl = create_dataloader(get_parity_mnist(train=False), batch_size=512, shuffle=False)
    fl = create_dataloader(get_fashion_mnist(train=False), batch_size=512, shuffle=False)

    results = {}
    methods = ["grpo", "grpo_kl", "sft1", "sft2", "oracle"]

    for method in methods:
        for mbits, wlabel in [(7, "bf16"), (23, "fp32")]:
            key = f"{method}_{wlabel}"
            print(f"Running {key}...", end=" ", flush=True)
            set_seed(SEED)
            m = run_method(method, mbits, base, train_loader, pl, fl, device)
            results[key] = m
            final = m[-1]
            print(f"parity={final['parity_acc']:.4f} fashion={final['fashion_acc']:.4f} KL={final['forward_kl']:.4f}")

    with open(OUTPUT / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Summary table
    print(f"\n{'Method':<12} {'Weights':<6} {'Parity':>8} {'Fashion':>8} {'KL':>10}")
    print("-" * 48)
    for method in methods:
        for wl in ["bf16", "fp32"]:
            key = f"{method}_{wl}"
            final = results[key][-1]
            print(f"{method:<12} {wl:<6} {final['parity_acc']:>7.4f} {final['fashion_acc']:>7.4f} {final['forward_kl']:>10.4f}")
