#!/usr/bin/env python3
"""ParityMNIST: m7-m10 + fp32 master weights, 5 methods, 5 seeds, CPU only."""
import os, sys, json, copy, torch, torch.nn as nn, numpy as np
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from rl_razor.model import MLP
from rl_razor.data import get_finetuning_data, get_parity_mnist, get_fashion_mnist, create_dataloader
from rl_razor.metrics import forward_kl
from rl_razor.utils import set_seed

BS = 64; LR = 1e-4; N_EPOCHS = 2; GROUP_SIZE = 8
SEEDS = [42, 123, 456, 789, 1024]
MANTISSA = [(7,"m7"), (8,"m8"), (9,"m9"), (10,"m10"), (23,"fp32")]
PRETRAIN = "experiments/mantissa_sweep/pretrain/pretrained_model.pt"
OUTPUT = Path("experiments/mantissa_7to10_fp32_multiseed")

def snap_rne(t, b):
    shift = 23-b; halfway = 1<<(shift-1); mask = ~((1<<shift)-1); lsb = 1<<shift
    raw = t.view(torch.int32); trunc = raw & mask; rem = raw & ((1<<shift)-1)
    up = (rem > halfway) | ((rem == halfway) & (((trunc >> shift) & 1) == 1))
    t.copy_((trunc + up.int() * lsb).view(torch.float32))

def snap_model(model, mbits):
    if mbits >= 23: return
    with torch.no_grad():
        for p in model.parameters(): snap_rne(p.data, mbits)

@torch.no_grad()
def evaluate(model, base, pl, fl):
    model.eval()
    c=t=0
    for x,y in pl: c+=((model(x).argmax(1)%2)==(y%2)).sum().item(); t+=y.size(0)
    par = c/t
    c=t=0
    for x,y in fl: c+=(model(x).argmax(1)==y).sum().item(); t+=y.size(0)
    fash = c/t
    kl = forward_kl(base, model, pl, "cpu")
    model.train()
    return par, fash, kl

def run_one(method, mbits, base_model, train_loader, pl, fl, seed):
    set_seed(seed)
    model = MLP()
    model.load_state_dict(copy.deepcopy(base_model.state_dict()))
    snap_model(model, mbits)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)

    for ep in range(1, N_EPOCHS+1):
        model.train()
        for x, y in train_loader:
            n = x.size(0); opt.zero_grad()
            if method in ("sft1","sft2","oracle"):
                if method == "sft1": targets = (y%2).long()
                elif method == "sft2":
                    targets = y.clone(); even = y%2==0; odd = ~even
                    targets[even] = torch.where(torch.rand(even.sum())<0.5, torch.zeros(even.sum(),dtype=torch.long), torch.full((even.sum(),),4,dtype=torch.long))
                    targets[odd] = torch.where(torch.rand(odd.sum())<0.5, torch.ones(odd.sum(),dtype=torch.long), torch.full((odd.sum(),),5,dtype=torch.long))
                else:
                    with torch.no_grad(): bp = torch.softmax(base_model(x),1)
                    parity = y%2; mask = torch.zeros(n,10)
                    for i in range(10): mask[:,i] = ((i%2)==parity).float()
                    op = bp*mask; op = op/op.sum(1,keepdim=True); targets = torch.multinomial(op,1).squeeze(1)
                nn.CrossEntropyLoss()(model(x), targets).backward()
            else:
                x_g = x.unsqueeze(1).expand(-1,GROUP_SIZE,-1).reshape(n*GROUP_SIZE,-1)
                y_g = y.unsqueeze(1).expand(-1,GROUP_SIZE).reshape(n*GROUP_SIZE)
                logits = model(x_g); probs = torch.softmax(logits,1); lp = torch.log_softmax(logits,1)
                actions = torch.multinomial(probs,1).squeeze(1)
                rewards = ((actions%2)==(y_g%2)).float().reshape(n,GROUP_SIZE)
                adv = (rewards-rewards.mean(1,keepdim=True))/(rewards.std(1,keepdim=True)+1e-8)
                sel = lp.gather(1,actions.unsqueeze(1)).squeeze(1)
                loss = -(adv.reshape(n*GROUP_SIZE)*sel).mean()
                if method == "grpo_kl":
                    with torch.no_grad(): blp = torch.log_softmax(base_model(x_g),1)
                    loss = loss + 0.1*(probs*(lp-blp)).sum(1).mean()
                loss.backward()
            opt.step(); snap_model(model, mbits)

    return evaluate(model, base_model, pl, fl)

if __name__ == "__main__":
    OUTPUT.mkdir(parents=True, exist_ok=True)
    base = MLP.from_checkpoint(PRETRAIN, device="cpu"); base.eval()
    train_data, _ = get_finetuning_data("rl")
    train_loader = create_dataloader(train_data, batch_size=BS, shuffle=True)
    pl = create_dataloader(get_parity_mnist(train=False), batch_size=512, shuffle=False)
    fl = create_dataloader(get_fashion_mnist(train=False), batch_size=512, shuffle=False)

    methods = ["sft1", "sft2", "oracle", "grpo", "grpo_kl"]
    all_results = {}

    total = len(methods) * len(MANTISSA) * len(SEEDS)
    done = 0

    for method in methods:
        for mbits, mlabel in MANTISSA:
            key = f"{method}_{mlabel}"
            pars, fashs, kls = [], [], []
            for seed in SEEDS:
                p, f, k = run_one(method, mbits, base, train_loader, pl, fl, seed)
                pars.append(p); fashs.append(f); kls.append(k)
                done += 1
            all_results[key] = {"parity": pars, "fashion": fashs, "kl": kls}
            pm, ps = np.mean(pars), np.std(pars)
            fm, fs = np.mean(fashs), np.std(fashs)
            km, ks = np.mean(kls), np.std(kls)
            print(f"[{done}/{total}] {key:<16} par={pm:.4f}±{ps:.4f}  fash={fm:.4f}±{fs:.4f}  KL={km:.4f}±{ks:.4f}")

    with open(OUTPUT / "results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {OUTPUT / 'results.json'}")
