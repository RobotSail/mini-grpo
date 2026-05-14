#!/usr/bin/env python3
"""GRPO (no KL) across m7-m10+fp32, 30 seeds, parallel on GPUs."""
import os, sys, json, copy, torch, torch.nn as nn, torch.multiprocessing as mp, numpy as np
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from rl_razor.model import MLP
from rl_razor.data import get_finetuning_data, get_parity_mnist, get_fashion_mnist, create_dataloader
from rl_razor.metrics import forward_kl
from rl_razor.utils import set_seed

BS = 64; LR = 1e-4; N_EPOCHS = 2; GROUP_SIZE = 8
SEEDS = list(range(30))
MANTISSA = [(7,"m7"), (8,"m8"), (9,"m9"), (10,"m10"), (23,"fp32")]
PRETRAIN = "experiments/mantissa_sweep/pretrain/pretrained_model.pt"
OUTPUT = Path("experiments/grpo_30seeds")

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

def worker(gpu_id, jobs, base_sd, results_dict):
    device = f"cuda:{gpu_id}"
    base = MLP().to(device); base.load_state_dict(base_sd); base.eval()

    train_data, _ = get_finetuning_data("rl")
    tl = create_dataloader(train_data, batch_size=BS, shuffle=True)
    pl = create_dataloader(get_parity_mnist(train=False), batch_size=512, shuffle=False)
    fl = create_dataloader(get_fashion_mnist(train=False), batch_size=512, shuffle=False)

    for mbits, mlabel, seed in jobs:
        set_seed(seed)
        model = MLP().to(device)
        model.load_state_dict(copy.deepcopy(base_sd))
        snap_model(model, mbits)
        opt = torch.optim.AdamW(model.parameters(), lr=LR)

        for ep in range(1, N_EPOCHS+1):
            model.train()
            for x, y in tl:
                x, y = x.to(device), y.to(device)
                n = x.size(0); opt.zero_grad()
                x_g = x.unsqueeze(1).expand(-1,GROUP_SIZE,-1).reshape(n*GROUP_SIZE,-1)
                y_g = y.unsqueeze(1).expand(-1,GROUP_SIZE).reshape(n*GROUP_SIZE)
                logits = model(x_g); probs = torch.softmax(logits,1); lp = torch.log_softmax(logits,1)
                actions = torch.multinomial(probs,1).squeeze(1)
                rewards = ((actions%2)==(y_g%2)).float().reshape(n,GROUP_SIZE)
                adv = (rewards-rewards.mean(1,keepdim=True))/(rewards.std(1,keepdim=True)+1e-8)
                sel = lp.gather(1,actions.unsqueeze(1)).squeeze(1)
                loss = -(adv.reshape(n*GROUP_SIZE)*sel).mean()
                loss.backward(); opt.step(); snap_model(model, mbits)

        par, fash, kl = evaluate(model, base, pl, fl, device)
        key = f"{mlabel}_seed{seed}"
        results_dict[key] = {"parity": par, "fashion": fash, "kl": kl, "mantissa": mlabel, "seed": seed}
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)

    base_sd = torch.load(PRETRAIN, map_location="cpu", weights_only=True)
    if "model_state_dict" in base_sd:
        base_sd = base_sd["model_state_dict"]

    # Build all jobs
    all_jobs = []
    for mbits, mlabel in MANTISSA:
        for seed in SEEDS:
            all_jobs.append((mbits, mlabel, seed))

    print(f"{len(all_jobs)} total jobs, 8 GPUs")

    # Distribute across GPUs
    n_gpus = 8
    gpu_jobs = [[] for _ in range(n_gpus)]
    for i, job in enumerate(all_jobs):
        gpu_jobs[i % n_gpus].append(job)

    manager = mp.Manager()
    results_dict = manager.dict()

    procs = []
    for gpu_id in range(n_gpus):
        p = mp.Process(target=worker, args=(gpu_id, gpu_jobs[gpu_id], base_sd, results_dict))
        p.start(); procs.append(p)

    for p in procs:
        p.join()

    # Aggregate
    results = dict(results_dict)
    with open(OUTPUT / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'Mantissa':<8} {'Parity':>14} {'Fashion':>16} {'KL':>16}")
    print("-" * 58)
    for mbits, mlabel in MANTISSA:
        pars = [v["parity"] for v in results.values() if v["mantissa"] == mlabel]
        fashs = [v["fashion"] for v in results.values() if v["mantissa"] == mlabel]
        kls = [v["kl"] for v in results.values() if v["mantissa"] == mlabel]
        print(f"{mlabel:<8} {np.mean(pars):.4f}±{np.std(pars):.4f}  "
              f"{np.mean(fashs):.4f}±{np.std(fashs):.4f}  "
              f"{np.mean(kls):.4f}±{np.std(kls):.4f}")

    print(f"\nSaved to {OUTPUT / 'results.json'}")
