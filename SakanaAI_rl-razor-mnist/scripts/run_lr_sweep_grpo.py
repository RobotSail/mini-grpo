#!/usr/bin/env python3
"""LR sweep for GRPO unregularized in fp32. 15 LRs, 30 seeds, cosine warmup, 8 GPUs."""
import os, sys, json, copy, math, torch, torch.multiprocessing as mp, numpy as np
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from rl_razor.model import MLP
from rl_razor.data import get_finetuning_data, get_parity_mnist, get_fashion_mnist, create_dataloader
from rl_razor.metrics import forward_kl
from rl_razor.utils import set_seed

BS = 64; N_EPOCHS = 2; GROUP_SIZE = 8; WARMUP_RATIO = 0.1
SEEDS = list(range(30))
LRS = [
    3.000000e-06, 4.542834e-06, 6.879114e-06, 1.041689e-05, 1.577407e-05,
    2.388633e-05, 3.617054e-05, 5.477226e-05, 8.294042e-05, 1.255949e-04,
    1.901855e-04, 2.879938e-04, 4.361027e-04, 6.603807e-04, 1.000000e-03,
]
PRETRAIN = "experiments/mantissa_sweep/pretrain/pretrained_model.pt"
OUTPUT = Path("experiments/lr_sweep_grpo_v2")
N_GPUS = 8
WORKERS_PER_GPU = 10


def make_scheduler(optimizer, total_steps):
    warmup_steps = int(WARMUP_RATIO * total_steps)
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


@torch.no_grad()
def evaluate(model, base, pl, fl, device):
    model.eval()
    c = t = 0
    for x, y in pl:
        x, y = x.to(device), y.to(device)
        c += ((model(x).argmax(1) % 2) == (y % 2)).sum().item()
        t += y.size(0)
    par = c / t
    c = t = 0
    for x, y in fl:
        x, y = x.to(device), y.to(device)
        c += (model(x).argmax(1) == y).sum().item()
        t += y.size(0)
    fash = c / t
    kl = forward_kl(base, model, pl, device)
    model.train()
    return par, fash, kl


def run_one(lr, base_model, base_sd, train_loader, pl, fl, seed, device, steps_per_epoch):
    set_seed(seed)
    model = MLP().to(device)
    model.load_state_dict(copy.deepcopy(base_sd))
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = make_scheduler(opt, N_EPOCHS * steps_per_epoch)

    for ep in range(N_EPOCHS):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            n = x.size(0)
            opt.zero_grad()

            x_g = x.unsqueeze(1).expand(-1, GROUP_SIZE, -1).reshape(n * GROUP_SIZE, -1)
            y_g = y.unsqueeze(1).expand(-1, GROUP_SIZE).reshape(n * GROUP_SIZE)
            logits = model(x_g)
            probs = torch.softmax(logits, 1)
            lp = torch.log_softmax(logits, 1)
            actions = torch.multinomial(probs, 1).squeeze(1)
            rewards = ((actions % 2) == (y_g % 2)).float().reshape(n, GROUP_SIZE)
            adv = (rewards - rewards.mean(1, keepdim=True)) / (rewards.std(1, keepdim=True) + 1e-8)
            sel = lp.gather(1, actions.unsqueeze(1)).squeeze(1)
            loss = -(adv.reshape(n * GROUP_SIZE) * sel).mean()
            loss.backward()

            opt.step()
            scheduler.step()

    return evaluate(model, base_model, pl, fl, device)


def worker(gpu_id, jobs, base_sd, results_dict):
    device = f"cuda:{gpu_id}"
    base = MLP().to(device)
    base.load_state_dict(base_sd)
    base.eval()

    train_data, _ = get_finetuning_data("rl")
    tl = create_dataloader(train_data, batch_size=BS, shuffle=True)
    steps_per_epoch = len(tl)
    pl = create_dataloader(get_parity_mnist(train=False), batch_size=512, shuffle=False)
    fl = create_dataloader(get_fashion_mnist(train=False), batch_size=512, shuffle=False)

    for lr, seed in jobs:
        par, fash, kl = run_one(lr, base, base_sd, tl, pl, fl, seed, device, steps_per_epoch)
        key = f"grpo_lr{lr:.2e}_seed{seed}"
        results_dict[key] = {
            "parity": par, "fashion": fash, "kl": kl,
            "lr": lr, "method": "grpo", "seed": seed,
        }
        print(f"  [GPU{gpu_id}] {key}: par={par:.4f} fash={fash:.4f} KL={kl:.4f}")
        torch.cuda.empty_cache()


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)

    base_sd = torch.load(PRETRAIN, map_location="cpu", weights_only=True)
    if "model_state_dict" in base_sd:
        base_sd = base_sd["model_state_dict"]

    all_jobs = []
    for lr in LRS:
        for seed in SEEDS:
            all_jobs.append((lr, seed))

    n_workers = N_GPUS * WORKERS_PER_GPU
    print(f"{len(all_jobs)} total jobs ({len(LRS)} LRs x {len(SEEDS)} seeds) across {n_workers} workers")

    worker_jobs = [[] for _ in range(n_workers)]
    for i, job in enumerate(all_jobs):
        worker_jobs[i % n_workers].append(job)

    manager = mp.Manager()
    results_dict = manager.dict()

    procs = []
    for w_id in range(n_workers):
        gpu_id = w_id % N_GPUS
        p = mp.Process(target=worker, args=(gpu_id, worker_jobs[w_id], base_sd, results_dict))
        p.start()
        procs.append(p)

    for p in procs:
        p.join()

    results = dict(results_dict)
    out_path = OUTPUT / "lr_sweep_grpo_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'LR':<14} {'Parity':>14} {'Fashion':>16} {'KL':>16}")
    print("-" * 64)
    for lr in LRS:
        vals = [v for v in results.values() if abs(v["lr"] - lr) < lr * 1e-6]
        if not vals: continue
        pars = [v["parity"] for v in vals]; fashs = [v["fashion"] for v in vals]; kls = [v["kl"] for v in vals]
        print(f"{lr:<14.2e} {np.mean(pars):.4f}±{np.std(pars):.4f}  "
              f"{np.mean(fashs):.4f}±{np.std(fashs):.4f}  {np.mean(kls):.4f}±{np.std(kls):.4f}")

    print(f"\nSaved {len(results)} results to {out_path}")
