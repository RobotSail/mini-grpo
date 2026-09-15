#!/usr/bin/env python3
"""Magnitude threshold sweep: fp32 GRPO with per-element update thresholding.

Cosine warmup scheduler, 26 thresholds (coarse+fine superset) + bf16/fp32 baselines.
GRPO unregularized, 30 seeds, 8 GPUs.
"""
import os, sys, json, copy, math, torch, torch.multiprocessing as mp, numpy as np
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from rl_razor.model import MLP
from rl_razor.data import get_finetuning_data, get_parity_mnist, get_fashion_mnist, create_dataloader
from rl_razor.metrics import forward_kl
from rl_razor.utils import set_seed

BS = 64; LR = 1e-4; N_EPOCHS = 2; GROUP_SIZE = 8; WARMUP_RATIO = 0.1
SEEDS = list(range(30))
t1 = np.logspace(np.log10(2.33e-10), np.log10(4.88e-04), 20).tolist()
t2 = np.arange(5e-5, 1e-4 + 1e-10, 1e-5).tolist()
THRESHOLDS = sorted(set(round(t, 15) for t in t1 + t2))
PRETRAIN = "experiments/mantissa_sweep/pretrain/pretrained_model.pt"
OUTPUT = Path("experiments/magnitude_threshold_sweep_v2")
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


def snap_rne(t, b):
    shift = 23 - b; halfway = 1 << (shift - 1); mask = ~((1 << shift) - 1); lsb = 1 << shift
    raw = t.view(torch.int32); trunc = raw & mask; rem = raw & ((1 << shift) - 1)
    up = (rem > halfway) | ((rem == halfway) & (((trunc >> shift) & 1) == 1))
    t.copy_((trunc + up.int() * lsb).view(torch.float32))


def snap_model(model):
    with torch.no_grad():
        for p in model.parameters():
            snap_rne(p.data, 7)


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


def run_one(mode, threshold, base_model, base_sd, train_loader, pl, fl, seed, device, steps_per_epoch):
    set_seed(seed)
    model = MLP().to(device)
    model.load_state_dict(copy.deepcopy(base_sd))

    if mode == "bf16":
        snap_model(model)

    opt = torch.optim.AdamW(model.parameters(), lr=LR)
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

            if mode == "threshold":
                w_old = {id(p): p.data.clone() for p in model.parameters()}

            opt.step()
            scheduler.step()

            if mode == "threshold":
                with torch.no_grad():
                    for p in model.parameters():
                        delta = p.data - w_old[id(p)]
                        mask = delta.abs() < threshold
                        p.data[mask] = w_old[id(p)][mask]
            elif mode == "bf16":
                snap_model(model)

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

    for mode, threshold, seed in jobs:
        par, fash, kl = run_one(mode, threshold, base, base_sd, tl, pl, fl, seed, device, steps_per_epoch)
        if mode == "threshold":
            key = f"thresh_{threshold:.2e}_seed{seed}"
        else:
            key = f"{mode}_seed{seed}"
        results_dict[key] = {
            "parity": par, "fashion": fash, "kl": kl,
            "threshold": threshold, "mode": mode, "seed": seed,
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
    for threshold in THRESHOLDS:
        for seed in SEEDS:
            all_jobs.append(("threshold", threshold, seed))
    for seed in SEEDS:
        all_jobs.append(("bf16", 0.0, seed))
        all_jobs.append(("fp32", 0.0, seed))

    n_workers = N_GPUS * WORKERS_PER_GPU
    print(f"{len(all_jobs)} total jobs across {n_workers} workers")
    print(f"Thresholds: {[f'{t:.2e}' for t in THRESHOLDS]}")

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
    out_path = OUTPUT / "threshold_sweep_all_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'Mode':<14} {'Threshold':<14} {'Parity':>14} {'Fashion':>16} {'KL':>16}")
    print("-" * 78)
    for mode in ["fp32", "bf16"]:
        vals = [v for v in results.values() if v["mode"] == mode]
        if not vals: continue
        pars = [v["parity"] for v in vals]; fashs = [v["fashion"] for v in vals]; kls = [v["kl"] for v in vals]
        print(f"{mode:<14} {'---':<14} {np.mean(pars):.4f}±{np.std(pars):.4f}  "
              f"{np.mean(fashs):.4f}±{np.std(fashs):.4f}  {np.mean(kls):.4f}±{np.std(kls):.4f}")
    for threshold in THRESHOLDS:
        vals = [v for v in results.values() if v["mode"] == "threshold" and abs(v["threshold"] - threshold) < threshold * 1e-6]
        if not vals: continue
        pars = [v["parity"] for v in vals]; fashs = [v["fashion"] for v in vals]; kls = [v["kl"] for v in vals]
        print(f"{'threshold':<14} {threshold:<14.2e} {np.mean(pars):.4f}±{np.std(pars):.4f}  "
              f"{np.mean(fashs):.4f}±{np.std(fashs):.4f}  {np.mean(kls):.4f}±{np.std(kls):.4f}")

    print(f"\nSaved {len(results)} results to {out_path}")
