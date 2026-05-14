#!/usr/bin/env python3
"""
Fixed LR sweep with best-checkpoint selection.
5 methods × 5 mantissa × {30 seeds for GRPO, 5 seeds for others}.
Evaluates every 0.2 epochs (10 checkpoints), selects best parity accuracy.
"""
import os, sys, json, copy, torch, torch.nn as nn, torch.multiprocessing as mp, numpy as np
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from rl_razor.model import MLP
from rl_razor.data import get_finetuning_data, get_parity_mnist, get_fashion_mnist, create_dataloader
from rl_razor.metrics import forward_kl
from rl_razor.utils import set_seed

BS = 64; LR = 1e-4; N_EPOCHS = 2; GROUP_SIZE = 8
EVAL_FRAC = 0.2  # evaluate every 0.2 epochs
MANTISSA = [(7,"m7"), (8,"m8"), (9,"m9"), (10,"m10"), (23,"fp32")]
METHODS = ["sft1", "sft2", "oracle", "grpo", "grpo_kl"]
SEEDS_GRPO = list(range(30))
SEEDS_OTHER = [42, 123, 456, 789, 1024]
PRETRAIN = "experiments/mantissa_sweep/pretrain/pretrained_model.pt"
OUTPUT = Path("experiments/bestckpt_sweep")

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

def run_one(method, mbits, seed, base_sd, tl, pl, fl, device):
    set_seed(seed)
    base = MLP().to(device); base.load_state_dict(base_sd); base.eval()
    model = MLP().to(device); model.load_state_dict(copy.deepcopy(base_sd))
    snap_model(model, mbits)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)

    steps_per_epoch = len(tl)
    ckpt_interval = max(1, int(steps_per_epoch * EVAL_FRAC))

    best = {"parity": 0, "fashion": 0, "kl": 0, "epoch": 0}
    all_checkpoints = []
    global_step = 0

    for ep in range(1, N_EPOCHS+1):
        model.train()
        for x, y in tl:
            x, y = x.to(device), y.to(device)
            n = x.size(0); opt.zero_grad()
            if method in ("sft1","sft2","oracle"):
                if method == "sft1": targets = (y%2).long()
                elif method == "sft2":
                    targets = y.clone(); even = y%2==0; odd = ~even
                    targets[even] = torch.where(torch.rand(even.sum(),device=device)<0.5, torch.zeros(even.sum(),device=device,dtype=torch.long), torch.full((even.sum(),),4,device=device,dtype=torch.long))
                    targets[odd] = torch.where(torch.rand(odd.sum(),device=device)<0.5, torch.ones(odd.sum(),device=device,dtype=torch.long), torch.full((odd.sum(),),5,device=device,dtype=torch.long))
                else:
                    with torch.no_grad(): bp = torch.softmax(base(x),1)
                    parity = y%2; mask = torch.zeros(n,10,device=device)
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
                    with torch.no_grad(): blp = torch.log_softmax(base(x_g),1)
                    loss = loss + 0.1*(probs*(lp-blp)).sum(1).mean()
                loss.backward()
            opt.step(); snap_model(model, mbits)
            global_step += 1

            if global_step % ckpt_interval == 0:
                par, fash, kl = evaluate(model, base, pl, fl, device)
                ep_frac = global_step / steps_per_epoch
                ckpt = {"parity": par, "fashion": fash, "kl": kl, "epoch": round(ep_frac, 2)}
                all_checkpoints.append(ckpt)
                if par > best["parity"]:
                    best = ckpt

    del model, base; torch.cuda.empty_cache()
    return best, all_checkpoints

def worker(gpu_id, jobs, base_sd, results_dict):
    device = f"cuda:{gpu_id}"
    train_data, _ = get_finetuning_data("rl")
    tl = create_dataloader(train_data, batch_size=BS, shuffle=True)
    pl = create_dataloader(get_parity_mnist(train=False), batch_size=512, shuffle=False)
    fl = create_dataloader(get_fashion_mnist(train=False), batch_size=512, shuffle=False)

    for i, (method, mbits, mlabel, seed) in enumerate(jobs):
        best, all_ckpts = run_one(method, mbits, seed, base_sd, tl, pl, fl, device)
        key = f"{method}_{mlabel}_seed{seed}"
        results_dict[key] = {
            "method": method, "mantissa": mlabel, "mantissa_bits": mbits, "seed": seed,
            "best_parity": best["parity"], "best_fashion": best["fashion"],
            "best_kl": best["kl"], "best_epoch": best["epoch"],
            "final_parity": all_ckpts[-1]["parity"], "final_fashion": all_ckpts[-1]["fashion"],
            "final_kl": all_ckpts[-1]["kl"],
        }
        if (i+1) % 20 == 0:
            print(f"  [GPU {gpu_id}] {i+1}/{len(jobs)} done", flush=True)

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)

    base_sd = torch.load(PRETRAIN, map_location="cpu", weights_only=True)
    if "model_state_dict" in base_sd: base_sd = base_sd["model_state_dict"]

    all_jobs = []
    for method in METHODS:
        seeds = SEEDS_GRPO if method == "grpo" else SEEDS_OTHER
        for mbits, mlabel in MANTISSA:
            for seed in seeds:
                all_jobs.append((method, mbits, mlabel, seed))

    print(f"{len(all_jobs)} total jobs (best-checkpoint selection, LR={LR})")

    n_gpus = 8
    gpu_jobs = [[] for _ in range(n_gpus)]
    for i, job in enumerate(all_jobs):
        gpu_jobs[i % n_gpus].append(job)

    manager = mp.Manager()
    results_dict = manager.dict()

    procs = []
    for gpu_id in range(n_gpus):
        print(f"  GPU {gpu_id}: {len(gpu_jobs[gpu_id])} jobs")
        p = mp.Process(target=worker, args=(gpu_id, gpu_jobs[gpu_id], base_sd, results_dict))
        p.start(); procs.append(p)

    for p in procs:
        p.join()

    results = dict(results_dict)
    with open(OUTPUT / "results.json", "w") as f:
        json.dump(results, f)

    # Summary
    print(f"\n{len(results)} results saved")
    print(f"\n{'Method':<12} {'Mantissa':<8} {'Best Parity':>14} {'Best Fashion':>16} {'Best KL':>14}  (seeds)")
    print("-" * 70)
    for method in METHODS:
        seeds = SEEDS_GRPO if method == "grpo" else SEEDS_OTHER
        for _, mlabel in MANTISSA:
            vals = [v for v in results.values() if v["method"]==method and v["mantissa"]==mlabel]
            bp = np.mean([v["best_parity"] for v in vals])
            bf = np.mean([v["best_fashion"] for v in vals])
            bk = np.mean([v["best_kl"] for v in vals])
            bps = np.std([v["best_parity"] for v in vals])
            bfs = np.std([v["best_fashion"] for v in vals])
            bks = np.std([v["best_kl"] for v in vals])
            print(f"{method:<12} {mlabel:<8} {bp:.4f}±{bps:.4f}  {bf:.4f}±{bfs:.4f}  {bk:.4f}±{bks:.4f}  ({len(vals)})")

    print(f"\nSaved to {OUTPUT / 'results.json'}")
