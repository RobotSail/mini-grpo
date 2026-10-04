#!/usr/bin/env python3
"""Gradient-clipping ablation for EMNLP rebuttal.

Tests whether the BF16-vs-FP32 precision effect is mediated by gradient clipping.
Does NOT modify the existing training code — uses hooks around optimizer.step().

Grid:
  Methods:   sft1, grpo (no KL)
  Precision: bf16 (m7), fp32 (m23)
  Clipping:  none, 0.5, 1.0, 2.0
  Seeds:     5 (sft1), 30 (grpo)
"""

import os, sys, json, copy, math, time, argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rl_razor.model import MLP
from rl_razor.data import (
    get_finetuning_data, get_parity_mnist, get_fashion_mnist, create_dataloader,
)
from rl_razor.metrics import forward_kl
from rl_razor.utils import set_seed, snap_to_lattice, save_weights, _snap_rne

# ── Constants matching mantissa-sweep recipe ────────────────────────────────
BS = 64
LR = 1e-4
N_EPOCHS = 2
GROUP_SIZE = 8
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.0

PRETRAIN = "experiments/mantissa_sweep/pretrain/pretrained_model.pt"

# ── Helpers ─────────────────────────────────────────────────────────────────

def make_scheduler(optimizer, total_steps):
    warmup_steps = int(WARMUP_RATIO * total_steps)
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def snap_model(model, mbits):
    if mbits >= 23:
        return
    with torch.no_grad():
        for p in model.parameters():
            _snap_rne(p.data, mbits)


def half_ulp(mantissa_bits):
    """Half-ULP at the typical weight magnitude for this precision.

    For fp32 (m23): 2^(-24) ≈ 5.96e-8
    For bf16 (m7):  2^(-8)  ≈ 3.91e-3
    These are at exponent 0 (|w| in [1,2)); actual ULP varies with magnitude,
    so we compute per-parameter below.
    """
    return 2.0 ** (-(mantissa_bits + 1))


def per_param_half_ulp(w, mantissa_bits):
    """Compute the half-ULP threshold for each element based on its exponent."""
    w_abs = w.abs().clamp(min=1e-45)
    exponent = torch.floor(torch.log2(w_abs))
    return 2.0 ** (exponent - mantissa_bits)


@torch.no_grad()
def evaluate(model, base, parity_loader, fashion_loader, device):
    model.eval()
    # Parity accuracy
    c = t = 0
    for x, y in parity_loader:
        x, y = x.to(device), y.to(device)
        c += ((model(x).argmax(1) % 2) == (y % 2)).sum().item()
        t += y.size(0)
    par = c / t
    # Fashion accuracy
    c = t = 0
    for x, y in fashion_loader:
        x, y = x.to(device), y.to(device)
        c += (model(x).argmax(1) == y).sum().item()
        t += y.size(0)
    fash = c / t
    # KL
    kl = forward_kl(base, model, parity_loader, device)
    model.train()
    return par, fash, kl


def compute_sparsity(model, base_sd):
    """Fraction of parameters with nonzero change vs pretrained."""
    nonzero = 0
    total = 0
    for name, p in model.named_parameters():
        delta = p.data - base_sd[name].to(p.device)
        nonzero += (delta != 0).sum().item()
        total += p.numel()
    return nonzero / total if total > 0 else 0.0


def compute_update_stats(w_old_dict, model, mantissa_bits):
    """Per-parameter update magnitude stats: median, p90, below-half-ULP fraction."""
    all_abs = []
    below_ulp_count = 0
    total_count = 0
    for p in model.parameters():
        w_old = w_old_dict[id(p)]
        delta = (p.data - w_old).abs()
        all_abs.append(delta.flatten())
        hulp = per_param_half_ulp(w_old, mantissa_bits)
        below_ulp_count += (delta < hulp).sum().item()
        total_count += p.numel()
    all_abs = torch.cat(all_abs)
    return {
        "median_abs_delta": all_abs.median().item(),
        "p90_abs_delta": all_abs.quantile(0.9).item(),
        "below_half_ulp_frac": below_ulp_count / total_count if total_count > 0 else 0.0,
    }


# ── Training with clipping instrumentation ──────────────────────────────────

def run_sft_clipped(model, base_model, base_sd_named, train_loader, parity_val, fashion_val,
                    seed, device, mantissa_bits, max_norm, steps_per_epoch):
    set_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = make_scheduler(opt, N_EPOCHS * steps_per_epoch)

    clip_fired = 0
    total_steps = 0
    update_stats_log = {}  # step -> stats
    report_steps = set()
    total = N_EPOCHS * steps_per_epoch
    for frac in [0.1, 0.5, 0.9]:
        report_steps.add(max(1, int(frac * total)))

    snap_model(model, mantissa_bits)

    for ep in range(N_EPOCHS):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()

            # Measure grad norm BEFORE clipping
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=float('inf')  # just measure
            )

            # Apply clipping if enabled
            if max_norm is not None and max_norm > 0:
                if grad_norm > max_norm:
                    clip_fired += 1
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)

            w_old = {id(p): p.data.clone() for p in model.parameters()}
            opt.step()
            snap_model(model, mantissa_bits)
            scheduler.step()

            total_steps += 1

            if total_steps in report_steps:
                update_stats_log[total_steps] = compute_update_stats(w_old, model, mantissa_bits)

    par, fash, kl = evaluate(model, base_model, parity_val, fashion_val, device)
    sparsity = compute_sparsity(model, base_sd_named)
    clip_frac = clip_fired / total_steps if total_steps > 0 else 0.0

    return {
        "parity_acc": par, "fashion_acc": fash, "forward_kl": kl,
        "clip_fired_fraction": clip_frac, "final_sparsity": sparsity,
        "update_stats": update_stats_log, "total_steps": total_steps,
        "grad_norm_at_clip": grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
    }


def run_grpo_clipped(model, base_model, base_sd_named, train_loader, parity_val, fashion_val,
                     seed, device, mantissa_bits, max_norm, steps_per_epoch):
    set_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = make_scheduler(opt, N_EPOCHS * steps_per_epoch)

    clip_fired = 0
    total_steps = 0
    update_stats_log = {}
    report_steps = set()
    total = N_EPOCHS * steps_per_epoch
    for frac in [0.1, 0.5, 0.9]:
        report_steps.add(max(1, int(frac * total)))

    snap_model(model, mantissa_bits)

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

            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=float('inf')
            )

            if max_norm is not None and max_norm > 0:
                if grad_norm > max_norm:
                    clip_fired += 1
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)

            w_old = {id(p): p.data.clone() for p in model.parameters()}
            opt.step()
            snap_model(model, mantissa_bits)
            scheduler.step()

            total_steps += 1

            if total_steps in report_steps:
                update_stats_log[total_steps] = compute_update_stats(w_old, model, mantissa_bits)

    par, fash, kl = evaluate(model, base_model, parity_val, fashion_val, device)
    sparsity = compute_sparsity(model, base_sd_named)
    clip_frac = clip_fired / total_steps if total_steps > 0 else 0.0

    return {
        "parity_acc": par, "fashion_acc": fash, "forward_kl": kl,
        "clip_fired_fraction": clip_frac, "final_sparsity": sparsity,
        "update_stats": update_stats_log, "total_steps": total_steps,
    }


# ── Worker ──────────────────────────────────────────────────────────────────

def worker(gpu_id, jobs, base_sd, base_sd_named, results_dict, data_dir):
    device = f"cuda:{gpu_id}"

    # SFT data
    sft_train, _ = get_finetuning_data("sft1", data_dir=data_dir)
    sft_loader = create_dataloader(sft_train, batch_size=BS, shuffle=True)
    # GRPO data
    grpo_train, _ = get_finetuning_data("rl", data_dir=data_dir)
    grpo_loader = create_dataloader(grpo_train, batch_size=BS, shuffle=True)

    parity_val = create_dataloader(get_parity_mnist(train=False, data_dir=data_dir), batch_size=512, shuffle=False)
    fashion_val = create_dataloader(get_fashion_mnist(train=False, data_dir=data_dir), batch_size=512, shuffle=False)

    for method, mbits, clip, seed in jobs:
        # Fresh model copy
        model = MLP().to(device)
        model.load_state_dict(copy.deepcopy(base_sd))
        base_model = MLP().to(device)
        base_model.load_state_dict(copy.deepcopy(base_sd))
        base_model.eval()
        base_named = {n: p.data.to(device) for n, p in base_model.named_parameters()}

        if method == "sft1":
            loader = sft_loader
            steps_per_epoch = len(loader)
            result = run_sft_clipped(
                model, base_model, base_named, loader, parity_val, fashion_val,
                seed, device, mbits, clip, steps_per_epoch)
        else:
            loader = grpo_loader
            steps_per_epoch = len(loader)
            result = run_grpo_clipped(
                model, base_model, base_named, loader, parity_val, fashion_val,
                seed, device, mbits, clip, steps_per_epoch)

        prec_label = "bf16" if mbits == 7 else "fp32"
        clip_label = "none" if clip is None else f"{clip}"
        key = f"{method}_{prec_label}_clip{clip_label}_seed{seed}"
        result["method"] = method
        result["precision"] = prec_label
        result["mantissa_bits"] = mbits
        result["clip_setting"] = clip_label
        result["seed"] = seed
        results_dict[key] = result
        print(f"  [GPU{gpu_id}] {key}: par={result['parity_acc']:.4f} "
              f"fash={result['fashion_acc']:.4f} KL={result['forward_kl']:.4f} "
              f"clip_fired={result['clip_fired_fraction']:.3f} "
              f"sparsity={result['final_sparsity']:.4f}")
        torch.cuda.empty_cache()


# ── Main ────────────────────────────────────────────────────────────────────

def build_grid(probe_only=False):
    methods_seeds = [("sft1", 5), ("grpo", 30)]
    precisions = [(7, "bf16"), (23, "fp32")]
    clips = [None, 0.5, 1.0, 2.0]

    if probe_only:
        return [("grpo", 23, 1.0, 0)]

    jobs = []
    for method, n_seeds in methods_seeds:
        for mbits, _ in precisions:
            for clip in clips:
                for seed in range(n_seeds):
                    jobs.append((method, mbits, clip, seed))
    return jobs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", action="store_true", help="Run probe only (1 config)")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--n-gpus", type=int, default=8)
    parser.add_argument("--workers-per-gpu", type=int, default=4)
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--pretrained", type=str, default=PRETRAIN)
    args = parser.parse_args()

    mp.set_start_method("spawn", force=True)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    # Load pretrained
    base_sd = torch.load(args.pretrained, map_location="cpu", weights_only=True)
    if "model_state_dict" in base_sd:
        base_sd = base_sd["model_state_dict"]
    base_sd_named = base_sd  # already a state_dict with named keys

    jobs = build_grid(probe_only=args.probe)
    n_workers = args.n_gpus * args.workers_per_gpu
    print(f"{len(jobs)} jobs, {n_workers} workers across {args.n_gpus} GPUs")

    worker_jobs = [[] for _ in range(n_workers)]
    for i, job in enumerate(jobs):
        worker_jobs[i % n_workers].append(job)

    manager = mp.Manager()
    results_dict = manager.dict()

    t0 = time.time()
    procs = []
    for w_id in range(n_workers):
        gpu_id = w_id % args.n_gpus
        bucket = worker_jobs[w_id]
        if not bucket:
            continue
        p = mp.Process(target=worker, args=(gpu_id, bucket, base_sd, base_sd_named, results_dict, args.data_dir))
        p.start()
        procs.append(p)

    for p in procs:
        p.join()

    results = dict(results_dict)
    elapsed = time.time() - t0

    # Convert update_stats keys from int to str for JSON
    for k, v in results.items():
        if "update_stats" in v:
            v["update_stats"] = {str(sk): sv for sk, sv in v["update_stats"].items()}

    out_file = output / ("probe_results.json" if args.probe else "clip_ablation_results.json")
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nDone in {elapsed:.0f}s ({elapsed/60:.1f}m). Saved to {out_file}")

    # Print summary
    if args.probe:
        for k, v in results.items():
            print(f"\n=== PROBE: {k} ===")
            print(f"  clip_fired_fraction: {v['clip_fired_fraction']:.4f}")
            print(f"  parity_acc: {v['parity_acc']:.4f}")
            print(f"  fashion_acc: {v['fashion_acc']:.4f}")
            print(f"  forward_kl: {v['forward_kl']:.4f}")
            if v['clip_fired_fraction'] < 0.05:
                print("\n  WARNING: Clipping fires on <5% of steps at max_norm=1.0!")
                print("  The clip ladder may need adjustment.")
    else:
        # Summary table
        print(f"\n{'Method':<8} {'Prec':<6} {'Clip':<6} {'Seeds':>5} "
              f"{'Parity':>14} {'Fashion':>16} {'KL':>12} {'ClipFired':>10} {'Sparsity':>10}")
        print("-" * 100)
        for method in ["sft1", "grpo"]:
            for prec in ["bf16", "fp32"]:
                for clip_label in ["none", "0.5", "1.0", "2.0"]:
                    vals = [v for v in results.values()
                            if v["method"] == method and v["precision"] == prec
                            and v["clip_setting"] == clip_label]
                    if not vals:
                        continue
                    pars = [v["parity_acc"] for v in vals]
                    fashs = [v["fashion_acc"] for v in vals]
                    kls = [v["forward_kl"] for v in vals]
                    clips_f = [v["clip_fired_fraction"] for v in vals]
                    spars = [v["final_sparsity"] for v in vals]
                    print(f"{method:<8} {prec:<6} {clip_label:<6} {len(vals):>5} "
                          f"{np.mean(pars):.4f}±{np.std(pars):.4f} "
                          f"{np.mean(fashs):.4f}±{np.std(fashs):.4f} "
                          f"{np.mean(kls):.6f} "
                          f"{np.mean(clips_f):.4f} "
                          f"{np.mean(spars):.4f}")


if __name__ == "__main__":
    main()
