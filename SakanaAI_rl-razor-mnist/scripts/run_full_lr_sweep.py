#!/usr/bin/env python3
"""Full LR sweep: 15 LRs × 11 mantissa (m1-m10+m23) × 5 methods × 30 seeds.

Reuses existing results in sakana_lr_sweep/ (skips completed jobs).
Sub-bf16 (m1-m6) uses the quantize_update approach automatically.
After completion, consolidates all results into a single JSON file.
"""
import os, sys, subprocess, time, json, numpy as np
from pathlib import Path
from collections import defaultdict

FINETUNE_SCRIPT = os.path.join(os.path.dirname(__file__), "finetune.py")
PRETRAIN = "experiments/mantissa_sweep/pretrain/pretrained_model.pt"
OUTPUT = "/mnt/nvme1n1/experiments/sakana_lr_sweep"
N_GPUS = 8
PER_GPU = 10

LRS = np.logspace(np.log10(3e-6), np.log10(1e-3), 15).tolist()
MANTISSA = [(1, "m1"), (2, "m2"), (3, "m3"), (4, "m4"), (5, "m5"), (6, "m6"),
            (7, "m7"), (8, "m8"), (9, "m9"), (10, "m10"), (23, "fp32")]
METHODS = ["sft1", "sft2", "oracle", "grpo", "grpo_kl"]
SEEDS = list(range(30))

os.makedirs(OUTPUT, exist_ok=True)

# Build set of completed run names upfront (one scan instead of 24k globs)
print("Scanning existing results...")
completed = set()
for scan_dir in [OUTPUT, "experiments/sakana_lr_sweep"]:
    scan_path = Path(scan_dir)
    if scan_path.exists():
        for d in scan_path.iterdir():
            if d.is_dir() and (d / "results.json").exists():
                completed.add(d.name.rsplit("_", 1)[0])
print(f"  Found {len(completed)} completed runs")

all_jobs = []
for method in METHODS:
    for mbits, mlabel in MANTISSA:
        for lr in LRS:
            for seed in SEEDS:
                run_name = f"{method}_{mlabel}_lr{lr:.2e}_seed{seed}"
                if run_name in completed:
                    continue
                all_jobs.append((method, mbits, mlabel, lr, seed, run_name))

batch_size = N_GPUS * PER_GPU
total_batches = (len(all_jobs) + batch_size - 1) // batch_size

print(f"{len(all_jobs)} jobs to run ({batch_size} per batch, {total_batches} batches)")
print(f"Estimated time: {total_batches * 2.5 / 60:.1f} hours")
t0 = time.time()

total_ok = total_fail = 0
for batch_start in range(0, len(all_jobs), batch_size):
    batch = all_jobs[batch_start:batch_start + batch_size]
    batch_num = batch_start // batch_size + 1
    elapsed = time.time() - t0
    if batch_num > 1:
        rate = elapsed / batch_start
        eta = rate * (len(all_jobs) - batch_start)
        print(f"\nBatch {batch_num}/{total_batches} ({len(batch)} jobs)... "
              f"[elapsed {elapsed/60:.0f}m, ETA {eta/60:.0f}m]")
    else:
        print(f"\nBatch {batch_num}/{total_batches} ({len(batch)} jobs)...")

    procs = []
    for i, (method, mbits, mlabel, lr, seed, run_name) in enumerate(batch):
        gpu_id = i % N_GPUS
        cmd = [
            sys.executable, FINETUNE_SCRIPT,
            "--pretrained-model", PRETRAIN,
            "--method", method,
            "--lr", str(lr),
            "--seed", str(seed),
            "--mantissa-bits", str(mbits),
            "--exp-dir", OUTPUT,
            "--wandb-name", run_name,
            "--checkpoint-every", "0.2",
            "--skip-alt-metrics",
        ]
        if method == "grpo_kl":
            cmd += ["--kl-coef", "0.1"]

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, env=env)
        procs.append((run_name, proc))

    ok = fail = 0
    for run_name, proc in procs:
        _, stderr = proc.communicate()
        if proc.returncode != 0:
            err = stderr.decode()[-200:] if stderr else ""
            print(f"  FAILED {run_name}: {err}")
            fail += 1
        else:
            ok += 1
    total_ok += ok; total_fail += fail
    print(f"  {ok} ok, {fail} failed (cumulative: {total_ok}/{total_ok+total_fail})")

elapsed = time.time() - t0
print(f"\nTraining done: {total_ok} ok, {total_fail} failed in {elapsed:.0f}s ({elapsed/60:.1f}m)")

# ── Consolidate all results into a single JSON ───────────────────────────
print("\nConsolidating results from both locations...")
all_results = []
for scan_dir in [OUTPUT, "experiments/sakana_lr_sweep"]:
  for d in sorted(Path(scan_dir).iterdir()):
    rfile = d / "results.json"
    if not rfile.exists():
        continue
    r = json.load(open(rfile))
    config = r.get("config", {})
    all_results.append({
        "method": r.get("method", config.get("method", "")),
        "mantissa_bits": config.get("mantissa_bits", 0),
        "lr": config.get("lr", 0),
        "seed": config.get("seed", 0),
        "final_parity_acc": r.get("final_parity_acc", 0),
        "final_fashion_acc": r.get("final_fashion_acc", 0),
        "final_kl_divergence": r.get("final_kl_divergence", 0),
        "checkpoints": r.get("checkpoints", []),
        "run_dir": str(d),
    })

out_file = os.path.join(OUTPUT, "lr_sweep_all_results.json")
with open(out_file, "w") as f:
    json.dump(all_results, f, indent=1)
print(f"Saved {len(all_results)} results to {out_file}")
print(f"Results in: {OUTPUT}/")
