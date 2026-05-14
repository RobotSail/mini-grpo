#!/usr/bin/env python3
"""
LR sweep using SakanaAI finetune.py.
15 LRs × 5 mantissa × {30 seeds GRPO, 5 seeds others}.
"""
import os, sys, subprocess, time, json, numpy as np
from pathlib import Path

FINETUNE_SCRIPT = os.path.join(os.path.dirname(__file__), "finetune.py")
PRETRAIN = "experiments/mantissa_sweep/pretrain/pretrained_model.pt"
OUTPUT = "experiments/sakana_lr_sweep"
N_GPUS = 8
PER_GPU = 10

LRS = np.logspace(np.log10(3e-6), np.log10(1e-3), 15).tolist()
MANTISSA = [(7, "m7"), (8, "m8"), (9, "m9"), (10, "m10"), (23, "fp32")]
METHODS = ["sft1", "sft2", "oracle", "grpo", "grpo_kl"]
SEEDS_GRPO = list(range(30))
SEEDS_OTHER = [42, 123, 456, 789, 1024]

os.makedirs(OUTPUT, exist_ok=True)

# Build all jobs
all_jobs = []
for method in METHODS:
    seeds = SEEDS_GRPO if method == "grpo" else SEEDS_OTHER
    for mbits, mlabel in MANTISSA:
        for lr in LRS:
            for seed in seeds:
                run_name = f"{method}_{mlabel}_lr{lr:.2e}_seed{seed}"
                # Skip existing
                matches = list(Path(OUTPUT).glob(f"{run_name}_*/results.json"))
                if matches:
                    continue
                all_jobs.append((method, mbits, mlabel, lr, seed, run_name))

batch_size = N_GPUS * PER_GPU
total_batches = (len(all_jobs) + batch_size - 1) // batch_size

print(f"{len(all_jobs)} jobs to run ({batch_size} per batch, {total_batches} batches)")
t0 = time.time()

total_ok = total_fail = 0
for batch_start in range(0, len(all_jobs), batch_size):
    batch = all_jobs[batch_start:batch_start + batch_size]
    batch_num = batch_start // batch_size + 1
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
            fail += 1
        else:
            ok += 1
    total_ok += ok; total_fail += fail
    print(f"  {ok} ok, {fail} failed (cumulative: {total_ok}/{total_ok+total_fail})")

elapsed = time.time() - t0
print(f"\nDone: {total_ok} ok, {total_fail} failed in {elapsed:.0f}s ({elapsed/60:.1f}m)")
print(f"Results in: {OUTPUT}/")
