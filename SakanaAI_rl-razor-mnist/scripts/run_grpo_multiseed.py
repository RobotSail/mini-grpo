#!/usr/bin/env python3
"""Run GRPO across all mantissa widths with 4 additional seeds (seed=42 already done)."""
import os, sys, subprocess, time
from pathlib import Path
from itertools import product

PRETRAIN = "experiments/mantissa_sweep/pretrain/pretrained_model.pt"
OUTPUT_ROOT = "/mnt/nvme3n1/workspace/osilkin/rl-razor-mnist/experiments/mantissa_full_sweep"
N_GPUS = 8
JOBS_PER_GPU = 10
LR = 1e-4
EPOCHS = 2
CKPT_EVERY = 0.2
MANTISSA_BITS = list(range(7, 24))
SEEDS = [123, 456, 789, 1024]  # seed=42 already done

os.makedirs(OUTPUT_ROOT, exist_ok=True)

jobs = []
for seed, mbits in product(SEEDS, MANTISSA_BITS):
    label = "fp32" if mbits == 23 else f"m{mbits}"
    run_name = f"grpo_{label}_seed{seed}"
    matches = list(Path(OUTPUT_ROOT).glob(f"{run_name}_*/results.json"))
    if matches:
        print(f"SKIP {run_name}")
        continue
    jobs.append((seed, mbits, run_name))

print(f"\n{len(jobs)} jobs, {N_GPUS} GPUs, {JOBS_PER_GPU}/GPU")
batch_size = N_GPUS * JOBS_PER_GPU
t0 = time.time()

for batch_start in range(0, len(jobs), batch_size):
    batch = jobs[batch_start:batch_start + batch_size]
    batch_num = batch_start // batch_size + 1
    total_batches = (len(jobs) + batch_size - 1) // batch_size
    print(f"\n=== Batch {batch_num}/{total_batches} ({len(batch)} jobs) ===")

    procs = []
    for i, (seed, mbits, run_name) in enumerate(batch):
        gpu_id = i % N_GPUS
        cmd = [
            sys.executable, "scripts/finetune.py",
            "--pretrained-model", PRETRAIN,
            "--method", "grpo",
            "--batch-size", "64",
            "--lr", str(LR),
            "--epochs", str(EPOCHS),
            "--scheduler", "cosine_with_warmup",
            "--warmup-ratio", "0.1",
            "--weight-decay", "0.0",
            "--seed", str(seed),
            "--checkpoint-every", str(CKPT_EVERY),
            "--mantissa-bits", str(mbits),
            "--exp-dir", OUTPUT_ROOT,
            "--wandb-name", run_name,
        ]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, env=env)
        procs.append((run_name, proc))

    for run_name, proc in procs:
        _, stderr = proc.communicate()
        if proc.returncode != 0:
            print(f"  FAILED {run_name}: {stderr.decode()[-200:]}")
        else:
            print(f"  OK {run_name}")

print(f"\nTotal time: {time.time()-t0:.0f}s ({(time.time()-t0)/60:.1f}m)")
