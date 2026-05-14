#!/usr/bin/env python3
"""
Run all 5 methods × 17 mantissa widths (m7-m23) using SakanaAI GRPO replication.
Batches 10 jobs per GPU across 8 GPUs.
All output written to NVMe drive.
"""
import os, sys, subprocess, time, json
from pathlib import Path
from itertools import product

PRETRAIN = "experiments/mantissa_sweep/pretrain/pretrained_model.pt"
OUTPUT_ROOT = "/mnt/nvme3n1/workspace/osilkin/rl-razor-mnist/experiments/mantissa_full_sweep"
N_GPUS = 8
JOBS_PER_GPU = 10
LR = 1e-4
EPOCHS = 2
SEED = 42
CKPT_EVERY = 0.2

METHODS = ["grpo", "grpo_kl", "sft1", "sft2", "oracle"]
MANTISSA_BITS = list(range(7, 24))  # 7 (bf16) through 23 (fp32/no snap)

os.makedirs(OUTPUT_ROOT, exist_ok=True)

# Build all jobs
jobs = []
for method, mbits in product(METHODS, MANTISSA_BITS):
    label = "fp32" if mbits == 23 else f"m{mbits}"
    run_name = f"{method}_{label}"
    # Check if already done
    matches = list(Path(OUTPUT_ROOT).glob(f"{run_name}_*/results.json"))
    if matches:
        print(f"SKIP {run_name}")
        continue
    jobs.append((method, mbits, run_name))

print(f"\n{len(jobs)} jobs to run, {N_GPUS} GPUs, {JOBS_PER_GPU} per GPU, "
      f"{N_GPUS * JOBS_PER_GPU} concurrent")
print(f"Batches needed: {(len(jobs) + N_GPUS * JOBS_PER_GPU - 1) // (N_GPUS * JOBS_PER_GPU)}")

t0 = time.time()
batch_size = N_GPUS * JOBS_PER_GPU

for batch_start in range(0, len(jobs), batch_size):
    batch = jobs[batch_start:batch_start + batch_size]
    batch_num = batch_start // batch_size + 1
    total_batches = (len(jobs) + batch_size - 1) // batch_size
    print(f"\n=== Batch {batch_num}/{total_batches} ({len(batch)} jobs) ===")

    procs = []
    for i, (method, mbits, run_name) in enumerate(batch):
        gpu_id = i % N_GPUS

        cmd = [
            sys.executable, "scripts/finetune.py",
            "--pretrained-model", PRETRAIN,
            "--method", method,
            "--batch-size", "64",
            "--lr", str(LR),
            "--epochs", str(EPOCHS),
            "--scheduler", "cosine_with_warmup",
            "--warmup-ratio", "0.1",
            "--weight-decay", "0.0",
            "--seed", str(SEED),
            "--checkpoint-every", str(CKPT_EVERY),
            "--mantissa-bits", str(mbits),
            "--exp-dir", OUTPUT_ROOT,
            "--wandb-name", run_name,
        ]
        if method == "grpo_kl":
            cmd += ["--kl-coef", "0.1"]

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        # Remove --device flag, let CUDA_VISIBLE_DEVICES handle GPU selection
        cmd = [c for c in cmd if c != f"cuda:{gpu_id}"]
        cmd = [c for c in cmd if c != "--device"]

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            env=env,
        )
        procs.append((run_name, proc))

    # Wait for all in batch
    for run_name, proc in procs:
        _, stderr = proc.communicate()
        if proc.returncode != 0:
            err_tail = stderr.decode()[-200:] if stderr else ""
            print(f"  FAILED {run_name}: {err_tail}")
        else:
            print(f"  OK {run_name}")

elapsed = time.time() - t0
print(f"\nTotal time: {elapsed:.0f}s ({elapsed/60:.1f}m)")
print(f"Results in: {OUTPUT_ROOT}/")
