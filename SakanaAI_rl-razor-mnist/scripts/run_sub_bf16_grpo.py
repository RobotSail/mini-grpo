#!/usr/bin/env python3
"""Batched GRPO runs with sub-bf16 mantissa widths (m1-m6) using quantize_update.

30 seeds × 6 mantissa widths = 180 jobs, batched across 8 GPUs.
"""
import os, sys, subprocess, time
from pathlib import Path

FINETUNE_SCRIPT = os.path.join(os.path.dirname(__file__), "finetune.py")
PRETRAIN = "experiments/mantissa_sweep/pretrain/pretrained_model.pt"
OUTPUT = "experiments/sakana_sub_bf16_grpo30"
N_GPUS = 8
PER_GPU = 10

MANTISSA = [(1, "m1"), (2, "m2"), (3, "m3"), (4, "m4"), (5, "m5"), (6, "m6")]
SEEDS = list(range(30))

os.makedirs(OUTPUT, exist_ok=True)

all_jobs = []
for mbits, mlabel in MANTISSA:
    for seed in SEEDS:
        run_name = f"grpo_{mlabel}_seed{seed}"
        matches = list(Path(OUTPUT).glob(f"{run_name}_*/results.json"))
        if matches:
            continue
        all_jobs.append((mbits, mlabel, seed, run_name))

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
    for i, (mbits, mlabel, seed, run_name) in enumerate(batch):
        gpu_id = i % N_GPUS
        cmd = [
            sys.executable, FINETUNE_SCRIPT,
            "--pretrained-model", PRETRAIN,
            "--method", "grpo",
            "--seed", str(seed),
            "--mantissa-bits", str(mbits),
            "--exp-dir", OUTPUT,
            "--wandb-name", run_name,
            "--checkpoint-every", "0.2",
            "--skip-alt-metrics",
        ]
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
print(f"\nDone: {total_ok} ok, {total_fail} failed in {elapsed:.0f}s ({elapsed/60:.1f}m)")
print(f"Results in: {OUTPUT}/")
