#!/usr/bin/env python3
"""
Batched launcher for SakanaAI finetune.py.
Runs the OFFICIAL script with all default settings, only varying method/mantissa/seed.
Batches across GPUs using CUDA_VISIBLE_DEVICES.
"""
import os, sys, subprocess, time, json, argparse
from pathlib import Path
from itertools import product

FINETUNE_SCRIPT = os.path.join(os.path.dirname(__file__), "finetune.py")
PRETRAIN = "experiments/mantissa_sweep/pretrain/pretrained_model.pt"

def run_batch(jobs, n_gpus, output_root):
    """Run a batch of jobs, distributing across GPUs."""
    procs = []
    for i, (method, mbits, mlabel, seed, run_name) in enumerate(jobs):
        gpu_id = i % n_gpus

        cmd = [
            sys.executable, FINETUNE_SCRIPT,
            "--pretrained-model", PRETRAIN,
            "--method", method,
            "--seed", str(seed),
            "--mantissa-bits", str(mbits),
            "--exp-dir", output_root,
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

    # Wait for all
    ok = 0; fail = 0
    for run_name, proc in procs:
        _, stderr = proc.communicate()
        if proc.returncode != 0:
            err = stderr.decode()[-200:] if stderr else ""
            print(f"  FAILED {run_name}: {err}")
            fail += 1
        else:
            ok += 1
    return ok, fail


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=["grpo30", "all5"])
    parser.add_argument("--output", required=True)
    parser.add_argument("--gpus", type=int, default=8)
    parser.add_argument("--per-gpu", type=int, default=10)
    args = parser.parse_args()

    MANTISSA = [(7, "m7"), (8, "m8"), (9, "m9"), (10, "m10"), (23, "fp32")]

    if args.mode == "grpo30":
        methods_seeds = [("grpo", list(range(30)))]
    else:
        methods_seeds = [
            ("grpo", list(range(30))),
            ("grpo_kl", [42, 123, 456, 789, 1024]),
            ("sft1", [42, 123, 456, 789, 1024]),
            ("sft2", [42, 123, 456, 789, 1024]),
            ("oracle", [42, 123, 456, 789, 1024]),
        ]

    # Build jobs, skip existing
    all_jobs = []
    for method, seeds in methods_seeds:
        for mbits, mlabel in MANTISSA:
            for seed in seeds:
                run_name = f"{method}_{mlabel}_seed{seed}"
                # Check if results exist
                matches = list(Path(args.output).glob(f"{run_name}_*/results.json"))
                if matches:
                    continue
                all_jobs.append((method, mbits, mlabel, seed, run_name))

    batch_size = args.gpus * args.per_gpu
    total_batches = (len(all_jobs) + batch_size - 1) // batch_size

    print(f"{len(all_jobs)} jobs to run ({batch_size} per batch, {total_batches} batches)")
    t0 = time.time()

    total_ok = total_fail = 0
    for batch_start in range(0, len(all_jobs), batch_size):
        batch = all_jobs[batch_start:batch_start + batch_size]
        batch_num = batch_start // batch_size + 1
        print(f"\nBatch {batch_num}/{total_batches} ({len(batch)} jobs)...")
        ok, fail = run_batch(batch, args.gpus, args.output)
        total_ok += ok; total_fail += fail
        print(f"  {ok} ok, {fail} failed")

    elapsed = time.time() - t0
    print(f"\nDone: {total_ok} ok, {total_fail} failed in {elapsed:.0f}s ({elapsed/60:.1f}m)")


if __name__ == "__main__":
    main()
