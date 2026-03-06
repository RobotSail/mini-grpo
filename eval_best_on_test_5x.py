#!/usr/bin/env python3
"""
Run the best checkpoint from each experiment on the GSM8K test set 5 times.
Reports mean and std of accuracy across runs.
"""

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np


def run_single_eval(checkpoint_path: str, gpu_id: int, run_id: int) -> dict | None:
    """Run a single evaluation and return results."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        out_path = f.name

    cmd = [
        sys.executable, "eval_gsm8k_parallel.py",
        "--checkpoints", checkpoint_path,
        "--gpus", str(gpu_id),
        "--output", out_path,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  [Run {run_id}] FAILED: {result.stderr[-300:]}")
        return None

    with open(out_path) as f:
        data = json.load(f)

    Path(out_path).unlink(missing_ok=True)
    return data


def main():
    parser = argparse.ArgumentParser(description="Evaluate best checkpoints on test set 5 times")
    parser.add_argument("--best-checkpoints", type=str, default="precision_validation_results/best_checkpoints.json")
    parser.add_argument("--gpus", type=str, default="0,1,2,3,4,5,6,7")
    parser.add_argument("--n-runs", type=int, default=5)
    parser.add_argument("--output-dir", type=str, default="precision_test_results")

    args = parser.parse_args()

    with open(args.best_checkpoints) as f:
        best_checkpoints = json.load(f)

    gpu_ids = [int(g.strip()) for g in args.gpus.split(",")]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}

    def eval_experiment(exp_key, info, gpu_id):
        """Run all n_runs for one experiment on one GPU."""
        ckpt_path = info["full_path"]
        print(f"[GPU {gpu_id}] Starting {exp_key}: {Path(ckpt_path).name}")

        accuracies = []
        for run in range(args.n_runs):
            data = run_single_eval(ckpt_path, gpu_id, run + 1)
            if data:
                ckpt_name = list(data.keys())[0]
                acc = data[ckpt_name]["accuracy"]
                accuracies.append(acc)
                print(f"  [GPU {gpu_id}] {exp_key} run {run+1}/{args.n_runs}: {acc:.2%}")

        if accuracies:
            mean_acc = np.mean(accuracies)
            std_acc = np.std(accuracies)
            result = {
                "checkpoint_path": ckpt_path,
                "validation_accuracy": info["accuracy"],
                "test_accuracies": accuracies,
                "test_mean": float(mean_acc),
                "test_std": float(std_acc),
                "n_runs": len(accuracies),
            }
            print(f"  [GPU {gpu_id}] {exp_key} => {mean_acc:.2%} ± {std_acc:.2%}")
            return exp_key, result
        return exp_key, None

    from concurrent.futures import ThreadPoolExecutor, as_completed

    sorted_exps = sorted(best_checkpoints.items())
    with ThreadPoolExecutor(max_workers=len(gpu_ids)) as executor:
        futures = {}
        for i, (exp_key, info) in enumerate(sorted_exps):
            gpu_id = gpu_ids[i % len(gpu_ids)]
            future = executor.submit(eval_experiment, exp_key, info, gpu_id)
            futures[future] = exp_key

        for future in as_completed(futures):
            exp_key, result = future.result()
            if result:
                all_results[exp_key] = result

    # Save results
    results_path = output_dir / "test_5x_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)

    # Save markdown log
    log_lines = ["# Test Set Results (5 runs per checkpoint)\n"]
    log_lines.append("| Experiment | Checkpoint | Val Acc | Test Mean | Test Std | Runs |")
    log_lines.append("|------------|-----------|--------:|----------:|---------:|-----:|")

    for exp_key, r in sorted(all_results.items()):
        ckpt_name = Path(r["checkpoint_path"]).name
        log_lines.append(
            f"| {exp_key} | {ckpt_name} | {r['validation_accuracy']:.2%} | "
            f"{r['test_mean']:.2%} | {r['test_std']:.2%} | {r['n_runs']} |"
        )

    log_lines.append("")
    log_lines.append("## Individual Run Scores\n")
    for exp_key, r in sorted(all_results.items()):
        log_lines.append(f"### {exp_key}")
        log_lines.append(f"- Checkpoint: `{r['checkpoint_path']}`")
        log_lines.append(f"- Validation: {r['validation_accuracy']:.2%}")
        log_lines.append(f"- Test runs: {', '.join(f'{a:.2%}' for a in r['test_accuracies'])}")
        log_lines.append(f"- **Mean: {r['test_mean']:.2%} ± {r['test_std']:.2%}**")
        log_lines.append("")

    log_path = output_dir / "test_5x_results_log.md"
    with open(log_path, "w") as f:
        f.write("\n".join(log_lines))

    print(f"\n{'='*60}")
    print(f"Results saved to: {results_path}")
    print(f"Log saved to: {log_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
