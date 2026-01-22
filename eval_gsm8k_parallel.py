#!/usr/bin/env python3
"""
Parallel GSM8K Evaluation Script

Evaluates multiple model checkpoints in parallel across multiple GPUs.
Each GPU runs a separate subprocess with its own vLLM instance.
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


# Default system message
DEFAULT_SYSTEM_MSG = "You are a helpful math assistant. Always provide your final numerical answer inside of the <answer>...</answer> tags, e.g.: <answer>42</answer>"


def get_step_from_path(path: Path | str) -> tuple[int, str]:
    """Extract step number and label from checkpoint path."""
    if isinstance(path, str):
        path = Path(path)

    name = path.name

    if name.startswith("step_"):
        step = int(name.split("_")[1])
        return step, f"step_{step}"
    elif "_step_" in name:
        step = int(name.split("_step_")[1])
        return step, f"step_{step}"
    else:
        return 0, name


def get_checkpoint_dirs(base_path: str) -> list[Path | str]:
    """Get all checkpoint directories sorted by step number."""
    base = Path(base_path)

    # Check if this is a HuggingFace model name
    if "/" in base_path and not base.exists():
        return [base_path]

    if not base.exists():
        raise ValueError(f"Checkpoint path does not exist: {base_path}")

    # Check if this is a direct model directory
    if (base / "config.json").exists():
        return [base]

    # Find checkpoint directories
    checkpoint_dirs = []
    for d in base.iterdir():
        if not d.is_dir():
            continue
        if d.name.startswith("step_") or "_step_" in d.name:
            checkpoint_dirs.append(d)

    def get_step_number(path: Path) -> int:
        name = path.name
        if name.startswith("step_"):
            return int(name.split("_")[1])
        elif "_step_" in name:
            return int(name.split("_step_")[1])
        return 0

    checkpoint_dirs = sorted(checkpoint_dirs, key=get_step_number)

    if not checkpoint_dirs:
        raise ValueError(f"No checkpoint directories found in {base_path}")

    return checkpoint_dirs


def run_single_eval(checkpoint_path: str, gpu_id: int, eval_path: str | None,
                    system_msg: str, eval_kwargs: dict, output_file: str) -> tuple[str, int, str, dict | None, str]:
    """
    Run evaluation for a single checkpoint in a subprocess.
    Returns (checkpoint_path, step, label, metrics, error_msg)
    """
    step, label = get_step_from_path(checkpoint_path)

    # Build command to run eval_gsm8k.py
    cmd = [
        sys.executable, "eval_gsm8k.py",
        "--checkpoint-dir", checkpoint_path,
        "--gpu", str(gpu_id),
        "--output", output_file,
        "--max-new-tokens", str(eval_kwargs["max_new_tokens"]),
        "--temperature", str(eval_kwargs["temperature"]),
        "--top-k", str(eval_kwargs["top_k"]),
        "--top-p", str(eval_kwargs["top_p"]),
        "--repetition-penalty", str(eval_kwargs["repetition_penalty"]),
        "--group-size", str(eval_kwargs["group_size"]),
        "--system-msg", system_msg,
    ]

    if eval_path:
        cmd.extend(["--eval-path", eval_path])

    print(f"[GPU {gpu_id}] Starting {label}: {checkpoint_path}")

    try:
        # Run subprocess
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent,
        )

        if result.returncode != 0:
            error_msg = f"Subprocess failed:\n{result.stderr[-2000:]}"
            print(f"[GPU {gpu_id}] {label} FAILED")
            return checkpoint_path, step, label, None, error_msg

        # Read results from output file
        with open(output_file) as f:
            results_data = json.load(f)

        # Get the single result (checkpoint was evaluated directly)
        if results_data:
            # The key might be the checkpoint name or step label
            metrics = list(results_data.values())[0]
            print(f"[GPU {gpu_id}] {label}: Accuracy={metrics['accuracy']:.2%}, Parsable={metrics['parsable_rate']:.2%}")
            return checkpoint_path, step, label, metrics, ""
        else:
            return checkpoint_path, step, label, None, "No results in output file"

    except Exception as e:
        error_msg = str(e)
        print(f"[GPU {gpu_id}] {label} ERROR: {error_msg}")
        return checkpoint_path, step, label, None, error_msg


def main():
    parser = argparse.ArgumentParser(
        description="Parallel GSM8K evaluation across multiple GPUs"
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        required=True,
        help="Path to checkpoint directory (single checkpoint or dir with step_* subdirs)",
    )
    parser.add_argument(
        "--gpus",
        type=str,
        required=True,
        help="Comma-separated list of GPU IDs (e.g., '0,1,2,3')",
    )
    parser.add_argument(
        "--eval-path",
        type=str,
        default=None,
        help="Path to evaluation data (jsonl). If not provided, loads GSM8K test split",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file for results",
    )
    parser.add_argument(
        "--steps",
        type=str,
        default=None,
        help="Comma-separated list of steps to evaluate (default: all)",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="Maximum new tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        "-t",
        type=float,
        default=0.7,
        help="Sampling temperature (default: 0.7)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=0,
        help="Top-k sampling (0 to disable, default: 0)",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=1.0,
        help="Top-p (nucleus) sampling (1.0 to disable, default: 1.0)",
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.0,
        help="Repetition penalty (1.0 to disable, default: 1.0)",
    )
    parser.add_argument(
        "--system-msg",
        type=str,
        default=DEFAULT_SYSTEM_MSG,
        help="System message for the chat template",
    )
    parser.add_argument(
        "--group-size",
        type=int,
        default=1,
        help="Number of samples per prompt (for pass@k evaluation)",
    )

    args = parser.parse_args()

    # Parse GPU IDs
    gpu_ids = [int(g.strip()) for g in args.gpus.split(",")]
    print(f"Using GPUs: {gpu_ids}")

    # Get checkpoints
    checkpoint_dirs = get_checkpoint_dirs(args.checkpoint_dir)
    print(f"Found {len(checkpoint_dirs)} checkpoint(s)")

    # Filter by steps if specified
    if args.steps:
        requested_steps = set(int(s.strip()) for s in args.steps.split(","))
        checkpoint_dirs = [
            d for d in checkpoint_dirs
            if get_step_from_path(d)[0] in requested_steps
        ]
        print(f"Filtered to {len(checkpoint_dirs)} checkpoint(s)")

    if not checkpoint_dirs:
        print("No checkpoints to evaluate!")
        return

    # Build evaluation kwargs
    eval_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "repetition_penalty": args.repetition_penalty,
        "group_size": args.group_size,
    }

    print(f"\nStarting parallel evaluation with {len(gpu_ids)} GPU(s)...")
    print(f"Total checkpoints: {len(checkpoint_dirs)}")
    print("=" * 60)

    # Create temp directory for intermediate results
    with tempfile.TemporaryDirectory() as tmpdir:
        # Build work items with round-robin GPU assignment
        work_items = []
        for i, ckpt in enumerate(checkpoint_dirs):
            gpu_id = gpu_ids[i % len(gpu_ids)]
            output_file = os.path.join(tmpdir, f"result_{i}.json")
            work_items.append((
                str(ckpt),
                gpu_id,
                args.eval_path,
                args.system_msg,
                eval_kwargs,
                output_file,
            ))

        # Run evaluations in parallel using ThreadPoolExecutor
        # (threads are fine since actual work is in subprocesses)
        results = {}
        errors = []

        with ThreadPoolExecutor(max_workers=len(gpu_ids)) as executor:
            futures = {
                executor.submit(run_single_eval, *item): item[0]
                for item in work_items
            }

            for future in as_completed(futures):
                checkpoint_path, step, label, metrics, error = future.result()
                if metrics:
                    results[label] = {
                        "step": step,
                        "path": checkpoint_path,
                        **metrics,
                    }
                else:
                    errors.append((label, error))

    # Print errors if any
    if errors:
        print(f"\n{'=' * 60}")
        print(f"ERRORS ({len(errors)} checkpoints failed)")
        print("=" * 60)
        for label, error in errors:
            print(f"{label}: {error[:200]}...")

    # Print summary
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    print(f"{'Checkpoint':<20} {'Accuracy':>12} {'Parsable':>12}")
    print("-" * 48)
    for label, metrics in sorted(results.items(), key=lambda x: x[1]["step"]):
        print(f"{label:<20} {metrics['accuracy']:>11.2%} {metrics['parsable_rate']:>11.2%}")

    # Save results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        checkpoint_path = Path(args.checkpoint_dir)
        if checkpoint_path.exists() and checkpoint_path.is_dir():
            output_path = checkpoint_path / "eval_results.json"
        else:
            safe_name = args.checkpoint_dir.replace("/", "_")
            output_path = Path(f"eval_results_{safe_name}.json")

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
