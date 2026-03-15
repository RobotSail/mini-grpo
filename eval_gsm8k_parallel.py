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


def get_step_from_path(path: Path | str, include_parent: bool = False) -> tuple[int, str]:
    """Extract step/token number and label from checkpoint path.

    Handles formats:
    - step_N (e.g., step_121)
    - *_step_N (e.g., checkpoint_step_121)
    - tokens_N (e.g., tokens_100000)
    - *_tokens_N (e.g., checkpoint_tokens_100000)
    - checkpoint-N (e.g., checkpoint-150943) - token count

    Args:
        path: Path to checkpoint
        include_parent: If True, include parent directory name in label to avoid collisions
    """
    if isinstance(path, str):
        path = Path(path)

    name = path.name

    # Get parent prefix for disambiguation (e.g., "sft_adamw" from ".../sft_adamw/hf_format/...")
    parent_prefix = ""
    if include_parent:
        # Walk up to find a meaningful parent name (skip hf_format, checkpoints, etc.)
        for parent in path.parents:
            pname = parent.name
            if pname and pname not in ("hf_format", "checkpoints", ""):
                # Extract short identifier (e.g., "grpo_adamw" -> "grpo_adamw")
                parent_prefix = pname + "_"
                break

    # step-based checkpoints
    if name.startswith("step_"):
        step = int(name.split("_")[1])
        return step, f"{parent_prefix}step_{step}"
    elif "_step_" in name:
        step = int(name.split("_step_")[1])
        return step, f"{parent_prefix}step_{step}"
    # token-based checkpoints
    elif name.startswith("tokens_"):
        tokens = int(name.split("_")[1])
        return tokens, f"{parent_prefix}tokens_{tokens}"
    elif "_tokens_" in name:
        tokens = int(name.split("_tokens_")[1])
        return tokens, f"{parent_prefix}tokens_{tokens}"
    # checkpoint-N format (rejection sampling)
    elif name.startswith("checkpoint-"):
        suffix = name.split("-", 1)[1]
        if suffix.isdigit():
            tokens = int(suffix)
            return tokens, f"{parent_prefix}checkpoint-{tokens}"
        return -1, f"{parent_prefix}{name}"
    else:
        return 0, f"{parent_prefix}{name}"


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

    # Find checkpoint directories (step-based, token-based, or checkpoint-N format)
    checkpoint_dirs = []
    for d in base.iterdir():
        if not d.is_dir():
            continue
        name = d.name
        is_step_ckpt = name.startswith("step_") or "_step_" in name
        is_token_ckpt = name.startswith("tokens_") or "_tokens_" in name
        is_checkpoint_ckpt = name.startswith("checkpoint-")
        if is_step_ckpt or is_token_ckpt or is_checkpoint_ckpt:
            checkpoint_dirs.append(d)

    def get_sort_key(path: Path) -> int:
        """Extract numeric value for sorting (works for both step and token checkpoints)."""
        name = path.name
        if name.startswith("step_"):
            return int(name.split("_")[1])
        elif "_step_" in name:
            return int(name.split("_step_")[1])
        elif name.startswith("tokens_"):
            return int(name.split("_")[1])
        elif "_tokens_" in name:
            return int(name.split("_tokens_")[1])
        elif name.startswith("checkpoint-"):
            suffix = name.split("-", 1)[1]
            if suffix.isdigit():
                return int(suffix)
            return -1  # e.g. checkpoint-initial
        return 0

    checkpoint_dirs = sorted(checkpoint_dirs, key=get_sort_key)

    if not checkpoint_dirs:
        raise ValueError(f"No checkpoint directories found in {base_path}")

    return checkpoint_dirs


def run_single_eval(
    checkpoint_path: str, gpu_id: int, eval_path: str | None, system_msg: str, eval_kwargs: dict, output_file: str
) -> tuple[str, int, str, dict | None, str]:
    """
    Run evaluation for a single checkpoint in a subprocess.
    Returns (checkpoint_path, step, label, metrics, error_msg)
    """
    step, label = get_step_from_path(checkpoint_path, include_parent=True)

    # Build command to run eval_gsm8k.py
    cmd = [
        sys.executable,
        "eval_gsm8k.py",
        "--checkpoint-dir",
        checkpoint_path,
        "--gpu",
        str(gpu_id),
        "--output",
        output_file,
        "--max-new-tokens",
        str(eval_kwargs["max_new_tokens"]),
        "--temperature",
        str(eval_kwargs["temperature"]),
        "--top-k",
        str(eval_kwargs["top_k"]),
        "--top-p",
        str(eval_kwargs["top_p"]),
        "--repetition-penalty",
        str(eval_kwargs["repetition_penalty"]),
        "--group-size",
        str(eval_kwargs["group_size"]),
        "--system-msg",
        system_msg,
        "--sampling-dtype",
        eval_kwargs["sampling_dtype"],
    ]

    if eval_path:
        cmd.extend(["--eval-path", eval_path])

    # KL divergence options
    if eval_kwargs.get("compute_kl") or eval_kwargs.get("kl_only"):
        cmd.extend(["--base-model", eval_kwargs["base_model"]])
        cmd.extend(["--kl-batch-size", str(eval_kwargs["kl_batch_size"])])
        cmd.extend(["--kl-max-new-tokens", str(eval_kwargs["kl_max_new_tokens"])])
        cmd.extend(["--kl-max-prompt-length", str(eval_kwargs["kl_max_prompt_length"])])
        cmd.extend(["--kl-temperature", str(eval_kwargs["kl_temperature"])])
    if eval_kwargs.get("compute_kl"):
        cmd.append("--compute-kl")
    if eval_kwargs.get("kl_only"):
        cmd.append("--kl-only")
    if eval_kwargs.get("reverse_kl"):
        cmd.append("--reverse-kl")
    if eval_kwargs.get("kl_dataset"):
        cmd.extend(["--kl-dataset", eval_kwargs["kl_dataset"]])
    if eval_kwargs.get("calibration"):
        cmd.append("--calibration")

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
            # Build status message based on what was computed
            status_parts = []
            if "accuracy" in metrics:
                status_parts.append(f"Accuracy={metrics['accuracy']:.2%}")
            if "parsable_rate" in metrics:
                status_parts.append(f"Parsable={metrics['parsable_rate']:.2%}")
            if "kl_divergence" in metrics:
                status_parts.append(f"KL={metrics['kl_divergence']:.4f}")
            if "forward_kl" in metrics:
                status_parts.append(f"FwdKL={metrics['forward_kl']:.4f}")
            elif "reverse_kl" in metrics:
                status_parts.append(f"RevKL={metrics['reverse_kl']:.4f}")
            if "ece" in metrics:
                status_parts.append(f"ECE={metrics['ece']:.4f}")
            print(f"[GPU {gpu_id}] {label}: {', '.join(status_parts)}")
            return checkpoint_path, step, label, metrics, ""
        else:
            return checkpoint_path, step, label, None, "No results in output file"

    except Exception as e:
        error_msg = str(e)
        print(f"[GPU {gpu_id}] {label} ERROR: {error_msg}")
        return checkpoint_path, step, label, None, error_msg


def main():
    parser = argparse.ArgumentParser(description="Parallel GSM8K evaluation across multiple GPUs")
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="Path to checkpoint directory (single checkpoint or dir with step_* subdirs)",
    )
    parser.add_argument(
        "--checkpoints",
        type=str,
        default=None,
        help="Comma-separated list of specific checkpoint paths to evaluate",
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
        help="Comma-separated list of steps/tokens to evaluate (default: all)",
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
        default=0.0,
        help="Sampling temperature (default: 0.0 for greedy decoding)",
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
    parser.add_argument(
        "--compute-kl",
        action="store_true",
        help="Compute KL divergence from base model",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="Qwen/Qwen2-1.5B-Instruct",
        help="Base model for KL divergence computation",
    )
    parser.add_argument(
        "--kl-batch-size",
        type=int,
        default=4,
        help="Batch size for KL divergence computation",
    )
    parser.add_argument(
        "--kl-max-new-tokens",
        type=int,
        default=256,
        help="Max new tokens to generate for KL computation",
    )
    parser.add_argument(
        "--kl-max-prompt-length",
        type=int,
        default=256,
        help="Max prompt length for KL computation",
    )
    parser.add_argument(
        "--kl-only",
        action="store_true",
        help="Only compute KL divergence, skip accuracy evaluation",
    )
    parser.add_argument(
        "--reverse-kl",
        action="store_true",
        help="[Legacy name] Forward KL: generate from base (π₀), compute KL(π₀||π). Use --forward-kl.",
    )
    parser.add_argument(
        "--forward-kl",
        action="store_true",
        help="Forward KL: generate from base (π₀), compute KL(π₀||π). Measures drift from base.",
    )
    parser.add_argument(
        "--kl-dataset",
        type=str,
        default=None,
        help="Dataset for KL computation. Options: 'ifeval' (recommended for drift), "
             "'gsm8k', or path to jsonl file.",
    )
    parser.add_argument(
        "--kl-temperature",
        type=float,
        default=0.0,
        help="Sampling temperature for KL rollout generation (default: 0.0 for greedy)",
    )
    parser.add_argument(
        "--calibration",
        action="store_true",
        help="Compute ECE (Expected Calibration Error) using answer token confidence",
    )
    parser.add_argument(
        "--sampling-dtype",
        type=str,
        default="float16",
        help="vLLM dtype for generation/sampling (default: float16). "
             "Use 'auto' to infer from model weights, 'bfloat16' for native Qwen precision.",
    )

    args = parser.parse_args()

    # Validate arguments
    if not args.checkpoint_dir and not args.checkpoints:
        parser.error("Either --checkpoint-dir or --checkpoints must be specified")

    # Parse GPU IDs
    gpu_ids = [int(g.strip()) for g in args.gpus.split(",")]
    print(f"Using GPUs: {gpu_ids}")

    # Get checkpoints
    if args.checkpoints:
        # Use explicitly specified checkpoint paths
        checkpoint_dirs = [p.strip() for p in args.checkpoints.split(",")]
        print(f"Using {len(checkpoint_dirs)} specified checkpoint(s)")
    else:
        # Discover checkpoints from directory
        checkpoint_dirs = get_checkpoint_dirs(args.checkpoint_dir)
        print(f"Found {len(checkpoint_dirs)} checkpoint(s)")

        # Filter by steps if specified
        if args.steps:
            requested_steps = set(int(s.strip()) for s in args.steps.split(","))
            checkpoint_dirs = [d for d in checkpoint_dirs if get_step_from_path(d)[0] in requested_steps]
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
        "compute_kl": args.compute_kl,
        "kl_only": args.kl_only,
        "base_model": args.base_model,
        "kl_batch_size": args.kl_batch_size,
        "kl_max_new_tokens": args.kl_max_new_tokens,
        "kl_max_prompt_length": args.kl_max_prompt_length,
        "reverse_kl": args.reverse_kl or args.forward_kl,
        "kl_dataset": args.kl_dataset,
        "kl_temperature": args.kl_temperature,
        "calibration": args.calibration,
        "sampling_dtype": args.sampling_dtype,
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
            work_items.append(
                (
                    str(ckpt),
                    gpu_id,
                    args.eval_path,
                    args.system_msg,
                    eval_kwargs,
                    output_file,
                )
            )

        # Run evaluations in parallel using ThreadPoolExecutor
        # (threads are fine since actual work is in subprocesses)
        results = {}
        errors = []

        with ThreadPoolExecutor(max_workers=len(gpu_ids)) as executor:
            futures = {executor.submit(run_single_eval, *item): item[0] for item in work_items}

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

    # Build header based on what was computed
    has_accuracy = not args.kl_only
    has_kl = args.compute_kl or args.kl_only

    kl_header = "Fwd KL" if (args.reverse_kl or args.forward_kl) else "Rev KL"
    header = f"{'Checkpoint':<20}"
    if has_accuracy:
        header += f" {'Accuracy':>12} {'Parsable':>12}"
    if has_kl:
        header += f" {kl_header:>12}"
    print(header)
    print("-" * len(header))

    kl_metric_key = "forward_kl" if (args.reverse_kl or args.forward_kl) else "kl_divergence"
    for label, metrics in sorted(results.items(), key=lambda x: x[1]["step"]):
        row = f"{label:<20}"
        if has_accuracy:
            row += f" {metrics.get('accuracy', 0):>11.2%} {metrics.get('parsable_rate', 0):>11.2%}"
        if has_kl:
            row += f" {metrics.get(kl_metric_key, 0):>11.4f}"
        print(row)

    # Save results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
    elif args.checkpoint_dir:
        checkpoint_path = Path(args.checkpoint_dir)
        if checkpoint_path.exists() and checkpoint_path.is_dir():
            output_path = checkpoint_path / "eval_results.json"
        else:
            safe_name = args.checkpoint_dir.replace("/", "_")
            output_path = Path(f"eval_results_{safe_name}.json")
    else:
        # Using --checkpoints, default to current directory
        output_path = Path("eval_results_selected.json")

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
