#!/usr/bin/env python3
"""
Analyze validation results and find the best checkpoint per experiment.
Outputs a markdown log and a JSON with the best checkpoint paths.
"""

import json
import sys
from pathlib import Path

RESULTS_DIR = Path("precision_validation_results")
SPARSITY_DIR = Path("/mnt/nvme2n1/checkpoints/verify-exps-variable-seeds/sparsity")


def find_best(results: dict) -> tuple[str, float]:
    """Find checkpoint with highest accuracy."""
    best_key = max(results, key=lambda k: results[k]["accuracy"])
    return best_key, results[best_key]["accuracy"]


def resolve_full_path(exp_key: str, ckpt_name: str) -> str:
    """Resolve the full checkpoint path from experiment key and checkpoint name.

    The eval script prepends the parent directory name as a prefix
    (e.g., 'adamw_fp32_checkpoint-314820' for dir 'checkpoint-314820' under 'adamw_fp32/').
    We need to strip that prefix to get the actual directory name.
    """
    opt, method, prec = exp_key.split("_")  # e.g. adamw_grpo_fp32

    if method == "grpo":
        parent_dir = f"{opt}_{prec}"
        # Strip parent prefix: "adamw_fp32_checkpoint-314820" -> "checkpoint-314820"
        actual_name = ckpt_name.replace(f"{parent_dir}_", "", 1)
        return str(SPARSITY_DIR / parent_dir / actual_name)
    else:  # sft
        parent_dir = f"sft_{opt}_{prec}"
        # Strip parent prefix: "sft_adamw_fp32_tokens_1204593" -> need to find the real dir name
        # The hf_format dirs are like "samples_11098.0_tokens_1204593"
        # The eval key has format: sft_adamw_fp32_tokens_NNNN -> extract tokens_NNNN
        # and find matching dir
        hf_dir = SPARSITY_DIR / parent_dir / "hf_format"
        # Extract the token count from the checkpoint name
        if "tokens_" in ckpt_name:
            token_suffix = "tokens_" + ckpt_name.split("tokens_")[-1]
            # Find matching directory
            for d in hf_dir.iterdir():
                if d.is_dir() and d.name.endswith(token_suffix):
                    return str(d)
        # Fallback: try stripping parent prefix directly
        actual_name = ckpt_name.replace(f"{parent_dir}_", "", 1)
        return str(hf_dir / actual_name)


def main():
    if not RESULTS_DIR.exists():
        print(f"Results directory not found: {RESULTS_DIR}")
        sys.exit(1)

    best_checkpoints = {}
    lines = []
    lines.append("# Precision Experiment Validation Results\n")
    lines.append("## Best Checkpoint Per Experiment\n")
    lines.append("| Experiment | Best Checkpoint | Validation Accuracy | Full Path |")
    lines.append("|------------|----------------|--------------------:|-----------|")

    for json_file in sorted(RESULTS_DIR.glob("*_validation.json")):
        exp_key = json_file.stem.replace("_validation", "")

        with open(json_file) as f:
            results = json.load(f)

        best_name, best_acc = find_best(results)
        full_path = resolve_full_path(exp_key, best_name)

        best_checkpoints[exp_key] = {
            "checkpoint_name": best_name,
            "accuracy": best_acc,
            "full_path": full_path,
        }

        lines.append(f"| {exp_key} | {best_name} | {best_acc:.2%} | `{full_path}` |")

    lines.append("")
    lines.append("## All Checkpoint Scores\n")

    for json_file in sorted(RESULTS_DIR.glob("*_validation.json")):
        exp_key = json_file.stem.replace("_validation", "")

        with open(json_file) as f:
            results = json.load(f)

        lines.append(f"### {exp_key}\n")
        lines.append("| Checkpoint | Accuracy | Parsable |")
        lines.append("|------------|--------:|---------:|")

        for ckpt_name, data in sorted(results.items(), key=lambda x: -x[1]["accuracy"]):
            acc = data["accuracy"]
            parsable = data.get("parsable_rate", data.get("parsable", "N/A"))
            if isinstance(parsable, float):
                parsable = f"{parsable:.2%}"
            lines.append(f"| {ckpt_name} | {acc:.2%} | {parsable} |")

        lines.append("")

    # Save markdown log
    log_path = RESULTS_DIR / "validation_results_log.md"
    with open(log_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Saved validation log to: {log_path}")

    # Save best checkpoints JSON
    best_path = RESULTS_DIR / "best_checkpoints.json"
    with open(best_path, "w") as f:
        json.dump(best_checkpoints, f, indent=2)
    print(f"Saved best checkpoints to: {best_path}")

    # Print summary
    print("\nBest checkpoints:")
    for exp_key, info in sorted(best_checkpoints.items()):
        print(f"  {exp_key:25s} -> {info['checkpoint_name']:45s} acc={info['accuracy']:.2%}")


if __name__ == "__main__":
    main()
