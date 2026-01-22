#!/usr/bin/env python3
"""
GSM8K Evaluation Script

Evaluates model checkpoints on the GSM8K dataset and reports accuracy metrics.
Uses vLLM for fast batched inference.
"""

import argparse
import os
import sys

# Parse --gpu argument early to set CUDA_VISIBLE_DEVICES before any CUDA initialization
def _get_gpu_arg():
    for i, arg in enumerate(sys.argv):
        if arg == "--gpu" and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
        if arg.startswith("--gpu="):
            return arg.split("=")[1]
    return "0"

os.environ["CUDA_VISIBLE_DEVICES"] = _get_gpu_arg()

import json
import re
from pathlib import Path

import datasets
from tqdm import tqdm
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer


# Regex pattern to match <answer>...</answer> tags
answer_pattern = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)


def parse_number(text: str) -> float:
    """
    Parse a string into a float, handling common formats from GSM8K answers.
    """
    if not text or not isinstance(text, str):
        raise ValueError(f"Empty or invalid input: {text}")

    text = text.strip()
    text = re.sub(r"[$\u20AC\u00A3\u00A5\u20B9]", "", text)
    text = text.replace("%", "")
    text = text.replace(",", "")
    text = text.strip()

    if not any(c.isdigit() for c in text):
        raise ValueError(f"No digits found in answer: {text}")

    match = re.search(r"-?\d+\.?\d*", text)
    if not match:
        raise ValueError(f"Could not extract number from: {text}")

    return float(match.group())


def get_checkpoint_dirs(base_path: str) -> list[Path | str]:
    """Get all checkpoint directories sorted by step number.

    Also accepts HuggingFace model names (e.g., 'Qwen/Qwen2-1.5B-Instruct').
    """
    base = Path(base_path)

    # Check if this is a HuggingFace model name (contains / but doesn't exist locally)
    if "/" in base_path and not base.exists():
        # Return as string (HuggingFace model name)
        return [base_path]

    if not base.exists():
        raise ValueError(f"Checkpoint path does not exist: {base_path}")

    # Check if this is a direct model directory (has config.json)
    if (base / "config.json").exists():
        return [base]

    # Find step_* directories (also handles samples_X_step_Y format from SFT)
    checkpoint_dirs = []
    for d in base.iterdir():
        if not d.is_dir():
            continue
        if d.name.startswith("step_"):
            checkpoint_dirs.append(d)
        elif "_step_" in d.name:
            # Handle samples_X_step_Y format
            checkpoint_dirs.append(d)

    def get_step_number(path: Path) -> int:
        name = path.name
        if name.startswith("step_"):
            return int(name.split("_")[1])
        elif "_step_" in name:
            # Extract step number from samples_X_step_Y format
            return int(name.split("_step_")[1])
        return 0

    checkpoint_dirs = sorted(checkpoint_dirs, key=get_step_number)

    if not checkpoint_dirs:
        raise ValueError(f"No checkpoint directories found in {base_path}")

    return checkpoint_dirs


def load_gsm8k_eval(
    data_path: str | None = None,
    system_msg: str = "You are a helpful math assistant. Always provide your final numerical answer inside of the <answer>...</answer> tags, e.g.: <answer>42</answer>",
) -> datasets.Dataset:
    """Load GSM8K evaluation dataset."""
    if data_path:
        # Load from local file
        dataset = datasets.load_dataset("json", data_files=data_path, split="train")
    else:
        # Load from HuggingFace
        dataset = datasets.load_dataset("openai/gsm8k", name="main", split="test")
        dataset = dataset.rename_columns({"question": "problem"})

        def _get_answers(sample):
            answers = re.findall(r"<<(.+?)>>", sample["answer"])
            alt_matches = re.findall(r"#### (.+)", sample["answer"])

            if answers:
                answer = answers[-1].split("=")[-1]
            elif alt_matches:
                answer = alt_matches[-1]
            else:
                raise ValueError(f"Failed to find answer in: {sample['answer']}")

            return {"answer": float(answer.replace(",", ""))}

        dataset = dataset.map(_get_answers)

        def _add_fields(sample):
            return {
                "operation": "gsm8k",
                "messages": [
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": sample["problem"]},
                ],
            }

        dataset = dataset.map(_add_fields)

    return dataset


def evaluate_checkpoint(
    model_path: str | Path,
    eval_dataset: datasets.Dataset,
    gpu: int,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_k: int = 0,
    top_p: float = 1.0,
    repetition_penalty: float = 1.0,
    group_size: int = 1,
) -> dict:
    """Evaluate a single checkpoint on the dataset using vLLM."""
    model_path = str(model_path)

    # Load tokenizer for chat template
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Initialize vLLM
    llm = LLM(
        model=model_path,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.9,
        dtype="bfloat16",
    )

    # Setup sampling params
    # vLLM uses top_k=-1 to disable, we use 0 as input convention
    sampling_params = SamplingParams(
        max_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k if top_k > 0 else -1,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        n=group_size,
    )

    # Prepare all prompts
    prompts = []
    for sample in eval_dataset:
        prompt = tokenizer.apply_chat_template(
            conversation=sample["messages"],
            tokenize=False,
            add_generation_prompt=True,
        )
        prompts.append(prompt)

    # Generate all responses in batch
    print(f"Generating {len(prompts)} responses with vLLM...")
    outputs = llm.generate(prompts, sampling_params)

    # Evaluate responses
    total_correct = 0
    total_parsable = 0
    total_samples = 0

    for idx, output in enumerate(tqdm(outputs, desc="Scoring")):
        sample = eval_dataset[idx]
        expected = float(sample["answer"])

        any_correct = False
        any_parsable = False

        for completion in output.outputs:
            response = completion.text
            matches = answer_pattern.findall(response)

            if matches:
                last_match = matches[-1]
                try:
                    parsed_answer = parse_number(last_match)
                    any_parsable = True

                    if abs(parsed_answer - expected) < 1e-6:
                        any_correct = True
                        break
                except ValueError:
                    pass

        if any_correct:
            total_correct += 1
        if any_parsable:
            total_parsable += 1
        total_samples += 1

    # Cleanup
    del llm

    return {
        "total_samples": total_samples,
        "correct": total_correct,
        "parsable": total_parsable,
        "accuracy": total_correct / total_samples if total_samples > 0 else 0.0,
        "parsable_rate": total_parsable / total_samples if total_samples > 0 else 0.0,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate model checkpoints on GSM8K using vLLM")
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        required=True,
        help="Path to checkpoint directory (single checkpoint or dir with step_* subdirs)",
    )
    parser.add_argument(
        "--eval-path",
        type=str,
        default=None,
        help="Path to evaluation data (jsonl). If not provided, loads GSM8K test split from HuggingFace",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file for results",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="GPU index to use",
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
        default="You are a helpful math assistant. Always provide your final numerical answer inside of the <answer>...</answer> tags, e.g.: <answer>42</answer>",
        help="System message for the chat template",
    )
    parser.add_argument(
        "--group-size",
        type=int,
        default=1,
        help="Number of samples per prompt (for pass@k evaluation)",
    )
    parser.add_argument(
        "--steps",
        type=str,
        default=None,
        help="Comma-separated list of steps to evaluate (default: all)",
    )

    args = parser.parse_args()

    # Load evaluation dataset
    print("Loading evaluation dataset...")
    eval_dataset = load_gsm8k_eval(args.eval_path, system_msg=args.system_msg)
    print(f"Loaded {len(eval_dataset)} evaluation samples")

    # Get checkpoints to evaluate
    checkpoint_dirs = get_checkpoint_dirs(args.checkpoint_dir)
    print(f"Found {len(checkpoint_dirs)} checkpoint(s) to evaluate")

    # Filter by steps if specified
    if args.steps:
        requested_steps = set(int(s.strip()) for s in args.steps.split(","))

        def get_step_from_dir(d):
            if isinstance(d, str):
                return -1
            if d.name.startswith("step_"):
                return int(d.name.split("_")[1])
            elif "_step_" in d.name:
                return int(d.name.split("_step_")[1])
            return 0

        checkpoint_dirs = [
            d for d in checkpoint_dirs
            if get_step_from_dir(d) in requested_steps
        ]
        print(f"Filtered to {len(checkpoint_dirs)} checkpoint(s)")

    # Evaluate each checkpoint
    results = {}
    for checkpoint_dir in checkpoint_dirs:
        # Handle both Path objects and HuggingFace model name strings
        if isinstance(checkpoint_dir, str):
            # HuggingFace model name
            step = -1  # Use -1 to indicate base model
            label = checkpoint_dir.replace("/", "_")
        elif checkpoint_dir.name.startswith("step_"):
            step = int(checkpoint_dir.name.split("_")[1])
            label = f"step_{step}"
        elif "_step_" in checkpoint_dir.name:
            # Handle samples_X_step_Y format
            step = int(checkpoint_dir.name.split("_step_")[1])
            label = f"step_{step}"
        else:
            step = 0
            label = checkpoint_dir.name

        print(f"\n{'=' * 60}")
        print(f"Evaluating: {checkpoint_dir}")
        print(f"{'=' * 60}")

        metrics = evaluate_checkpoint(
            model_path=checkpoint_dir,
            eval_dataset=eval_dataset,
            gpu=args.gpu,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            group_size=args.group_size,
        )

        results[label] = {
            "step": step,
            "path": str(checkpoint_dir),
            **metrics,
        }

        print(f"\nResults for {label}:")
        print(f"  Accuracy: {metrics['accuracy']:.2%} ({metrics['correct']}/{metrics['total_samples']})")
        print(f"  Parsable: {metrics['parsable_rate']:.2%} ({metrics['parsable']}/{metrics['total_samples']})")

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
        # Default output location - use checkpoint dir if local, else current directory
        checkpoint_path = Path(args.checkpoint_dir)
        if checkpoint_path.exists() and checkpoint_path.is_dir():
            output_path = checkpoint_path / "eval_results.json"
        else:
            # HuggingFace model or non-existent path - save to current directory
            safe_name = args.checkpoint_dir.replace("/", "_")
            output_path = Path(f"eval_results_{safe_name}.json")

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
