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
import torch
import torch.nn.functional as F
from tqdm import tqdm
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer, AutoModelForCausalLM


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

        # IMPORTANT: Strip assistant messages from eval data to avoid leaking answers
        # SFT training data includes assistant responses, but eval should only have
        # system + user messages for the model to generate a response
        def _strip_assistant_messages(sample):
            if "messages" in sample:
                # Keep only system and user messages
                filtered = [m for m in sample["messages"] if m["role"] in ("system", "user")]
                return {"messages": filtered}
            return {}

        dataset = dataset.map(_strip_assistant_messages)
        print(f"[INFO] Stripped assistant messages from eval data to prevent answer leakage")
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


def load_ifeval(
    system_msg: str = "You are a helpful assistant.",
) -> datasets.Dataset:
    """Load IFEval dataset for KL divergence measurement.

    IFEval (Instruction Following Eval) contains ~500 single-turn prompts
    that test instruction-following capabilities. Good for measuring
    drift on general instruction-following.
    """
    dataset = datasets.load_dataset("google/IFEval", split="train")

    def _format_sample(sample):
        return {
            "operation": "ifeval",
            "messages": [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": sample["prompt"]},
            ],
        }

    dataset = dataset.map(_format_sample)
    return dataset


def load_kl_dataset(
    dataset_name: str,
    system_msg: str = "You are a helpful assistant.",
) -> datasets.Dataset:
    """Load a dataset for KL divergence computation.

    Args:
        dataset_name: Either "ifeval", "gsm8k", or a path to a jsonl file
        system_msg: System message to use for chat formatting

    Returns:
        Dataset with 'messages' field for chat formatting
    """
    if dataset_name.lower() == "ifeval":
        return load_ifeval(system_msg=system_msg)
    elif dataset_name.lower() == "gsm8k":
        return load_gsm8k_eval(system_msg=system_msg)
    elif Path(dataset_name).exists():
        # Load from local jsonl file
        return datasets.load_dataset("json", data_files=dataset_name, split="train")
    else:
        raise ValueError(
            f"Unknown dataset: {dataset_name}. "
            "Use 'ifeval', 'gsm8k', or provide a path to a jsonl file."
        )


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


def compute_kl_divergence(
    base_model_path: str,
    checkpoint_path: str,
    eval_dataset: datasets.Dataset,
    batch_size: int = 8,
    max_new_tokens: int = 256,
    max_prompt_length: int = 256,
    reverse: bool = False,
) -> dict:
    """
    Compute KL divergence on generated rollouts.

    By default (reverse=False):
        Generate from checkpoint, compute KL(checkpoint || base)
        = E_{x ~ checkpoint}[log checkpoint(x) - log base(x)]
        Measures: "How different is checkpoint's behavior from base?"

    With reverse=True:
        Generate from base, compute KL(base || checkpoint)
        = E_{x ~ base}[log base(x) - log checkpoint(x)]
        Measures: "How much has checkpoint drifted from base behavior?"

    Args:
        base_model_path: Path to the reference/base model
        checkpoint_path: Path to the checkpoint model
        eval_dataset: Dataset with 'messages' field
        batch_size: Batch size for log prob computation
        max_new_tokens: Max tokens to generate per prompt
        max_prompt_length: Max prompt length (unused, vLLM handles this)
        reverse: If True, generate from base and compute KL(base || checkpoint)

    Returns:
        Dictionary with KL divergence metrics
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Determine which model generates rollouts
    generator_path = base_model_path if reverse else checkpoint_path

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(generator_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Prepare prompts
    prompts = []
    for sample in eval_dataset:
        prompt = tokenizer.apply_chat_template(
            conversation=sample["messages"],
            tokenize=False,
            add_generation_prompt=True,
        )
        prompts.append(prompt)

    # Step 1: Fast generation with vLLM
    gen_source = "base model" if reverse else "checkpoint"
    print(f"Generating rollouts from {gen_source} with vLLM ({len(prompts)} prompts)...")
    llm = LLM(
        model=generator_path,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.45,  # Leave room for HF models
        dtype="bfloat16",
    )

    sampling_params = SamplingParams(
        max_tokens=max_new_tokens,
        temperature=0.7,
        top_p=1.0,
    )

    outputs = llm.generate(prompts, sampling_params)

    # Collect generated sequences (prompt + completion)
    all_sequences = []
    all_prompt_lengths = []
    for i, output in enumerate(outputs):
        prompt_ids = output.prompt_token_ids
        generated_ids = output.outputs[0].token_ids
        full_seq = list(prompt_ids) + list(generated_ids)
        all_sequences.append(full_seq)
        all_prompt_lengths.append(len(prompt_ids))

    # Free vLLM memory
    del llm
    torch.cuda.empty_cache()

    # Step 2: Load both HF models for log prob computation
    print(f"Loading checkpoint model for log probs: {checkpoint_path}")
    checkpoint_model = AutoModelForCausalLM.from_pretrained(
        checkpoint_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    checkpoint_model.eval()

    print(f"Loading base model for log probs: {base_model_path}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    base_model.eval()

    # Step 3: Compute KL divergence on generated tokens
    kl_direction = "KL(base || checkpoint)" if reverse else "KL(checkpoint || base)"
    print(f"Computing {kl_direction}...")
    total_kl = 0.0
    total_tokens = 0
    all_seq_kl = []

    with torch.no_grad():
        for i in tqdm(range(0, len(all_sequences), batch_size), desc="KL divergence"):
            batch_seqs = all_sequences[i:i + batch_size]
            batch_prompt_lens = all_prompt_lengths[i:i + batch_size]

            # Pad sequences to same length
            max_len = max(len(seq) for seq in batch_seqs)
            padded_ids = []
            attention_masks = []
            for seq in batch_seqs:
                pad_len = max_len - len(seq)
                padded = seq + [tokenizer.pad_token_id] * pad_len
                mask = [1] * len(seq) + [0] * pad_len
                padded_ids.append(padded)
                attention_masks.append(mask)

            input_ids = torch.tensor(padded_ids, device=device)
            attention_mask = torch.tensor(attention_masks, device=device)

            # Get logits from both models
            checkpoint_outputs = checkpoint_model(input_ids=input_ids, attention_mask=attention_mask)
            base_outputs = base_model(input_ids=input_ids, attention_mask=attention_mask)

            # Get log probabilities for the actual next tokens
            checkpoint_logprobs = F.log_softmax(checkpoint_outputs.logits[:, :-1], dim=-1)
            base_logprobs = F.log_softmax(base_outputs.logits[:, :-1], dim=-1)

            # Get log prob of actual tokens
            target_ids = input_ids[:, 1:]

            # Gather log probs for actual tokens
            checkpoint_token_logprobs = checkpoint_logprobs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
            base_token_logprobs = base_logprobs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)

            # KL per token depends on direction
            if reverse:
                # KL(base || checkpoint) = log base(x) - log checkpoint(x)
                kl_per_token = base_token_logprobs - checkpoint_token_logprobs
            else:
                # KL(checkpoint || base) = log checkpoint(x) - log base(x)
                kl_per_token = checkpoint_token_logprobs - base_token_logprobs

            # Accumulate KL for generated tokens only (not prompt)
            for j, prompt_len in enumerate(batch_prompt_lens):
                seq_len = sum(attention_masks[j])
                gen_start = prompt_len - 1  # -1 because we shifted
                gen_end = seq_len - 1

                if gen_end > gen_start:
                    seq_kl = kl_per_token[j, gen_start:gen_end]
                    seq_kl_mean = seq_kl.mean().item()
                    all_seq_kl.append(seq_kl_mean)
                    total_kl += seq_kl.sum().item()
                    total_tokens += (gen_end - gen_start)

    # Cleanup
    del base_model, checkpoint_model
    torch.cuda.empty_cache()

    mean_kl = total_kl / total_tokens if total_tokens > 0 else 0.0
    std_kl = torch.tensor(all_seq_kl).std().item() if all_seq_kl else 0.0

    kl_key = "reverse_kl" if reverse else "kl_divergence"
    return {
        kl_key: mean_kl,
        f"{kl_key}_std": std_kl,
        "total_tokens": int(total_tokens),
        "num_sequences": len(prompts),
        "avg_generated_tokens": total_tokens / len(prompts) if prompts else 0,
        "kl_direction": kl_direction,
        "generator": "base" if reverse else "checkpoint",
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
    parser.add_argument(
        "--compute-kl",
        action="store_true",
        help="Compute KL divergence from base model (requires --base-model)",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="Qwen/Qwen2-1.5B-Instruct",
        help="Base model for KL divergence computation (default: Qwen/Qwen2-1.5B-Instruct)",
    )
    parser.add_argument(
        "--kl-batch-size",
        type=int,
        default=4,
        help="Batch size for KL divergence computation (default: 4)",
    )
    parser.add_argument(
        "--kl-max-new-tokens",
        type=int,
        default=256,
        help="Max new tokens to generate for KL computation (default: 256)",
    )
    parser.add_argument(
        "--kl-max-prompt-length",
        type=int,
        default=256,
        help="Max prompt length for KL computation (default: 256)",
    )
    parser.add_argument(
        "--kl-only",
        action="store_true",
        help="Only compute KL divergence, skip accuracy evaluation",
    )
    parser.add_argument(
        "--reverse-kl",
        action="store_true",
        help="Compute reverse KL: generate from base, compute KL(base || checkpoint). "
             "Measures drift from base model behavior.",
    )
    parser.add_argument(
        "--kl-dataset",
        type=str,
        default=None,
        help="Dataset for KL computation. Options: 'ifeval' (recommended for drift), "
             "'gsm8k', or path to jsonl file. If not provided, uses eval dataset.",
    )

    args = parser.parse_args()

    # Load evaluation dataset
    print("Loading evaluation dataset...")
    eval_dataset = load_gsm8k_eval(args.eval_path, system_msg=args.system_msg)
    print(f"Loaded {len(eval_dataset)} evaluation samples")

    # Load KL dataset if specified (for held-out distribution)
    if args.kl_dataset:
        print(f"Loading KL dataset: {args.kl_dataset}")
        kl_dataset = load_kl_dataset(args.kl_dataset, system_msg=args.system_msg)
        print(f"Loaded {len(kl_dataset)} KL evaluation samples")
    else:
        kl_dataset = eval_dataset

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

        metrics = {}

        # Compute accuracy (unless --kl-only)
        if not args.kl_only:
            accuracy_metrics = evaluate_checkpoint(
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
            metrics.update(accuracy_metrics)
            print(f"\nAccuracy Results for {label}:")
            print(f"  Accuracy: {metrics['accuracy']:.2%} ({metrics['correct']}/{metrics['total_samples']})")
            print(f"  Parsable: {metrics['parsable_rate']:.2%} ({metrics['parsable']}/{metrics['total_samples']})")

        # Compute KL divergence (if requested)
        if args.compute_kl or args.kl_only:
            kl_metrics = compute_kl_divergence(
                base_model_path=args.base_model,
                checkpoint_path=str(checkpoint_dir),
                eval_dataset=kl_dataset,
                batch_size=args.kl_batch_size,
                max_new_tokens=args.kl_max_new_tokens,
                max_prompt_length=args.kl_max_prompt_length,
                reverse=args.reverse_kl,
            )
            metrics.update(kl_metrics)
            print(f"\nKL Divergence Results for {label}:")
            kl_key = "reverse_kl" if args.reverse_kl else "kl_divergence"
            kl_std_key = f"{kl_key}_std"
            kl_direction = kl_metrics.get("kl_direction", "KL")
            print(f"  {kl_direction}: {metrics[kl_key]:.4f} (± {metrics[kl_std_key]:.4f})")
            print(f"  Generator: {kl_metrics.get('generator', 'unknown')}")
            print(f"  Generated tokens: {metrics['total_tokens']} ({metrics['avg_generated_tokens']:.1f} avg/seq)")

        results[label] = {
            "step": step,
            "path": str(checkpoint_dir),
            **metrics,
        }

    # Print summary
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")

    # Build header based on what was computed
    has_accuracy = not args.kl_only
    has_kl = args.compute_kl or args.kl_only

    kl_header = "Rev KL" if args.reverse_kl else "KL Div"
    header = f"{'Checkpoint':<20}"
    if has_accuracy:
        header += f" {'Accuracy':>12} {'Parsable':>12}"
    if has_kl:
        header += f" {kl_header:>12}"
    print(header)
    print("-" * len(header))

    kl_metric_key = "reverse_kl" if args.reverse_kl else "kl_divergence"
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
