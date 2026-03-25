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



def extract_answer_confidence(completion) -> float | None:
    """Extract confidence for the answer tokens from a vLLM completion.

    Confidence = geometric mean of token probabilities for tokens inside
    the last <answer>...</answer> span.

    Returns None if no answer tags found or no logprobs available.
    """
    if not completion.logprobs:
        return None

    text = completion.text
    # Find the last <answer>...</answer> span by character position
    matches = list(re.finditer(r"<answer>(.*?)</answer>", text, re.DOTALL | re.IGNORECASE))
    if not matches:
        return None

    last_match = matches[-1]
    # Only the content between tags, not the tags themselves
    content_start_char = last_match.start(1)  # start of group 1 (content after <answer>)
    content_end_char = last_match.end(1)      # end of group 1 (content before </answer>)

    # Map character positions to token indices.
    # completion.logprobs is a list of dicts, one per generated token.
    # We reconstruct character offsets by walking through token texts.
    char_offset = 0
    answer_logprobs = []

    for token_logprob_dict in completion.logprobs:
        # Each entry is a dict {token_id: Logprob} for the sampled token
        if not token_logprob_dict:
            char_offset += 1  # approximate
            continue

        # Get the sampled token's logprob (first entry is the sampled token)
        sampled = next(iter(token_logprob_dict.values()))
        token_text = sampled.decoded_token if hasattr(sampled, 'decoded_token') else ""
        token_len = len(token_text)
        token_end = char_offset + token_len

        # Check if this token is fully within the answer content (between tags)
        if char_offset >= content_start_char and token_end <= content_end_char:
            answer_logprobs.append(sampled.logprob)

        char_offset = token_end

    if not answer_logprobs:
        return None

    # Geometric mean of token probabilities = exp(mean(log_probs))
    import math
    mean_logprob = sum(answer_logprobs) / len(answer_logprobs)
    return math.exp(mean_logprob)


def compute_ece(confidences: list[float], correctness: list[bool], n_bins: int = 10) -> dict:
    """Compute calibration metrics: ECE, MCE, Brier score, and NLL.

    ECE = Σ |B_m|/n × |acc(B_m) - conf(B_m)|
    MCE = max_m |acc(B_m) - conf(B_m)|
    Brier = (1/n) Σ (c_i - z_i)²
    NLL = -(1/n) Σ [z_i log(c_i) + (1-z_i) log(1-c_i)]

    Args:
        confidences: Per-sample confidence scores in [0, 1]
        correctness: Per-sample correctness (True/False)
        n_bins: Number of equal-width bins

    Returns:
        Dict with ece, mce, brier, nll, mean_confidence, and per_bin_data.
    """
    import numpy as np

    confidences = np.array(confidences)
    correctness = np.array(correctness, dtype=float)
    n = len(confidences)

    if n == 0:
        return {"ece": 0.0, "mce": 0.0, "brier": 0.0, "nll": 0.0,
                "mean_confidence": 0.0, "per_bin_data": []}

    # Brier score: mean squared error between confidence and correctness
    brier = float(np.mean((confidences - correctness) ** 2))

    # NLL (log loss): clip to avoid log(0)
    eps = 1e-12
    c_clipped = np.clip(confidences, eps, 1 - eps)
    nll = float(-np.mean(
        correctness * np.log(c_clipped) + (1 - correctness) * np.log(1 - c_clipped)
    ))

    # Binned metrics: ECE and MCE
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    mce = 0.0
    per_bin_data = []

    for m in range(n_bins):
        lo, hi = bin_edges[m], bin_edges[m + 1]
        mask = (confidences > lo) & (confidences <= hi) if m > 0 else (confidences >= lo) & (confidences <= hi)
        bin_size = mask.sum()

        if bin_size == 0:
            per_bin_data.append({
                "bin_lo": float(lo), "bin_hi": float(hi),
                "count": 0, "accuracy": 0.0, "avg_confidence": 0.0,
            })
            continue

        bin_acc = correctness[mask].mean()
        bin_conf = confidences[mask].mean()
        bin_gap = abs(bin_acc - bin_conf)
        ece += (bin_size / n) * bin_gap
        mce = max(mce, bin_gap)

        per_bin_data.append({
            "bin_lo": float(lo), "bin_hi": float(hi),
            "count": int(bin_size),
            "accuracy": float(bin_acc),
            "avg_confidence": float(bin_conf),
        })

    return {
        "ece": float(ece),
        "mce": float(mce),
        "brier": brier,
        "nll": nll,
        "mean_confidence": float(confidences.mean()),
        "n_calibration_samples": n,
        "per_bin_data": per_bin_data,
    }


def compute_answer_confidences_hf(
    model_path: str,
    tokenizer,
    prompts: list[str],
    responses: list[str],
    batch_size: int = 8,
) -> list[float | None]:
    """Compute answer token confidence via HuggingFace forward pass in FP32.

    For each (prompt, response) pair:
    1. Tokenize prompt + response
    2. Forward pass to get logits
    3. Extract logprobs of tokens corresponding to answer content (between <answer> tags)
    4. Confidence = exp(mean(logprobs))

    Returns list of confidence values (None if no answer found).
    """
    import math

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading model for calibration (FP32 logprobs): {model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float32,
        device_map=device,
    )
    model.eval()

    confidences = [None] * len(prompts)

    with torch.no_grad():
        for i in tqdm(range(0, len(prompts), batch_size), desc="Calibration (FP32)"):
            batch_prompts = prompts[i:i + batch_size]
            batch_responses = responses[i:i + batch_size]

            for j, (prompt, response) in enumerate(zip(batch_prompts, batch_responses)):
                idx = i + j

                # Find answer content in response
                matches = list(re.finditer(
                    r"<answer>(.*?)</answer>", response, re.DOTALL | re.IGNORECASE
                ))
                if not matches:
                    continue

                last_match = matches[-1]
                answer_content = last_match.group(1)
                if not answer_content.strip():
                    continue

                # Tokenize full sequence (prompt + response)
                full_text = prompt + response
                full_ids = tokenizer.encode(full_text, return_tensors="pt").to(device)
                prompt_ids = tokenizer.encode(prompt, return_tensors="pt")
                prompt_len = prompt_ids.shape[1]

                # Tokenize just the answer content to know its token length
                # We find where the answer content appears in the response tokens
                pre_answer = response[:last_match.start(1)]
                pre_answer_ids = tokenizer.encode(prompt + pre_answer, return_tensors="pt")
                pre_answer_len = pre_answer_ids.shape[1]

                post_answer = response[:last_match.end(1)]
                post_answer_ids = tokenizer.encode(prompt + post_answer, return_tensors="pt")
                post_answer_len = post_answer_ids.shape[1]

                answer_token_start = pre_answer_len
                answer_token_end = post_answer_len

                if answer_token_start >= answer_token_end:
                    continue

                # Forward pass
                outputs = model(input_ids=full_ids)
                # logits[t] predicts token[t+1], so for token at position p,
                # its logprob comes from logits[p-1]
                logprobs = F.log_softmax(outputs.logits[0, :-1], dim=-1)
                target_ids = full_ids[0, 1:]

                # Gather logprobs of actual tokens in the answer span
                # Token at position p has its logprob at logprobs[p-1]
                answer_logprobs = []
                for p in range(answer_token_start, min(answer_token_end, len(target_ids))):
                    token_id = target_ids[p].item()
                    lp = logprobs[p, token_id].item()
                    answer_logprobs.append(lp)

                if answer_logprobs:
                    mean_lp = sum(answer_logprobs) / len(answer_logprobs)
                    confidences[idx] = math.exp(mean_lp)

    del model
    torch.cuda.empty_cache()
    return confidences


def evaluate_checkpoint(
    model_path: str | Path,
    eval_dataset: datasets.Dataset,
    gpu: int,
    max_new_tokens: int = 512,
    temperature: float = 0.0,
    top_k: int = 0,
    top_p: float = 1.0,
    repetition_penalty: float = 1.0,
    group_size: int = 1,
    calibration: bool = False,
    sampling_dtype: str = "float16",
) -> dict:
    """Evaluate a single checkpoint on the dataset using vLLM."""
    model_path = str(model_path)

    # Load tokenizer for chat template
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Initialize vLLM
    llm = LLM(
        model=model_path,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.9 if not calibration else 0.45,
        dtype=sampling_dtype,
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

    # Evaluate responses — collect text and correctness
    total_correct = 0
    total_parsable = 0
    total_samples = 0
    per_sample_correct = []
    per_sample_response = []

    for idx, output in enumerate(tqdm(outputs, desc="Scoring")):
        sample = eval_dataset[idx]
        expected = float(sample["answer"])

        any_correct = False
        any_parsable = False
        best_response = output.outputs[0].text  # default to first

        for completion in output.outputs:
            response = completion.text
            matches = answer_pattern.findall(response)

            if matches:
                last_match = matches[-1]
                try:
                    parsed_answer = parse_number(last_match)
                    any_parsable = True
                    best_response = response

                    if abs(parsed_answer - expected) < 1e-6:
                        any_correct = True
                        best_response = response
                        break
                except ValueError:
                    pass

        if any_correct:
            total_correct += 1
        if any_parsable:
            total_parsable += 1
        total_samples += 1
        per_sample_correct.append(any_correct)
        per_sample_response.append(best_response)

    # Free vLLM before loading HF model for calibration
    del llm
    torch.cuda.empty_cache()

    result = {
        "total_samples": total_samples,
        "correct": total_correct,
        "parsable": total_parsable,
        "accuracy": total_correct / total_samples if total_samples > 0 else 0.0,
        "parsable_rate": total_parsable / total_samples if total_samples > 0 else 0.0,
    }

    # Compute calibration via HF forward pass in FP32
    if calibration:
        confidences = compute_answer_confidences_hf(
            model_path, tokenizer, prompts, per_sample_response, batch_size=1,
        )

        calibration_data = [
            (conf, correct)
            for conf, correct in zip(confidences, per_sample_correct)
            if conf is not None
        ]

        if calibration_data:
            conf_list = [c for c, _ in calibration_data]
            corr_list = [c for _, c in calibration_data]
            ece_results = compute_ece(conf_list, corr_list)
            result.update(ece_results)

    return result


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
            calibration=args.calibration,
            sampling_dtype=args.sampling_dtype,
        )
        print(f"\nAccuracy Results for {label}:")
        print(f"  Accuracy: {metrics['accuracy']:.2%} ({metrics['correct']}/{metrics['total_samples']})")
        print(f"  Parsable: {metrics['parsable_rate']:.2%} ({metrics['parsable']}/{metrics['total_samples']})")
        if args.calibration and "ece" in metrics:
            print(f"  ECE: {metrics['ece']:.4f} | MCE: {metrics['mce']:.4f} | Brier: {metrics['brier']:.4f} | NLL: {metrics['nll']:.4f}")
            print(f"  Mean confidence: {metrics['mean_confidence']:.4f}")

        results[label] = {
            "step": step,
            "path": str(checkpoint_dir),
            **metrics,
        }

    # Print summary
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")

    header = f"{'Checkpoint':<20} {'Accuracy':>12} {'Parsable':>12}"
    print(header)
    print("-" * len(header))

    for label, metrics in sorted(results.items(), key=lambda x: x[1]["step"]):
        print(f"{label:<20} {metrics.get('accuracy', 0):>11.2%} {metrics.get('parsable_rate', 0):>11.2%}")

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
