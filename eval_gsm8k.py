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


def compute_kl_divergence(
    base_model_path: str,
    checkpoint_path: str,
    eval_dataset: datasets.Dataset,
    batch_size: int = 8,
    max_new_tokens: int = 256,
    max_prompt_length: int = 256,
    reverse: bool = False,
    temperature: float = 0.0,
    sampling_dtype: str = "float16",
) -> dict:
    """
    Compute KL divergence on generated rollouts.

    By default (reverse=False):
        Generate from checkpoint (π), compute KL(π || π₀)
        = E_{y ~ π}[log π(y) - log π₀(y)]
        This is the REVERSE KL (mode-seeking).

    With reverse=True:
        Generate from base (π₀), compute KL(π₀ || π)
        = E_{y ~ π₀}[log π₀(y) - log π(y)]
        This is the FORWARD KL (mean-seeking).
        Use --forward-kl on the CLI for this.

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

    # Always generate from checkpoint (finetuned model) — determines conditioning context.
    # For forward KL(base || checkpoint), we compute the full-vocabulary KL at each position.
    generator_path = checkpoint_path

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

    # Step 1: Fast generation with vLLM (always from checkpoint)
    gen_source = "checkpoint"
    print(f"Generating rollouts from {gen_source} with vLLM ({len(prompts)} prompts)...")
    llm = LLM(
        model=generator_path,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.9,
        dtype=sampling_dtype,
    )

    sampling_params = SamplingParams(
        max_tokens=max_new_tokens,
        temperature=temperature,
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

    # Step 2: Load both HF models for log prob computation in FP32
    print(f"Loading checkpoint model for log probs (FP32): {checkpoint_path}")
    checkpoint_model = AutoModelForCausalLM.from_pretrained(
        checkpoint_path,
        torch_dtype=torch.float32,
        device_map=device,
    )
    checkpoint_model.eval()

    print(f"Loading base model for log probs (FP32): {base_model_path}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float32,
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

            # Full-vocabulary KL at each token position
            # KL(p || q) = Σ_v p(v) · [log p(v) - log q(v)]
            base_logits = base_outputs.logits[:, :-1].float()
            ft_logits = checkpoint_outputs.logits[:, :-1].float()

            if reverse:
                # KL(base || checkpoint) = Σ_v p_base(v) · [log p_base(v) - log p_ft(v)]
                p_log = F.log_softmax(base_logits, dim=-1)
                q_log = F.log_softmax(ft_logits, dim=-1)
            else:
                # KL(checkpoint || base) = Σ_v p_ft(v) · [log p_ft(v) - log p_base(v)]
                p_log = F.log_softmax(ft_logits, dim=-1)
                q_log = F.log_softmax(base_logits, dim=-1)

            p = p_log.exp()
            kl_per_token = (p * (p_log - q_log)).sum(dim=-1)  # [batch, seq]
            del base_logits, ft_logits, p_log, q_log, p

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

    # Output keys: use correct statistical names + legacy aliases for compat
    if reverse:
        # Forward KL: KL(π₀ || π), y ~ π₀
        result = {
            "forward_kl": mean_kl,
            "forward_kl_std": std_kl,
            "reverse_kl": mean_kl,       # legacy alias
            "reverse_kl_std": std_kl,     # legacy alias
        }
    else:
        # Reverse KL: KL(π || π₀), y ~ π
        result = {
            "reverse_kl_actual": mean_kl,
            "reverse_kl_actual_std": std_kl,
            "kl_divergence": mean_kl,     # legacy alias
            "kl_divergence_std": std_kl,  # legacy alias
        }
    result.update({
        "total_tokens": int(total_tokens),
        "num_sequences": len(prompts),
        "avg_generated_tokens": total_tokens / len(prompts) if prompts else 0,
        "kl_direction": kl_direction,
        "generator": "checkpoint",
    })
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
        "--compute-kl",
        action="store_true",
        help="Compute reverse KL: generate from checkpoint, compute KL(π||π₀). "
             "Legacy name; use --reverse-kl for clarity.",
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
        "--kl-temperature",
        type=float,
        default=0.0,
        help="Sampling temperature for KL rollout generation (default: 0.0 for greedy)",
    )
    parser.add_argument(
        "--kl-only",
        action="store_true",
        help="Only compute KL divergence, skip accuracy evaluation",
    )
    parser.add_argument(
        "--reverse-kl",
        action="store_true",
        help="[MISLEADING NAME — this is actually the FORWARD KL] "
             "Generate from base (π₀), compute KL(π₀||π). "
             "Use --forward-kl instead for clarity.",
    )
    parser.add_argument(
        "--forward-kl",
        action="store_true",
        help="Compute forward KL: generate from base (π₀), compute KL(π₀||π). "
             "Measures how much the checkpoint has drifted from base behavior.",
    )
    parser.add_argument(
        "--kl-dataset",
        type=str,
        default=None,
        help="Dataset for KL computation. Options: 'ifeval' (recommended for drift), "
             "'gsm8k', or path to jsonl file. If not provided, uses eval dataset.",
    )
    parser.add_argument(
        "--sampling-dtype",
        type=str,
        default="float16",
        help="vLLM dtype for generation/sampling (default: float16). "
             "Use 'auto' to infer from model weights, 'bfloat16' for native Qwen precision.",
    )

    args = parser.parse_args()

    # --forward-kl is the correct name; --reverse-kl is kept for backwards compat
    if args.forward_kl:
        args.reverse_kl = True

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
                calibration=args.calibration,
                sampling_dtype=args.sampling_dtype,
            )
            metrics.update(accuracy_metrics)
            print(f"\nAccuracy Results for {label}:")
            print(f"  Accuracy: {metrics['accuracy']:.2%} ({metrics['correct']}/{metrics['total_samples']})")
            print(f"  Parsable: {metrics['parsable_rate']:.2%} ({metrics['parsable']}/{metrics['total_samples']})")
            if args.calibration and "ece" in metrics:
                print(f"  ECE: {metrics['ece']:.4f} | MCE: {metrics['mce']:.4f} | Brier: {metrics['brier']:.4f} | NLL: {metrics['nll']:.4f}")
                print(f"  Mean confidence: {metrics['mean_confidence']:.4f}")

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
                temperature=args.kl_temperature,
                sampling_dtype=args.sampling_dtype,
            )
            metrics.update(kl_metrics)
            print(f"\nKL Divergence Results for {label}:")
            kl_key = "forward_kl" if args.reverse_kl else "kl_divergence"
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

    kl_header = "Fwd KL" if args.reverse_kl else "Rev KL"
    header = f"{'Checkpoint':<20}"
    if has_accuracy:
        header += f" {'Accuracy':>12} {'Parsable':>12}"
    if has_kl:
        header += f" {kl_header:>12}"
    print(header)
    print("-" * len(header))

    kl_metric_key = "forward_kl" if args.reverse_kl else "kl_divergence"
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
