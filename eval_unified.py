#!/usr/bin/env python3
"""
Unified evaluation: accuracy (N runs averaged), forward KL, and ECE
all computed from the same generation pass.

Flow per checkpoint:
1. Generate responses with vLLM (N times if --n-runs > 1 for accuracy averaging)
2. Score accuracy for each run, average across runs
3. Compute logprobs offline (HF, fp16) for both trained and base model
   on ALL generated sequences across all runs
4. Compute forward KL from logprobs
5. Compute ECE from answer token confidences

Usage:
    python eval_unified.py \
        --checkpoint /path/to/checkpoint \
        --gpu 0 \
        --n-runs 3 \
        --output results.json
"""

import argparse
import json
import math
import os
import re
from pathlib import Path

import datasets
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_SYSTEM_MSG = (
    "You are a helpful math assistant. Always provide your final numerical answer "
    "inside of the <answer>...</answer> tags, e.g.: <answer>42</answer>"
)


def load_eval_dataset(eval_path: str | None = None, max_samples: int | None = None):
    """Load evaluation dataset."""
    if eval_path:
        ds = datasets.load_dataset("json", data_files=eval_path, split="train")
    else:
        ds = datasets.load_dataset("openai/gsm8k", "main", split="test")
        data = []
        for item in ds:
            messages = [
                {"role": "system", "content": DEFAULT_SYSTEM_MSG},
                {"role": "user", "content": item["question"]},
            ]
            data.append({"messages": messages, "answer": float(item["answer"].split("####")[-1].strip().replace(",", ""))})
        ds = datasets.Dataset.from_list(data)

    # Strip assistant messages if present
    def _strip(sample):
        if "messages" in sample:
            sample["messages"] = [m for m in sample["messages"] if m["role"] in ("system", "user")]
        return sample
    ds = ds.map(_strip)

    if max_samples:
        ds = ds.select(range(min(max_samples, len(ds))))
    return ds


def extract_answer(text: str) -> float | None:
    """Extract numerical answer from <answer>...</answer> tags."""
    matches = re.findall(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if not matches:
        return None
    answer_str = matches[-1].strip().replace(",", "").replace("$", "")
    try:
        return float(answer_str)
    except ValueError:
        return None


def generate_and_score(llm, sampling_params, prompts, answers):
    """Generate responses and score accuracy. Returns (accuracy, outputs)."""
    outputs = llm.generate(prompts, sampling_params)

    correct = 0
    parsable = 0
    for output, gold in zip(outputs, answers):
        text = output.outputs[0].text
        pred = extract_answer(text)
        if pred is not None:
            parsable += 1
            if abs(pred - gold) < 1e-3:
                correct += 1

    accuracy = correct / len(answers) if answers else 0
    parsable_rate = parsable / len(answers) if answers else 0
    return accuracy, parsable_rate, outputs


def compute_logprobs(model_path, sequences, prompt_lengths, tokenizer, batch_size=4, device=None):
    """Compute per-token logprobs for generated tokens using HF in fp16."""
    if device is None:
        device = torch.device("cuda")

    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float32, device_map=device,
    )
    model.eval()

    all_gen_logprobs = []
    with torch.no_grad():
        for i in range(0, len(sequences), batch_size):
            batch_seqs = sequences[i:i + batch_size]
            batch_prompt_lens = prompt_lengths[i:i + batch_size]

            max_len = max(len(s) for s in batch_seqs)
            padded_ids, attention_masks = [], []
            for seq in batch_seqs:
                pad_len = max_len - len(seq)
                padded_ids.append(seq + [tokenizer.pad_token_id] * pad_len)
                attention_masks.append([1] * len(seq) + [0] * pad_len)

            input_ids = torch.tensor(padded_ids, device=device)
            attention_mask = torch.tensor(attention_masks, device=device)

            out = model(input_ids=input_ids, attention_mask=attention_mask)
            logprobs = F.log_softmax(out.logits[:, :-1].float(), dim=-1)
            target_ids = input_ids[:, 1:]
            token_lp = logprobs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)

            for j, pl in enumerate(batch_prompt_lens):
                seq_len = sum(attention_masks[j])
                gen_start = pl - 1
                gen_end = seq_len - 1
                if gen_end > gen_start:
                    all_gen_logprobs.append(token_lp[j, gen_start:gen_end].cpu())
                else:
                    all_gen_logprobs.append(torch.tensor([]))

    del model
    torch.cuda.empty_cache()
    return all_gen_logprobs


def compute_answer_confidence(tokenizer, prompt_ids, generated_ids, logprobs_tensor):
    """Compute confidence as geometric mean of answer token probabilities."""
    # Decode the generated text to find <answer>...</answer> span
    gen_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    matches = list(re.finditer(r"<answer>(.*?)</answer>", gen_text, re.DOTALL))
    if not matches:
        return None

    # Find the token positions corresponding to the answer content
    last_match = matches[-1]
    answer_start_char = last_match.start(1)
    answer_end_char = last_match.end(1)

    # Map character positions to token positions in the generated portion
    cumulative = 0
    token_start = None
    token_end = None
    for idx, tid in enumerate(generated_ids):
        token_text = tokenizer.decode([tid])
        token_len = len(token_text)
        if token_start is None and cumulative + token_len > answer_start_char:
            token_start = idx
        if cumulative >= answer_end_char:
            token_end = idx
            break
        cumulative += token_len

    if token_start is None:
        return None
    if token_end is None:
        token_end = len(generated_ids)

    # Get logprobs for answer tokens
    if token_start >= len(logprobs_tensor) or token_end <= token_start:
        return None

    answer_logprobs = logprobs_tensor[token_start:token_end]
    if len(answer_logprobs) == 0:
        return None

    # Geometric mean of probabilities = exp(mean of logprobs)
    confidence = torch.exp(answer_logprobs.mean()).item()
    return confidence


def compute_ece(confidences, correctness, n_bins=10):
    """Compute Expected Calibration Error."""
    if not confidences:
        return {"ece": 0.0, "mce": 0.0, "n_calibration_samples": 0, "per_bin_data": []}

    n = len(confidences)
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    per_bin = []
    ece = 0.0
    mce = 0.0

    for b in range(n_bins):
        lo, hi = bin_boundaries[b], bin_boundaries[b + 1]
        mask = [(lo <= c < hi) if b < n_bins - 1 else (lo <= c <= hi) for c in confidences]
        bin_confs = [c for c, m in zip(confidences, mask) if m]
        bin_corrs = [c for c, m in zip(correctness, mask) if m]

        count = len(bin_confs)
        if count > 0:
            avg_conf = np.mean(bin_confs)
            avg_acc = np.mean(bin_corrs)
            gap = abs(avg_acc - avg_conf)
            ece += (count / n) * gap
            mce = max(mce, gap)
        else:
            avg_conf = 0.0
            avg_acc = 0.0

        per_bin.append({
            "bin_lo": float(lo), "bin_hi": float(hi),
            "count": count, "accuracy": float(avg_acc), "avg_confidence": float(avg_conf),
        })

    return {
        "ece": float(ece),
        "mce": float(mce),
        "mean_confidence": float(np.mean(confidences)),
        "n_calibration_samples": n,
        "per_bin_data": per_bin,
    }


def main():
    parser = argparse.ArgumentParser(description="Unified eval: accuracy + forward KL + ECE")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--base-model", type=str, default="Qwen/Qwen2-1.5B-Instruct")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--n-runs", type=int, default=1, help="Number of generation passes for accuracy averaging")
    parser.add_argument("--eval-path", type=str, default=None, help="Path to eval JSONL (default: GSM8K test)")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--output", type=str, required=True)

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device("cuda:0")

    from vllm import LLM, SamplingParams

    # Load dataset
    print("Loading dataset...")
    ds = load_eval_dataset(args.eval_path, args.max_samples)
    print(f"  {len(ds)} samples")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Format prompts
    prompts = []
    answers = []
    for sample in ds:
        prompt = tokenizer.apply_chat_template(
            conversation=sample["messages"], tokenize=False, add_generation_prompt=True,
        )
        prompts.append(prompt)
        answers.append(float(sample["answer"]))

    # Load vLLM
    print(f"Loading checkpoint: {args.checkpoint}")
    llm = LLM(
        model=args.checkpoint, tensor_parallel_size=1,
        gpu_memory_utilization=0.9, dtype="float32",
    )
    sampling_params = SamplingParams(
        max_tokens=args.max_new_tokens, temperature=args.temperature, top_p=1.0,
    )

    # Step 1: Generate N times, score accuracy each time
    all_accuracies = []
    all_parsable_rates = []
    all_outputs_runs = []

    for run in range(args.n_runs):
        acc, parsable, outputs = generate_and_score(llm, sampling_params, prompts, answers)
        all_accuracies.append(acc)
        all_parsable_rates.append(parsable)
        all_outputs_runs.append(outputs)
        print(f"  Run {run+1}/{args.n_runs}: acc={acc:.2%}, parsable={parsable:.2%}")

    mean_acc = float(np.mean(all_accuracies))
    std_acc = float(np.std(all_accuracies))
    mean_parsable = float(np.mean(all_parsable_rates))

    # Collect ALL generated sequences across all runs for KL/ECE
    all_sequences = []
    all_prompt_lengths = []
    all_generated_ids = []
    all_correct = []
    all_gold = []

    for run_idx, outputs in enumerate(all_outputs_runs):
        for out_idx, (output, gold) in enumerate(zip(outputs, answers)):
            prompt_ids = list(output.prompt_token_ids)
            gen_ids = list(output.outputs[0].token_ids)
            full_seq = prompt_ids + gen_ids
            all_sequences.append(full_seq)
            all_prompt_lengths.append(len(prompt_ids))
            all_generated_ids.append(gen_ids)
            all_gold.append(gold)

            pred = extract_answer(output.outputs[0].text)
            all_correct.append(pred is not None and abs(pred - gold) < 1e-3)

    del llm
    torch.cuda.empty_cache()

    n_total_seqs = len(all_sequences)
    print(f"\n  Total sequences for KL/ECE: {n_total_seqs}")

    # Step 2: Compute logprobs for trained model
    print("  Computing trained model logprobs...")
    trained_lp = compute_logprobs(
        args.checkpoint, all_sequences, all_prompt_lengths,
        tokenizer, batch_size=args.batch_size, device=device,
    )

    # Step 3: Compute logprobs for base model
    print("  Computing base model logprobs...")
    base_lp = compute_logprobs(
        args.base_model, all_sequences, all_prompt_lengths,
        tokenizer, batch_size=args.batch_size, device=device,
    )

    # Step 4: Forward KL = E_π[log π - log π₀]
    total_kl = 0.0
    total_tokens = 0
    per_seq_kl = []
    for tlp, blp in zip(trained_lp, base_lp):
        if len(tlp) > 0 and len(blp) > 0:
            kl_per_token = tlp - blp
            seq_kl = kl_per_token.mean().item()
            per_seq_kl.append(seq_kl)
            total_kl += kl_per_token.sum().item()
            total_tokens += len(kl_per_token)

    forward_kl = total_kl / total_tokens if total_tokens > 0 else 0.0
    forward_kl_std = float(np.std(per_seq_kl)) if per_seq_kl else 0.0

    # Step 5: ECE from trained model logprobs
    confidences = []
    correctness = []
    for i, (gen_ids, tlp, is_correct) in enumerate(zip(all_generated_ids, trained_lp, all_correct)):
        if len(tlp) == 0:
            continue
        conf = compute_answer_confidence(tokenizer, all_sequences[i][:all_prompt_lengths[i]], gen_ids, tlp)
        if conf is not None:
            confidences.append(conf)
            correctness.append(float(is_correct))

    ece_results = compute_ece(confidences, correctness)

    # Build result
    result = {
        "checkpoint": args.checkpoint,
        "n_samples": len(ds),
        "n_runs": args.n_runs,
        "accuracy_mean": mean_acc,
        "accuracy_std": std_acc,
        "accuracy_runs": all_accuracies,
        "parsable_rate": mean_parsable,
        "forward_kl": forward_kl,
        "forward_kl_std": forward_kl_std,
        "forward_kl_total_tokens": total_tokens,
        "forward_kl_n_sequences": len(per_seq_kl),
        **ece_results,
    }

    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Checkpoint: {Path(args.checkpoint).name}")
    print(f"Accuracy: {mean_acc:.2%} ± {std_acc:.2%} (n={args.n_runs})")
    print(f"Forward KL: {forward_kl:.4f} ± {forward_kl_std:.4f}")
    print(f"ECE: {ece_results['ece']:.4f} (n={ece_results['n_calibration_samples']})")
    print(f"Mean confidence: {ece_results['mean_confidence']:.4f}")
    print(f"Saved: {args.output}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
