"""Evaluate a countdown checkpoint: accuracy, forward KL, and ECE."""
import argparse
import json
import math
import re
import sys
sys.path.insert(0, "/mnt/4TB/workspace/oleg/mini-grpo")

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from countdown_utils import countdown_reward_fn

ANSWER_PATTERN = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)


def load_test_data(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def generate_responses(checkpoint, tokenizer, test_samples, max_new_tokens=512,
                       temperature=1.0, top_p=1.0):
    """Generate responses using vLLM."""
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=checkpoint,
        dtype="float16",
        gpu_memory_utilization=0.45,
        trust_remote_code=True,
    )
    sampling_params = SamplingParams(
        temperature=temperature, top_p=top_p, max_tokens=max_new_tokens,
    )

    prompts = []
    for sample in test_samples:
        prompt_ids = tokenizer.apply_chat_template(
            sample["messages"], add_generation_prompt=True,
        )
        prompts.append(tokenizer.decode(prompt_ids))

    outputs = llm.generate(prompts, sampling_params)

    responses = []
    prompt_lengths = []
    sequences = []
    for output in outputs:
        text = output.outputs[0].text
        prompt_ids = list(output.prompt_token_ids)
        gen_ids = list(output.outputs[0].token_ids)
        responses.append(text)
        prompt_lengths.append(len(prompt_ids))
        sequences.append(prompt_ids + gen_ids)

    del llm
    torch.cuda.empty_cache()

    return responses, sequences, prompt_lengths


def compute_accuracy(test_samples, responses):
    """Score responses and return accuracy + per-sample results."""
    correct = 0
    parsable = 0
    results = []

    for sample, text in zip(test_samples, responses):
        r = countdown_reward_fn(text, sample["answer"], {"numbers": sample.get("numbers")})
        results.append(r)
        if r["is_parsable"]:
            parsable += 1
        if r["is_correct"]:
            correct += 1

    total = len(test_samples)
    return {
        "correct": correct,
        "parsable": parsable,
        "total": total,
        "accuracy": correct / total,
        "parsable_rate": parsable / total,
    }, results


def compute_logprobs_batched(model_path, sequences, prompt_lengths, tokenizer, batch_size=4):
    """Compute per-token log probabilities for generated tokens."""
    device = torch.device("cuda")
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16, device_map=device,
    )
    model.eval()

    all_gen_logprobs = []

    with torch.no_grad():
        for i in range(0, len(sequences), batch_size):
            batch_seqs = sequences[i:i + batch_size]
            batch_prompt_lens = prompt_lengths[i:i + batch_size]

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

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logprobs = F.log_softmax(outputs.logits[:, :-1].float(), dim=-1)

            target_ids = input_ids[:, 1:]
            token_logprobs = logprobs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)

            for j, prompt_len in enumerate(batch_prompt_lens):
                seq_len = sum(attention_masks[j])
                gen_start = prompt_len - 1
                gen_end = seq_len - 1
                if gen_end > gen_start:
                    all_gen_logprobs.append(token_logprobs[j, gen_start:gen_end].cpu())
                else:
                    all_gen_logprobs.append(torch.tensor([]))

    del model
    torch.cuda.empty_cache()
    return all_gen_logprobs


def compute_forward_kl(trained_lp, base_lp):
    """KL(π || π₀) = E_π[log π - log π₀]."""
    total_kl = 0.0
    total_tokens = 0
    all_seq_kl = []

    for p_lp, q_lp in zip(trained_lp, base_lp):
        if len(p_lp) > 0 and len(q_lp) > 0:
            kl_per_token = p_lp - q_lp
            all_seq_kl.append(kl_per_token.mean().item())
            total_kl += kl_per_token.sum().item()
            total_tokens += len(kl_per_token)

    mean_kl = total_kl / total_tokens if total_tokens > 0 else 0.0
    std_kl = torch.tensor(all_seq_kl).std().item() if all_seq_kl else 0.0
    return {"forward_kl": mean_kl, "forward_kl_std": std_kl, "kl_tokens": total_tokens}


def compute_answer_confidences(model_path, tokenizer, prompts, responses, batch_size=4):
    """Compute confidence = exp(mean logprob of answer tokens) via HF forward pass."""
    device = torch.device("cuda")
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float32, device_map=device,
    )
    model.eval()

    confidences = []

    with torch.no_grad():
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i + batch_size]
            batch_responses = responses[i:i + batch_size]

            for prompt, response in zip(batch_prompts, batch_responses):
                # Find answer tags
                match = ANSWER_PATTERN.search(response)
                if not match:
                    confidences.append(None)
                    continue

                answer_content = match.group(1).strip()
                if not answer_content:
                    confidences.append(None)
                    continue

                # Tokenize full sequence
                full_text = prompt + response
                full_ids = tokenizer.encode(full_text, return_tensors="pt").to(device)
                prompt_ids = tokenizer.encode(prompt, return_tensors="pt")
                prompt_len = prompt_ids.shape[1]

                # Tokenize just the answer content to find its tokens
                answer_ids = tokenizer.encode(answer_content, add_special_tokens=False)
                answer_len = len(answer_ids)

                if full_ids.shape[1] <= prompt_len:
                    confidences.append(None)
                    continue

                outputs = model(input_ids=full_ids)
                logprobs = F.log_softmax(outputs.logits[:, :-1].float(), dim=-1)
                target_ids = full_ids[:, 1:]
                token_logprobs = logprobs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)

                # Extract response logprobs (after prompt)
                gen_logprobs = token_logprobs[0, prompt_len - 1:]

                # Use last `answer_len` tokens of the response as the answer region
                # (approximate — the answer tags are near the end)
                if answer_len > 0 and answer_len <= len(gen_logprobs):
                    answer_logprobs = gen_logprobs[-answer_len:]
                    confidence = math.exp(answer_logprobs.mean().item())
                    confidences.append(min(confidence, 1.0))
                else:
                    # Fall back to mean of all generated logprobs
                    if len(gen_logprobs) > 0:
                        confidence = math.exp(gen_logprobs.mean().item())
                        confidences.append(min(confidence, 1.0))
                    else:
                        confidences.append(None)

    del model
    torch.cuda.empty_cache()
    return confidences


def compute_ece(confidences, correctness, n_bins=10):
    """ECE, MCE, Brier, NLL."""
    import numpy as np

    valid = [(c, r) for c, r in zip(confidences, correctness) if c is not None]
    if not valid:
        return {"ece": 0.0, "mce": 0.0, "brier": 0.0, "nll": 0.0, "n_calibration_samples": 0}

    conf = np.array([c for c, _ in valid])
    corr = np.array([float(r) for _, r in valid])
    n = len(conf)

    brier = float(np.mean((conf - corr) ** 2))
    eps = 1e-12
    c_clipped = np.clip(conf, eps, 1 - eps)
    nll = float(-np.mean(corr * np.log(c_clipped) + (1 - corr) * np.log(1 - c_clipped)))

    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    mce = 0.0
    for m in range(n_bins):
        lo, hi = bin_edges[m], bin_edges[m + 1]
        mask = (conf > lo) & (conf <= hi) if m > 0 else (conf >= lo) & (conf <= hi)
        bin_size = mask.sum()
        if bin_size == 0:
            continue
        bin_gap = abs(corr[mask].mean() - conf[mask].mean())
        ece += (bin_size / n) * bin_gap
        mce = max(mce, bin_gap)

    return {
        "ece": float(ece), "mce": float(mce), "brier": brier, "nll": nll,
        "mean_confidence": float(conf.mean()), "n_calibration_samples": n,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--base-model", default="Qwen/Qwen2-1.5B-Instruct")
    parser.add_argument("--test-data", required=True)
    parser.add_argument("--output", required=True, help="Output JSON path")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    test_samples = load_test_data(args.test_data)
    print(f"Loaded {len(test_samples)} test samples")

    # 1. Generate responses
    print("Generating responses...")
    responses, sequences, prompt_lengths = generate_responses(
        args.checkpoint, tokenizer, test_samples, args.max_new_tokens,
        temperature=args.temperature, top_p=args.top_p,
    )

    # 2. Accuracy
    print("Computing accuracy...")
    acc_metrics, per_sample = compute_accuracy(test_samples, responses)
    print(f"  Accuracy: {acc_metrics['accuracy']:.1%} ({acc_metrics['correct']}/{acc_metrics['total']})")
    print(f"  Parsable: {acc_metrics['parsable_rate']:.1%}")

    # 3. Forward KL
    print("Computing checkpoint logprobs...")
    trained_lp = compute_logprobs_batched(
        args.checkpoint, sequences, prompt_lengths, tokenizer, args.batch_size,
    )
    print("Computing base model logprobs...")
    base_lp = compute_logprobs_batched(
        args.base_model, sequences, prompt_lengths, tokenizer, args.batch_size,
    )
    kl_metrics = compute_forward_kl(trained_lp, base_lp)
    print(f"  Forward KL: {kl_metrics['forward_kl']:.4f}")

    # 4. ECE
    print("Computing answer confidences for ECE...")
    prompts = []
    for sample in test_samples:
        prompt_ids = tokenizer.apply_chat_template(
            sample["messages"], add_generation_prompt=True,
        )
        prompts.append(tokenizer.decode(prompt_ids))

    confidences = compute_answer_confidences(
        args.checkpoint, tokenizer, prompts, responses, args.batch_size,
    )
    correctness = [r["is_correct"] for r in per_sample]
    ece_metrics = compute_ece(confidences, correctness)
    print(f"  ECE: {ece_metrics['ece']:.4f}")

    # Combine and save
    result = {
        "checkpoint": args.checkpoint,
        "base_model": args.base_model,
        **acc_metrics,
        **kl_metrics,
        **ece_metrics,
    }

    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved results to {args.output}")

    print(f"\n{'='*60}")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Accuracy:   {acc_metrics['accuracy']:.1%}")
    print(f"  Parsable:   {acc_metrics['parsable_rate']:.1%}")
    print(f"  Forward KL: {kl_metrics['forward_kl']:.4f}")
    print(f"  ECE:        {ece_metrics['ece']:.4f}")
    print(f"  Brier:      {ece_metrics['brier']:.4f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
