#!/usr/bin/env python3
"""
Efficient KL divergence computation across multiple checkpoints.

Generates sequences from base model ONCE, then computes KL(base || checkpoint)
against all specified checkpoints using the same generated sequences.
"""

import argparse
import json
from pathlib import Path

import datasets
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams


DEFAULT_SYSTEM_MSG = (
    "You are a helpful math assistant. Always provide your final numerical answer "
    "inside of the <answer>...</answer> tags, e.g.: <answer>42</answer>"
)


def load_eval_dataset(eval_path: str | None, system_msg: str) -> datasets.Dataset:
    """Load evaluation dataset."""
    if eval_path:
        data = []
        with open(eval_path) as f:
            for line in f:
                item = json.loads(line)
                if "messages" in item:
                    # IMPORTANT: Strip assistant messages to avoid leaking answers
                    # SFT training data includes assistant responses, but eval should only
                    # have system + user messages for the model to generate a response
                    filtered_messages = [m for m in item["messages"] if m["role"] in ("system", "user")]
                    item["messages"] = filtered_messages
                    data.append(item)
                elif "question" in item:
                    messages = [
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": item["question"]},
                    ]
                    data.append({"messages": messages, "answer": item.get("answer", "")})
        print(f"[INFO] Loaded {len(data)} samples (stripped assistant messages to prevent answer leakage)")
        return datasets.Dataset.from_list(data)
    else:
        ds = datasets.load_dataset("openai/gsm8k", "main", split="test")
        data = []
        for item in ds:
            messages = [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": item["question"]},
            ]
            data.append({"messages": messages, "answer": item["answer"]})
        return datasets.Dataset.from_list(data)


def generate_from_base(
    base_model_path: str,
    eval_dataset: datasets.Dataset,
    max_new_tokens: int = 256,
    temperature: float = 0.0,
    gpu_memory_util: float = 0.4,
) -> tuple[list[list[int]], list[int], AutoTokenizer]:
    """Generate sequences from base model using vLLM."""
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    prompts = []
    for sample in eval_dataset:
        prompt = tokenizer.apply_chat_template(
            conversation=sample["messages"],
            tokenize=False,
            add_generation_prompt=True,
        )
        prompts.append(prompt)

    print(f"Generating {len(prompts)} sequences from base model: {base_model_path}")
    llm = LLM(
        model=base_model_path,
        tensor_parallel_size=1,
        gpu_memory_utilization=gpu_memory_util,
        dtype="bfloat16",
    )

    sampling_params = SamplingParams(
        max_tokens=max_new_tokens,
        temperature=temperature,
        top_p=1.0,
    )

    outputs = llm.generate(prompts, sampling_params)

    all_sequences = []
    all_prompt_lengths = []
    for output in outputs:
        prompt_ids = output.prompt_token_ids
        generated_ids = output.outputs[0].token_ids
        full_seq = list(prompt_ids) + list(generated_ids)
        all_sequences.append(full_seq)
        all_prompt_lengths.append(len(prompt_ids))

    # Free vLLM memory
    del llm
    torch.cuda.empty_cache()

    print(f"Generated {len(all_sequences)} sequences, avg length: {sum(len(s) for s in all_sequences) / len(all_sequences):.1f}")
    return all_sequences, all_prompt_lengths, tokenizer


def compute_logprobs_for_sequences(
    model_path: str,
    sequences: list[list[int]],
    prompt_lengths: list[int],
    tokenizer: AutoTokenizer,
    batch_size: int = 4,
) -> tuple[list[torch.Tensor], int]:
    """Compute log probabilities for generated tokens using a model."""
    device = torch.device("cuda")

    print(f"Loading model: {model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    model.eval()

    all_gen_logprobs = []
    total_gen_tokens = 0

    with torch.no_grad():
        for i in tqdm(range(0, len(sequences), batch_size), desc=f"Computing logprobs"):
            batch_seqs = sequences[i:i + batch_size]
            batch_prompt_lens = prompt_lengths[i:i + batch_size]

            # Pad sequences
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
            logprobs = F.log_softmax(outputs.logits[:, :-1], dim=-1)

            target_ids = input_ids[:, 1:]
            token_logprobs = logprobs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)

            # Extract only generated token logprobs
            for j, prompt_len in enumerate(batch_prompt_lens):
                seq_len = sum(attention_masks[j])
                gen_start = prompt_len - 1
                gen_end = seq_len - 1

                if gen_end > gen_start:
                    gen_logprobs = token_logprobs[j, gen_start:gen_end].cpu()
                    all_gen_logprobs.append(gen_logprobs)
                    total_gen_tokens += len(gen_logprobs)
                else:
                    all_gen_logprobs.append(torch.tensor([]))

    del model
    torch.cuda.empty_cache()

    return all_gen_logprobs, total_gen_tokens


def compute_kl_divergence(
    base_logprobs: list[torch.Tensor],
    checkpoint_logprobs: list[torch.Tensor],
) -> dict:
    """Compute KL(base || checkpoint) = E[log base - log checkpoint]."""
    total_kl = 0.0
    total_tokens = 0
    all_seq_kl = []

    for base_lp, ckpt_lp in zip(base_logprobs, checkpoint_logprobs):
        if len(base_lp) > 0 and len(ckpt_lp) > 0:
            # KL(base || checkpoint) = log base(x) - log checkpoint(x)
            kl_per_token = base_lp - ckpt_lp
            seq_kl = kl_per_token.mean().item()
            all_seq_kl.append(seq_kl)
            total_kl += kl_per_token.sum().item()
            total_tokens += len(kl_per_token)

    mean_kl = total_kl / total_tokens if total_tokens > 0 else 0.0
    std_kl = torch.tensor(all_seq_kl).std().item() if all_seq_kl else 0.0

    return {
        "kl_divergence": mean_kl,
        "kl_std": std_kl,
        "total_tokens": total_tokens,
        "num_sequences": len(all_seq_kl),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Compute KL(base || checkpoint) for multiple checkpoints efficiently"
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="Qwen/Qwen2-1.5B-Instruct",
        help="Base model path",
    )
    # Individual checkpoint arguments for each method + optimizer
    parser.add_argument(
        "--grpo-adamw",
        type=str,
        default=None,
        help="Path to GRPO + AdamW checkpoint",
    )
    parser.add_argument(
        "--grpo-muon",
        type=str,
        default=None,
        help="Path to GRPO + Muon checkpoint",
    )
    parser.add_argument(
        "--sft-adamw",
        type=str,
        default=None,
        help="Path to SFT + AdamW checkpoint",
    )
    parser.add_argument(
        "--sft-muon",
        type=str,
        default=None,
        help="Path to SFT + Muon checkpoint",
    )
    parser.add_argument(
        "--eval-path",
        type=str,
        default=None,
        help="Path to evaluation data (jsonl). If not provided, uses GSM8K test",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="kl_results.json",
        help="Output JSON file",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Max tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (default: 0.0 for greedy decoding)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for log prob computation",
    )
    parser.add_argument(
        "--system-msg",
        type=str,
        default=DEFAULT_SYSTEM_MSG,
        help="System message",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Max samples to evaluate (for testing)",
    )

    args = parser.parse_args()

    # Build checkpoint dict from individual arguments
    checkpoints = {}
    if args.grpo_adamw:
        checkpoints["grpo_adamw"] = args.grpo_adamw
    if args.grpo_muon:
        checkpoints["grpo_muon"] = args.grpo_muon
    if args.sft_adamw:
        checkpoints["sft_adamw"] = args.sft_adamw
    if args.sft_muon:
        checkpoints["sft_muon"] = args.sft_muon

    if not checkpoints:
        parser.error("At least one checkpoint must be specified (--grpo-adamw, --grpo-muon, --sft-adamw, --sft-muon)")

    print(f"Will compute KL for {len(checkpoints)} checkpoints: {list(checkpoints.keys())}")

    # Load dataset
    eval_dataset = load_eval_dataset(args.eval_path, args.system_msg)
    if args.max_samples:
        eval_dataset = eval_dataset.select(range(min(args.max_samples, len(eval_dataset))))
    print(f"Loaded {len(eval_dataset)} evaluation samples")

    # Step 1: Generate from base model
    sequences, prompt_lengths, tokenizer = generate_from_base(
        args.base_model,
        eval_dataset,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )

    # Step 2: Compute base model logprobs
    print("\nComputing base model log probabilities...")
    base_logprobs, _ = compute_logprobs_for_sequences(
        args.base_model,
        sequences,
        prompt_lengths,
        tokenizer,
        batch_size=args.batch_size,
    )

    # Step 3: Compute KL for each checkpoint
    results = {}
    for ckpt_name, ckpt_path in checkpoints.items():
        print(f"\n{'='*60}")
        print(f"Processing checkpoint: {ckpt_name}")
        print(f"{'='*60}")

        # Compute checkpoint logprobs
        ckpt_logprobs, _ = compute_logprobs_for_sequences(
            ckpt_path,
            sequences,
            prompt_lengths,
            tokenizer,
            batch_size=args.batch_size,
        )

        # Compute KL divergence
        kl_metrics = compute_kl_divergence(base_logprobs, ckpt_logprobs)

        results[ckpt_name] = {
            "path": ckpt_path,
            **kl_metrics,
        }

        print(f"KL(base || {ckpt_name}): {kl_metrics['kl_divergence']:.4f} (± {kl_metrics['kl_std']:.4f})")

    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY: KL(base || checkpoint)")
    print(f"{'='*60}")
    print(f"{'Checkpoint':<40} {'KL Div':>12} {'Std':>12}")
    print("-" * 66)
    for name, metrics in sorted(results.items(), key=lambda x: x[1]["kl_divergence"]):
        print(f"{name:<40} {metrics['kl_divergence']:>12.4f} {metrics['kl_std']:>12.4f}")

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
