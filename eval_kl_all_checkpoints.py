#!/usr/bin/env python3
"""
Compute KL divergence for all checkpoints in a directory.
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


def get_checkpoint_dirs(base_path: str) -> list[Path]:
    """Get all checkpoint directories sorted by token count."""
    base = Path(base_path)

    if not base.exists():
        raise ValueError(f"Path does not exist: {base_path}")

    # Check if this is a direct model directory
    if (base / "config.json").exists():
        return [base]

    checkpoint_dirs = []
    for d in base.iterdir():
        if not d.is_dir():
            continue
        # Skip non-checkpoint directories
        if d.name in ("epoch_0", "_internal_data_processing"):
            continue
        # Check for config.json (valid checkpoint)
        if (d / "config.json").exists():
            checkpoint_dirs.append(d)

    def get_tokens(path: Path) -> int:
        name = path.name
        if "tokens_" in name:
            try:
                return int(name.split("tokens_")[-1])
            except ValueError:
                return 0
        return 0

    return sorted(checkpoint_dirs, key=get_tokens)


def load_eval_dataset(system_msg: str) -> datasets.Dataset:
    """Load GSM8K test dataset."""
    ds = datasets.load_dataset("openai/gsm8k", "main", split="test")
    data = []
    for item in ds:
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": item["question"]},
        ]
        data.append({"messages": messages})
    return datasets.Dataset.from_list(data)


def generate_from_base(
    base_model_path: str,
    eval_dataset: datasets.Dataset,
    max_new_tokens: int = 256,
    temperature: float = 0.0,
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
        gpu_memory_utilization=0.4,
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

    del llm
    torch.cuda.empty_cache()

    return all_sequences, all_prompt_lengths, tokenizer


def compute_logprobs_for_sequences(
    model_path: str,
    sequences: list[list[int]],
    prompt_lengths: list[int],
    tokenizer: AutoTokenizer,
    batch_size: int = 4,
) -> list[torch.Tensor]:
    """Compute log probabilities for generated tokens."""
    device = torch.device("cuda")

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    model.eval()

    all_gen_logprobs = []

    with torch.no_grad():
        for i in tqdm(range(0, len(sequences), batch_size), desc=f"Logprobs", leave=False):
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
            logprobs = F.log_softmax(outputs.logits[:, :-1], dim=-1)

            target_ids = input_ids[:, 1:]
            token_logprobs = logprobs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)

            for j, prompt_len in enumerate(batch_prompt_lens):
                seq_len = sum(attention_masks[j])
                gen_start = prompt_len - 1
                gen_end = seq_len - 1

                if gen_end > gen_start:
                    gen_logprobs = token_logprobs[j, gen_start:gen_end].cpu()
                    all_gen_logprobs.append(gen_logprobs)
                else:
                    all_gen_logprobs.append(torch.tensor([]))

    del model
    torch.cuda.empty_cache()

    return all_gen_logprobs


def compute_kl(base_logprobs: list[torch.Tensor], ckpt_logprobs: list[torch.Tensor]) -> dict:
    """Compute KL(base || checkpoint)."""
    total_kl = 0.0
    total_tokens = 0
    all_seq_kl = []

    for base_lp, ckpt_lp in zip(base_logprobs, ckpt_logprobs):
        if len(base_lp) > 0 and len(ckpt_lp) > 0:
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
    parser = argparse.ArgumentParser(description="Compute KL for all checkpoints in a directory")
    parser.add_argument("--checkpoint-dir", type=str, required=True, help="Directory containing checkpoints")
    parser.add_argument("--base-model", type=str, default="Qwen/Qwen2-1.5B-Instruct")
    parser.add_argument("--output", type=str, required=True, help="Output JSON file")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--max-samples", type=int, default=None, help="Limit samples for testing")

    args = parser.parse_args()

    # Find all checkpoints
    checkpoint_dirs = get_checkpoint_dirs(args.checkpoint_dir)
    print(f"Found {len(checkpoint_dirs)} checkpoints")

    # Load dataset
    eval_dataset = load_eval_dataset(DEFAULT_SYSTEM_MSG)
    if args.max_samples:
        eval_dataset = eval_dataset.select(range(min(args.max_samples, len(eval_dataset))))
    print(f"Using {len(eval_dataset)} samples")

    # Generate from base model once
    sequences, prompt_lengths, tokenizer = generate_from_base(
        args.base_model,
        eval_dataset,
        max_new_tokens=args.max_new_tokens,
    )

    # Compute base model logprobs once
    print("\nComputing base model log probabilities...")
    base_logprobs = compute_logprobs_for_sequences(
        args.base_model,
        sequences,
        prompt_lengths,
        tokenizer,
        batch_size=args.batch_size,
    )

    # Compute KL for each checkpoint
    results = {}
    for ckpt_path in checkpoint_dirs:
        ckpt_name = ckpt_path.name
        print(f"\nProcessing: {ckpt_name}")

        ckpt_logprobs = compute_logprobs_for_sequences(
            str(ckpt_path),
            sequences,
            prompt_lengths,
            tokenizer,
            batch_size=args.batch_size,
        )

        kl_metrics = compute_kl(base_logprobs, ckpt_logprobs)

        # Extract token count for sorting
        tokens = 0
        if "tokens_" in ckpt_name:
            try:
                tokens = int(ckpt_name.split("tokens_")[-1])
            except ValueError:
                pass

        results[ckpt_name] = {
            "path": str(ckpt_path),
            "tokens": tokens,
            **kl_metrics,
        }

        print(f"  KL: {kl_metrics['kl_divergence']:.4f} (± {kl_metrics['kl_std']:.4f})")

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
