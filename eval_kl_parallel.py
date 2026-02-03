#!/usr/bin/env python3
"""
Parallel KL divergence evaluation across multiple GPUs.

Key design:
- Generates base model rollouts ONCE and caches them
- All checkpoints in a variant are compared against the SAME rollouts
- Distributes checkpoint evaluation across multiple GPUs
"""

import argparse
import json
import multiprocessing as mp
import os
import pickle
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


def load_eval_dataset(system_msg: str, max_samples: int | None = None) -> datasets.Dataset:
    """Load GSM8K test dataset."""
    ds = datasets.load_dataset("openai/gsm8k", "main", split="test")
    data = []
    for item in ds:
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": item["question"]},
        ]
        data.append({"messages": messages})
    dataset = datasets.Dataset.from_list(data)
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    return dataset


def generate_base_rollouts(
    base_model_path: str,
    eval_dataset: datasets.Dataset,
    cache_path: Path,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
) -> tuple[list[list[int]], list[int], AutoTokenizer]:
    """Generate sequences from base model, with caching."""

    # Check cache
    if cache_path.exists():
        print(f"Loading cached rollouts from: {cache_path}")
        with open(cache_path, "rb") as f:
            cached = pickle.load(f)
        tokenizer = AutoTokenizer.from_pretrained(base_model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return cached["sequences"], cached["prompt_lengths"], tokenizer

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

    # Cache the rollouts
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump({
            "sequences": all_sequences,
            "prompt_lengths": all_prompt_lengths,
        }, f)
    print(f"Cached rollouts to: {cache_path}")

    return all_sequences, all_prompt_lengths, tokenizer


def compute_logprobs_for_sequences(
    model_path: str,
    sequences: list[list[int]],
    prompt_lengths: list[int],
    tokenizer: AutoTokenizer,
    batch_size: int = 4,
    device: torch.device | None = None,
) -> list[torch.Tensor]:
    """Compute log probabilities for generated tokens."""
    if device is None:
        device = torch.device("cuda")

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    model.eval()

    all_gen_logprobs = []

    with torch.no_grad():
        for i in tqdm(range(0, len(sequences), batch_size), desc="Logprobs", leave=False):
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
        if d.name in ("epoch_0", "_internal_data_processing"):
            continue
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


def worker_evaluate_checkpoint(args: tuple) -> tuple[str, dict]:
    """Worker function to evaluate KL for one checkpoint."""
    (
        checkpoint_path,
        sequences,
        prompt_lengths,
        base_logprobs_path,
        tokenizer_path,
        batch_size,
        gpu_id,
    ) = args

    # Set GPU for this process
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load cached base logprobs
    with open(base_logprobs_path, "rb") as f:
        base_logprobs = pickle.load(f)

    # Compute checkpoint logprobs
    ckpt_logprobs = compute_logprobs_for_sequences(
        checkpoint_path,
        sequences,
        prompt_lengths,
        tokenizer,
        batch_size=batch_size,
        device=torch.device("cuda:0"),
    )

    # Compute KL
    kl_metrics = compute_kl(base_logprobs, ckpt_logprobs)

    # Extract token count
    ckpt_name = Path(checkpoint_path).name
    tokens = 0
    if "tokens_" in ckpt_name:
        try:
            tokens = int(ckpt_name.split("tokens_")[-1])
        except ValueError:
            pass

    result = {
        "path": checkpoint_path,
        "tokens": tokens,
        **kl_metrics,
    }

    return ckpt_name, result


def main():
    parser = argparse.ArgumentParser(description="Parallel KL divergence evaluation")
    parser.add_argument(
        "--checkpoint-dirs",
        type=str,
        required=True,
        help="Comma-separated list of checkpoint directories to evaluate",
    )
    parser.add_argument(
        "--gpus",
        type=str,
        required=True,
        help="Comma-separated GPU IDs (e.g., '0,1,2,3')",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="Qwen/Qwen2-1.5B-Instruct",
        help="Base model for generating rollouts",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        required=True,
        help="Directory to cache base model rollouts and logprobs",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save per-experiment KL results",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0.7)

    args = parser.parse_args()

    gpu_ids = [int(g.strip()) for g in args.gpus.split(",")]
    checkpoint_dirs = [d.strip() for d in args.checkpoint_dirs.split(",")]

    cache_dir = Path(args.cache_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Load dataset
    print("Loading evaluation dataset...")
    eval_dataset = load_eval_dataset(DEFAULT_SYSTEM_MSG, args.max_samples)
    print(f"Using {len(eval_dataset)} samples")

    # Step 2: Generate base model rollouts (cached)
    rollouts_cache = cache_dir / "base_rollouts.pkl"
    sequences, prompt_lengths, tokenizer = generate_base_rollouts(
        args.base_model,
        eval_dataset,
        rollouts_cache,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )

    # Step 3: Compute base model logprobs (cached)
    base_logprobs_cache = cache_dir / "base_logprobs.pkl"
    if base_logprobs_cache.exists():
        print(f"Loading cached base logprobs from: {base_logprobs_cache}")
    else:
        print("Computing base model log probabilities...")
        base_logprobs = compute_logprobs_for_sequences(
            args.base_model,
            sequences,
            prompt_lengths,
            tokenizer,
            batch_size=args.batch_size,
        )
        with open(base_logprobs_cache, "wb") as f:
            pickle.dump(base_logprobs, f)
        print(f"Cached base logprobs to: {base_logprobs_cache}")

    # Step 4: Collect all checkpoints from all experiment directories
    all_checkpoints = []  # (exp_name, ckpt_path)
    for exp_dir in checkpoint_dirs:
        exp_path = Path(exp_dir)
        exp_name = exp_path.name

        # Handle SFT hf_format subdirectory
        if (exp_path / "hf_format").exists():
            ckpt_base = exp_path / "hf_format"
        else:
            ckpt_base = exp_path

        ckpts = get_checkpoint_dirs(str(ckpt_base))
        print(f"Found {len(ckpts)} checkpoints in {exp_name}")
        for ckpt in ckpts:
            all_checkpoints.append((exp_name, str(ckpt)))

    print(f"\nTotal checkpoints to evaluate: {len(all_checkpoints)}")

    # Step 5: Build work items with round-robin GPU assignment
    work_items = []
    for i, (exp_name, ckpt_path) in enumerate(all_checkpoints):
        gpu_id = gpu_ids[i % len(gpu_ids)]
        work_items.append((
            ckpt_path,
            sequences,
            prompt_lengths,
            str(base_logprobs_cache),
            args.base_model,
            args.batch_size,
            gpu_id,
        ))

    # Step 6: Run in parallel
    print(f"\nEvaluating checkpoints across {len(gpu_ids)} GPUs...")
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=len(gpu_ids)) as pool:
        results_list = list(tqdm(
            pool.imap(worker_evaluate_checkpoint, work_items),
            total=len(work_items),
            desc="KL evaluation",
        ))

    # Step 7: Group results by experiment and save
    results_by_exp = {}
    for (exp_name, ckpt_path), (ckpt_name, result) in zip(all_checkpoints, results_list):
        if exp_name not in results_by_exp:
            results_by_exp[exp_name] = {}
        results_by_exp[exp_name][ckpt_name] = result
        print(f"  {exp_name}/{ckpt_name}: KL={result['kl_divergence']:.4f}")

    # Save per-experiment results
    for exp_name, results in results_by_exp.items():
        # Find the original experiment directory
        for exp_dir in checkpoint_dirs:
            if Path(exp_dir).name == exp_name:
                output_path = Path(exp_dir) / "all_checkpoints_kl.json"
                break
        else:
            output_path = output_dir / f"{exp_name}_kl.json"

        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved: {output_path}")

    print("\nDone!")


if __name__ == "__main__":
    main()
