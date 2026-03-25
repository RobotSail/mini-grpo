#!/usr/bin/env python3
"""
Full-vocabulary forward KL divergence evaluation with 5-shot ICL.

Computes the true forward KL divergence over the entire vocabulary:
    D_KL(t) = Σ_{v∈V} π_0(v|x,y_{<t}) [log π_0(v|x,y_{<t}) - log π(v|x,y_{<t})]

Uses the standard 5 GSM8K few-shot examples (from the train split, same as
lm-eval-harness / opencompass), reformatted as multi-turn user/assistant
exchanges with <answer>...</answer> tags.

KL is computed only on the generated response tokens (not the few-shot prompt),
matching what happens during training.

Usage:
    python eval_kl_full_vocab.py \
        --base-model Qwen/Qwen2-1.5B-Instruct \
        --checkpoint-paths path/to/ckpt1,path/to/ckpt2 \
        --checkpoint-names name1,name2 \
        --gpus 0,1,2,3 \
        --output results_full_vocab_kl.json
"""

import argparse
import json
import multiprocessing as mp
import os
import pickle
import re
from pathlib import Path

import datasets
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_SYSTEM_MSG = (
    "You are a helpful math assistant. Always provide your final numerical answer "
    "inside of the <answer>...</answer> tags, e.g.: <answer>42</answer>"
)

NUM_FEWSHOT = 5


def _strip_calc_annotations(text: str) -> str:
    """Remove <<calc=result>> annotations from GSM8K answers."""
    return re.sub(r"<<.*?>>", "", text)


def _convert_answer_format(answer: str) -> str:
    """Convert '#### X' format to '<answer>X</answer>' format."""
    answer = _strip_calc_annotations(answer)
    answer = re.sub(r"####\s*(.+)", r"<answer>\1</answer>", answer)
    return answer.strip()


def build_fewshot_messages() -> list[dict]:
    """Build the 5-shot ICL messages from GSM8K train split."""
    train_ds = datasets.load_dataset("openai/gsm8k", "main", split="train")

    messages = [{"role": "system", "content": DEFAULT_SYSTEM_MSG}]

    for i in range(NUM_FEWSHOT):
        question = train_ds[i]["question"]
        answer = _convert_answer_format(train_ds[i]["answer"])
        messages.append({"role": "user", "content": question})
        messages.append({"role": "assistant", "content": answer})

    return messages


def load_eval_dataset(max_samples: int | None = None) -> datasets.Dataset:
    """Load GSM8K test split with 5-shot ICL prefix as multi-turn messages."""
    ds = datasets.load_dataset("openai/gsm8k", "main", split="test")
    fewshot_prefix = build_fewshot_messages()

    data = []
    for item in ds:
        messages = fewshot_prefix + [{"role": "user", "content": item["question"]}]
        data.append({"messages": messages, "answer": item["answer"]})

    ds = datasets.Dataset.from_list(data)
    if max_samples:
        ds = ds.select(range(min(max_samples, len(ds))))
    return ds


def generate_base_rollouts(
    base_model_path: str,
    eval_dataset: datasets.Dataset,
    cache_path: Path,
    max_new_tokens: int = 256,
    temperature: float = 0.0,
) -> tuple[list[list[int]], list[int], AutoTokenizer]:
    """Generate sequences from base model, with caching."""
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
    from vllm import LLM, SamplingParams
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

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump({
            "sequences": all_sequences,
            "prompt_lengths": all_prompt_lengths,
        }, f)
    print(f"Cached rollouts to: {cache_path}")

    return all_sequences, all_prompt_lengths, tokenizer


def compute_full_vocab_kl(
    base_model_path: str,
    checkpoint_path: str,
    sequences: list[list[int]],
    prompt_lengths: list[int],
    tokenizer: AutoTokenizer,
    batch_size: int = 4,
    device: torch.device | None = None,
) -> dict:
    """
    Compute full-vocabulary forward KL: D_KL(π_0 || π) at each generated position.

    KL is computed only on generated tokens (after the prompt), matching
    how KL is computed during RL training.
    """
    if device is None:
        device = torch.device("cuda")

    print(f"  Loading base model: {base_model_path}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    base_model.eval()

    print(f"  Loading checkpoint: {checkpoint_path}")
    ckpt_model = AutoModelForCausalLM.from_pretrained(
        checkpoint_path,
        torch_dtype="auto",
        device_map=device,
    )
    ckpt_model.eval()

    all_seq_kl = []
    total_kl = 0.0
    total_tokens = 0

    with torch.no_grad():
        for i in tqdm(range(0, len(sequences), batch_size), desc="Full-vocab KL", leave=False):
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

            # Forward pass through both models
            base_outputs = base_model(input_ids=input_ids, attention_mask=attention_mask)
            ckpt_outputs = ckpt_model(input_ids=input_ids, attention_mask=attention_mask)

            # Log-softmax over full vocabulary at each position (in fp32)
            base_log_probs = F.log_softmax(base_outputs.logits[:, :-1].float(), dim=-1)
            ckpt_log_probs = F.log_softmax(ckpt_outputs.logits[:, :-1].float(), dim=-1)

            # Full-vocab KL at each position:
            # KL(t) = Σ_v p_base(v) * (log p_base(v) - log p_ckpt(v))
            base_probs = base_log_probs.exp()
            kl_per_position = (base_probs * (base_log_probs - ckpt_log_probs)).sum(dim=-1)

            for j, prompt_len in enumerate(batch_prompt_lens):
                seq_len = sum(attention_masks[j])
                # Only score generated tokens (after prompt)
                gen_start = prompt_len - 1  # logits shifted by 1
                gen_end = seq_len - 1

                if gen_end > gen_start:
                    gen_kl = kl_per_position[j, gen_start:gen_end]
                    seq_kl = gen_kl.mean().item()
                    all_seq_kl.append(seq_kl)
                    total_kl += gen_kl.sum().item()
                    total_tokens += gen_end - gen_start

            del base_outputs, ckpt_outputs, base_log_probs, ckpt_log_probs, base_probs, kl_per_position

    del base_model, ckpt_model
    torch.cuda.empty_cache()

    mean_kl = total_kl / total_tokens if total_tokens > 0 else 0.0
    kl_tensor = torch.tensor(all_seq_kl)
    std_kl = kl_tensor.std().item() if len(all_seq_kl) > 1 else 0.0
    median_kl = kl_tensor.median().item() if all_seq_kl else 0.0

    return {
        "full_vocab_kl": mean_kl,
        "full_vocab_kl_std": std_kl,
        "full_vocab_kl_median": median_kl,
        "total_tokens": total_tokens,
        "num_sequences": len(all_seq_kl),
    }


def worker_fn(args: tuple) -> tuple[str, dict]:
    """Worker function for multiprocessing."""
    (
        name,
        base_model_path,
        checkpoint_path,
        sequences,
        prompt_lengths,
        tokenizer_path,
        batch_size,
        gpu_id,
    ) = args

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    device = torch.device("cuda:0")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"\n[GPU {gpu_id}] Evaluating: {name}")
    result = compute_full_vocab_kl(
        base_model_path=base_model_path,
        checkpoint_path=checkpoint_path,
        sequences=sequences,
        prompt_lengths=prompt_lengths,
        tokenizer=tokenizer,
        batch_size=batch_size,
        device=device,
    )

    return name, result


def main():
    parser = argparse.ArgumentParser(description="Full-vocabulary forward KL divergence evaluation (5-shot ICL)")
    parser.add_argument("--base-model", type=str, default="Qwen/Qwen2-1.5B-Instruct")
    parser.add_argument(
        "--checkpoint-paths", type=str, required=True,
        help="Comma-separated checkpoint paths",
    )
    parser.add_argument(
        "--checkpoint-names", type=str, required=True,
        help="Comma-separated names for each checkpoint",
    )
    parser.add_argument("--gpus", type=str, required=True, help="Comma-separated GPU IDs")
    parser.add_argument("--output", type=str, required=True, help="Output JSON path")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--cache-dir", type=str, default="kl_cache", help="Cache directory for base rollouts")

    args = parser.parse_args()

    gpu_ids = [int(g.strip()) for g in args.gpus.split(",")]
    ckpt_paths = [p.strip() for p in args.checkpoint_paths.split(",")]
    ckpt_names = [n.strip() for n in args.checkpoint_names.split(",")]

    assert len(ckpt_paths) == len(ckpt_names), "Must have same number of paths and names"

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Load dataset with 5-shot ICL and generate base rollouts
    print("Loading evaluation dataset with 5-shot ICL prompts...")
    eval_dataset = load_eval_dataset(args.max_samples)
    print(f"Using {len(eval_dataset)} test samples")

    # Print an example prompt for verification
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    example_prompt = tokenizer.apply_chat_template(
        eval_dataset[0]["messages"], tokenize=False, add_generation_prompt=True,
    )
    print(f"\n--- Example prompt (first 500 chars) ---\n{example_prompt[:500]}\n---\n")

    rollouts_cache = cache_dir / "base_rollouts_5shot.pkl"
    sequences, prompt_lengths, tokenizer = generate_base_rollouts(
        args.base_model,
        eval_dataset,
        rollouts_cache,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )

    print(f"Average prompt length: {sum(prompt_lengths) / len(prompt_lengths):.0f} tokens")
    gen_lens = [len(s) - p for s, p in zip(sequences, prompt_lengths)]
    print(f"Average generation length: {sum(gen_lens) / len(gen_lens):.0f} tokens")

    # Step 2: Build work items with round-robin GPU assignment
    work_items = []
    for i, (name, path) in enumerate(zip(ckpt_names, ckpt_paths)):
        gpu_id = gpu_ids[i % len(gpu_ids)]
        work_items.append((
            name,
            args.base_model,
            path,
            sequences,
            prompt_lengths,
            args.base_model,
            args.batch_size,
            gpu_id,
        ))

    # Step 3: Run in parallel
    print(f"\nEvaluating {len(work_items)} checkpoints across {len(gpu_ids)} GPUs...")
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=min(len(gpu_ids), len(work_items))) as pool:
        results_list = list(tqdm(
            pool.imap(worker_fn, work_items),
            total=len(work_items),
            desc="Full-vocab KL evaluation",
        ))

    # Step 4: Collect and save results
    results = {}
    for name, result in results_list:
        results[name] = result
        print(f"  {name}: full_vocab_kl={result['full_vocab_kl']:.6f} "
              f"(±{result['full_vocab_kl_std']:.6f}), "
              f"median={result['full_vocab_kl_median']:.6f}, "
              f"tokens={result['total_tokens']}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to: {output_path}")


if __name__ == "__main__":
    main()
