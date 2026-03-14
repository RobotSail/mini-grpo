#!/usr/bin/env python3
"""
Parallel forward KL divergence evaluation across multiple GPUs.

Computes KL(π || π₀) = E_{x~π}[log π(x) - log π₀(x)]
where π is the trained model and π₀ is the base model,
with samples drawn from the trained model.

Key difference from reverse KL (eval_kl_parallel.py):
- Reverse KL: generate from base model ONCE, reuse across checkpoints
- Forward KL: generate from EACH checkpoint separately

Each worker:
1. Generates rollouts from checkpoint (vLLM)
2. Computes checkpoint logprobs (HF transformers, fp16)
3. Computes base model logprobs (HF transformers, fp16)
4. KL(π || π₀) = mean(log π - log π₀) over generated tokens
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
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


def format_prompts(eval_dataset: datasets.Dataset, tokenizer: AutoTokenizer) -> list[str]:
    """Format dataset into prompt strings for vLLM."""
    prompts = []
    for sample in eval_dataset:
        prompt = tokenizer.apply_chat_template(
            conversation=sample["messages"],
            tokenize=False,
            add_generation_prompt=True,
        )
        prompts.append(prompt)
    return prompts


def compute_logprobs_for_sequences(
    model_path: str,
    sequences: list[list[int]],
    prompt_lengths: list[int],
    tokenizer: AutoTokenizer,
    batch_size: int = 4,
    device: torch.device | None = None,
) -> list[torch.Tensor]:
    """Compute log probabilities for generated tokens in fp16."""
    if device is None:
        device = torch.device("cuda")

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map=device,
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
                    gen_logprobs = token_logprobs[j, gen_start:gen_end].cpu()
                    all_gen_logprobs.append(gen_logprobs)
                else:
                    all_gen_logprobs.append(torch.tensor([]))

    del model
    torch.cuda.empty_cache()

    return all_gen_logprobs


def compute_kl(p_logprobs: list[torch.Tensor], q_logprobs: list[torch.Tensor]) -> dict:
    """Compute KL(p || q) = E_p[log p - log q].

    When called as compute_kl(trained_lp, base_lp): forward KL(π || π₀)
    When called as compute_kl(base_lp, trained_lp): reverse KL(π₀ || π)
    """
    total_kl = 0.0
    total_tokens = 0
    all_seq_kl = []

    for p_lp, q_lp in zip(p_logprobs, q_logprobs):
        if len(p_lp) > 0 and len(q_lp) > 0:
            kl_per_token = p_lp - q_lp  # log p - log q
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


def extract_tokens(name: str) -> int:
    """Extract token count from checkpoint name."""
    if "tokens_" in name:
        try:
            return int(name.split("tokens_")[-1])
        except ValueError:
            return 0
    elif name.startswith("checkpoint-"):
        try:
            return int(name.split("-")[1])
        except ValueError:
            return 0
    return 0


def get_checkpoint_dirs(base_path: str) -> list[Path]:
    """Get all checkpoint directories sorted by token count."""
    base = Path(base_path)

    if not base.exists():
        raise ValueError(f"Path does not exist: {base_path}")

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
        elif name.startswith("checkpoint-"):
            try:
                return int(name.split("-")[1])
            except ValueError:
                return 0
        return 0

    return sorted(checkpoint_dirs, key=get_tokens)


def run_single_checkpoint(
    checkpoint_path: str,
    base_model_path: str,
    gpu_id: int,
    output_file: str,
    batch_size: int = 4,
    max_new_tokens: int = 256,
    max_samples: int | None = None,
    temperature: float = 0.0,
):
    """Evaluate forward KL for a single checkpoint. Runs in its own process."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    device = torch.device("cuda:0")

    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load dataset and format prompts
    eval_dataset = load_eval_dataset(DEFAULT_SYSTEM_MSG, max_samples)
    prompts = format_prompts(eval_dataset, tokenizer)

    ckpt_name = Path(checkpoint_path).name
    print(f"  [GPU {gpu_id}] Generating from {ckpt_name}...")

    # Step 1: Generate rollouts from checkpoint
    llm = LLM(
        model=checkpoint_path,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.4,
        dtype="float16",
    )
    sampling_params = SamplingParams(
        max_tokens=max_new_tokens,
        temperature=temperature,
        top_p=1.0,
    )
    outputs = llm.generate(prompts, sampling_params)

    sequences = []
    prompt_lengths = []
    for output in outputs:
        prompt_ids = output.prompt_token_ids
        gen_ids = output.outputs[0].token_ids
        sequences.append(list(prompt_ids) + list(gen_ids))
        prompt_lengths.append(len(prompt_ids))

    del llm
    torch.cuda.empty_cache()

    # Step 2: Compute checkpoint logprobs (trained model π)
    print(f"  [GPU {gpu_id}] Computing π logprobs for {ckpt_name}...")
    trained_logprobs = compute_logprobs_for_sequences(
        checkpoint_path, sequences, prompt_lengths,
        tokenizer, batch_size=batch_size, device=device,
    )

    # Step 3: Compute base model logprobs (base model π₀)
    print(f"  [GPU {gpu_id}] Computing π₀ logprobs for {ckpt_name}...")
    base_logprobs = compute_logprobs_for_sequences(
        base_model_path, sequences, prompt_lengths,
        tokenizer, batch_size=batch_size, device=device,
    )

    # Step 4: Forward KL(π || π₀) = E_π[log π - log π₀]
    forward_kl = compute_kl(trained_logprobs, base_logprobs)

    tokens = extract_tokens(ckpt_name)

    result = {
        ckpt_name: {
            "path": checkpoint_path,
            "tokens": tokens,
            "forward_kl": forward_kl["kl_divergence"],
            "forward_kl_std": forward_kl["kl_std"],
            "total_tokens": forward_kl["total_tokens"],
            "num_sequences": forward_kl["num_sequences"],
        }
    }

    with open(output_file, "w") as f:
        json.dump(result, f, indent=2)

    print(f"  [GPU {gpu_id}] {ckpt_name}: fwd_KL={forward_kl['kl_divergence']:.4f}")


def launch_subprocess(
    checkpoint_path: str,
    base_model_path: str,
    gpu_id: int,
    output_file: str,
    batch_size: int,
    max_new_tokens: int,
    max_samples: int | None,
    temperature: float,
) -> tuple[str, str, subprocess.CompletedProcess]:
    """Launch a subprocess to evaluate one checkpoint."""
    cmd = [
        sys.executable, __file__,
        "--mode", "single",
        "--checkpoint", checkpoint_path,
        "--base-model", base_model_path,
        "--gpu", str(gpu_id),
        "--output-file", output_file,
        "--batch-size", str(batch_size),
        "--max-new-tokens", str(max_new_tokens),
        "--temperature", str(temperature),
    ]
    if max_samples is not None:
        cmd.extend(["--max-samples", str(max_samples)])

    result = subprocess.run(cmd, capture_output=True, text=True)
    return checkpoint_path, output_file, result


def main():
    parser = argparse.ArgumentParser(
        description="Parallel forward KL divergence: KL(π || π₀) with samples from π"
    )
    parser.add_argument("--mode", type=str, default="parallel", choices=["parallel", "single"])

    # Single mode args
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--output-file", type=str, default=None)

    # Parallel mode args
    parser.add_argument(
        "--checkpoint-dirs",
        type=str,
        default=None,
        help="Comma-separated experiment directories containing checkpoints",
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default="0",
        help="Comma-separated GPU IDs (e.g., '0,1,2,3')",
    )

    # Shared args
    parser.add_argument("--base-model", type=str, default="Qwen/Qwen2-1.5B-Instruct")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0.0)

    args = parser.parse_args()

    # ---- Single checkpoint mode (called by subprocess) ----
    if args.mode == "single":
        run_single_checkpoint(
            checkpoint_path=args.checkpoint,
            base_model_path=args.base_model,
            gpu_id=args.gpu,
            output_file=args.output_file,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
            max_samples=args.max_samples,
            temperature=args.temperature,
        )
        return

    # ---- Parallel mode (orchestrator) ----
    if not args.checkpoint_dirs:
        parser.error("--checkpoint-dirs is required in parallel mode")

    gpu_ids = [int(g.strip()) for g in args.gpus.split(",")]
    checkpoint_dirs = [d.strip() for d in args.checkpoint_dirs.split(",")]

    # Collect all checkpoints
    all_checkpoints = []  # (exp_name, ckpt_path)
    for exp_dir in checkpoint_dirs:
        exp_path = Path(exp_dir)
        exp_name = exp_path.name

        if (exp_path / "hf_format").exists():
            ckpt_base = exp_path / "hf_format"
        else:
            ckpt_base = exp_path

        ckpts = get_checkpoint_dirs(str(ckpt_base))
        print(f"Found {len(ckpts)} checkpoints in {exp_name}")
        for ckpt in ckpts:
            all_checkpoints.append((exp_name, str(ckpt)))

    print(f"\nTotal checkpoints to evaluate: {len(all_checkpoints)}")
    print(f"Using {len(gpu_ids)} GPUs: {gpu_ids}")

    # Create temp directory for per-checkpoint results
    tmp_dir = tempfile.mkdtemp(prefix="forward_kl_")

    # Launch subprocesses with ThreadPoolExecutor (one thread per GPU)
    futures = {}
    with ThreadPoolExecutor(max_workers=len(gpu_ids)) as executor:
        for i, (exp_name, ckpt_path) in enumerate(all_checkpoints):
            gpu_id = gpu_ids[i % len(gpu_ids)]
            output_file = os.path.join(tmp_dir, f"result_{i}.json")

            future = executor.submit(
                launch_subprocess,
                ckpt_path, args.base_model, gpu_id, output_file,
                args.batch_size, args.max_new_tokens, args.max_samples, args.temperature,
            )
            futures[future] = (i, exp_name, ckpt_path, output_file)

        # Collect results
        results_by_exp = {}
        n_done = 0
        n_total = len(all_checkpoints)

        for future in as_completed(futures):
            idx, exp_name, ckpt_path, output_file = futures[future]
            ckpt_name = Path(ckpt_path).name
            n_done += 1

            try:
                _, _, proc = future.result()
                if proc.returncode != 0:
                    print(f"[{n_done}/{n_total}] FAILED {ckpt_name}: {proc.stderr[-500:]}")
                    continue

                with open(output_file) as f:
                    result = json.load(f)

                if exp_name not in results_by_exp:
                    results_by_exp[exp_name] = {}
                results_by_exp[exp_name].update(result)

                fwd_kl = list(result.values())[0]["forward_kl"]
                print(f"[{n_done}/{n_total}] {exp_name}/{ckpt_name}: fwd_KL={fwd_kl:.4f}")

            except Exception as e:
                print(f"[{n_done}/{n_total}] ERROR {ckpt_name}: {e}")

    # Save per-experiment results
    for exp_name, results in results_by_exp.items():
        for exp_dir in checkpoint_dirs:
            if Path(exp_dir).name == exp_name:
                output_path = Path(exp_dir) / "all_checkpoints_forward_kl.json"
                break
        else:
            output_path = Path(checkpoint_dirs[0]).parent / f"{exp_name}_forward_kl.json"

        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved: {output_path}")

    # Print summary
    print(f"\n{'='*60}")
    print("Forward KL Summary: KL(π || π₀)")
    print(f"{'='*60}")
    for exp_name, results in results_by_exp.items():
        print(f"\n{exp_name}:")
        for ckpt_name, r in sorted(results.items(), key=lambda x: x[1]["tokens"]):
            print(f"  {ckpt_name}: fwd_KL={r['forward_kl']:.4f}")

    # Cleanup
    import shutil
    shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
