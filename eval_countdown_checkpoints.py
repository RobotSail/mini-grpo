#!/usr/bin/env python
"""
Evaluate countdown GRPO checkpoints: accuracy + forward KL divergence.

For each checkpoint:
  - Accuracy: generate completions via vLLM, score with countdown_reward_fn
  - Forward KL: compute KL(base || checkpoint) on base model rollouts

Base model rollouts and logprobs are computed once and cached.

Usage:
    python eval_countdown_checkpoints.py \
        --checkpoint-dir /mnt/nvme2n1/checkpoints/countdown/grpo/adamw \
        --base-model Qwen/Qwen2-1.5B-Instruct \
        --test-data generated_data_thinking/countdown_grpo_test.jsonl \
        --num-samples 2000 \
        --gpus 0,1,2,3,4,5,6,7 \
        --output-dir eval_results/adamw
"""

import argparse
import json
import os
import pickle
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import torch
import torch.nn.functional as F
import datasets
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

from countdown_utils import countdown_reward_fn, R1_ASSISTANT_PREFIX


# ── Helpers ──────────────────────────────────────────────────────────────────

def get_sorted_checkpoints(checkpoint_dir: str) -> list[Path]:
    """Return checkpoint dirs sorted by token count (ascending)."""
    base = Path(checkpoint_dir)
    dirs = []
    for d in base.iterdir():
        if not d.is_dir():
            continue
        name = d.name
        if name == "checkpoint-initial":
            dirs.append((0, d))
        elif name.startswith("checkpoint-"):
            try:
                n = int(name.split("-")[1])
                dirs.append((n, d))
            except (ValueError, IndexError):
                continue
    dirs.sort(key=lambda x: x[0])
    return [d for _, d in dirs]


def load_test_prompts(test_data_path: str, num_samples: int, tokenizer) -> list[dict]:
    """Load test data and format prompts with the R1-style chat template."""
    ds = datasets.load_dataset("json", data_files=test_data_path, split="train")
    if num_samples > 0 and num_samples < len(ds):
        ds = ds.select(range(num_samples))

    prompts = []
    for sample in ds:
        messages = sample["messages"]
        has_assistant_prefix = messages and messages[-1]["role"] == "assistant"

        prompt_text = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=not has_assistant_prefix,
            continue_final_message=has_assistant_prefix,
            tokenize=False,
        )
        prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)

        prompts.append({
            "prompt_text": prompt_text,
            "prompt_ids": prompt_ids,
            "prompt_len": len(prompt_ids),
            "answer": sample["answer"],
            "numbers": sample.get("numbers"),
            "problem": sample.get("problem", ""),
        })
    return prompts


# ── Base model rollouts ──────────────────────────────────────────────────────

def generate_base_rollouts(
    base_model: str,
    prompts: list[dict],
    cache_dir: Path,
    gpu: int = 0,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
) -> list[dict]:
    """Generate rollouts from the base model via vLLM. Cached to disk."""
    cache_file = cache_dir / "base_rollouts.pkl"
    if cache_file.exists():
        print(f"Loading cached base rollouts from {cache_file}")
        with open(cache_file, "rb") as f:
            return pickle.load(f)

    print(f"Generating base rollouts ({len(prompts)} samples) on GPU {gpu}...")
    from vllm import LLM, SamplingParams

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    llm = LLM(
        model=base_model,
        gpu_memory_utilization=0.45,
        dtype="bfloat16",
        max_model_len=2048,
        enforce_eager=True,
    )

    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_new_tokens,
        top_p=1.0,
    )

    prompt_texts = [p["prompt_text"] for p in prompts]
    outputs = llm.generate(prompt_texts, sampling_params)

    rollouts = []
    for prompt, output in zip(prompts, outputs):
        generated_text = output.outputs[0].text
        generated_ids = list(output.outputs[0].token_ids)
        full_ids = prompt["prompt_ids"] + generated_ids
        rollouts.append({
            "prompt_ids": prompt["prompt_ids"],
            "prompt_len": prompt["prompt_len"],
            "generated_text": generated_text,
            "generated_ids": generated_ids,
            "full_ids": full_ids,
            "answer": prompt["answer"],
            "numbers": prompt["numbers"],
        })

    del llm
    torch.cuda.empty_cache()

    cache_dir.mkdir(parents=True, exist_ok=True)
    with open(cache_file, "wb") as f:
        pickle.dump(rollouts, f)
    print(f"Cached base rollouts to {cache_file}")

    return rollouts


# ── Logprob computation ──────────────────────────────────────────────────────

@torch.no_grad()
def compute_logprobs(
    model_path: str,
    rollouts: list[dict],
    cache_path: Path | None = None,
    gpu: int = 0,
    batch_size: int = 8,
) -> list[torch.Tensor]:
    """Compute per-token logprobs for the completion portion of each rollout.

    Returns a list of 1-D float tensors, one per rollout, containing
    log p(token_t | token_{<t}) for each generated token.
    """
    if cache_path and cache_path.exists():
        print(f"Loading cached logprobs from {cache_path}")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    print(f"Computing logprobs for {len(rollouts)} sequences using {model_path}...")
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device)

    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2",
    ).to(device).eval()

    all_logprobs = []
    for i in tqdm(range(0, len(rollouts), batch_size), desc="logprobs"):
        batch = rollouts[i : i + batch_size]

        # Pad sequences
        max_len = max(len(r["full_ids"]) for r in batch)
        input_ids = torch.zeros(len(batch), max_len, dtype=torch.long, device=device)
        attention_mask = torch.zeros(len(batch), max_len, dtype=torch.long, device=device)
        for j, r in enumerate(batch):
            seq = r["full_ids"]
            input_ids[j, : len(seq)] = torch.tensor(seq, dtype=torch.long)
            attention_mask[j, : len(seq)] = 1

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        log_probs = F.log_softmax(outputs.logits, dim=-1)

        for j, r in enumerate(batch):
            prompt_len = r["prompt_len"]
            seq_len = len(r["full_ids"])
            # Logprobs for generated tokens: p(token_t | token_{<t})
            # logits at position t-1 predict token at position t
            gen_start = prompt_len - 1  # logit index for first generated token
            gen_end = seq_len - 1
            if gen_start >= gen_end:
                all_logprobs.append(torch.tensor([], dtype=torch.float32))
                continue

            token_ids = input_ids[j, prompt_len:seq_len]  # generated token ids
            token_logprobs = log_probs[j, gen_start:gen_end]  # logits predicting those tokens
            per_token_lp = token_logprobs.gather(1, token_ids.unsqueeze(1)).squeeze(1)
            all_logprobs.append(per_token_lp.float().cpu())

    del model
    torch.cuda.empty_cache()

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(all_logprobs, f)
        print(f"Cached logprobs to {cache_path}")

    return all_logprobs


# ── Forward KL ───────────────────────────────────────────────────────────────

def compute_forward_kl(
    checkpoint_logprobs: list[torch.Tensor],
    base_logprobs: list[torch.Tensor],
) -> dict:
    """Compute forward KL(checkpoint || base) = E_checkpoint[log p_ckpt - log p_base].

    Samples are drawn from the checkpoint distribution (checkpoint-generated text).
    Averaged per-token, then averaged across sequences.
    """
    seq_kls = []
    total_kl = 0.0
    total_tokens = 0
    for clp, blp in zip(checkpoint_logprobs, base_logprobs):
        n = min(len(clp), len(blp))
        if n == 0:
            continue
        kl_per_token = clp[:n] - blp[:n]  # log p_ckpt - log p_base
        seq_kl = kl_per_token.mean().item()
        seq_kls.append(seq_kl)
        total_kl += kl_per_token.sum().item()
        total_tokens += n

    return {
        "kl_mean": total_kl / max(total_tokens, 1),
        "kl_seq_mean": sum(seq_kls) / max(len(seq_kls), 1) if seq_kls else 0.0,
        "num_sequences": len(seq_kls),
        "total_tokens": total_tokens,
    }


# ── Accuracy evaluation ─────────────────────────────────────────────────────

def generate_and_score(
    model_path: str,
    prompts: list[dict],
    gpu: int = 0,
    max_new_tokens: int = 512,
    temperature: float = 0.0,
) -> tuple[dict, list[dict]]:
    """Generate from a checkpoint, score accuracy, and return rollouts for KL.

    Expects CUDA_VISIBLE_DEVICES to already be set by the caller.

    Returns:
        (accuracy_dict, rollouts) where rollouts have prompt_ids, prompt_len,
        full_ids for logprob computation.
    """
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=model_path,
        gpu_memory_utilization=0.45,
        dtype="bfloat16",
        max_model_len=2048,
        enforce_eager=True,
    )

    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_new_tokens,
    )

    prompt_texts = [p["prompt_text"] for p in prompts]
    outputs = llm.generate(prompt_texts, sampling_params)

    del llm
    torch.cuda.empty_cache()

    total = len(outputs)
    correct = 0
    formatted = 0
    parsable = 0
    rollouts = []

    for prompt, output in zip(prompts, outputs):
        response_text = output.outputs[0].text
        generated_ids = list(output.outputs[0].token_ids)

        result = countdown_reward_fn(
            response_text, prompt["answer"],
            {"numbers": prompt["numbers"]},
        )
        if result.get("has_format", False):
            formatted += 1
        if result.get("is_parsable", False):
            parsable += 1
        if result.get("is_correct", False):
            correct += 1

        rollouts.append({
            "prompt_ids": prompt["prompt_ids"],
            "prompt_len": prompt["prompt_len"],
            "full_ids": prompt["prompt_ids"] + generated_ids,
        })

    acc = {
        "correct": correct,
        "formatted": formatted,
        "parsable": parsable,
        "total": total,
        "correct_rate": correct / max(total, 1),
        "format_rate": formatted / max(total, 1),
        "parsable_rate": parsable / max(total, 1),
    }
    return acc, rollouts


# ── Per-checkpoint evaluation (subprocess mode) ────────────────────────────

def eval_single_checkpoint_cli(
    checkpoint: str,
    base_model: str,
    test_data: str,
    num_samples: int,
    max_new_tokens: int,
    cache_dir: str,
    gpu: int,
    output_file: str,
):
    """Run as a subprocess: evaluate one checkpoint, write result to JSON file.

    For each checkpoint:
      1. Generate completions via vLLM → accuracy scores + rollouts
      2. Compute checkpoint logprobs on its own completions
      3. Compute base model logprobs on the same completions
      4. KL(checkpoint || base) = E_ckpt[log p_ckpt - log p_base]
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    torch.cuda.set_device(0)

    ckpt_name = Path(checkpoint).name
    if ckpt_name == "checkpoint-initial":
        step = 0
    else:
        try:
            step = int(ckpt_name.split("-")[1])
        except (ValueError, IndexError):
            step = -1

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    prompts = load_test_prompts(test_data, num_samples, tokenizer)

    # 1. Generate from checkpoint → accuracy + rollouts
    acc, rollouts = generate_and_score(
        checkpoint, prompts, gpu=0, max_new_tokens=max_new_tokens,
    )

    # 2. Compute checkpoint logprobs on its own generations
    ckpt_logprobs = compute_logprobs(checkpoint, rollouts, cache_path=None, gpu=0)

    # 3. Compute base model logprobs on the same generations
    base_logprobs = compute_logprobs(base_model, rollouts, cache_path=None, gpu=0)

    # 4. KL(checkpoint || base)
    kl = compute_forward_kl(ckpt_logprobs, base_logprobs)

    result = {
        "checkpoint": ckpt_name,
        "tokens": step,
        "accuracy": acc,
        "kl": kl,
    }
    with open(output_file, "w") as f:
        json.dump(result, f)

    print(
        f"[GPU {gpu}] {ckpt_name}: correct={acc['correct_rate']:.3f}, "
        f"format={acc['format_rate']:.3f}, kl={kl['kl_mean']:.4f}"
    )


def launch_checkpoint_eval(
    checkpoint: str, base_model: str, test_data: str, num_samples: int,
    max_new_tokens: int, cache_dir: str, gpu: int, output_file: str,
) -> subprocess.Popen:
    """Launch a subprocess to evaluate one checkpoint."""
    cmd = [
        sys.executable, __file__, "--eval-single",
        "--checkpoint", checkpoint,
        "--base-model", base_model,
        "--test-data", test_data,
        "--num-samples", str(num_samples),
        "--max-new-tokens", str(max_new_tokens),
        "--cache-dir", cache_dir,
        "--gpu", str(gpu),
        "--single-output", output_file,
    ]
    return subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stderr)


# ── Plotting ─────────────────────────────────────────────────────────────────

def plot_results(results: list[dict], output_path: str, title: str = ""):
    """Plot accuracy and KL vs training tokens."""
    results = sorted(results, key=lambda r: r["tokens"])

    tokens = [r["tokens"] / 1e6 for r in results]  # in millions
    correct_rates = [r["accuracy"]["correct_rate"] * 100 for r in results]
    format_rates = [r["accuracy"]["format_rate"] * 100 for r in results]
    kl_means = [r["kl"]["kl_mean"] for r in results]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold")

    # Accuracy plot
    ax1.plot(tokens, correct_rates, "o-", color="tab:green", label="Correct %", markersize=3)
    ax1.plot(tokens, format_rates, "s-", color="tab:blue", label="Format %", markersize=2, alpha=0.6)
    ax1.set_ylabel("Rate (%)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_title("Accuracy & Format Rate")

    # KL plot
    ax2.plot(tokens, kl_means, "o-", color="tab:red", markersize=3)
    ax2.set_ylabel("Forward KL (nats/token)")
    ax2.set_xlabel("Training Tokens (millions)")
    ax2.grid(True, alpha=0.3)
    ax2.set_title("Forward KL Divergence (base || checkpoint)")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot to {output_path}")
    plt.close()


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate countdown checkpoints")
    parser.add_argument("--checkpoint-dir", default=None, help="Directory containing checkpoints")
    parser.add_argument("--base-model", default="Qwen/Qwen2-1.5B-Instruct")
    parser.add_argument("--test-data", default="generated_data_thinking/countdown_grpo_test.jsonl")
    parser.add_argument("--num-samples", type=int, default=2000,
                        help="Number of test samples (0 = all)")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7",
                        help="Comma-separated GPU IDs for parallel eval")
    parser.add_argument("--output-dir", default=None, help="Output directory for results and plots")
    parser.add_argument("--cache-dir", default=None, help="Cache dir (default: output-dir/cache)")
    parser.add_argument("--title", default=None, help="Plot title")
    # Single-checkpoint subprocess mode
    parser.add_argument("--eval-single", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--checkpoint", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--gpu", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--single-output", default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    # ── Single-checkpoint subprocess mode ──
    if args.eval_single:
        eval_single_checkpoint_cli(
            checkpoint=args.checkpoint,
            base_model=args.base_model,
            test_data=args.test_data,
            num_samples=args.num_samples,
            max_new_tokens=args.max_new_tokens,
            cache_dir=args.cache_dir,
            gpu=args.gpu,
            output_file=args.single_output,
        )
        return

    # ── Main orchestrator mode ──
    assert args.checkpoint_dir, "--checkpoint-dir is required"
    assert args.output_dir, "--output-dir is required"

    gpu_ids = [int(g) for g in args.gpus.split(",")]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir) if args.cache_dir else output_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = output_dir / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # 1. Verify test data exists
    print(f"Test data: {args.test_data}")
    print(f"Base model: {args.base_model}")
    print(f"Num samples: {args.num_samples}")

    # 2. Get sorted checkpoints
    checkpoints = get_sorted_checkpoints(args.checkpoint_dir)
    print(f"Found {len(checkpoints)} checkpoints in {args.checkpoint_dir}")

    # 5. Check existing results (resume support)
    results_path = output_dir / "results.json"
    existing_results = {}
    if results_path.exists():
        with open(results_path) as f:
            for r in json.load(f):
                existing_results[r["checkpoint"]] = r
        print(f"Loaded {len(existing_results)} existing results, will skip those")

    remaining = [c for c in checkpoints if c.name not in existing_results]
    print(f"{len(remaining)} checkpoints to evaluate")

    if remaining:
        # 6. Evaluate in parallel via subprocesses (one per GPU)
        # Process in waves: each wave fills all GPUs
        for wave_start in range(0, len(remaining), len(gpu_ids)):
            wave = remaining[wave_start : wave_start + len(gpu_ids)]
            procs = []
            result_files = []

            for i, ckpt in enumerate(wave):
                gpu = gpu_ids[i % len(gpu_ids)]
                result_file = str(tmp_dir / f"{ckpt.name}.json")
                result_files.append(result_file)

                proc = launch_checkpoint_eval(
                    checkpoint=str(ckpt),
                    base_model=args.base_model,
                    test_data=args.test_data,
                    num_samples=args.num_samples,
                    max_new_tokens=args.max_new_tokens,
                    cache_dir=str(cache_dir),
                    gpu=gpu,
                    output_file=result_file,
                )
                procs.append(proc)
                print(f"Launched {ckpt.name} on GPU {gpu} (pid {proc.pid})")

            # Wait for all procs in this wave
            for proc in procs:
                proc.wait()

            # Collect results from this wave
            for rf in result_files:
                if Path(rf).exists():
                    with open(rf) as f:
                        r = json.load(f)
                    existing_results[r["checkpoint"]] = r

            # Save incrementally
            with open(results_path, "w") as f:
                json.dump(list(existing_results.values()), f, indent=2)

            done = len(existing_results)
            total = len(checkpoints)
            print(f"Progress: {done}/{total} checkpoints evaluated")

    # 7. Final results
    results = list(existing_results.values())
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {len(results)} results to {results_path}")

    # 8. Plot
    if results:
        title = args.title or Path(args.checkpoint_dir).name
        plot_results(results, str(output_dir / "accuracy_kl.png"), title=title)
    else:
        print("No results to plot.")


if __name__ == "__main__":
    main()
