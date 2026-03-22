# seed here for reproducibility
import os

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

from trainer import RSTrainer

from typing import Any
import requests
from transformers import GenerationConfig
import json
import random
import subprocess
import sys
import time
import httpx
from typer import Typer
import typer
import re
import pydantic
import datasets
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Qwen2Tokenizer,
    PreTrainedModel,
)
import torch
from torch.optim import AdamW
import os
from IPython import embed
from tqdm import tqdm
import numpy as np
import torch.distributed as dist

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None

from instructlab.training.data_process import (
    configure_tokenizer,
)

from data_utils import (
    generate_dataset,
    dataset_from_groups,
    create_grpo_data_loader,
    load_gsm8k,
    split_batch_into_microbatches,
)
from utils import preview_tokenization, display_scorecard, set_determinism
from optimizers import create_optimizer, create_fsdp2_muon_optimizer
from type_defs import (
    Problem,
    SamplingParams,
    TokenSample,
    RolloutResult,
    Sample,
    TrainingComponents,
    Hyperparameters,
)

import logging
from rich.logging import RichHandler
from rich.console import Console

# Create a rich console for consistent formatting
console = Console()

# Configure rich logging handler
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=console, rich_tracebacks=True, markup=True)]
)

# Create a logger that can be imported by other modules
logger = logging.getLogger("mini-grpo")


# Regex pattern to match <answer>...</answer> tags
answer_pattern = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)


app = Typer()


def send_chat_completion(
    prompt: str,
    system_prompt: str,
    model: str = "qwen/Qwen2-1.5B-Instruct",
    base_url: str = "http://localhost:8000/v1",
    temperature: float = 0.7,
    max_tokens: int = 512,
):
    """Send a chat completion request to vLLM server."""
    url = f"{base_url}/chat/completions"

    headers = {"Content-Type": "application/json"}

    data = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    response = requests.post(url, headers=headers, json=data)
    response.raise_for_status()

    return response.json()


def parse_number(text: str) -> float:
    """
    Parse a string into a float, handling common formats from GSM8K answers.

    Handles:
    - Whitespace (leading/trailing/internal)
    - Percentage signs (42% -> 42.0)
    - Currency symbols ($100, EUR50, etc.)
    - Comma separators (1,000,000 -> 1000000)
    - Negative numbers (-42, negative prefix)
    - Decimal numbers (3.14)

    Returns: float
    Raises: ValueError if no valid number can be parsed
    """
    if not text or not isinstance(text, str):
        raise ValueError(f"Empty or invalid input: {text}")

    # Strip whitespace
    text = text.strip()

    # Remove currency symbols ($, EUR, GBP, JPY, etc.)
    text = re.sub(r"[$\u20AC\u00A3\u00A5\u20B9]", "", text)

    # Remove percentage sign (keep the number)
    text = text.replace("%", "")

    # Remove commas (thousand separators)
    text = text.replace(",", "")

    # Strip remaining whitespace after removals
    text = text.strip()

    # Check for digits
    if not any(c.isdigit() for c in text):
        raise ValueError(f"No digits found in answer: {text}")

    # Extract the numeric portion (handles cases like "42 dollars" -> "42")
    match = re.search(r"-?\d+\.?\d*", text)
    if not match:
        raise ValueError(f"Could not extract number from: {text}")

    return float(match.group())


@app.command()
def generate_data(
    # system_msg: str,
    system_msg="You are a helpful math assistant. Always provide your final numerical answer inside of the <answer>...</answer> tags, e.g.: <answer>42</answer>",
    num_problems: int = 20,
    min_num: int = -100,
    max_num: int = 100,
    seed: int = 42,
    model_name: str = "qwen/Qwen2-1.5B-Instruct",
    output_dir: str = "generated_data",
    test_split: float = 0.0,
    max_seq_len: int = 8192,
):
    # this is the dataset
    dataset: datasets.Dataset = generate_dataset(
        system_msg=system_msg,
        seed=seed,
        num_problems=num_problems,
        min_num=min_num,
        max_num=max_num,
    )
    if test_split > 0:
        dataset_dict = dataset.train_test_split(test_split)
        train, test = dataset_dict["train"], dataset_dict["test"]
    else:
        train = dataset
        test = None
    os.makedirs(output_dir, exist_ok=True)

    # write out training data
    train_path = os.path.join(output_dir, "train.jsonl")
    train.to_json(train_path)
    typer.secho(
        f"✓ Generated {len(train)} training examples",
        fg=typer.colors.GREEN,
    )
    typer.secho(
        f"✓ Saved training data to '{train_path}'",
        fg=typer.colors.BLUE,
    )

    # write out test data if it exists
    if test:
        test_path = os.path.join(output_dir, "test.jsonl")
        test.to_json(test_path)
        typer.secho(
            f"✓ Generated {len(test)} test examples",
            fg=typer.colors.GREEN,
        )
        typer.secho(
            f"✓ Saved test data to '{test_path}'",
            fg=typer.colors.BLUE,
        )


@app.command()
def generate_gsm8k(
    system_msg: str = typer.Option(
        "You are a helpful math assistant. Always provide your final numerical answer inside of the <answer>...</answer> tags, e.g.: <answer>42</answer>",
        "--system-msg",
        help="System message to use for the chat format",
    ),
    seed: int = typer.Option(67, help="Random seed for train/test split"),
    output_dir: str = typer.Option("generated_data", help="Directory to save the dataset"),
    test_split: float = typer.Option(0.0, help="Fraction of data to use for test set"),
):
    """Load GSM8K dataset and save it in the format expected by this repo."""
    train_dataset, test_dataset = load_gsm8k(
        system_msg=system_msg,
        eval_split=test_split,
        seed=seed,
    )

    os.makedirs(output_dir, exist_ok=True)

    # Write out training data
    train_path = os.path.join(output_dir, "gsm8k_train.jsonl")
    train_dataset.to_json(train_path)
    typer.secho(
        f"✓ Generated {len(train_dataset)} training examples from GSM8K",
        fg=typer.colors.GREEN,
    )
    typer.secho(
        f"✓ Saved training data to '{train_path}'",
        fg=typer.colors.BLUE,
    )

    # Write out test data if it exists
    if test_dataset:
        test_path = os.path.join(output_dir, "gsm8k_test.jsonl")
        test_dataset.to_json(test_path)
        typer.secho(
            f"✓ Generated {len(test_dataset)} test examples from GSM8K",
            fg=typer.colors.GREEN,
        )
        typer.secho(
            f"✓ Saved test data to '{test_path}'",
            fg=typer.colors.BLUE,
        )


def _clean_calculator_annotations(text: str) -> str:
    """Remove <<a op b=c>> calculator annotation patterns from GSM8K answers."""
    return re.sub(r"<<[^>]+>>", "", text)


def _reformat_to_answer_tags(answer: str) -> str:
    """Replace GSM8K's '#### <ans>' format with '<answer>{ans}</answer>' tags."""
    pattern = r"####\s*(.+)$"
    match = re.search(pattern, answer, re.MULTILINE)
    if match:
        final_ans = match.group(1).strip()
        return re.sub(pattern, f"<answer>{final_ans}</answer>", answer, flags=re.MULTILINE)
    return answer


def _create_sft_message(question: str, answer: str, system_msg: str) -> dict:
    """Create a single SFT sample in messages format with answer for evaluation."""
    cleaned = _clean_calculator_annotations(answer)
    reformatted = _reformat_to_answer_tags(cleaned)

    # Extract the numerical answer for evaluation
    # GSM8K uses "#### <ans>" format for final answers
    pattern = r"####\s*(.+)$"
    match = re.search(pattern, answer, re.MULTILINE)
    if match:
        final_ans = match.group(1).strip().replace(",", "")
    else:
        # Fallback - no numerical answer found
        final_ans = "0"

    try:
        numerical_answer = float(final_ans)
    except ValueError:
        numerical_answer = 0.0

    return {
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": question},
            {"role": "assistant", "content": reformatted},
        ],
        "answer": numerical_answer,  # For evaluation
    }


@app.command()
def generate_sft_gsm8k(
    system_msg: str = typer.Option(
        "You are a helpful math assistant. Always provide your final numerical answer inside of the <answer>...</answer> tags, e.g.: <answer>42</answer>",
        "--system-msg",
        help="System message to use for the chat format",
    ),
    seed: int = typer.Option(67, help="Random seed for train/test split"),
    output_dir: str = typer.Option("generated_data", help="Directory to save the dataset"),
    test_split: float = typer.Option(0.0, help="Fraction of data to use for test set"),
):
    """
    Generate SFT training data from GSM8K in messages format.

    This command processes GSM8K to create SFT-ready data by:
    - Removing calculator annotations (<<a+b=c>>)
    - Converting #### answers to <answer>X</answer> format
    - Formatting as chat messages with system/user/assistant roles
    """
    # load GSM8K
    gsm8k = datasets.load_dataset("openai/gsm8k", "main", split="train")
    typer.secho(f"✓ Loaded {len(gsm8k)} samples from GSM8K", fg=typer.colors.GREEN)

    # process into SFT format
    sft_samples = []
    for i in range(len(gsm8k)):
        sample = _create_sft_message(
            question=gsm8k["question"][i],
            answer=gsm8k["answer"][i],
            system_msg=system_msg,
        )
        sft_samples.append(sample)

    # convert to HF dataset for easy splitting and saving
    sft_dataset = datasets.Dataset.from_list(sft_samples)

    # handle train/test split
    if test_split > 0:
        split_data = sft_dataset.train_test_split(test_size=test_split, seed=seed)
        train_dataset = split_data["train"]
        test_dataset = split_data["test"]
    else:
        train_dataset = sft_dataset
        test_dataset = None

    os.makedirs(output_dir, exist_ok=True)

    # write training data
    train_path = os.path.join(output_dir, "gsm8k_sft_train.jsonl")
    train_dataset.to_json(train_path)
    typer.secho(
        f"✓ Generated {len(train_dataset)} SFT training examples",
        fg=typer.colors.GREEN,
    )
    typer.secho(f"✓ Saved to '{train_path}'", fg=typer.colors.BLUE)

    # write test data if split was requested
    if test_dataset:
        test_path = os.path.join(output_dir, "gsm8k_sft_test.jsonl")
        test_dataset.to_json(test_path)
        typer.secho(
            f"✓ Generated {len(test_dataset)} SFT test examples",
            fg=typer.colors.GREEN,
        )
        typer.secho(f"✓ Saved to '{test_path}'", fg=typer.colors.BLUE)

    # show a sample for verification
    typer.secho("\n--- Sample Output ---", fg=typer.colors.BRIGHT_CYAN)
    sample = sft_samples[0]
    for msg in sample["messages"]:
        role = msg["role"].upper()
        content = msg["content"][:150] + "..." if len(msg["content"]) > 150 else msg["content"]
        typer.secho(f"[{role}]: {content}", fg=typer.colors.WHITE)


@app.command()
def generate_gsm8k_datasets(
    system_msg: str = typer.Option(
        "You are a helpful math assistant. Always provide your final numerical answer inside of the <answer>...</answer> tags, e.g.: <answer>42</answer>",
        "--system-msg",
        help="System message to use for the chat format",
    ),
    seed: int = typer.Option(67, "--seed", help="Random seed for train/test split"),
    output_dir: str = typer.Option("generated_data", "--output-dir", help="Directory to save the datasets"),
    test_split: float = typer.Option(0.1, "--test-split", help="Fraction of data to use for test set"),
):
    """
    Generate both GRPO and SFT datasets from GSM8K using the same train/test split.

    This ensures both training approaches use identical samples for fair comparison.
    Outputs:
      - gsm8k_grpo_train.jsonl / gsm8k_grpo_test.jsonl (for GRPO training)
      - gsm8k_sft_train.jsonl / gsm8k_sft_test.jsonl (for SFT training)
    """
    # Load GSM8K once
    gsm8k = datasets.load_dataset("openai/gsm8k", "main", split="train")
    typer.secho(f"✓ Loaded {len(gsm8k)} samples from GSM8K", fg=typer.colors.GREEN)

    # Do the train/test split FIRST on raw data indices
    if test_split > 0:
        split_data = gsm8k.train_test_split(test_size=test_split, seed=seed)
        train_gsm8k = split_data["train"]
        test_gsm8k = split_data["test"]
        typer.secho(
            f"✓ Split with seed={seed}: {len(train_gsm8k)} train, {len(test_gsm8k)} test",
            fg=typer.colors.GREEN,
        )
    else:
        train_gsm8k = gsm8k
        test_gsm8k = None

    os.makedirs(output_dir, exist_ok=True)

    def _create_grpo_sample(question: str, answer: str) -> dict:
        """Create a GRPO sample (prompt-only, with numerical answer for grading)."""
        # Extract numerical answer from GSM8K format
        alt_matches = re.findall(r"#### (.+)", answer)
        if alt_matches:
            numerical_answer = float(alt_matches[-1].replace(",", ""))
        else:
            numerical_answer = 0.0

        return {
            "messages": [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": question},
            ],
            "answer": numerical_answer,
            "problem": question,
            "operation": "gsm8k",
        }

    def process_split(gsm8k_split, split_name: str):
        """Process a GSM8K split into both GRPO and SFT formats."""
        grpo_samples = []
        sft_samples = []

        for i in range(len(gsm8k_split)):
            question = gsm8k_split["question"][i]
            answer = gsm8k_split["answer"][i]

            # GRPO format (prompt only)
            grpo_samples.append(_create_grpo_sample(question, answer))

            # SFT format (prompt + response)
            sft_samples.append(_create_sft_message(question, answer, system_msg))

        # Save GRPO dataset
        grpo_dataset = datasets.Dataset.from_list(grpo_samples)
        grpo_path = os.path.join(output_dir, f"gsm8k_grpo_{split_name}.jsonl")
        grpo_dataset.to_json(grpo_path)
        typer.secho(f"✓ Saved {len(grpo_dataset)} GRPO {split_name} samples to '{grpo_path}'", fg=typer.colors.BLUE)

        # Save SFT dataset
        sft_dataset = datasets.Dataset.from_list(sft_samples)
        sft_path = os.path.join(output_dir, f"gsm8k_sft_{split_name}.jsonl")
        sft_dataset.to_json(sft_path)
        typer.secho(f"✓ Saved {len(sft_dataset)} SFT {split_name} samples to '{sft_path}'", fg=typer.colors.BLUE)

    # Process train split
    process_split(train_gsm8k, "train")

    # Process test split if it exists
    if test_gsm8k is not None:
        process_split(test_gsm8k, "test")

    typer.secho(
        f"\n✓ Generated GRPO and SFT datasets with identical samples (seed={seed})",
        fg=typer.colors.GREEN,
        bold=True,
    )


@torch.no_grad
def generate_rollouts(
    ctx: TrainingComponents,
    model: PreTrainedModel,
    tokenizer: AutoTokenizer,
    batch: dict[str, list[any]],
    batch_size: int,
    group_size: int,
    sampling_params: SamplingParams,
    show_tqdm=False,
) -> list[Sample]:
    model.eval()
    device = next(p.device for p in model.parameters())
    # here we need to create a set of rollouts for each prompt
    groups: list[Sample] = []

    iterator = range(batch_size)
    if show_tqdm:
        iterator = tqdm(
            iterator,
            desc="Generating rollouts",
            leave=False,  # Don't leave the bar after completion
            position=1,  # Nested position to avoid conflicts with outer bar
        )

    for i in iterator:
        # TODO: optimize this
        # Preview the messages for this batch item
        # if i == 0:  # Only preview the first item to avoid clutter
        #     typer.secho(f"\n[Batch {i}] Messages:", fg=typer.colors.BRIGHT_CYAN)
        #     for msg in batch["messages"][i]:
        #         typer.secho(
        #             f"  [{msg['role']}]: {msg['content']}", fg=typer.colors.CYAN
        #         )
        input_ids = tokenizer.apply_chat_template(
            conversation=batch["messages"][i],
            return_tensors="pt",
            add_generation_prompt=True,
        ).to(device=device)

        # now we sample
        outputs = model.generate(
            input_ids,
            attention_mask=torch.ones_like(input_ids),
            max_new_tokens=sampling_params.max_new_tokens,
            num_return_sequences=group_size,
            do_sample=True,
            temperature=sampling_params.temperature,
            top_k=sampling_params.top_k,
            top_p=sampling_params.top_p,
            repetition_penalty=sampling_params.repetition_penalty,
            # output_logits=True,
            output_scores=True,
            return_dict_in_generate=True,
        )
        input_len = input_ids.numel()
        new_tokens = outputs.sequences[:, input_len:]

        # for each sample in the batch we append the generated responses as they're parsed back from the model
        # this should align with the rollout ordering that we get from the batch
        # TODO: vectorize logprob gathering

        # embed()

        # we recollect the sample by combining across the column dimension
        seed_sample = {k: v[i] for k, v in batch.items()}
        rollout_data: list[RolloutResult] = []
        problem = Problem(
            answer=seed_sample["answer"],
            operation=seed_sample["operation"],
            problem=seed_sample["problem"],
        )

        # go through each sequence and grab the respective logprob
        # TODO: optimize this part
        for seq_idx, seq in enumerate(new_tokens.tolist()):
            logprobs: list[TokenSample] = []

            # stop processing after the model generated EOS token
            try:
                seq_end = seq.index(tokenizer.eos_token_id) + 1
            except ValueError:
                # fallback to full sequence
                seq_end = len(seq)

            # next, we just need to select the probs for our specific tokens
            # Cast to FP32 for precise log_softmax over large vocab (model forward stays in BF16)
            processed_logits = torch.stack([t[seq_idx] for t in outputs.scores[:seq_end]])
            processed_logits_f32 = processed_logits.float()
            ref_logprobs = processed_logits_f32.log_softmax(dim=-1)
            index = torch.tensor(seq[:seq_end], dtype=torch.long, device=processed_logits.device)
            index = index.unsqueeze(-1)  # extend from (T,) into (T, 1)
            probs = ref_logprobs.gather(dim=-1, index=index)
            probs = probs.squeeze(-1)  # (T, 1) --> (T,)
            index = index.squeeze(-1)  # (T, 1) --> (T,)

            for tok, prob in zip(index.tolist(), probs.tolist()):
                logprobs.append(
                    TokenSample(
                        token=tok,
                        logprob=prob,
                    )
                )

            # here we append the rollout data
            policy_response = tokenizer.decode(new_tokens[seq_idx], skip_special_tokens=True)
            rollout_data.append(
                RolloutResult(
                    logprobs=logprobs,
                    response=policy_response,
                    seed_messages=seed_sample["messages"],
                )
            )

        assert input_ids.ndim > 1
        groups.append(
            Sample(
                problem=problem,
                rollouts=rollout_data,
                input_ids=input_ids.tolist()[0],  # record the input ids so we can reuse them later
            )
        )

    for group in groups:
        grade_groups(group)
        calculate_advantage(group)

    # empty cache
    torch.cuda.empty_cache()
    return groups


@torch.no_grad
def grade_groups(group: Sample):
    """
    Given a batch of samples, calculates the advantage for each one.
    Modifies objects in place.

    Grading rules:
    - Use the LAST <answer>...</answer> tag if multiple present (final answer after reasoning)
    - +0.1 reward for parsable format
    - +1.0 reward for correct answer
    """
    for rollout in group.rollouts:
        # Defaults
        rollout.is_parsable = False
        rollout.is_correct = False
        rollout.reward = 0

        # Find all answer tags
        matches = answer_pattern.findall(rollout.response)

        if not matches:
            # No answer tags found - no reward
            continue

        # Take the LAST answer (final answer after reasoning)
        last_match = matches[-1]

        try:
            parsed_answer = parse_number(last_match)
            rollout.is_parsable = True

            # Format reward for proper answer structure
            rollout.reward += 0.1

            # Check correctness with tolerance for floating point comparison
            expected = float(group.problem.answer)
            if abs(parsed_answer - expected) < 1e-6:
                rollout.is_correct = True
                rollout.reward += 1.0

        except ValueError:
            # Could not parse the last answer - no parsable reward
            pass


def _truncate_response(response: str, max_len: int = 80) -> str:
    """Truncate response text, keeping the end if too long."""
    # collapse whitespace and newlines for table display
    cleaned = " ".join(response.split())
    if len(cleaned) <= max_len:
        return cleaned
    return "..." + cleaned[-(max_len - 3) :]


def _extract_last_answer(response: str) -> str | None:
    """Extract content from the last <answer>...</answer> tag, or None if not found."""
    matches = answer_pattern.findall(response)
    if not matches:
        return None
    return matches[-1].strip()


def _print_group_detail(sample: Sample, label: str, color):
    """Print detailed information for a single group of rollouts."""
    typer.secho(f"\n{'─' * 70}", fg=color)
    typer.secho(f"  [{label}]", fg=color, bold=True)
    typer.secho(f"{'─' * 70}", fg=color)

    # print the problem
    typer.secho("\n  Problem:", fg=typer.colors.BRIGHT_CYAN, bold=True)
    problem_text = sample.problem.problem
    # wrap long problems for readability
    for line in problem_text.split("\n"):
        typer.secho(f"    {line}", fg=typer.colors.WHITE)
    typer.secho(f"    Expected Answer: {sample.problem.answer}", fg=typer.colors.YELLOW)

    # calculate group stats
    rewards = [r.reward for r in sample.rollouts]
    avg_reward = sum(rewards) / len(rewards) if rewards else 0.0
    typer.secho(f"    Group Avg Reward: {avg_reward:.4f}", fg=typer.colors.CYAN)

    # print table header
    typer.secho("\n  Rollouts Table:", fg=typer.colors.BRIGHT_CYAN, bold=True)
    header = f"  {'#':>3} | {'Response (truncated)':<60} | {'Reward':>8} | {'Adv':>8} | {'Parse':>5} | {'Answer':<20} | {'Correct':>7}"
    typer.secho(header, fg=typer.colors.WHITE, bold=True)
    typer.secho(
        f"  {'-' * 3}-+-{'-' * 60}-+-{'-' * 8}-+-{'-' * 8}-+-{'-' * 5}-+-{'-' * 20}-+-{'-' * 7}", fg=typer.colors.WHITE
    )

    # track best and worst rollouts
    best_rollout = None
    worst_rollout = None
    best_reward = float("-inf")
    worst_reward = float("inf")

    for i, rollout in enumerate(sample.rollouts):
        truncated = _truncate_response(rollout.response, 60)
        correct_str = "✓" if rollout.is_correct else "✗"
        correct_color = typer.colors.GREEN if rollout.is_correct else typer.colors.RED

        # extract answer tag content
        answer_content = _extract_last_answer(rollout.response)
        parsable_str = "✓" if answer_content is not None else "✗"
        parsable_color = typer.colors.GREEN if answer_content is not None else typer.colors.RED
        answer_display = (
            (answer_content[:17] + "...") if answer_content and len(answer_content) > 20 else (answer_content or "-")
        )

        row = f"  {i + 1:>3} | {truncated:<60} | {rollout.reward:>8.4f} | {rollout.advantage:>8.4f} | "
        typer.echo(row, nl=False)
        typer.secho(f"{parsable_str:>5}", fg=parsable_color, nl=False)
        typer.echo(f" | {answer_display:<20} | ", nl=False)
        typer.secho(f"{correct_str:>7}", fg=correct_color)

        if rollout.reward > best_reward:
            best_reward = rollout.reward
            best_rollout = rollout
        if rollout.reward < worst_reward:
            worst_reward = rollout.reward
            worst_rollout = rollout

    # print full conversation for best rollout
    if best_rollout:
        typer.secho(f"\n  Best Rollout (reward={best_reward:.4f}):", fg=typer.colors.GREEN, bold=True)
        for msg in best_rollout.seed_messages:
            typer.secho(f"    [{msg.role}]:", fg=typer.colors.CYAN)
            typer.secho(msg.content, fg=typer.colors.WHITE)
        typer.secho("    [assistant]:", fg=typer.colors.CYAN)
        typer.secho(best_rollout.response, fg=typer.colors.GREEN)

    # print full conversation for worst rollout (only if different from best)
    if worst_rollout and worst_rollout is not best_rollout:
        typer.secho(f"\n  Worst Rollout (reward={worst_reward:.4f}):", fg=typer.colors.RED, bold=True)
        for msg in worst_rollout.seed_messages:
            typer.secho(f"    [{msg.role}]:", fg=typer.colors.CYAN)
            typer.secho(msg.content, fg=typer.colors.WHITE)
        typer.secho("    [assistant]:", fg=typer.colors.CYAN)
        typer.secho(worst_rollout.response, fg=typer.colors.RED)


def print_example_rollout(samples: list[Sample], step: int = 0, verbose: bool = False):
    """Print batch statistics and optionally detailed rollouts for best/worst groups."""
    if not samples:
        return

    # calculate batch statistics
    total_rollouts = sum(len(s.rollouts) for s in samples)
    total_rewards = sum(r.reward for s in samples for r in s.rollouts)
    parsable_count = sum(1 for s in samples for r in s.rollouts if r.is_parsable)
    correct_count = sum(1 for s in samples for r in s.rollouts if r.is_correct)

    avg_reward = total_rewards / total_rollouts if total_rollouts > 0 else 0.0
    parsable_rate = parsable_count / total_rollouts if total_rollouts > 0 else 0.0
    correct_rate = correct_count / total_rollouts if total_rollouts > 0 else 0.0

    # print batch statistics header
    typer.secho(f"\n{'=' * 120}", fg=typer.colors.BRIGHT_MAGENTA)
    typer.secho(f"  ROLLOUT SUMMARY (Step {step})", fg=typer.colors.BRIGHT_MAGENTA, bold=True)
    typer.secho(f"{'=' * 120}", fg=typer.colors.BRIGHT_MAGENTA)

    typer.secho("\n[BATCH STATISTICS]:", fg=typer.colors.BRIGHT_CYAN)
    typer.secho(
        f"  Prompts: {len(samples)} | Rollouts: {total_rollouts} | Rollouts/Prompt: {total_rollouts // len(samples)}",
        fg=typer.colors.WHITE,
    )
    typer.secho(
        f"  Parsable: {parsable_count}/{total_rollouts} ({parsable_rate:.1%})",
        fg=typer.colors.GREEN if parsable_rate > 0.5 else typer.colors.YELLOW,
    )
    typer.secho(
        f"  Correct:  {correct_count}/{total_rollouts} ({correct_rate:.1%})",
        fg=typer.colors.GREEN if correct_rate > 0.3 else typer.colors.YELLOW,
    )
    typer.secho(f"  Avg Reward: {avg_reward:.4f}", fg=typer.colors.CYAN)

    # print per-group summary table
    typer.secho("\n[PER-GROUP SUMMARY]:", fg=typer.colors.BRIGHT_CYAN)
    header = f"  {'#':>3} | {'Question':<50} | {'Ans':>8} | {'Reward':>8} | {'Adv':>8} | {'Parse':>6} | {'Acc':>6}"
    typer.secho(header, fg=typer.colors.WHITE, bold=True)
    typer.secho(
        f"  {'-' * 3}-+-{'-' * 50}-+-{'-' * 8}-+-{'-' * 8}-+-{'-' * 8}-+-{'-' * 6}-+-{'-' * 6}", fg=typer.colors.WHITE
    )

    for idx, sample in enumerate(samples, 1):
        num_rollouts = len(sample.rollouts)
        if num_rollouts == 0:
            continue

        # calculate per-group stats
        group_rewards = [r.reward for r in sample.rollouts]
        group_advantages = [r.advantage for r in sample.rollouts]
        group_parsable = sum(1 for r in sample.rollouts if r.is_parsable)
        group_correct = sum(1 for r in sample.rollouts if r.is_correct)

        avg_grp_reward = sum(group_rewards) / num_rollouts
        avg_grp_adv = sum(group_advantages) / num_rollouts
        parse_rate = group_parsable / num_rollouts
        acc_rate = group_correct / num_rollouts

        # truncate question for display
        question = sample.problem.problem
        if len(question) > 50:
            question = question[:47] + "..."

        # format answer (handle floats that are actually ints)
        ans = sample.problem.answer
        ans_str = str(int(ans)) if ans == int(ans) else f"{ans:.2f}"
        if len(ans_str) > 8:
            ans_str = ans_str[:8]

        # color based on accuracy
        if acc_rate >= 0.5:
            row_color = typer.colors.GREEN
        elif acc_rate > 0:
            row_color = typer.colors.YELLOW
        else:
            row_color = typer.colors.RED

        row = f"  {idx:>3} | {question:<50} | {ans_str:>8} | {avg_grp_reward:>8.3f} | {avg_grp_adv:>+8.3f} | {parse_rate:>5.0%} | {acc_rate:>5.0%}"
        typer.secho(row, fg=row_color)

    # detailed group info only if verbose
    if verbose:

        def group_avg_reward(sample: Sample) -> float:
            if not sample.rollouts:
                return 0.0
            return sum(r.reward for r in sample.rollouts) / len(sample.rollouts)

        sorted_samples = sorted(samples, key=group_avg_reward, reverse=True)

        best_group = sorted_samples[0]
        worst_group = sorted_samples[-1]

        _print_group_detail(best_group, "BEST GROUP (highest avg reward)", typer.colors.GREEN)

        if worst_group is not best_group:
            _print_group_detail(worst_group, "WORST GROUP (lowest avg reward)", typer.colors.RED)

    typer.secho(f"\n{'=' * 120}\n", fg=typer.colors.BRIGHT_MAGENTA)


# i dont think we even have tensors flowing through this function but you
# can never be too sure.
@torch.no_grad
def calculate_advantage(group: Sample):
    r"""
    This is the fun part, we have to implement the GRPO-style
    advantage calculation. Basically we take each set of rollouts as a single
    group and we calculate a group-level advantage as a workaround for
    not being able to calculate RTG or step-level advantage as in vanilla REINFORCE.

    Formula looks like this:

    $$
    A_i = \frac{r_i - \mean(r)}{\std(r) + \epsilon}
    $$
    """
    eps = 1e-8
    avg = sum(r.reward for r in group.rollouts) / len(group.rollouts)
    var = sum((r.reward - avg) ** 2 for r in group.rollouts) / len(group.rollouts)
    std = var**0.5

    # if std < eps (because all rewards are equal) we use the std trick
    # of setting group advantage to 0
    enable_std_trick = std < eps

    # GRPO simple advantage with clamping to prevent extreme values
    for rollout in group.rollouts:
        if enable_std_trick:
            rollout.advantage = 0.0
        else:
            adv = (rollout.reward - avg) / (std + eps)
            # Clamp advantages to prevent extreme policy updates
            rollout.advantage = max(-10.0, min(10.0, adv))


@torch.no_grad
def eval_model(
    eval_dataset: datasets.Dataset,
    comps: TrainingComponents,
    return_metrics: bool = False,
) -> dict | None:
    """
    Evaluate model on dataset.

    Args:
        eval_dataset: Dataset to evaluate on
        comps: Training components
        return_metrics: If True, return metrics dict instead of just printing

    Returns:
        If return_metrics=True, returns dict with metrics
    """
    comps.model.eval()

    # we generate all the rollouts
    eval_data = eval_dataset.batch(eval_dataset.num_rows)
    pass_at = [
        1,
    ]  #  3,#  5, 10]
    results = []

    for npass in pass_at:
        samples = generate_rollouts(
            comps,
            comps.model,
            comps.tokenizer,
            batch=next(iter(eval_data)),
            batch_size=eval_dataset.num_rows,
            group_size=npass,
            sampling_params=comps.sampling_params,
            show_tqdm=True,
        )

        # now we go and determine the passing rate
        percent_scores = []
        for sample in samples:
            passing_rate = sum(1 if r.is_correct else 0 for r in sample.rollouts) / len(sample.rollouts)
            percent_scores.append(passing_rate)
        # Calculate statistics
        percent_above_50 = sum(1 if score > 0.5 else 0 for score in percent_scores) / len(percent_scores) * 100
        percent_at_100 = sum(1 if score == 1.0 else 0 for score in percent_scores) / len(percent_scores) * 100

        results.append((npass, percent_above_50, percent_at_100))

    # Print all results at the end
    typer.secho("\n=== Evaluation Scorecard ===", fg=typer.colors.BRIGHT_MAGENTA)
    typer.secho(f"Total samples evaluated: {len(samples)}", fg=typer.colors.BRIGHT_BLUE)
    for npass, percent_above_50, percent_at_100 in results:
        typer.secho(
            f"Pass@{npass}: {percent_above_50:.1f}% above 50% | {percent_at_100:.1f}% at 100% (across {len(samples)} samples with {npass} rollout(s) each)",
            fg=typer.colors.CYAN,
        )

    if return_metrics and results:
        _, above_50, at_100 = results[0]  # Return first pass@k metrics
        return {
            "above_50": above_50,
            "at_100": at_100,
            "samples": len(samples),
        }
    return None


@app.command()
def eval(
    eval_path: str = typer.Option(..., "--eval-path", help="Path to the evaluation dataset (jsonl)"),
    model_name: str = typer.Option(..., "--model", "-m", help="Model name or path"),
    gpu: int = typer.Option(0, "--gpu", "-g", help="CUDA GPU index to use"),
    max_new_tokens: int = typer.Option(128, help="Maximum number of new tokens to generate"),
    max_seq_len: int = typer.Option(8192, "--msl", "--max-seq-len", help="Maximum sequence length"),
    temperature: float = typer.Option(0.7, "-t", "--temp", help="Sampling temperature"),
    group_size: int = typer.Option(1, "-G", "--group-size", help="Number of rollouts per prompt (for pass@k)"),
):
    raise NotImplementedError("this path currently isn't implemented or being used")

    """Run evaluation on a dataset without training."""
    device = torch.device("cuda", gpu)

    eval_dataset = datasets.load_dataset("json", data_files=eval_path, split="train")
    typer.secho(f"✓ Loaded {len(eval_dataset)} evaluation samples", fg=typer.colors.GREEN)

    model = AutoModelForCausalLM.from_pretrained(model_name, device_map=device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token_id and not model.config.pad_token_id:
        model.config.pad_token_id = tokenizer.pad_token_id

    sampling_params = SamplingParams(
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        max_tokens=max_seq_len,
        top_p=1.0,
        top_k=0.0,
        repetition_penalty=1.0,
    )

    eval_data = eval_dataset.batch(eval_dataset.num_rows)
    samples = generate_rollouts(
        ctx,
        model,
        tokenizer,
        batch=next(iter(eval_data)),
        batch_size=eval_dataset.num_rows,
        group_size=group_size,
        sampling_params=sampling_params,
        show_tqdm=True,
    )

    percent_scores = []
    for sample in samples:
        passing_rate = sum(1 if r.is_correct else 0 for r in sample.rollouts) / len(sample.rollouts)
        percent_scores.append(passing_rate)

    percent_above_50 = sum(1 if score > 0.5 else 0 for score in percent_scores) / len(percent_scores) * 100
    percent_at_100 = sum(1 if score == 1.0 else 0 for score in percent_scores) / len(percent_scores) * 100

    typer.secho("\n=== Evaluation Results ===", fg=typer.colors.BRIGHT_MAGENTA)
    typer.secho(f"Model: {model_name}", fg=typer.colors.BRIGHT_BLUE)
    typer.secho(f"Samples: {len(samples)}", fg=typer.colors.BRIGHT_BLUE)
    typer.secho(
        f"Pass@{group_size}: {percent_above_50:.1f}% above 50% | {percent_at_100:.1f}% at 100%",
        fg=typer.colors.CYAN,
    )


def check_model_health(model: PreTrainedModel) -> bool:
    """Check if model weights contain NaN or Inf values."""
    for name, param in model.named_parameters():
        if torch.isnan(param).any() or torch.isinf(param).any():
            typer.secho(f"WARNING: NaN/Inf detected in model parameter: {name}", fg=typer.colors.RED)
            return False
    return True


def train_policy_on_rollouts(
    samples: list[Sample],
    comps: TrainingComponents,
    use_wandb: bool = False,
    global_step: int = 0,
    use_packed: bool = False,
    max_tokens_per_microbatch: int = 0,
    current_optim_step: int = 0,
    max_steps: int = 0,
    current_tokens_trained: int = 0,
    token_train_budget: int = 0,
) -> tuple[int, int, bool]:
    """
    Train the policy model on generated rollouts using GRPO.

    Args:
        samples: List of Sample objects containing rollouts
        comps: Training components
        use_wandb: Whether to log to wandb
        global_step: Current global training step
        use_packed: Use padding-free packed sequences (requires Flash Attention 2)
        max_tokens_per_microbatch: Max tokens per microbatch (0 = no limit, process full batch)
        current_optim_step: Current optimizer step count
        max_steps: Maximum optimizer steps (0 = no limit)
        current_tokens_trained: Current count of tokens trained on
        token_train_budget: Maximum tokens to train on (0 = no limit)

    Returns:
        Tuple of (updated_optim_step, tokens_trained, should_stop) where should_stop is True if budget reached
    """
    comps.model.train()

    # Create dataset from rollouts
    dataset = dataset_from_groups(samples, comps.train_tokenizer)

    # Track optimizer steps and tokens
    optim_step = current_optim_step
    tokens_trained = current_tokens_trained

    # Training loop over inner epochs
    for epoch in range(comps.hyperparams.inner_epochs):
        data_loader = create_grpo_data_loader(dataset, comps, seed=comps.seed + epoch, use_packed=use_packed)

        for batch in data_loader:
            # Clear cache at start of each batch
            torch.cuda.empty_cache()

            # Split batch into microbatches if max_tokens specified
            if max_tokens_per_microbatch > 0:
                microbatches = list(split_batch_into_microbatches(batch, max_tokens_per_microbatch))
            else:
                # No splitting - process full batch
                batch["total_tokens_in_batch"] = batch["num_tokens"]
                batch["num_microbatches"] = 1
                microbatches = [batch]

            num_microbatches = len(microbatches)
            accumulated_loss = 0.0
            accumulated_metrics = {"kl_div": 0.0, "importance_ratio": 0.0}
            valid_microbatches = 0
            batch_tokens = 0  # Tokens trained in this batch (sum of rollout_lens)

            # Accumulate gradients across microbatches
            for micro_idx, microbatch in enumerate(microbatches):
                # Track tokens being trained on (rollout_lens = completion tokens with gradients)
                batch_tokens += microbatch["rollout_lens"].sum().item()
                if use_packed:
                    grpo_loss, metrics = _train_step_packed(microbatch, comps)
                else:
                    grpo_loss, metrics = _train_step_padded(microbatch, comps)

                # Check for NaN/Inf in loss before backward
                if torch.isnan(grpo_loss) or torch.isinf(grpo_loss):
                    typer.secho(
                        f"WARNING: NaN/Inf loss detected in microbatch {micro_idx + 1}/{num_microbatches}! Skipping. "
                        f"KL: {metrics.get('kl_div', 'N/A')}, IR: {metrics.get('importance_ratio', 'N/A')}",
                        fg=typer.colors.RED,
                    )
                    continue

                # Scale loss by number of microbatches for correct gradient accumulation
                scaled_loss = grpo_loss / num_microbatches
                scaled_loss.backward()

                accumulated_loss += grpo_loss.item()
                accumulated_metrics["kl_div"] += metrics["kl_div"]
                accumulated_metrics["importance_ratio"] += metrics["importance_ratio"]
                valid_microbatches += 1

                # Clear intermediate tensors
                del grpo_loss, scaled_loss
                torch.cuda.empty_cache()

            # Skip optimizer step if no valid microbatches
            if valid_microbatches == 0:
                comps.optimizer.zero_grad()
                continue

            # Average metrics
            avg_loss = accumulated_loss / valid_microbatches
            avg_metrics = {k: v / valid_microbatches for k, v in accumulated_metrics.items()}

            # Check for NaN in gradients before optimizer step
            has_nan_grad = False
            for name, param in comps.model.named_parameters():
                if param.grad is not None and (torch.isnan(param.grad).any() or torch.isinf(param.grad).any()):
                    typer.secho(f"WARNING: NaN/Inf gradient in {name}! Skipping optimizer step.", fg=typer.colors.RED)
                    has_nan_grad = True
                    break

            if has_nan_grad:
                comps.optimizer.zero_grad()
                continue

            # Gradient clipping and optimization
            gradnorm = torch.nn.utils.clip_grad_norm_(comps.model.parameters(), 1.0)
            comps.optimizer.step()
            comps.optimizer.zero_grad()
            optim_step += 1
            tokens_trained += batch_tokens

            # Clear cache after optimizer step
            torch.cuda.empty_cache()

            # Log metrics (including KL divergence and tokens)
            kl_div = avg_metrics.get("kl_div", 0.0)
            ir_mean = avg_metrics.get("importance_ratio", 1.0)
            typer.secho(
                f"Inner Epoch {epoch + 1}/{comps.hyperparams.inner_epochs} | "
                f"Step {optim_step} | "
                f"Loss: {avg_loss:.4f} | "
                f"KL: {kl_div:.4f} | "
                f"IR: {ir_mean:.4f} | "
                f"Grad Norm: {gradnorm.item():.4f} | "
                f"Tokens: {tokens_trained:,}",
                fg=typer.colors.YELLOW,
            )

            # Log to wandb if enabled
            if use_wandb and wandb is not None:
                wandb.log(
                    {
                        "train/loss": avg_loss,
                        "train/grad_norm": gradnorm.item(),
                        "train/kl_divergence": avg_metrics["kl_div"],
                        "train/importance_ratio_mean": avg_metrics["importance_ratio"],
                        "train/microbatches": num_microbatches,
                        "train/optim_step": optim_step,
                        "train/tokens_trained": tokens_trained,
                        "train/batch_tokens": batch_tokens,
                    },
                    step=optim_step,
                )

            # Check if we've reached max_steps or token budget
            if max_steps > 0 and optim_step >= max_steps:
                return optim_step, tokens_trained, True
            if token_train_budget > 0 and tokens_trained >= token_train_budget:
                return optim_step, tokens_trained, True

        # Clear cache after each inner epoch
        torch.cuda.empty_cache()

    return optim_step, tokens_trained, False


def _train_step_padded(batch: dict, comps: TrainingComponents) -> tuple[torch.Tensor, dict]:
    """Training step for padded (standard) batch format with mixed precision."""
    # Send everything to GPU
    input_ids = batch["input_ids"].to(comps.device)
    advantages = batch["advantages"].to(comps.device)
    old_logprobs = batch["logprobs"].to(comps.device)
    old_logprob_ids = batch["logprob_ids"].to(comps.device)
    rollout_lens = batch["rollout_lens"].to(comps.device)
    attn_mask = batch["attention_mask"].to(comps.device)
    grpo_logit_mask = batch["grpo_mask"].to(comps.device)

    # Forward pass on policy model with autocast (FP32 weights, BF16 forward)
    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        new_outputs = comps.model(input_ids=input_ids, attention_mask=attn_mask)
        new_logits = new_outputs.logits

    # Temperature scaling
    if comps.sampling_params.temperature > 0:
        new_logits = new_logits / comps.sampling_params.temperature

    # Forward pass on frozen reference model (already in BF16)
    # Compute ref logprobs immediately and discard ref_logits to save memory
    gather_indices = old_logprob_ids.unsqueeze(-1)  # (B, T) -> (B, T, 1)
    with torch.no_grad():
        ref_outputs = comps.ref_model(input_ids, attention_mask=attn_mask)
        ref_logits = ref_outputs.logits
        if comps.sampling_params.temperature > 0:
            ref_logits = ref_logits / comps.sampling_params.temperature

        # Compute ref logprobs in original dtype to avoid massive FP32 allocation
        # logsumexp is numerically stable, cast only the final scalar result
        ref_gathered = ref_logits.gather(dim=-1, index=gather_indices)
        ref_logsumexp = ref_logits.logsumexp(dim=-1, keepdim=True)
        ref_logprobs = (ref_gathered - ref_logsumexp).squeeze(-1).float()
        del ref_logits, ref_outputs
        torch.cuda.empty_cache()

    # Compute policy logprobs in original dtype to avoid massive FP32 allocation
    new_gathered = new_logits.gather(dim=-1, index=gather_indices)
    new_logsumexp = new_logits.logsumexp(dim=-1, keepdim=True)
    new_logprobs = (new_gathered - new_logsumexp).squeeze(-1).float()
    del new_logits, new_gathered, new_logsumexp, new_outputs

    # Importance ratio (keep in FP32 for stability with exp)
    # Clamp log ratio to prevent exp() from exploding/underflowing
    log_ratio = (new_logprobs - old_logprobs.float()).clamp(-20, 20)
    importance_ratio = log_ratio.exp()

    # Clipped surrogate objective
    advantages = advantages.unsqueeze(-1)  # (B,) -> (B, 1)
    unclipped = advantages * importance_ratio
    clipped = advantages * importance_ratio.clamp(1 - comps.hyperparams.eps, 1 + comps.hyperparams.eps)
    clipped_surrogate = torch.minimum(unclipped, clipped)

    # KL penalty with numerical stability
    # Clamp log diff before exp() to prevent overflow
    log_diff = (ref_logprobs - new_logprobs).clamp(-20, 20)
    dkl_approx = log_diff.exp() - log_diff - 1
    # KL should be non-negative; clamp to prevent outliers from dominating
    dkl_approx = dkl_approx.clamp(min=0, max=100)

    # Per-token loss
    per_token_loss = clipped_surrogate - comps.hyperparams.kl_penalty_strength * dkl_approx
    grpo_token_loss = per_token_loss * grpo_logit_mask.float()

    # Sequence-level averaging (clamp rollout_lens to avoid division by zero)
    safe_rollout_lens = rollout_lens.float().clamp(min=1.0)
    grpo_sequence_loss = grpo_token_loss.sum(dim=-1) / safe_rollout_lens
    grpo_loss = -grpo_sequence_loss.mean()

    # Check for NaN in intermediate values for debugging
    metrics = {
        "kl_div": dkl_approx.mean().item() if not torch.isnan(dkl_approx).any() else float("nan"),
        "importance_ratio": importance_ratio.mean().item() if not torch.isnan(importance_ratio).any() else float("nan"),
    }
    return grpo_loss, metrics


def _train_step_packed(batch: dict, comps: TrainingComponents) -> tuple[torch.Tensor, dict]:
    """
    Training step for packed (padding-free) batch format.

    With Flash Attention 2, we can process variable-length sequences
    packed into a single tensor, avoiding wasted computation on padding.
    """
    # Send everything to GPU
    input_ids = batch["input_ids"].to(comps.device)  # (T_total,)
    position_ids = batch["position_ids"].to(comps.device)  # (T_total,)
    old_logprobs = batch["logprobs"].to(comps.device)  # (T_total,)
    old_logprob_ids = batch["logprob_ids"].to(comps.device)  # (T_total,)
    grpo_mask = batch["grpo_mask"].to(comps.device)  # (T_total,)
    seq_indices = batch["seq_indices"].to(comps.device)  # (T_total,)
    advantages = batch["advantages"].to(comps.device)  # (num_seqs,)
    rollout_lens = batch["rollout_lens"].to(comps.device)  # (num_seqs,)
    cu_seqlens = batch["cu_seqlens"].to(comps.device)  # (num_seqs + 1,)

    num_seqs = batch["num_sequences"]

    # Reshape for model: (1, T_total) - batch size 1 with all sequences packed
    input_ids_2d = input_ids.unsqueeze(0)
    position_ids_2d = position_ids.unsqueeze(0)

    # Forward pass on policy model with autocast (FP32 weights, BF16 forward)
    # Note: For Flash Attention 2, we don't need attention_mask when using position_ids
    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        new_outputs = comps.model(
            input_ids=input_ids_2d,
            position_ids=position_ids_2d,
        )
        new_logits = new_outputs.logits.squeeze(0)  # (T_total, V)

    # Temperature scaling
    if comps.sampling_params.temperature > 0:
        new_logits = new_logits / comps.sampling_params.temperature

    # Forward pass on frozen reference model (already in BF16)
    # Compute ref logprobs immediately and discard ref_logits to save memory
    with torch.no_grad():
        ref_outputs = comps.ref_model(
            input_ids=input_ids_2d,
            position_ids=position_ids_2d,
        )
        ref_logits = ref_outputs.logits.squeeze(0)  # (T_total, V)
        if comps.sampling_params.temperature > 0:
            ref_logits = ref_logits / comps.sampling_params.temperature

        # Compute ref logprobs in original dtype to avoid massive FP32 allocation
        # logsumexp is numerically stable, cast only the final scalar result
        gather_indices = old_logprob_ids.unsqueeze(-1)  # (T_total,) -> (T_total, 1)
        ref_gathered = ref_logits.gather(dim=-1, index=gather_indices).squeeze(-1)
        ref_logsumexp = ref_logits.logsumexp(dim=-1)
        ref_logprobs = (ref_gathered - ref_logsumexp).float()
        del ref_logits, ref_outputs
        torch.cuda.empty_cache()

    # Compute policy logprobs in original dtype to avoid massive FP32 allocation
    gather_indices = old_logprob_ids.unsqueeze(-1)  # (T_total,) -> (T_total, 1)
    new_gathered = new_logits.gather(dim=-1, index=gather_indices).squeeze(-1)  # (T_total,)
    new_logsumexp = new_logits.logsumexp(dim=-1)  # (T_total,)
    new_logprobs = (new_gathered - new_logsumexp).float()
    del new_logits, new_gathered, new_logsumexp, new_outputs

    # Importance ratio (per-token, keep in FP32 for stability with exp)
    # Clamp log ratio to prevent exp() from exploding/underflowing
    log_ratio = (new_logprobs - old_logprobs.float()).clamp(-20, 20)
    importance_ratio = log_ratio.exp()

    # Get per-token advantages using seq_indices
    token_advantages = advantages[seq_indices]  # (T_total,)

    # Clipped surrogate objective (per-token)
    unclipped = token_advantages * importance_ratio
    clipped = token_advantages * importance_ratio.clamp(1 - comps.hyperparams.eps, 1 + comps.hyperparams.eps)
    clipped_surrogate = torch.minimum(unclipped, clipped)

    # KL penalty (per-token) with numerical stability
    # Clamp log diff before exp() to prevent overflow
    log_diff = (ref_logprobs - new_logprobs).clamp(-20, 20)
    dkl_approx = log_diff.exp() - log_diff - 1
    # KL should be non-negative; clamp to prevent outliers from dominating
    dkl_approx = dkl_approx.clamp(min=0, max=100)

    # Per-token loss
    per_token_loss = clipped_surrogate - comps.hyperparams.kl_penalty_strength * dkl_approx
    grpo_token_loss = per_token_loss * grpo_mask.float()

    # Aggregate losses per sequence using scatter_add
    # Sum token losses for each sequence
    seq_loss_sum = torch.zeros(num_seqs, device=comps.device, dtype=grpo_token_loss.dtype)
    seq_loss_sum.scatter_add_(0, seq_indices, grpo_token_loss)

    # Average by sequence length (clamp to avoid division by zero)
    safe_rollout_lens = rollout_lens.float().clamp(min=1.0)
    grpo_sequence_loss = seq_loss_sum / safe_rollout_lens
    grpo_loss = -grpo_sequence_loss.mean()

    # Check for NaN in intermediate values for debugging
    kl_masked = dkl_approx[grpo_mask] if grpo_mask.any() else dkl_approx
    ir_masked = importance_ratio[grpo_mask] if grpo_mask.any() else importance_ratio
    metrics = {
        "kl_div": kl_masked.mean().item() if not torch.isnan(kl_masked).any() else float("nan"),
        "importance_ratio": ir_masked.mean().item() if not torch.isnan(ir_masked).any() else float("nan"),
    }
    return grpo_loss, metrics


@app.command()
def train(
    # dataset parameters, we'll eventually move these to a data generation command
    train_path: str = typer.Option(..., "--train-path", help="Path to the training data"),
    eval_path: str = typer.Option(None, "--eval-path", help="Path to the evaluation data"),
    seed: int = typer.Option(67, help="Random seed"),
    num_problems: int = typer.Option(20, help="Number of problems"),
    min_num: int = typer.Option(-100, help="Minimum number for problems"),
    max_num: int = typer.Option(100, help="Maximum number for problems"),
    # model
    model_name: str = typer.Option("qwen/Qwen2-1.5B-Instruct", help="Model name or path"),
    # training params
    epochs: int = typer.Option(1, help="Number of training epochs (ignored if --max-steps is set)"),
    max_steps: int = typer.Option(0, "--max-steps", help="Maximum optimizer steps (0 = use epochs instead)"),
    max_new_tokens: int = typer.Option(512, help="Maximum new tokens to generate (512 recommended for GSM8K)."),
    max_seq_len: int = typer.Option(
        8192,
        "--msl",
        "--max-seq-len",
        help="maximum length of the sequences that we work with",
    ),
    # optimizer params
    optimizer_type: str = typer.Option("adamw", "-O", "--optimizer", help="Optimizer type: 'adamw' or 'muon'"),
    lr: float = typer.Option(1e-5, "--lr", help="Learning rate (used for all parameters)"),
    beta1: float = typer.Option(0.9, help="Adam beta1 parameter"),
    beta2: float = typer.Option(0.95, help="Adam beta2 parameter"),
    wd: float = typer.Option(0.0, "--wd", help="Weight decay"),
    # device selection
    gpu: int = typer.Option(0, "--gpu", "-g", help="CUDA GPU index to use for training"),
    # GRPO params
    inner_epochs: int = typer.Option(2, help="Number of passes on inner generation"),
    inner_batch_size: int = typer.Option(32, "--inner-batch-size", help="Batch size during the GRPO inner loop."),
    batch_size: int = typer.Option(
        64, "-B", "--batch-size", help="Number of prompts to batch together when generating GRPO rollouts."
    ),
    group_size: int = typer.Option(
        16, "-G", "--group-size", help="Group size / number of rollouts to generate from a single prompt"
    ),
    temperature: float = typer.Option(0.7, "-t", "--temp", help="sampling temperature"),
    clip_eps: float = typer.Option(0.2, "--clip-eps", help="epsilon used for GRPO clip"),
    kl_strength: float = typer.Option(0.01, "--kl", help="strength of the kl penalty to the reference policy"),
    # memory optimization
    max_tokens_per_microbatch: int = typer.Option(
        0,
        "--max-tokens-per-microbatch",
        help="Max tokens per microbatch for gradient accumulation (0 = no limit, use full inner batch). "
        "Lower values reduce memory usage but increase training time.",
    ),
    # eval params
    eval_split: float = typer.Option(
        0.0, "--eval-split", help="portion of training samples to use for the eval dataset"
    ),
    eval_every: int = typer.Option(
        0, "--eval-every", help="Run evaluation every N training steps (0 = only at epoch end)"
    ),
    output_dir: str | None = typer.Option(None, "--output-dir", help="Directory for saving checkpoints"),
    save_every: int = typer.Option(
        0, "--save-every", help="Save checkpoint every N optimizer steps (0 = only at end of epoch/training)"
    ),
    token_train_budget: int = typer.Option(
        0, "--token-train-budget", help="Total token budget for training (loss-counted tokens backpropped on). 0 = disabled."
    ),
    save_every_n_tokens: int = typer.Option(
        0, "--save-every-n-tokens", help="Save checkpoint every N loss-counted tokens (tokens backpropped on, 0 = disabled)"
    ),
    # flash attention / memory optimization
    use_flash_attn: bool = typer.Option(
        False, "--flash-attn", help="Enable Flash Attention 2 with padding-free training"
    ),
    # wandb params
    use_wandb: bool = typer.Option(False, "--wandb", help="Enable wandb logging"),
    wandb_project: str = typer.Option("mini-grpo-gsm8k", "--wandb-project", help="Wandb project name"),
    wandb_run_name: str = typer.Option(None, "--wandb-run", help="Wandb run name (auto-generated if not set)"),
    wandb_entity: str = typer.Option(None, "--wandb-entity", help="Wandb entity/team name"),
    # verbosity
    verbose_rollouts: bool = typer.Option(
        False, "--verbose-rollouts", help="Show detailed best/worst group rollouts after each batch"
    ),
):
    set_determinism(seed)

    # load the raw dataset
    # train_dataset = JsonlDataset(data_path)
    train_dataset = datasets.load_dataset("json", data_files=train_path, split="train")
    eval_dataset = None

    if eval_split > 0:
        dataset_dict = train_dataset.train_test_split(test_size=eval_split, seed=seed)
        train_dataset = dataset_dict["train"]
        eval_dataset = dataset_dict["test"]

    # Print dataset statistics
    typer.secho(f"\n✓ Loaded {len(train_dataset)} training samples", fg=typer.colors.GREEN)
    if eval_dataset:
        typer.secho(f"✓ Loaded {len(eval_dataset)} evaluation samples", fg=typer.colors.GREEN)

    # Initialize wandb if enabled
    if use_wandb:
        if not WANDB_AVAILABLE:
            typer.secho("Warning: wandb is not installed. Install with 'pip install wandb'", fg=typer.colors.YELLOW)
            use_wandb = False
        else:
            run_config = {
                "model_name": model_name,
                "epochs": epochs,
                "max_steps": max_steps,
                "token_train_budget": token_train_budget,
                "batch_size": batch_size,
                "group_size": group_size,
                "inner_batch_size": inner_batch_size,
                "inner_epochs": inner_epochs,
                "lr": lr,
                "kl_strength": kl_strength,
                "clip_eps": clip_eps,
                "max_new_tokens": max_new_tokens,
                "max_seq_len": max_seq_len,
                "temperature": temperature,
                "optimizer": optimizer_type,
                "max_tokens_per_microbatch": max_tokens_per_microbatch,
                "save_every": save_every,
                "save_every_n_tokens": save_every_n_tokens,
            }
            wandb.init(
                project=wandb_project,
                entity=wandb_entity,
                name=wandb_run_name,
                config=run_config,
            )
            typer.secho("✓ Wandb initialized", fg=typer.colors.GREEN)

    # device setup
    train_device = torch.device("cuda", gpu)
    torch.cuda.set_device(gpu)  # required for NCCL to use the correct GPU

    # Model loading kwargs
    # Flash Attention 2 requires bf16/fp16 weights, otherwise use FP32 for mixed precision
    if use_flash_attn:
        # Load in FP32 first, then apply FSDP2 MixedPrecisionPolicy for FP32 master weights
        policy_model_kwargs = {
            "device_map": train_device,
            "torch_dtype": torch.float32,  # Load FP32, FSDP2 will handle bf16 forward
            "attn_implementation": "flash_attention_2",
        }
        ref_model_kwargs = {
            "device_map": train_device,
            "torch_dtype": torch.float16,  # Reference model in fp16 (inference only, better precision)
            "attn_implementation": "flash_attention_2",
        }
        typer.secho(
            "✓ Using Flash Attention 2 with FSDP2 mixed precision (FP32 master weights, bf16 forward)",
            fg=typer.colors.GREEN,
        )
    else:
        # Without Flash Attention: FP32 policy for mixed precision, FP16 reference
        policy_model_kwargs = {
            "device_map": train_device,
            "torch_dtype": torch.float32,  # FP32 master weights
        }
        ref_model_kwargs = {
            "device_map": train_device,
            "torch_dtype": torch.float16,  # Frozen, FP16 for better precision
        }

    # Initialize policy model
    model = AutoModelForCausalLM.from_pretrained(model_name, **policy_model_kwargs)

    # Apply FSDP2 with MixedPrecisionPolicy for Flash Attention mode
    if use_flash_attn:
        import torch.distributed as dist
        from torch.distributed._composable.fsdp import fully_shard, MixedPrecisionPolicy

        # Initialize process group for single-GPU FSDP2 (if not already initialized by torchrun)
        if not dist.is_initialized():
            os.environ.setdefault("MASTER_ADDR", "localhost")
            os.environ.setdefault("MASTER_PORT", "29500")
            os.environ.setdefault("RANK", "0")
            os.environ.setdefault("WORLD_SIZE", "1")
            dist.init_process_group(backend="nccl")
            typer.secho("✓ Initialized single-GPU distributed process group", fg=typer.colors.GREEN)

        mp_policy = MixedPrecisionPolicy(
            param_dtype=torch.bfloat16,  # Forward/backward in bf16 (Flash Attention compatible)
            reduce_dtype=torch.float32,  # Gradient reduction in fp32
        )
        # Apply FSDP2 to each transformer layer for memory efficiency
        for layer in model.model.layers:
            fully_shard(layer, mp_policy=mp_policy)
        fully_shard(model, mp_policy=mp_policy)
        typer.secho("✓ Policy model wrapped with FSDP2 MixedPrecisionPolicy", fg=typer.colors.GREEN)
    else:
        typer.secho("✓ Policy model loaded with FP32 master weights (mixed precision training)", fg=typer.colors.GREEN)

    # Reference model (frozen)
    ref_model = AutoModelForCausalLM.from_pretrained(model_name, **ref_model_kwargs)
    ref_model.eval()
    ref_model.requires_grad_(False)
    if use_flash_attn:
        typer.secho("✓ Reference model loaded in BF16 (frozen)", fg=typer.colors.GREEN)
    else:
        typer.secho("✓ Reference model loaded in FP16 (frozen)", fg=typer.colors.GREEN)
    tokenizer: Qwen2Tokenizer = AutoTokenizer.from_pretrained(model_name)

    # align tokenizer and tokens
    for m in [model, ref_model]:
        if tokenizer.pad_token_id and not m.config.pad_token_id:
            m.config.pad_token_id = tokenizer.pad_token_id
            typer.secho(
                f"model '{model_name}' doesn't have a pad_token_id, setting it to {tokenizer.pad_token_id}",
                fg=typer.colors.BRIGHT_BLUE,
            )

    # create optimizer
    if use_flash_attn and optimizer_type.lower() == "muon":
        # Use FSDP2-compatible Muon optimizer
        optimizer = create_fsdp2_muon_optimizer(
            model=model,
            muon_lr=lr,
            adamw_lr=lr,
            beta1=beta1,
            beta2=beta2,
            weight_decay=wd,
        )
        typer.secho(f"✓ Using MUON optimizer (FSDP2-compatible via muon-fsdp2, lr={lr})", fg=typer.colors.GREEN)
    else:
        optimizer = create_optimizer(
            model=model,
            optimizer_type=optimizer_type,
            lr=lr,
            beta1=beta1,
            beta2=beta2,
            weight_decay=wd,
            muon_lr=lr,
        )
        if use_flash_attn:
            typer.secho(f"✓ Using {optimizer_type.upper()} optimizer (FP32 states via FSDP2)", fg=typer.colors.GREEN)
        else:
            typer.secho(f"✓ Using {optimizer_type.upper()} optimizer", fg=typer.colors.GREEN)

    # create training components
    ctx = TrainingComponents(
        seed=seed,
        optimizer=optimizer,
        model=model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        device=train_device,
        hyperparams=Hyperparameters(
            lr=lr,
            model_name=model_name,
            max_seq_len=max_seq_len,
            batch_size=batch_size,
            group_size=group_size,
            epochs=epochs,
            inner_epochs=inner_epochs,
            inner_batch_size=inner_batch_size,
            eps=clip_eps,
            kl_penalty_strength=kl_strength,
        ),
        train_tokenizer=configure_tokenizer(model_name),
        sampling_params=SamplingParams(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            max_tokens=max_seq_len,
            top_p=1.0,
            top_k=0.0,
            repetition_penalty=1.0,
        ),
        output_dir=output_dir,
    )

    # check if we need to write into output dir
    if output_dir is not None and not ctx.valid_save_dir():
        typer.secho(
            f"Error: Cannot write to output directory '{output_dir}'",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1)

    preview_tokenization(train_dataset, tokenizer)

    # Baseline evaluation before training
    if eval_dataset is not None and len(eval_dataset) > 0:
        typer.secho("\n" + "=" * 60, fg=typer.colors.BRIGHT_YELLOW)
        typer.secho("  BASELINE EVALUATION (Before Training)", fg=typer.colors.BRIGHT_YELLOW, bold=True)
        typer.secho("=" * 60, fg=typer.colors.BRIGHT_YELLOW)
        baseline_metrics = eval_model(eval_dataset, ctx, return_metrics=True)
        if use_wandb and baseline_metrics:
            wandb.log(
                {
                    "baseline/pass_above_50": baseline_metrics.get("above_50", 0),
                    "baseline/pass_at_100": baseline_metrics.get("at_100", 0),
                },
                step=0,
            )

    # Step counters
    global_step = 0  # Counts batches of prompts processed
    optim_step = 0  # Counts optimizer.step() calls
    tokens_trained = 0  # Counts tokens backpropped on (cumulative)
    tokens_since_checkpoint = 0  # Tokens since last checkpoint (resets after save)

    # Determine training mode
    use_step_based = max_steps > 0
    use_token_based = token_train_budget > 0
    if use_token_based:
        typer.secho(f"Training for {token_train_budget:,} tokens", fg=typer.colors.CYAN)
    elif use_step_based:
        typer.secho(f"Training for {max_steps} optimizer steps", fg=typer.colors.CYAN)
    else:
        typer.secho(f"Training for {epochs} epoch(s)", fg=typer.colors.CYAN)

    # Training loop
    epoch = 0
    training_complete = False
    while not training_complete:
        minibatches: list[Sample] = []

        # Set up progress bar
        if use_token_based:
            desc = f"Tokens {tokens_trained:,}/{token_train_budget:,}"
        elif use_step_based:
            desc = f"Step {optim_step}/{max_steps}"
        else:
            desc = f"Epoch {epoch + 1}/{epochs}"

        pbar = tqdm(
            train_dataset.shuffle(
                seed=ctx.seed + epoch,
            ).iter(batch_size),
            desc=desc,
            total=len(train_dataset) // batch_size,
        )

        for batch in pbar:
            global_step += 1

            # Check model health before generating rollouts
            if not check_model_health(model):
                typer.secho(
                    "FATAL: Model weights contain NaN/Inf! Training cannot continue.",
                    fg=typer.colors.RED,
                    bold=True,
                )
                typer.secho(
                    "This typically happens when learning rate is too high or numerical instability occurs.",
                    fg=typer.colors.YELLOW,
                )
                if use_wandb:
                    wandb.finish(exit_code=1)
                raise typer.Exit(code=1)

            # Preview questions in this batch
            num_samples = len(batch["problem"])
            typer.secho(f"\n📋 Batch Preview ({num_samples} questions):", fg=typer.colors.CYAN, bold=True)
            for i, problem in enumerate(batch["problem"][:3], 1):
                question = problem if len(problem) <= 100 else problem[:97] + "..."
                typer.secho(f"  {i}. {question}", fg=typer.colors.WHITE)
            if num_samples > 3:
                typer.secho(f"  ... and {num_samples - 3} more", fg=typer.colors.WHITE, dim=True)

            # Generate rollouts for each prompt
            rollouts = generate_rollouts(
                ctx,
                model,
                tokenizer,
                batch,
                ctx.hyperparams.batch_size,
                ctx.hyperparams.group_size,
                sampling_params=ctx.sampling_params,
                show_tqdm=True,
            )

            # Print rollout summary with statistics and examples
            print_example_rollout(rollouts, step=optim_step, verbose=verbose_rollouts)

            # Calculate batch metrics
            total_rewards = sum(rollout.reward for sample in rollouts for rollout in sample.rollouts)
            total_rollouts = sum(len(sample.rollouts) for sample in rollouts)
            avg_reward = total_rewards / total_rollouts if total_rollouts > 0 else 0.0
            parsable_count = sum(1 for s in rollouts for r in s.rollouts if r.is_parsable)
            correct_count = sum(1 for s in rollouts for r in s.rollouts if r.is_correct)
            parsable_rate = parsable_count / total_rollouts if total_rollouts > 0 else 0.0
            correct_rate = correct_count / total_rollouts if total_rollouts > 0 else 0.0

            # Update tqdm postfix (will be updated again after training with final step count)
            pbar.set_postfix({"avg_reward": f"{avg_reward:.4f}", "acc": f"{correct_rate:.2%}"})

            # Log batch metrics to wandb
            if use_wandb:
                wandb.log(
                    {
                        "rollout/avg_reward": avg_reward,
                        "rollout/parsable_rate": parsable_rate,
                        "rollout/correct_rate": correct_rate,
                        "rollout/parsable_count": parsable_count,
                        "rollout/correct_count": correct_count,
                        "rollout/total_rollouts": total_rollouts,
                        "rollout/num_prompts": len(rollouts),
                    },
                    step=optim_step,
                )

            # Train policy on rollouts
            prev_optim_step = optim_step
            prev_tokens = tokens_trained
            optim_step, tokens_trained, should_stop = train_policy_on_rollouts(
                rollouts,
                ctx,
                use_wandb=use_wandb,
                global_step=global_step,
                use_packed=use_flash_attn,
                max_tokens_per_microbatch=max_tokens_per_microbatch,
                current_optim_step=optim_step,
                max_steps=max_steps,
                current_tokens_trained=tokens_trained,
                token_train_budget=token_train_budget,
            )
            steps_this_batch = optim_step - prev_optim_step
            tokens_this_batch = tokens_trained - prev_tokens
            tokens_since_checkpoint += tokens_this_batch
            token_budget_str = f"{token_train_budget:,}" if token_train_budget > 0 else "∞"
            typer.secho(
                f"Completed {steps_this_batch} optimizer steps (total: {optim_step}/{max_steps if max_steps > 0 else '∞'}) | "
                f"Tokens: +{tokens_this_batch:,} (total: {tokens_trained:,}/{token_budget_str})",
                fg=typer.colors.CYAN,
            )
            minibatches.extend(rollouts)

            # Update progress bar with current step/tokens
            if use_token_based:
                pbar.set_description(f"Tokens {tokens_trained:,}/{token_train_budget:,}")
            elif use_step_based:
                pbar.set_description(f"Step {optim_step}/{max_steps}")
            pbar.set_postfix({"avg_reward": f"{avg_reward:.4f}", "acc": f"{correct_rate:.2%}"})

            # Clear cache after training step before next rollout generation
            torch.cuda.empty_cache()

            # Check if we've reached max_steps or token budget
            if should_stop:
                if token_train_budget > 0 and tokens_trained >= token_train_budget:
                    typer.secho(
                        f"\nReached {tokens_trained:,} tokens (budget: {token_train_budget:,}). Stopping training.",
                        fg=typer.colors.GREEN,
                    )
                else:
                    typer.secho(f"\nReached {max_steps} optimizer steps. Stopping training.", fg=typer.colors.GREEN)
                training_complete = True
                # Save final checkpoint before breaking
                if output_dir:
                    ctx.save_checkpoint(optim_step, is_step=True)
                break

            # Save checkpoint at step intervals
            if save_every > 0 and optim_step % save_every == 0 and output_dir:
                typer.secho(f"\n[Step {optim_step}] Saving checkpoint...", fg=typer.colors.CYAN)
                ctx.save_checkpoint(optim_step, is_step=True)

            # Save checkpoint at token intervals (resetting counter approach)
            if save_every_n_tokens > 0 and output_dir:
                if tokens_since_checkpoint >= save_every_n_tokens:
                    typer.secho(f"\n[Tokens {tokens_trained:,}] Saving checkpoint...", fg=typer.colors.CYAN)
                    ctx.save_checkpoint(tokens_trained, is_step=True, suffix=f"tokens_{tokens_trained}")
                    # Reset counter, keeping the overflow
                    tokens_since_checkpoint = tokens_since_checkpoint - save_every_n_tokens

            # Intermediate evaluation (based on optim_step)
            if eval_every > 0 and optim_step % eval_every == 0:
                if eval_dataset is not None and len(eval_dataset) > 0:
                    typer.secho(f"\n[Step {optim_step}] Running intermediate evaluation...", fg=typer.colors.CYAN)
                    metrics = eval_model(eval_dataset, ctx, return_metrics=True)
                    if use_wandb and metrics:
                        wandb.log(
                            {
                                "eval/pass_above_50": metrics.get("above_50", 0),
                                "eval/pass_at_100": metrics.get("at_100", 0),
                            },
                            step=optim_step,
                        )
                    torch.cuda.empty_cache()

        # End of epoch handling
        if minibatches:
            # Calculate and display epoch scorecard
            epoch_metrics = display_scorecard(
                minibatches, epoch, epochs if not use_step_based else 0, return_metrics=True
            )

            # Log epoch metrics to wandb
            if use_wandb and epoch_metrics:
                wandb.log(
                    {
                        "epoch": epoch + 1,
                        "epoch/parsable_pct": epoch_metrics.get("parsable_pct", 0),
                        "epoch/correct_pct": epoch_metrics.get("correct_pct", 0),
                        "epoch/accuracy_pct": epoch_metrics.get("accuracy_pct", 0),
                    },
                    step=optim_step,
                )

            # End-of-epoch evaluation
            if eval_dataset is not None and len(eval_dataset) > 0:
                metrics = eval_model(eval_dataset, ctx, return_metrics=True)
                if use_wandb and metrics:
                    wandb.log(
                        {
                            "eval/pass_above_50": metrics.get("above_50", 0),
                            "eval/pass_at_100": metrics.get("at_100", 0),
                        },
                        step=optim_step,
                    )
                torch.cuda.empty_cache()

        # Save checkpoint at end of epoch/training (only if not using interval saving)
        if save_every == 0 and output_dir:
            if use_step_based:
                ctx.save_checkpoint(optim_step, is_step=True)
            else:
                ctx.save_checkpoint(epoch, is_step=False)

        epoch += 1

        # Check epoch-based termination
        if not use_step_based and epoch >= epochs:
            training_complete = True

    # Finish wandb run
    if use_wandb:
        wandb.finish()
        typer.secho("✓ Wandb run finished", fg=typer.colors.GREEN)
        

@app.command()
def rs_train(
    data_path: str = typer.Option(..., "--data-path", help="Path to the training data"),
    output_dir: str = typer.Option(..., "--output-dir", help="Path to the output directory"),
    model_name: str = typer.Option("Qwen/Qwen2-1.5B-Instruct", "--model", "-m", help="Model name or path"),

    max_tokens: int = typer.Option(..., "--max-tokens", help="Maximum loss-counted tokens to train on (tokens backpropped on, 0 = use epochs or steps)"),
    num_inner_epochs: int = typer.Option(1, "--inner-epochs", help="Number of inner epochs"),

    max_seq_len: int = typer.Option(8192, "--msl", "--max-seq-len", help="Maximum sequence length for a single sample"),
    max_tokens_per_gpu: int = typer.Option(8192, "--max-tokens-per-gpu", help="Max tokens per GPU"),
    save_every_n_tokens: int = typer.Option(
        0, "--save-every-n-tokens", help="Save checkpoint every N loss-counted tokens (tokens backpropped on, 0 = disabled)"
    ),

    # number of samples that we'd accept in a given batch
    samples_to_accept: int = typer.Option(1, "--samples-to-accept", help="Number of samples to accept per rollout batch"),
    inference_batch_size: int = typer.Option(32, "--inference-batch-size", help="Number of prompts to batch together when generating GRPO rollouts."),
    inference_group_size: int = typer.Option(16, "--inference-group-size", help="Group size / number of rollouts to generate from a single prompt"),

    # sampling params
    temperature: float = typer.Option(0.7, "-t", "--temp", help="sampling temperature"),
    max_new_tokens: int = typer.Option(512, "--max-new-tokens", help="Maximum number of new tokens to generate"),
    top_p: float = typer.Option(1.0, "--top-p", help="The proportion of the probability mass which we should consider for sampling."),
    top_k: int = typer.Option(0, "--top-k", help="sample only the top k highest probability tokens"),
    
    # wandb options
    use_wandb: bool = typer.Option(False, "--wandb", help="Enable wandb logging"),
    wandb_project: str = typer.Option("gsm8k-comparison", "--wandb-project", help="Wandb project name"),
    wandb_run_name: str = typer.Option(None, "--wandb-run", help="Wandb run name (auto-generated if not set)"),
    wandb_entity: str = typer.Option(None, "--wandb-entity", help="Wandb entity/team name"),
    
    seed: int = typer.Option(67, "--seed", help="Random seed"),

    optimizer_type: str = typer.Option("adamw", "-O", "--optimizer", help="Optimizer type: 'adamw' or 'muon'"),
    lr: float = typer.Option(1e-5, "--lr", help="Learning rate (used for all parameters)"),
    beta1: float = typer.Option(0.9, help="Adam beta1 parameter"),
    beta2: float = typer.Option(0.95, help="Adam beta2 parameter"),
    wd: float = typer.Option(0.0, "--wd", help="Weight decay"),

    # device selection
    gpu: int = typer.Option(0, "--gpu", "-g", help="CUDA GPU index to use for training"),
    vllm_gpus: str = typer.Option("1", "--vllm-gpus", help="Comma-separated GPU indices for vLLM inference (e.g. '1,2,3' for data parallel)"),

    # validation
    validation_path: str = typer.Option(None, "--validation-path", help="Path to validation data (same schema as training)"),
):
    """
    Train a model with Rejection Sampling on the given dataset.
    """

    # loads the trainer
    trainer = RSTrainer(
        data_path=data_path,
        model_name=model_name,
        output_dir=output_dir,
        token_budget=max_tokens,
        inner_epochs=num_inner_epochs,
        inner_batch_size=samples_to_accept,  # this isn't exactly correct
        save_every_n_tokens=save_every_n_tokens,
        samples_to_accept=samples_to_accept,
        inference_batch_size=inference_batch_size,
        inference_group_size=inference_group_size,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
        max_seq_len=max_seq_len,
        max_tokens_per_gpu=max_tokens_per_gpu,
        use_wandb=use_wandb,
        wandb_project=wandb_project,
        wandb_run_name=wandb_run_name,
        wandb_entity=wandb_entity,
        seed=seed,
        optimizer_type=optimizer_type,
        lr=lr,
        beta1=beta1,
        beta2=beta2,
        weight_decay=wd,
        gpu=gpu,
        vllm_gpus=vllm_gpus,
        vllm_gpu_memory_utilization=0.9,
        validation_path=validation_path,
    )
    # runs the training loop
    trainer.train()
    dist.barrier()
    dist.destroy_process_group()


@app.command()
def grpo_train(
    data_path: str = typer.Option(..., "--data-path", help="Path to training data (jsonl)"),
    output_dir: str = typer.Option(..., "--output-dir", help="Path to the output directory"),
    model_name: str = typer.Option(
        "Qwen/Qwen2-1.5B-Instruct", "--model", "-m", help="Model name or path",
    ),

    max_tokens: int = typer.Option(0, "--max-tokens", help="Total token budget (0 = use --max-steps)"),
    max_steps: int = typer.Option(0, "--max-steps", help="Max optimizer steps (0 = use --max-tokens)"),
    inner_epochs: int = typer.Option(2, "--inner-epochs", help="Inner epochs per rollout batch"),
    inner_batch_size: int = typer.Option(32, "--inner-batch-size", help="Training batch size for GRPO inner loop"),

    save_every_n_tokens: int = typer.Option(
        0, "--save-every-n-tokens", help="Save checkpoint every N tokens (0 = disabled)"
    ),
    save_every_n_steps: int = typer.Option(
        0, "--save-every-n-steps", help="Save checkpoint every N optimizer steps (0 = disabled)"
    ),

    # GRPO settings
    group_size: int = typer.Option(16, "-G", "--group-size", help="Rollouts per prompt"),
    batch_size: int = typer.Option(64, "-B", "--batch-size", help="Prompts per rollout iteration"),
    clip_eps: float = typer.Option(0.2, "--clip-eps", help="GRPO clip epsilon"),
    kl_strength: float = typer.Option(0.01, "--kl", help="KL penalty strength"),
    entropy_strength: float = typer.Option(0.0, "--entropy-strength", help="Entropy bonus strength (0 = disabled)"),
    gradient_clip: float = typer.Option(1.0, "--gradient-clip", help="Gradient clipping max norm"),

    # Sampling
    temperature: float = typer.Option(0.7, "-t", "--temp", help="Sampling temperature"),
    max_new_tokens: int = typer.Option(512, "--max-new-tokens", help="Max tokens to generate per response"),
    top_p: float = typer.Option(1.0, "--top-p", help="Top-p sampling threshold"),
    top_k: int = typer.Option(0, "--top-k", help="Top-k sampling (0 = disabled)"),
    max_seq_len: int = typer.Option(8192, "--msl", "--max-seq-len", help="Maximum sequence length"),

    # Memory optimization
    max_tokens_per_microbatch: int = typer.Option(
        0, "--max-tokens-per-microbatch",
        help="Max tokens per microbatch for gradient accumulation (0 = no limit)"
    ),

    # Optimizer
    optimizer_type: str = typer.Option("adamw", "-O", "--optimizer", help="Optimizer type: 'adamw' or 'muon'"),
    lr: float = typer.Option(1e-5, "--lr", help="Learning rate"),
    beta1: float = typer.Option(0.9, "--beta1", help="Adam beta1"),
    beta2: float = typer.Option(0.95, "--beta2", help="Adam beta2"),
    wd: float = typer.Option(0.0, "--wd", help="Weight decay"),

    # Precision
    precision: str = typer.Option(
        "fp32", "--precision", "-P",
        help="'fp32' | 'bf16' | 'mixed' (FP32 master weights + BF16 fwd via FSDP2)",
    ),

    # Device
    gpu: int = typer.Option(0, "--gpu", "-g", help="CUDA GPU for training"),
    vllm_gpus: str = typer.Option("1", "--vllm-gpus", help="Comma-separated GPU indices for vLLM inference"),

    # Wandb
    use_wandb: bool = typer.Option(False, "--wandb", help="Enable wandb logging"),
    wandb_project: str = typer.Option("mini-grpo", "--wandb-project", help="Wandb project name"),
    wandb_run_name: str = typer.Option(None, "--wandb-run", help="Wandb run name"),
    wandb_entity: str = typer.Option(None, "--wandb-entity", help="Wandb entity"),

    seed: int = typer.Option(67, "--seed", help="Random seed"),

    validation_path: str = typer.Option(None, "--validation-path", help="Path to validation data"),

    # Task
    task: str = typer.Option("gsm8k", "--task", help="Task name: 'gsm8k' or 'countdown'"),
):
    """
    Single-GPU GRPO training with vLLM inference and configurable precision.

    Uses vLLM on separate GPU(s) for fast rollout generation. Supports
    multiple tasks via --task flag (gsm8k, countdown).

    Use --precision to toggle between FP32 and BF16 for paired sparsity
    experiments. An initial checkpoint is saved automatically.
    """
    from grpo_trainer import GRPOTrainer
    from tasks import get_reward_fn

    trainer = GRPOTrainer(
        data_path=data_path,
        model_name=model_name,
        output_dir=output_dir,
        token_budget=max_tokens if max_tokens > 0 else 0,
        inner_epochs=inner_epochs,
        inner_batch_size=inner_batch_size,
        save_every_n_tokens=save_every_n_tokens,
        group_size=group_size,
        batch_size=batch_size,
        clip_eps=clip_eps,
        kl_strength=kl_strength,
        entropy_strength=entropy_strength,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
        max_seq_len=max_seq_len,
        max_tokens_per_microbatch=max_tokens_per_microbatch,
        optimizer_type=optimizer_type,
        lr=lr,
        beta1=beta1,
        beta2=beta2,
        weight_decay=wd,
        gradient_clip=gradient_clip,
        precision=precision,
        gpu=gpu,
        vllm_gpus=vllm_gpus,
        use_wandb=use_wandb,
        wandb_project=wandb_project,
        wandb_run_name=wandb_run_name,
        wandb_entity=wandb_entity,
        seed=seed,
        validation_path=validation_path,
        reward_fn=get_reward_fn(task),
    )
    trainer.train()


@app.command()
def sft_train(
    # Data paths
    data_path: str = typer.Option(..., "--data-path", help="Path to the training data (jsonl with 'messages' field)"),
    model_name: str = typer.Option("Qwen/Qwen2-1.5B-Instruct", "--model", "-m", help="Model name or path"),
    output_dir: str = typer.Option(..., "--output-dir", help="Path to the output directory"),
    # Training mode
    max_steps: int = typer.Option(0, "--max-steps", help="Maximum training steps (0 = use epochs or tokens)"),
    max_tokens: int = typer.Option(0, "--max-tokens", help="Maximum loss-counted tokens to train on (tokens backpropped on, 0 = use epochs or steps)"),
    num_epochs: int = typer.Option(1, "--epochs", help="Number of epochs (ignored if --max-steps or --max-tokens > 0)"),
    # Batch settings
    effective_batch_size: int = typer.Option(32, "-B", "--batch-size", help="Effective batch size"),
    max_tokens_per_gpu: int = typer.Option(8192, "--max-tokens-per-gpu", help="Max tokens per GPU"),
    max_seq_len: int = typer.Option(2048, "--max-seq-len", help="Maximum sequence length"),
    # Optimizer settings
    optimizer_type: str = typer.Option("adamw", "-O", "--optimizer", help="Optimizer: 'adamw' or 'muon'"),
    lr: float = typer.Option(1e-5, "--lr", help="Learning rate"),
    muon_lr: float = typer.Option(None, "--muon-lr", help="Muon-specific LR (defaults to --lr)"),
    beta1: float = typer.Option(0.9, "--beta1", help="Adam beta1"),
    beta2: float = typer.Option(0.95, "--beta2", help="Adam beta2"),
    weight_decay: float = typer.Option(0.0, "--wd", help="Weight decay"),
    # LR scheduler
    lr_scheduler: str = typer.Option("cosine", "--lr-scheduler", help="LR scheduler type"),
    warmup_steps: int = typer.Option(0, "--warmup-steps", help="Number of warmup steps"),
    # Checkpointing
    save_final_checkpoint: bool = typer.Option(True, "--save-final/--no-save-final", help="Save final checkpoint"),
    checkpoint_at_epoch: bool = typer.Option(False, "--checkpoint-at-epoch", help="Save checkpoint each epoch"),
    save_every_steps: int = typer.Option(
        0, "--save-every", help="Save checkpoint every N optimizer steps (0 = disabled)"
    ),
    save_every_n_tokens: int = typer.Option(
        0, "--save-every-n-tokens", help="Save checkpoint every N loss-counted tokens (tokens backpropped on, 0 = disabled)"
    ),
    # Data processing
    use_processed_dataset: bool = typer.Option(False, "--use-processed", help="Data is already tokenized"),
    unmask_messages: bool = typer.Option(False, "--unmask", help="Train on all tokens (not just assistant)"),
    # Wandb
    use_wandb: bool = typer.Option(False, "--wandb", help="Enable wandb logging"),
    wandb_project: str = typer.Option("muon-rl", "--wandb-project", help="Wandb project name"),
    wandb_run_name: str = typer.Option(None, "--wandb-run", help="Wandb run name"),
    wandb_entity: str = typer.Option(None, "--wandb-entity", help="Wandb entity"),
    # Precision
    precision: str = typer.Option(
        "mixed", "--precision", "-P",
        help="'fp32' | 'bf16' | 'mixed' (FP32 master weights + BF16 fwd via FSDP2)",
    ),
    # Misc
    seed: int = typer.Option(67, "--seed", help="Random seed"),
    use_liger: bool = typer.Option(False, "--liger", help="Use Liger kernels"),
    num_gpus: int = typer.Option(1, "--num-gpus", help="Number of GPUs to use"),
    validation_split: float = typer.Option(0.0, "--validation-split", help="Fraction of data to use for validation"),
    validation_frequency: int = typer.Option(0, "--validation-frequency", help="Frequency of validation (in steps)"),
    # GSM8K evaluation
    gsm8k_eval_path: str = typer.Option(None, "--gsm8k-eval-path", help="Path to GSM8K eval dataset"),
    gsm8k_eval_frequency: int = typer.Option(None, "--gsm8k-eval-frequency", help="GSM8K eval frequency (steps)"),
    gsm8k_max_new_tokens: int = typer.Option(512, "--gsm8k-max-new-tokens", help="Max tokens for GSM8K eval"),
    gsm8k_temperature: float = typer.Option(0.0, "--gsm8k-temperature", help="Temperature for GSM8K eval"),
    gsm8k_eval_samples: int = typer.Option(None, "--gsm8k-eval-samples", help="Number of GSM8K eval samples"),
    gsm8k_use_vllm: bool = typer.Option(
        False, "--gsm8k-use-vllm", help="Use vLLM for fast GSM8K eval (runs after checkpoint saves)"
    ),
    gsm8k_vllm_gpu_memory_utilization: float = typer.Option(
        0.8, "--gsm8k-vllm-gpu-memory-utilization", help="GPU memory utilization for vLLM"
    ),
    disable_kl: bool = typer.Option(False, "--disable-kl", help="Disable KL divergence tracking"),
):
    """
    Run SFT training using mini_trainer via training_hub.

    Supports the same optimizer options as the GRPO train command for fair comparison.

    Example:
        python cli.py sft-train \\
            --data-path gsm8k-data/gsm8k_sft_train.jsonl \\
            --model Qwen/Qwen2-1.5B-Instruct \\
            --output-dir /path/to/checkpoints \\
            --optimizer muon --lr 1e-5 \\
            --max-steps 1000 \\
            --wandb --wandb-run "muon-sft-baseline"
    """
    from training_hub import osft
    from mini_trainer import TrainingMode

    # Determine training mode (priority: tokens > steps > epochs)
    if max_tokens > 0:
        training_mode = TrainingMode.TOKEN
        typer.secho(f"Training for {max_tokens:,} tokens", fg=typer.colors.CYAN)
    elif max_steps > 0:
        training_mode = TrainingMode.STEP
        typer.secho(f"Training for {max_steps} steps", fg=typer.colors.CYAN)
    else:
        training_mode = TrainingMode.EPOCH
        typer.secho(f"Training for {num_epochs} epoch(s)", fg=typer.colors.CYAN)

    # Build optional kwargs
    optional_kwargs = {}
    if use_wandb:
        optional_kwargs["wandb_project"] = wandb_project
        if wandb_run_name:
            optional_kwargs["wandb_run_name"] = wandb_run_name
        if wandb_entity:
            optional_kwargs["wandb_entity"] = wandb_entity
    if validation_frequency > 0:
        optional_kwargs["validation_frequency"] = validation_frequency
    if validation_split > 0:
        optional_kwargs["validation_split"] = validation_split

    if muon_lr is not None:
        optional_kwargs["muon_lr"] = muon_lr

    # GSM8K evaluation parameters
    if gsm8k_eval_path:
        optional_kwargs["gsm8k_eval_path"] = gsm8k_eval_path
        if gsm8k_eval_frequency:
            optional_kwargs["gsm8k_eval_frequency"] = gsm8k_eval_frequency
        if gsm8k_max_new_tokens != 512:
            optional_kwargs["gsm8k_max_new_tokens"] = gsm8k_max_new_tokens
        if gsm8k_temperature > 0:
            optional_kwargs["gsm8k_temperature"] = gsm8k_temperature
        if gsm8k_eval_samples:
            optional_kwargs["gsm8k_eval_samples"] = gsm8k_eval_samples
        if gsm8k_use_vllm:
            optional_kwargs["gsm8k_use_vllm"] = gsm8k_use_vllm
        if gsm8k_vllm_gpu_memory_utilization != 0.8:
            optional_kwargs["gsm8k_vllm_gpu_memory_utilization"] = gsm8k_vllm_gpu_memory_utilization

    # Derive train_dtype from precision
    train_dtype = "float32" if precision in ("fp32", "mixed") else "bfloat16"

    osft(
        model_path=model_name,
        data_path=data_path,
        ckpt_output_dir=output_dir,
        # SFT mode (not OSFT)
        unfreeze_rank_ratio=1.0,  # 1.0 = full fine-tuning (no freezing)
        osft=False,
        # Batch settings
        effective_batch_size=effective_batch_size,
        max_tokens_per_gpu=max_tokens_per_gpu,
        max_seq_len=max_seq_len,
        # Optimizer
        optimizer_type=optimizer_type,
        learning_rate=lr,
        beta1=beta1,
        beta2=beta2,
        weight_decay=weight_decay,
        # LR scheduler
        lr_scheduler=lr_scheduler,
        warmup_steps=warmup_steps,
        # Training mode
        training_mode=training_mode,
        num_epochs=num_epochs,
        max_steps=max_steps,
        max_tokens=max_tokens,
        # Checkpointing
        save_final_checkpoint=save_final_checkpoint,
        checkpoint_at_epoch=checkpoint_at_epoch,
        save_every_steps=save_every_steps if save_every_steps > 0 else None,
        save_every_n_tokens=save_every_n_tokens if save_every_n_tokens > 0 else None,
        # Data processing
        use_processed_dataset=use_processed_dataset,
        unmask_messages=unmask_messages,
        # KL divergence tracking (enabled by default for comparison with GRPO)
        compute_kl=not disable_kl,
        # Misc
        seed=seed,
        use_liger=use_liger,
        nproc_per_node=num_gpus,
        # Precision control
        precision=precision,
        train_dtype=train_dtype,
        # Ensures training saves FP32 checkpoints
        save_dtype='float32',
        **optional_kwargs,
    )


# ─────────────────────────────────────────────────────────────────────────────
# COUNTDOWN TASK COMMANDS
# ─────────────────────────────────────────────────────────────────────────────


@app.command()
def generate_hard_countdown(
    n_train: int = typer.Option(50000, "--n-train", help="Number of training samples to generate"),
    n_val: int = typer.Option(1000, "--n-val", help="Number of validation samples to generate"),
    n_test: int = typer.Option(0, "--n-test", help="Number of held-out test samples (0 = none)"),
    few_shot: int = typer.Option(0, "--few-shot", help="Number of solved ICL examples to prepend (0 = none)"),
    think: bool = typer.Option(False, "--think/--no-think", help="Require <think> traces (ICL + reward)"),
    r1_prompt: bool = typer.Option(False, "--r1-prompt/--no-r1-prompt", help="Prepend R1 system prompt (independent of --think)"),
    hard: bool = typer.Option(True, "--hard/--all", help="Only problems requiring * or / (default: --hard)"),
    synthetic: bool = typer.Option(False, "--synthetic/--hf", help="Generate synthetic data (default: use HuggingFace dataset)"),
    seed: int = typer.Option(42, "--seed", help="Random seed"),
    output_dir: str = typer.Option("generated_data", "--output-dir", help="Directory to save datasets"),
):
    """
    Generate countdown training data.

    By default, loads from Jiayi-Pan/Countdown-Tasks-3to4 on HuggingFace
    and formats with ICL examples. Use --synthetic to generate problems
    via backward decomposition instead.

    Use --hard to keep only problems requiring * or / (default).
    Use --few-shot N to prepend N solved examples as multi-turn ICL context.
    Use --think to enable R1-style thinking.

    Outputs:
      - countdown_hard_train.jsonl
      - countdown_hard_val.jsonl
    """
    from countdown_utils import generate_synthetic_countdown, generate_countdown_from_hf

    if synthetic:
        result = generate_synthetic_countdown(
            n_train=n_train,
            n_val=n_val,
            seed=seed,
            few_shot=few_shot,
            think=think,
            hard=hard,
            r1_prompt=r1_prompt,
        )
    else:
        result = generate_countdown_from_hf(
            n_train=n_train,
            n_val=n_val,
            n_test=n_test,
            seed=seed,
            few_shot=few_shot,
            think=think,
            hard=hard,
            r1_prompt=r1_prompt,
        )

    os.makedirs(output_dir, exist_ok=True)

    for split_name in ["train", "val", "test"]:
        if split_name not in result:
            continue
        samples = result[split_name]
        ds = datasets.Dataset.from_list(samples)
        path = os.path.join(output_dir, f"countdown_hard_{split_name}.jsonl")
        ds.to_json(path)
        typer.secho(f"  Saved {len(samples)} {split_name} samples to '{path}'", fg=typer.colors.BLUE)

    parts = [f"{len(result['train'])} train", f"{len(result['val'])} val"]
    if "test" in result:
        parts.append(f"{len(result['test'])} test")
    typer.secho(
        f"\nGenerated {' + '.join(parts)} countdown samples (seed={seed})",
        fg=typer.colors.GREEN,
    )

    # Show examples
    if result["train"]:
        for i, ex in enumerate(result["train"][:3]):
            typer.secho(f"\nExample {i+1}:", fg=typer.colors.CYAN)
            typer.secho(f"  Numbers: {ex['numbers']}, Target: {ex['answer']}", fg=typer.colors.WHITE)


@app.command()
def generate_countdown_datasets(
    system_msg: str = typer.Option(
        None,
        "--system-msg",
        help="System message (default: countdown_utils.DEFAULT_COUNTDOWN_SYSTEM_MSG)",
    ),
    total_samples: int = typer.Option(5000, "--total-samples", help="Total samples to take from dataset (0 = all ~490K)"),
    seed: int = typer.Option(67, "--seed", help="Random seed for shuffling"),
    output_dir: str = typer.Option("generated_data", "--output-dir", help="Directory to save datasets"),
    val_split: float = typer.Option(0.05, "--val-split", help="Fraction of data for validation set"),
    test_split: float = typer.Option(0.05, "--test-split", help="Fraction of data for test set"),
    grpo_only: bool = typer.Option(False, "--grpo-only", help="Skip solving (much faster, GRPO data only)"),
    skip_verify: bool = typer.Option(False, "--skip-verify", help="Skip verification of SFT solutions"),
):
    """
    Generate paired GRPO and SFT countdown datasets from Jiayi-Pan/Countdown-Tasks-3to4.

    Loads the full dataset from HuggingFace, shuffles with seed, takes the first
    --total-samples, solves each for SFT, splits into train/val/test, verifies all
    solutions through the reward function, and saves to disk.

    Use --grpo-only to skip solving (much faster, useful when you only need GRPO data).

    Outputs:
      - countdown_{grpo,sft}_{train,val,test}.jsonl
    """
    from countdown_utils import generate_countdown_dataset, DEFAULT_COUNTDOWN_SYSTEM_MSG

    if system_msg is None:
        system_msg = DEFAULT_COUNTDOWN_SYSTEM_MSG

    desc = "Loading" if grpo_only else "Loading and solving"
    typer.secho(f"{desc} countdown problems from HuggingFace...", fg=typer.colors.CYAN)
    result = generate_countdown_dataset(
        total_samples=total_samples,
        system_msg=system_msg,
        seed=seed,
        val_split=val_split,
        test_split=test_split,
        verify=not skip_verify,
        grpo_only=grpo_only,
    )

    os.makedirs(output_dir, exist_ok=True)

    for split_name in ["grpo_train", "grpo_val", "grpo_test", "sft_train", "sft_val", "sft_test"]:
        samples = result[split_name]
        ds = datasets.Dataset.from_list(samples)
        path = os.path.join(output_dir, f"countdown_{split_name}.jsonl")
        ds.to_json(path)
        typer.secho(f"  Saved {len(samples)} {split_name} samples to '{path}'", fg=typer.colors.BLUE)

    n_train = len(result["grpo_train"])
    n_val = len(result["grpo_val"])
    n_test = len(result["grpo_test"])
    typer.secho(
        f"\nGenerated {n_train} train + {n_val} val + {n_test} test countdown samples (seed={seed})",
        fg=typer.colors.GREEN,
        bold=True,
    )

    # Show examples
    num_examples = min(3, n_train)
    typer.secho(f"\n{'=' * 70}", fg=typer.colors.BRIGHT_CYAN)
    typer.secho(f"  DATASET EXAMPLES ({num_examples} samples)", fg=typer.colors.BRIGHT_CYAN, bold=True)
    typer.secho(f"{'=' * 70}", fg=typer.colors.BRIGHT_CYAN)

    for ex_idx in range(num_examples):
        grpo_sample = result["grpo_train"][ex_idx]
        sft_sample = result["sft_train"][ex_idx]

        typer.secho(f"\n{'─' * 70}", fg=typer.colors.WHITE)
        typer.secho(f"  Example {ex_idx + 1}", fg=typer.colors.BRIGHT_CYAN, bold=True)
        typer.secho(f"{'─' * 70}", fg=typer.colors.WHITE)

        typer.secho(f"  Numbers: {grpo_sample['numbers']}", fg=typer.colors.YELLOW)
        typer.secho(f"  Target:  {grpo_sample['answer']}", fg=typer.colors.YELLOW)

        typer.secho(f"\n  [GRPO Format] (prompt only)", fg=typer.colors.GREEN, bold=True)
        for msg in grpo_sample["messages"]:
            role = msg["role"].upper()
            content = msg["content"]
            if role == "SYSTEM":
                content = content[:80] + "..." if len(content) > 80 else content
            typer.secho(f"    [{role}]: {content}", fg=typer.colors.WHITE)

        typer.secho(f"\n  [SFT Format] (includes assistant response)", fg=typer.colors.GREEN, bold=True)
        for msg in sft_sample["messages"]:
            role = msg["role"].upper()
            content = msg["content"]
            if role == "SYSTEM":
                content = content[:80] + "..." if len(content) > 80 else content
            color = typer.colors.BRIGHT_GREEN if role == "ASSISTANT" else typer.colors.WHITE
            typer.secho(f"    [{role}]: {content}", fg=color)

    typer.secho(f"\n{'=' * 70}\n", fg=typer.colors.BRIGHT_CYAN)


def _is_vllm_line_important(text: str) -> bool:
    """Filter vLLM output to only show important lines."""
    # Always show errors and warnings
    if "ERROR" in text or "error" in text.lower() or "FAILED" in text:
        return True
    # Show reload events
    if "Reloading weights" in text or "Loading weights took" in text:
        return True
    # Show startup/shutdown
    if "Starting vLLM" in text or "vLLM API server version" in text:
        return True
    if "Shutting down" in text or "Application startup complete" in text:
        return True
    # Hide everything else (request logs, throughput stats, cache info, sleep/wake details)
    return False


def _stream_output(stream, prefix, file=sys.stdout, filter_fn=None):
    """Stream subprocess output line by line with a prefix.

    If filter_fn is provided, only lines where filter_fn(text) returns True are printed.
    """
    buf = b""
    try:
        while True:
            chunk = stream.read(4096)
            if not chunk:
                if buf:
                    text = buf.decode("utf-8", errors="replace").rstrip()
                    if filter_fn is None or filter_fn(text):
                        print(f"[{prefix}] {text}", file=file, flush=True)
                break
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                text = line.decode("utf-8", errors="replace").rstrip("\r")
                if filter_fn is None or filter_fn(text):
                    print(f"[{prefix}] {text}", file=file, flush=True)
    except (ValueError, OSError):
        pass


def _kill_process_tree(pid: int, sig: int):
    """Send a signal to a process and all its descendants."""
    import signal
    try:
        result = subprocess.run(
            ["ps", "--no-headers", "-o", "pid", "--ppid", str(pid)],
            capture_output=True, text=True, timeout=5,
        )
        child_pids = [int(p.strip()) for p in result.stdout.strip().split("\n") if p.strip()]
        for cpid in child_pids:
            _kill_process_tree(cpid, sig)
    except (subprocess.TimeoutExpired, ValueError, FileNotFoundError):
        pass
    try:
        os.kill(pid, sig)
    except (ProcessLookupError, PermissionError):
        pass


def _kill_process(proc, name, timeout=5):
    """Terminate a subprocess and its descendants, escalate to SIGKILL."""
    import signal
    if proc is None or proc.poll() is not None:
        return
    typer.secho(f"Terminating {name} (pid={proc.pid})...", fg=typer.colors.YELLOW)
    _kill_process_tree(proc.pid, signal.SIGTERM)
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        typer.secho(f"Force killing {name}...", fg=typer.colors.RED)
        _kill_process_tree(proc.pid, signal.SIGKILL)
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            pass


def _wait_for_vllm_health(url: str, timeout: int = 300):
    """Wait for vLLM to become ready."""
    start = time.time()
    health_ok = False
    while time.time() - start < timeout:
        try:
            resp = httpx.get(f"{url}/health", timeout=10)
            if resp.status_code == 200:
                health_ok = True
            if health_ok:
                resp = httpx.get(f"{url}/v1/models", timeout=10)
                if resp.status_code == 200 and resp.json().get("data"):
                    return True
        except (httpx.ConnectError, httpx.ReadTimeout, httpx.ConnectTimeout):
            pass
        time.sleep(2)
    return False


def _setup_vllm_checkpoint(model_name: str, checkpoint_dir: str):
    """Save initial model weights + config to checkpoint dir for vLLM."""
    import shutil
    from transformers import AutoConfig, AutoModelForCausalLM
    from safetensors.torch import save_file

    if os.path.exists(checkpoint_dir):
        shutil.rmtree(checkpoint_dir)
    os.makedirs(checkpoint_dir, exist_ok=True)

    config = AutoConfig.from_pretrained(model_name)
    config.save_pretrained(checkpoint_dir)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained(checkpoint_dir)

    typer.secho("Loading model weights for vLLM checkpoint...", fg=typer.colors.CYAN)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.save_pretrained(checkpoint_dir)
    del model
    torch.cuda.empty_cache()

    typer.secho(f"vLLM checkpoint saved to {checkpoint_dir}", fg=typer.colors.GREEN)


@app.command()
def distributed_grpo_train(
    data_path: str = typer.Option(..., "--data-path", help="Path to GRPO training data (jsonl)"),
    output_dir: str = typer.Option(..., "--output-dir", help="Path to the output directory"),
    model_name: str = typer.Option(
        "Qwen/Qwen2-1.5B-Instruct", "--model", "-m", help="Model name or path",
    ),

    max_tokens: int = typer.Option(0, "--max-tokens", help="Total token budget (0 = use --max-steps)"),
    max_steps: int = typer.Option(0, "--max-steps", help="Max optimizer steps (0 = use --max-tokens)"),
    inner_epochs: int = typer.Option(2, "--inner-epochs", help="Inner epochs per rollout batch"),
    inner_batch_size: int = typer.Option(32, "--inner-batch-size", help="Training batch size for GRPO inner loop"),

    save_every_n_tokens: int = typer.Option(
        0, "--save-every-n-tokens", help="Save checkpoint every N tokens (0 = disabled)"
    ),
    save_every_n_steps: int = typer.Option(
        0, "--save-every-n-steps", help="Save checkpoint every N optimizer steps (0 = disabled)"
    ),

    # GRPO settings
    group_size: int = typer.Option(16, "-G", "--group-size", help="Rollouts per prompt"),
    batch_size: int = typer.Option(64, "-B", "--batch-size", help="Prompts per rollout iteration"),
    clip_eps: float = typer.Option(0.2, "--clip-eps", help="GRPO clip epsilon"),
    kl_strength: float = typer.Option(0.01, "--kl", help="KL penalty strength"),
    format_reward: float = typer.Option(0.1, "--format-reward", help="Reward for correct format only"),
    gradient_clip: float = typer.Option(1.0, "--gradient-clip", help="Gradient clipping max norm"),
    update_ref_every: int = typer.Option(0, "--update-ref-every", help="Update ref policy every N steps (0 = never)"),
    token_level_averaging: bool = typer.Option(False, "--token-level-avg/--seq-level-avg", help="Token-level loss averaging (default: sequence-level)"),
    think: bool = typer.Option(False, "--think/--no-think", help="Require <think> traces for correct reward"),
    best_val_ckpt_only: bool = typer.Option(False, "--best-val-ckpt-only", help="Only save checkpoint when validation accuracy improves"),

    # Sampling
    temperature: float = typer.Option(0.7, "-t", "--temp", help="Sampling temperature"),
    max_new_tokens: int = typer.Option(512, "--max-new-tokens", help="Max tokens to generate per response"),
    top_p: float = typer.Option(1.0, "--top-p", help="Top-p sampling threshold"),
    top_k: int = typer.Option(0, "--top-k", help="Top-k sampling (0 = disabled)"),
    max_seq_len: int = typer.Option(8192, "--msl", "--max-seq-len", help="Maximum sequence length"),

    # Memory / gradient accumulation
    max_tokens_per_gpu: int = typer.Option(
        4096, "--max-tokens-per-gpu",
        help="Max tokens per GPU per microbatch (controls gradient accumulation)"
    ),

    # Optimizer
    optimizer_type: str = typer.Option("adamw", "-O", "--optimizer", help="Optimizer type: 'adamw' or 'muon'"),
    lr: float = typer.Option(1e-5, "--lr", help="Learning rate"),
    beta1: float = typer.Option(0.9, "--beta1", help="Adam beta1"),
    beta2: float = typer.Option(0.95, "--beta2", help="Adam beta2"),
    wd: float = typer.Option(0.0, "--wd", help="Weight decay"),

    # GPU allocation
    train_gpus: str = typer.Option("0,1", "--train-gpus", help="Comma-separated GPU indices for training"),
    vllm_gpus: str = typer.Option("2", "--vllm-gpus", help="Comma-separated GPU indices for vLLM inference"),
    vllm_gpu_memory_utilization: float = typer.Option(0.9, "--vllm-mem", help="vLLM GPU memory utilization"),
    ref_cpu_offload: bool = typer.Option(False, "--ref-cpu-offload", help="CPU offload reference model to save GPU memory"),

    # Wandb
    use_wandb: bool = typer.Option(False, "--wandb", help="Enable wandb logging"),
    wandb_project: str = typer.Option("grpo-distributed", "--wandb-project", help="Wandb project name"),
    wandb_run_name: str = typer.Option(None, "--wandb-run", help="Wandb run name"),
    wandb_entity: str = typer.Option(None, "--wandb-entity", help="Wandb entity"),

    seed: int = typer.Option(67, "--seed", help="Random seed"),

    validation_path: str = typer.Option(None, "--validation-path", help="Path to validation data"),

    # Task
    task: str = typer.Option("gsm8k", "--task", help="Task name: 'gsm8k' or 'countdown'"),
):
    """
    Multi-GPU GRPO training with FSDP2.

    Orchestrator: starts vLLM on --vllm-gpus, then launches distributed
    training via torchrun on --train-gpus. Single command, no manual torchrun.

    Supports multiple tasks via --task flag (gsm8k, countdown).

    Example:
        python cli.py distributed-grpo-train \\
            --data-path data/grpo_train.jsonl \\
            --output-dir /out --max-tokens 1000000 \\
            --train-gpus 0,1 --vllm-gpus 2 --task countdown
    """
    import socket as sock
    import threading

    n_train_gpus = len(train_gpus.split(","))
    n_vllm_gpus = len(vllm_gpus.split(","))

    typer.secho(f"Distributed GRPO Orchestrator (task={task})", fg=typer.colors.BRIGHT_CYAN, bold=True)
    typer.secho(f"  Training GPUs: {train_gpus} ({n_train_gpus} workers)", fg=typer.colors.WHITE)
    typer.secho(f"  vLLM GPUs:     {vllm_gpus} ({n_vllm_gpus} data-parallel)", fg=typer.colors.WHITE)
    typer.secho(f"  Model:         {model_name}", fg=typer.colors.WHITE)
    if max_tokens > 0:
        typer.secho(f"  Token budget:  {max_tokens:,}", fg=typer.colors.WHITE)
    if max_steps > 0:
        typer.secho(f"  Step budget:   {max_steps:,}", fg=typer.colors.WHITE)

    # 1. Allocate port and set up checkpoint dir
    with sock.socket(sock.AF_INET, sock.SOCK_STREAM) as s:
        s.bind(("", 0))
        vllm_port = s.getsockname()[1]

    vllm_url = f"http://localhost:{vllm_port}"
    checkpoint_dir = f"/dev/shm/active-policy-{vllm_port}"

    typer.secho(f"\nSaving initial weights for vLLM...", fg=typer.colors.CYAN)
    _setup_vllm_checkpoint(model_name, checkpoint_dir)

    # 2. Start vLLM
    vllm_cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", checkpoint_dir,
        "--served-model-name", "policy",
        "--port", str(vllm_port),
        "--gpu-memory-utilization", str(vllm_gpu_memory_utilization),
        "--max-model-len", str(max_seq_len),
        "--seed", str(seed),
        "--dtype", "bfloat16",
        "--trust-remote-code",
        "--no-enable-log-requests",
        "--data-parallel-size", str(n_vllm_gpus),
        "--enable-sleep-mode",
    ]

    vllm_env = os.environ.copy()
    vllm_env["CUDA_VISIBLE_DEVICES"] = vllm_gpus
    vllm_env["VLLM_SERVER_DEV_MODE"] = "1"

    vllm_process = None
    training_process = None

    try:
        typer.secho(f"\nStarting vLLM on GPU(s) {vllm_gpus} (port {vllm_port})...", fg=typer.colors.CYAN)
        vllm_process = subprocess.Popen(
            vllm_cmd, env=vllm_env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            start_new_session=True,
        )

        # Stream vLLM output in background (filtered to important lines only)
        vllm_thread = threading.Thread(
            target=_stream_output,
            args=(vllm_process.stdout, "VLLM"),
            kwargs={"filter_fn": _is_vllm_line_important},
            daemon=True,
        )
        vllm_thread.start()

        # Wait for vLLM health
        typer.secho("Waiting for vLLM to be ready...", fg=typer.colors.CYAN)
        if not _wait_for_vllm_health(vllm_url):
            typer.secho("vLLM failed to start!", fg=typer.colors.RED, bold=True)
            raise RuntimeError("vLLM not ready within timeout")
        typer.secho(f"vLLM ready at {vllm_url}", fg=typer.colors.GREEN, bold=True)

        # 3. Launch torchrun for training (allocate unique master port)
        with sock.socket(sock.AF_INET, sock.SOCK_STREAM) as s:
            s.bind(("", 0))
            master_port = s.getsockname()[1]

        train_cmd = [
            sys.executable, "-m", "torch.distributed.run",
            "--nproc_per_node", str(n_train_gpus),
            "--master_port", str(master_port),
            "cli.py", "distributed-grpo-worker",
            "--data-path", data_path,
            "--output-dir", output_dir,
            "--model", model_name,
            "--max-tokens", str(max_tokens),
            "--max-steps", str(max_steps),
            "--inner-epochs", str(inner_epochs),
            "--inner-batch-size", str(inner_batch_size),
            "--save-every-n-tokens", str(save_every_n_tokens),
            "--save-every-n-steps", str(save_every_n_steps),
            "--group-size", str(group_size),
            "--batch-size", str(batch_size),
            "--clip-eps", str(clip_eps),
            "--kl", str(kl_strength),
            "--format-reward", str(format_reward),
            "--gradient-clip", str(gradient_clip),
            "--update-ref-every", str(update_ref_every),
            "--temp", str(temperature),
            "--max-new-tokens", str(max_new_tokens),
            "--top-p", str(top_p),
            "--top-k", str(top_k),
            "--max-seq-len", str(max_seq_len),
            "--max-tokens-per-gpu", str(max_tokens_per_gpu),
            "--optimizer", optimizer_type,
            "--lr", str(lr),
            "--beta1", str(beta1),
            "--beta2", str(beta2),
            "--wd", str(wd),
            "--vllm-url", vllm_url,
            "--vllm-checkpoint-dir", checkpoint_dir,
            "--seed", str(seed),
            "--task", task,
        ]
        if use_wandb:
            train_cmd.append("--wandb")
        if wandb_project:
            train_cmd += ["--wandb-project", wandb_project]
        if wandb_run_name:
            train_cmd += ["--wandb-run", wandb_run_name]
        if wandb_entity:
            train_cmd += ["--wandb-entity", wandb_entity]
        if validation_path:
            train_cmd += ["--validation-path", validation_path]
        if ref_cpu_offload:
            train_cmd.append("--ref-cpu-offload")
        if token_level_averaging:
            train_cmd.append("--token-level-avg")
        if think:
            train_cmd.append("--think")
        if best_val_ckpt_only:
            train_cmd.append("--best-val-ckpt-only")

        train_env = os.environ.copy()
        train_env["CUDA_VISIBLE_DEVICES"] = train_gpus

        typer.secho(
            f"\nLaunching training ({n_train_gpus} workers on GPU(s) {train_gpus})...",
            fg=typer.colors.CYAN,
        )
        training_process = subprocess.Popen(
            train_cmd, env=train_env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            start_new_session=True,
        )

        train_thread = threading.Thread(
            target=_stream_output, args=(training_process.stdout, "TRAIN"), daemon=True,
        )
        train_thread.start()

        # 4. Wait for training to complete
        while training_process.poll() is None:
            try:
                training_process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                continue

        exit_code = training_process.returncode
        if exit_code == 0:
            typer.secho("\nTraining completed successfully!", fg=typer.colors.GREEN, bold=True)
        else:
            typer.secho(f"\nTraining exited with code {exit_code}", fg=typer.colors.RED, bold=True)

    except KeyboardInterrupt:
        typer.secho("\nCtrl+C received, shutting down...", fg=typer.colors.YELLOW)
    except Exception as e:
        typer.secho(f"\nError: {e}", fg=typer.colors.RED)
    finally:
        _kill_process(training_process, "training")
        _kill_process(vllm_process, "vLLM")

        # Clean up checkpoint dir
        import shutil
        if os.path.exists(checkpoint_dir):
            shutil.rmtree(checkpoint_dir, ignore_errors=True)


@app.command()
def distributed_grpo_worker(
    data_path: str = typer.Option(..., "--data-path"),
    output_dir: str = typer.Option(..., "--output-dir"),
    model_name: str = typer.Option("Qwen/Qwen2-1.5B-Instruct", "--model", "-m"),
    max_tokens: int = typer.Option(0, "--max-tokens"),
    max_steps: int = typer.Option(0, "--max-steps"),
    inner_epochs: int = typer.Option(2, "--inner-epochs"),
    inner_batch_size: int = typer.Option(32, "--inner-batch-size"),
    save_every_n_tokens: int = typer.Option(0, "--save-every-n-tokens"),
    save_every_n_steps: int = typer.Option(0, "--save-every-n-steps"),
    group_size: int = typer.Option(16, "-G", "--group-size"),
    batch_size: int = typer.Option(64, "-B", "--batch-size"),
    clip_eps: float = typer.Option(0.2, "--clip-eps"),
    kl_strength: float = typer.Option(0.01, "--kl"),
    format_reward: float = typer.Option(0.1, "--format-reward"),
    gradient_clip: float = typer.Option(1.0, "--gradient-clip"),
    update_ref_every: int = typer.Option(0, "--update-ref-every"),
    token_level_averaging: bool = typer.Option(False, "--token-level-avg/--seq-level-avg"),
    think: bool = typer.Option(False, "--think/--no-think"),
    best_val_ckpt_only: bool = typer.Option(False, "--best-val-ckpt-only"),
    temperature: float = typer.Option(0.7, "-t", "--temp"),
    max_new_tokens: int = typer.Option(512, "--max-new-tokens"),
    top_p: float = typer.Option(1.0, "--top-p"),
    top_k: int = typer.Option(0, "--top-k"),
    max_seq_len: int = typer.Option(8192, "--msl", "--max-seq-len"),
    max_tokens_per_gpu: int = typer.Option(4096, "--max-tokens-per-gpu"),
    optimizer_type: str = typer.Option("adamw", "-O", "--optimizer"),
    lr: float = typer.Option(1e-5, "--lr"),
    beta1: float = typer.Option(0.9, "--beta1"),
    beta2: float = typer.Option(0.95, "--beta2"),
    wd: float = typer.Option(0.0, "--wd"),
    vllm_url: str = typer.Option(..., "--vllm-url"),
    vllm_checkpoint_dir: str = typer.Option(..., "--vllm-checkpoint-dir"),
    ref_cpu_offload: bool = typer.Option(False, "--ref-cpu-offload"),
    use_wandb: bool = typer.Option(False, "--wandb"),
    wandb_project: str = typer.Option("grpo-distributed", "--wandb-project"),
    wandb_run_name: str = typer.Option(None, "--wandb-run"),
    wandb_entity: str = typer.Option(None, "--wandb-entity"),
    seed: int = typer.Option(67, "--seed"),
    validation_path: str = typer.Option(None, "--validation-path"),
    task: str = typer.Option("gsm8k", "--task"),
):
    """
    Internal: GRPO training worker launched by distributed-grpo-train via torchrun.
    Do not call directly.
    """
    from distributed_grpo_trainer import DistributedGRPOTrainer
    from tasks import get_reward_fn

    trainer = DistributedGRPOTrainer(
        data_path=data_path,
        model_name=model_name,
        output_dir=output_dir,
        token_budget=max_tokens,
        max_steps=max_steps,
        inner_epochs=inner_epochs,
        inner_batch_size=inner_batch_size,
        save_every_n_tokens=save_every_n_tokens,
        save_every_n_steps=save_every_n_steps,
        group_size=group_size,
        batch_size=batch_size,
        clip_eps=clip_eps,
        kl_strength=kl_strength,
        format_reward=format_reward,
        update_ref_every=update_ref_every,
        token_level_averaging=token_level_averaging,
        require_think=think,
        best_val_ckpt_only=best_val_ckpt_only,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
        max_seq_len=max_seq_len,
        max_tokens_per_gpu=max_tokens_per_gpu,
        optimizer_type=optimizer_type,
        lr=lr,
        beta1=beta1,
        beta2=beta2,
        weight_decay=wd,
        gradient_clip=gradient_clip,
        vllm_url=vllm_url,
        vllm_checkpoint_dir=vllm_checkpoint_dir,
        ref_cpu_offload=ref_cpu_offload,
        use_wandb=use_wandb,
        wandb_project=wandb_project,
        wandb_run_name=wandb_run_name,
        wandb_entity=wandb_entity,
        seed=seed,
        validation_path=validation_path,
        task=task,
    )
    trainer.train()


@app.command()
def countdown_rs_train(
    data_path: str = typer.Option(..., "--data-path", help="Path to the training data"),
    output_dir: str = typer.Option(..., "--output-dir", help="Path to the output directory"),
    model_name: str = typer.Option("Qwen/Qwen2-1.5B-Instruct", "--model", "-m", help="Model name or path"),

    max_tokens: int = typer.Option(..., "--max-tokens", help="Maximum loss-counted tokens to train on"),
    num_inner_epochs: int = typer.Option(1, "--inner-epochs", help="Number of inner epochs"),

    max_seq_len: int = typer.Option(8192, "--msl", "--max-seq-len", help="Maximum sequence length"),
    max_tokens_per_gpu: int = typer.Option(8192, "--max-tokens-per-gpu", help="Max tokens per GPU"),
    save_every_n_tokens: int = typer.Option(
        0, "--save-every-n-tokens", help="Save checkpoint every N tokens (0 = disabled)"
    ),

    samples_to_accept: int = typer.Option(1, "--samples-to-accept", help="Number of samples to accept per rollout batch"),
    inference_batch_size: int = typer.Option(32, "--inference-batch-size", help="Inference batch size"),
    inference_group_size: int = typer.Option(16, "--inference-group-size", help="Rollouts per prompt"),

    # sampling params
    temperature: float = typer.Option(0.7, "-t", "--temp", help="Sampling temperature"),
    max_new_tokens: int = typer.Option(512, "--max-new-tokens", help="Max new tokens to generate"),
    top_p: float = typer.Option(1.0, "--top-p", help="Top-p sampling"),
    top_k: int = typer.Option(0, "--top-k", help="Top-k sampling (0 = disabled)"),

    # wandb
    use_wandb: bool = typer.Option(False, "--wandb", help="Enable wandb logging"),
    wandb_project: str = typer.Option("countdown-rs", "--wandb-project", help="Wandb project name"),
    wandb_run_name: str = typer.Option(None, "--wandb-run", help="Wandb run name"),
    wandb_entity: str = typer.Option(None, "--wandb-entity", help="Wandb entity"),

    seed: int = typer.Option(67, "--seed", help="Random seed"),

    optimizer_type: str = typer.Option("adamw", "-O", "--optimizer", help="Optimizer type: 'adamw' or 'muon'"),
    lr: float = typer.Option(1e-5, "--lr", help="Learning rate"),
    beta1: float = typer.Option(0.9, help="Adam beta1"),
    beta2: float = typer.Option(0.95, help="Adam beta2"),
    wd: float = typer.Option(0.0, "--wd", help="Weight decay"),

    gpu: int = typer.Option(0, "--gpu", "-g", help="CUDA GPU for training"),
    vllm_gpus: str = typer.Option("1", "--vllm-gpus", help="Comma-separated GPU indices for vLLM inference"),

    validation_path: str = typer.Option(None, "--validation-path", help="Path to validation data"),
):
    """
    Rejection Sampling training on countdown problems.

    Same RS loop as rs_train but uses the countdown reward function which
    validates arithmetic expressions in addition to checking the final answer.
    """
    from tasks import get_reward_fn

    trainer = RSTrainer(
        data_path=data_path,
        model_name=model_name,
        output_dir=output_dir,
        token_budget=max_tokens,
        inner_epochs=num_inner_epochs,
        inner_batch_size=samples_to_accept,
        save_every_n_tokens=save_every_n_tokens,
        samples_to_accept=samples_to_accept,
        inference_batch_size=inference_batch_size,
        inference_group_size=inference_group_size,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
        max_seq_len=max_seq_len,
        max_tokens_per_gpu=max_tokens_per_gpu,
        use_wandb=use_wandb,
        wandb_project=wandb_project,
        wandb_run_name=wandb_run_name,
        wandb_entity=wandb_entity,
        seed=seed,
        optimizer_type=optimizer_type,
        lr=lr,
        beta1=beta1,
        beta2=beta2,
        weight_decay=wd,
        gpu=gpu,
        vllm_gpus=vllm_gpus,
        vllm_gpu_memory_utilization=0.9,
        validation_path=validation_path,
        reward_fn=get_reward_fn("countdown"),
    )
    trainer.train()
    dist.barrier()
    dist.destroy_process_group()


@app.command()
def countdown_sft_train(
    # Data paths
    data_path: str = typer.Option(..., "--data-path", help="Path to countdown SFT training data (jsonl with 'messages')"),
    model_name: str = typer.Option("Qwen/Qwen2-1.5B-Instruct", "--model", "-m", help="Model name or path"),
    output_dir: str = typer.Option(..., "--output-dir", help="Path to the output directory"),
    # Training mode
    max_steps: int = typer.Option(0, "--max-steps", help="Maximum training steps"),
    max_tokens: int = typer.Option(0, "--max-tokens", help="Maximum loss-counted tokens"),
    num_epochs: int = typer.Option(1, "--epochs", help="Number of epochs"),
    # Batch settings
    effective_batch_size: int = typer.Option(32, "-B", "--batch-size", help="Effective batch size"),
    max_tokens_per_gpu: int = typer.Option(8192, "--max-tokens-per-gpu", help="Max tokens per GPU"),
    max_seq_len: int = typer.Option(2048, "--max-seq-len", help="Maximum sequence length"),
    # Optimizer settings
    optimizer_type: str = typer.Option("adamw", "-O", "--optimizer", help="Optimizer: 'adamw' or 'muon'"),
    lr: float = typer.Option(1e-5, "--lr", help="Learning rate"),
    muon_lr: float = typer.Option(None, "--muon-lr", help="Muon-specific LR"),
    beta1: float = typer.Option(0.9, "--beta1", help="Adam beta1"),
    beta2: float = typer.Option(0.95, "--beta2", help="Adam beta2"),
    weight_decay: float = typer.Option(0.0, "--wd", help="Weight decay"),
    # LR scheduler
    lr_scheduler: str = typer.Option("cosine", "--lr-scheduler", help="LR scheduler type"),
    warmup_steps: int = typer.Option(0, "--warmup-steps", help="Number of warmup steps"),
    # Checkpointing
    save_final_checkpoint: bool = typer.Option(True, "--save-final/--no-save-final", help="Save final checkpoint"),
    checkpoint_at_epoch: bool = typer.Option(False, "--checkpoint-at-epoch", help="Save checkpoint each epoch"),
    save_every_steps: int = typer.Option(0, "--save-every", help="Save every N optimizer steps (0 = disabled)"),
    save_every_n_tokens: int = typer.Option(0, "--save-every-n-tokens", help="Save every N tokens (0 = disabled)"),
    # Data processing
    use_processed_dataset: bool = typer.Option(False, "--use-processed", help="Data is already tokenized"),
    unmask_messages: bool = typer.Option(False, "--unmask", help="Train on all tokens (not just assistant)"),
    # Wandb
    use_wandb: bool = typer.Option(False, "--wandb", help="Enable wandb logging"),
    wandb_project: str = typer.Option("countdown-sft", "--wandb-project", help="Wandb project name"),
    wandb_run_name: str = typer.Option(None, "--wandb-run", help="Wandb run name"),
    wandb_entity: str = typer.Option(None, "--wandb-entity", help="Wandb entity"),
    # Precision
    precision: str = typer.Option(
        "mixed", "--precision", "-P",
        help="'fp32' | 'bf16' | 'mixed'",
    ),
    # Misc
    seed: int = typer.Option(67, "--seed", help="Random seed"),
    use_liger: bool = typer.Option(False, "--liger", help="Use Liger kernels"),
    num_gpus: int = typer.Option(1, "--num-gpus", help="Number of GPUs"),
    validation_split: float = typer.Option(0.0, "--validation-split", help="Fraction for validation"),
    validation_frequency: int = typer.Option(0, "--validation-frequency", help="Validation frequency (steps)"),
    disable_kl: bool = typer.Option(False, "--disable-kl", help="Disable KL divergence tracking"),
):
    """
    SFT training on countdown problems.

    Uses the same SFT pipeline as sft_train -- the data format (messages with
    system/user/assistant) is identical, just the content is countdown problems.
    """
    from training_hub import osft
    from mini_trainer import TrainingMode

    # Determine training mode
    if max_tokens > 0:
        training_mode = TrainingMode.TOKEN
        typer.secho(f"Training for {max_tokens:,} tokens", fg=typer.colors.CYAN)
    elif max_steps > 0:
        training_mode = TrainingMode.STEP
        typer.secho(f"Training for {max_steps} steps", fg=typer.colors.CYAN)
    else:
        training_mode = TrainingMode.EPOCH
        typer.secho(f"Training for {num_epochs} epoch(s)", fg=typer.colors.CYAN)

    optional_kwargs = {}
    if use_wandb:
        optional_kwargs["wandb_project"] = wandb_project
        if wandb_run_name:
            optional_kwargs["wandb_run_name"] = wandb_run_name
        if wandb_entity:
            optional_kwargs["wandb_entity"] = wandb_entity
    if validation_frequency > 0:
        optional_kwargs["validation_frequency"] = validation_frequency
    if validation_split > 0:
        optional_kwargs["validation_split"] = validation_split
    if muon_lr is not None:
        optional_kwargs["muon_lr"] = muon_lr

    train_dtype = "float32" if precision in ("fp32", "mixed") else "bfloat16"

    osft(
        model_path=model_name,
        data_path=data_path,
        ckpt_output_dir=output_dir,
        unfreeze_rank_ratio=1.0,
        osft=False,
        effective_batch_size=effective_batch_size,
        max_tokens_per_gpu=max_tokens_per_gpu,
        max_seq_len=max_seq_len,
        optimizer_type=optimizer_type,
        learning_rate=lr,
        beta1=beta1,
        beta2=beta2,
        weight_decay=weight_decay,
        lr_scheduler=lr_scheduler,
        warmup_steps=warmup_steps,
        training_mode=training_mode,
        num_epochs=num_epochs,
        max_steps=max_steps,
        max_tokens=max_tokens,
        save_final_checkpoint=save_final_checkpoint,
        checkpoint_at_epoch=checkpoint_at_epoch,
        save_every_steps=save_every_steps if save_every_steps > 0 else None,
        save_every_n_tokens=save_every_n_tokens if save_every_n_tokens > 0 else None,
        use_processed_dataset=use_processed_dataset,
        unmask_messages=unmask_messages,
        compute_kl=not disable_kl,
        seed=seed,
        use_liger=use_liger,
        nproc_per_node=num_gpus,
        precision=precision,
        train_dtype=train_dtype,
        save_dtype='float32',
        **optional_kwargs,
    )


if __name__ == "__main__":
    app()
