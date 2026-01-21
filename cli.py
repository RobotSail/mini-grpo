import requests
from transformers import GenerationConfig
import json
import random
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
from utils import preview_tokenization, display_scorecard
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
    """Create a single SFT sample in messages format."""
    cleaned = _clean_calculator_annotations(answer)
    reformatted = _reformat_to_answer_tags(cleaned)
    return {
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": question},
            {"role": "assistant", "content": reformatted},
        ]
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


@torch.no_grad
def generate_rollouts(
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


def print_example_rollout(samples: list[Sample], step: int = 0):
    """Print batch statistics and example rollouts after generation."""
    if not samples:
        return

    # Calculate batch statistics
    total_rollouts = sum(len(s.rollouts) for s in samples)
    total_rewards = sum(r.reward for s in samples for r in s.rollouts)
    parsable_count = sum(1 for s in samples for r in s.rollouts if r.is_parsable)
    correct_count = sum(1 for s in samples for r in s.rollouts if r.is_correct)

    avg_reward = total_rewards / total_rollouts if total_rollouts > 0 else 0.0
    parsable_rate = parsable_count / total_rollouts if total_rollouts > 0 else 0.0
    correct_rate = correct_count / total_rollouts if total_rollouts > 0 else 0.0

    # Print batch statistics
    typer.secho(f"\n{'=' * 70}", fg=typer.colors.BRIGHT_MAGENTA)
    typer.secho(f"  ROLLOUT SUMMARY (Step {step})", fg=typer.colors.BRIGHT_MAGENTA, bold=True)
    typer.secho(f"{'=' * 70}", fg=typer.colors.BRIGHT_MAGENTA)

    typer.secho(f"\n[BATCH STATISTICS]:", fg=typer.colors.BRIGHT_CYAN)
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

    # Find one correct and one incorrect example for comparison
    correct_example = None
    incorrect_example = None

    for sample in samples:
        for rollout in sample.rollouts:
            if rollout.is_correct and correct_example is None:
                correct_example = (sample, rollout)
            elif not rollout.is_correct and incorrect_example is None:
                incorrect_example = (sample, rollout)
            if correct_example and incorrect_example:
                break
        if correct_example and incorrect_example:
            break

    # Print examples
    examples_to_print = []
    if correct_example:
        examples_to_print.append(("CORRECT", correct_example, typer.colors.GREEN))
    if incorrect_example:
        examples_to_print.append(("INCORRECT", incorrect_example, typer.colors.RED))

    # Fallback: if no correct/incorrect distinction, just show first rollout
    if not examples_to_print and samples and samples[0].rollouts:
        examples_to_print.append(("EXAMPLE", (samples[0], samples[0].rollouts[0]), typer.colors.WHITE))

    for label, (sample, rollout), color in examples_to_print:
        typer.secho(f"\n[{label} ROLLOUT]:", fg=color, bold=True)

        # Print the user prompt (skip system message for brevity)
        user_msg = next((m for m in rollout.seed_messages if m.role == "user"), None)
        if user_msg:
            prompt_preview = user_msg.content[:150] + ("..." if len(user_msg.content) > 150 else "")
            typer.secho(f"  Prompt: {prompt_preview}", fg=typer.colors.YELLOW)

        # Print the response (truncated)
        response_preview = rollout.response[:400] + ("..." if len(rollout.response) > 400 else "")
        typer.secho(f"  Response: {response_preview}", fg=typer.colors.WHITE)

        # Print grading
        typer.secho(
            f"  Expected: {sample.problem.answer} | Parsable: {rollout.is_parsable} | Correct: {rollout.is_correct} | Reward: {rollout.reward:.2f}",
            fg=color,
        )

    typer.secho(f"{'=' * 70}\n", fg=typer.colors.BRIGHT_MAGENTA)


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
) -> tuple[int, bool]:
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

    Returns:
        Tuple of (updated_optim_step, should_stop) where should_stop is True if max_steps reached
    """
    comps.model.train()

    # Create dataset from rollouts
    dataset = dataset_from_groups(samples, comps.train_tokenizer)

    # Track optimizer steps
    optim_step = current_optim_step

    # Training loop over inner epochs
    for epoch in range(comps.hyperparams.inner_epochs):
        data_loader = create_grpo_data_loader(dataset, comps, use_packed=use_packed)

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

            # Accumulate gradients across microbatches
            for micro_idx, microbatch in enumerate(microbatches):
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

            # Clear cache after optimizer step
            torch.cuda.empty_cache()

            # Log metrics (including KL divergence)
            kl_div = avg_metrics.get("kl_div", 0.0)
            ir_mean = avg_metrics.get("importance_ratio", 1.0)
            typer.secho(
                f"Inner Epoch {epoch + 1}/{comps.hyperparams.inner_epochs} | "
                f"Step {optim_step} | "
                f"Loss: {avg_loss:.4f} | "
                f"KL: {kl_div:.4f} | "
                f"IR: {ir_mean:.4f} | "
                f"Grad Norm: {gradnorm.item():.4f}",
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
                    },
                    step=optim_step,
                )

            # Check if we've reached max_steps
            if max_steps > 0 and optim_step >= max_steps:
                return optim_step, True

        # Clear cache after each inner epoch
        torch.cuda.empty_cache()

    return optim_step, False


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
    # flash attention / memory optimization
    use_flash_attn: bool = typer.Option(
        False, "--flash-attn", help="Enable Flash Attention 2 with padding-free training"
    ),
    # wandb params
    use_wandb: bool = typer.Option(False, "--wandb", help="Enable wandb logging"),
    wandb_project: str = typer.Option("mini-grpo-gsm8k", "--wandb-project", help="Wandb project name"),
    wandb_run_name: str = typer.Option(None, "--wandb-run", help="Wandb run name (auto-generated if not set)"),
    wandb_entity: str = typer.Option(None, "--wandb-entity", help="Wandb entity/team name"),
):
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
            import os

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
    training_comps = TrainingComponents(
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
    if output_dir is not None and not training_comps.valid_save_dir():
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
        baseline_metrics = eval_model(eval_dataset, training_comps, return_metrics=True)
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

    # Determine training mode
    use_step_based = max_steps > 0
    if use_step_based:
        typer.secho(f"Training for {max_steps} optimizer steps", fg=typer.colors.CYAN)
    else:
        typer.secho(f"Training for {epochs} epoch(s)", fg=typer.colors.CYAN)

    # Training loop
    epoch = 0
    training_complete = False
    while not training_complete:
        minibatches: list[Sample] = []

        # Set up progress bar
        if use_step_based:
            desc = f"Step {optim_step}/{max_steps}"
        else:
            desc = f"Epoch {epoch + 1}/{epochs}"

        pbar = tqdm(
            train_dataset.shuffle().iter(batch_size),
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

            # Generate rollouts for each prompt
            rollouts = generate_rollouts(
                model,
                tokenizer,
                batch,
                training_comps.hyperparams.batch_size,
                training_comps.hyperparams.group_size,
                sampling_params=training_comps.sampling_params,
                show_tqdm=True,
            )

            # Print rollout summary with statistics and examples
            print_example_rollout(rollouts, step=optim_step)

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
            optim_step, should_stop = train_policy_on_rollouts(
                rollouts,
                training_comps,
                use_wandb=use_wandb,
                global_step=global_step,
                use_packed=use_flash_attn,
                max_tokens_per_microbatch=max_tokens_per_microbatch,
                current_optim_step=optim_step,
                max_steps=max_steps,
            )
            steps_this_batch = optim_step - prev_optim_step
            typer.secho(
                f"Completed {steps_this_batch} optimizer steps (total: {optim_step}/{max_steps if max_steps > 0 else '∞'})",
                fg=typer.colors.CYAN,
            )
            minibatches.extend(rollouts)

            # Update progress bar with current step
            if use_step_based:
                pbar.set_description(f"Step {optim_step}/{max_steps}")
            pbar.set_postfix({"avg_reward": f"{avg_reward:.4f}", "acc": f"{correct_rate:.2%}"})

            # Clear cache after training step before next rollout generation
            torch.cuda.empty_cache()

            # Check if we've reached max_steps
            if should_stop:
                typer.secho(f"\nReached {max_steps} optimizer steps. Stopping training.", fg=typer.colors.GREEN)
                training_complete = True
                # Save final checkpoint before breaking
                if output_dir:
                    training_comps.save_checkpoint(optim_step, is_step=True)
                break

            # Save checkpoint at intervals
            if save_every > 0 and optim_step % save_every == 0 and output_dir:
                typer.secho(f"\n[Step {optim_step}] Saving checkpoint...", fg=typer.colors.CYAN)
                training_comps.save_checkpoint(optim_step, is_step=True)

            # Intermediate evaluation (based on optim_step)
            if eval_every > 0 and optim_step % eval_every == 0:
                if eval_dataset is not None and len(eval_dataset) > 0:
                    typer.secho(f"\n[Step {optim_step}] Running intermediate evaluation...", fg=typer.colors.CYAN)
                    metrics = eval_model(eval_dataset, training_comps, return_metrics=True)
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
                metrics = eval_model(eval_dataset, training_comps, return_metrics=True)
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
                training_comps.save_checkpoint(optim_step, is_step=True)
            else:
                training_comps.save_checkpoint(epoch, is_step=False)

        epoch += 1

        # Check epoch-based termination
        if not use_step_based and epoch >= epochs:
            training_complete = True

    # Finish wandb run
    if use_wandb:
        wandb.finish()
        typer.secho("✓ Wandb run finished", fg=typer.colors.GREEN)


if __name__ == "__main__":
    app()
