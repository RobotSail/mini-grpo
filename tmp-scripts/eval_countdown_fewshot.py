#!/usr/bin/env python3
"""
Evaluate countdown task with few-shot prompting via multi-turn chat messages.

Uses the same system prompt and format as countdown_utils.py training.
Few-shot examples are provided as (user, assistant) turns in the chat history.
"""

import json
import re
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from countdown_utils import (
    validate_expression, validate_ast, solve_countdown,
    COUNTDOWN_SYSTEM_MSG, ANSWER_PATTERN,
)
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from tqdm import tqdm


def create_fewshot_examples(train_path: str, n: int = 5, seed: int = 42) -> list[dict]:
    """Pick n training problems and solve them to create few-shot examples."""
    import random
    rng = random.Random(seed)

    with open(train_path) as f:
        train = [json.loads(line) for line in f]

    rng.shuffle(train)
    examples = []
    for sample in train:
        if len(examples) >= n:
            break
        numbers = sample["numbers"]
        target = sample["answer"]
        solution = solve_countdown(numbers, target)
        if solution is None:
            continue
        # Build the assistant response in the expected format
        nums_str = ", ".join(str(n) for n in numbers)
        assistant_response = (
            f"<think>\nI need to find an expression using [{nums_str}] that equals {target}.\n"
            f"Let me try: {solution} = {target}\n"
            f"</think>\n<answer>{solution}</answer>"
        )
        examples.append({
            "user": sample["messages"][1]["content"],  # the user prompt
            "assistant": assistant_response,
        })

    return examples


def evaluate(
    model_path: str,
    val_path: str,
    train_path: str,
    n_shots: int = 5,
    max_samples: int = 0,
    gpu: int = 0,
    dtype: str = "auto",
):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

    # Load validation data
    with open(val_path) as f:
        val_data = [json.loads(line) for line in f]
    if max_samples > 0:
        val_data = val_data[:max_samples]
    print(f"Evaluating {len(val_data)} validation samples with {n_shots}-shot prompting")

    # Create few-shot examples
    fewshot = create_fewshot_examples(train_path, n=n_shots)
    print(f"Created {len(fewshot)} few-shot examples")
    for i, ex in enumerate(fewshot):
        print(f"  Example {i}: {ex['user'][:80]}...")

    # Build few-shot message prefix (system + n turns of user/assistant)
    system_msg = COUNTDOWN_SYSTEM_MSG
    fewshot_messages = [{"role": "system", "content": system_msg}]
    for ex in fewshot:
        fewshot_messages.append({"role": "user", "content": ex["user"]})
        fewshot_messages.append({"role": "assistant", "content": ex["assistant"]})

    # Load tokenizer and format prompts
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    prompts = []
    for sample in val_data:
        user_prompt = sample["messages"][1]["content"]
        messages = fewshot_messages + [{"role": "user", "content": user_prompt}]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        prompts.append(prompt)

    # Initialize vLLM
    llm = LLM(
        model=model_path,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.9,
        dtype=dtype,
        max_model_len=4096,
    )
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=1024,
        stop=["<|im_end|>", "<|endoftext|>"],
    )

    # Generate
    print("Generating responses...")
    outputs = llm.generate(prompts, sampling_params)

    # Score
    correct = 0
    has_format = 0
    parsable = 0

    for i, (output, sample) in enumerate(zip(outputs, val_data)):
        response = output.outputs[0].text
        numbers = sample["numbers"]
        target = sample["answer"]

        # Check format: <think>...</think> before <answer>...</answer>
        resp_lower = response.lower()
        think_start = resp_lower.find("<think>")
        think_end = resp_lower.find("</think>")
        answer_start = resp_lower.find("<answer>")
        has_think = think_start != -1 and think_end != -1 and think_start < think_end
        has_answer = answer_start != -1
        if has_think and has_answer and think_end < answer_start:
            has_format += 1

        # Extract and validate answer
        matches = ANSWER_PATTERN.findall(response)
        if matches:
            content = matches[-1].strip()
            if content and validate_ast(content, numbers):
                parsable += 1
                if validate_expression(content, numbers, int(target)):
                    correct += 1

    total = len(val_data)
    print(f"\n{'='*60}")
    print(f"Model: {model_path}")
    print(f"Samples: {total}, {n_shots}-shot")
    print(f"  Format correct: {has_format}/{total} ({100*has_format/total:.1f}%)")
    print(f"  Parsable:       {parsable}/{total} ({100*parsable/total:.1f}%)")
    print(f"  Correct:        {correct}/{total} ({100*correct/total:.1f}%)")
    print(f"{'='*60}")

    return {
        "model": model_path,
        "n_shots": n_shots,
        "total": total,
        "has_format": has_format,
        "parsable": parsable,
        "correct": correct,
        "accuracy": correct / total,
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen2-1.5B-Instruct")
    parser.add_argument("--val-path", type=str, default="generated_data/countdown_hard_val.jsonl")
    parser.add_argument("--train-path", type=str, default="generated_data/countdown_hard_train.jsonl")
    parser.add_argument("--n-shots", type=int, default=5)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--dtype", type=str, default="auto")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    result = evaluate(
        model_path=args.model,
        val_path=args.val_path,
        train_path=args.train_path,
        n_shots=args.n_shots,
        max_samples=args.max_samples,
        gpu=args.gpu,
        dtype=args.dtype,
    )

    if args.output:
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Saved to {args.output}")
