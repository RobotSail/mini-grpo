#!/usr/bin/env python3
"""
Evaluate countdown task using GSM8K-style few-shot prompting.

No <think> tags -- just Q/A format like GSM8K:
  Q: Find expression = target using [numbers]
  A: step-by-step reasoning, then #### expression
"""

import json
import re
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from countdown_utils import validate_expression, validate_ast, solve_countdown
from vllm import LLM, SamplingParams
from tqdm import tqdm


def format_question(numbers, target):
    nums_str = ", ".join(str(n) for n in numbers)
    return (
        f"Find a mathematical expression that equals {target} "
        f"using the numbers [{nums_str}]. "
        f"Use each number exactly once. You may use +, -, *, / and parentheses."
    )


def format_answer(numbers, target, expression):
    nums_str = ", ".join(str(n) for n in numbers)
    return (
        f"I need to find an expression using [{nums_str}] that equals {target}. "
        f"Let me try: {expression}. "
        f"Checking: {expression} = {target}. That works.\n"
        f"#### {expression}"
    )


def create_fewshot_examples(train_path, n=5, seed=42):
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
        examples.append({
            "question": format_question(numbers, target),
            "answer": format_answer(numbers, target, solution),
        })
    return examples


def evaluate(
    model_path, val_path, train_path,
    n_shots=5, max_samples=0, gpu=0, dtype="auto",
):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

    with open(val_path) as f:
        val_data = [json.loads(line) for line in f]
    if max_samples > 0:
        val_data = val_data[:max_samples]
    print(f"Evaluating {len(val_data)} samples with {n_shots}-shot GSM8K-style prompting")

    fewshot = create_fewshot_examples(train_path, n=n_shots)
    print(f"Created {len(fewshot)} few-shot examples")

    # Build GSM8K-style prompt prefix
    prefix = ""
    for ex in fewshot:
        prefix += f"Q: {ex['question']}\nA: {ex['answer']}\n\n"

    prompts = []
    for sample in val_data:
        numbers = sample["numbers"]
        target = sample["answer"]
        q = format_question(numbers, target)
        prompt = prefix + f"Q: {q}\nA:"
        prompts.append(prompt)

    llm = LLM(
        model=model_path,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.9,
        dtype=dtype,
        max_model_len=4096,
    )
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=256,
        stop=["Q:", "\n\n"],
    )

    print("Generating responses...")
    outputs = llm.generate(prompts, sampling_params)

    # Score -- extract expression after ####
    hash_pattern = re.compile(r"####\s*(.+?)(?:\s*$|\n)", re.MULTILINE)
    # Allowed AST nodes for a valid arithmetic expression
    import ast
    ALLOWED_NODES = (
        ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant,
        ast.Add, ast.Sub, ast.Mult, ast.Div, ast.USub,
    )

    extracted = 0
    parsable = 0
    nums_match = 0
    correct = 0

    for output, sample in zip(outputs, val_data):
        response = output.outputs[0].text
        numbers = sample["numbers"]
        target = int(sample["answer"])

        m = hash_pattern.search(response)
        if not m:
            continue
        expr = m.group(1).strip()
        extracted += 1

        # Parsable: can we parse it into an arithmetic AST at all?
        try:
            tree = ast.parse(expr, mode='eval')
            valid_ast = all(isinstance(n, ALLOWED_NODES) for n in ast.walk(tree))
            if not valid_ast:
                continue
        except SyntaxError:
            continue
        parsable += 1

        # Numbers match: does the expression use exactly the given numbers?
        ast_nums = sorted(
            int(n.value) for n in ast.walk(tree) if isinstance(n, ast.Constant)
        )
        if ast_nums != sorted(numbers):
            continue
        nums_match += 1

        # Correct: does it evaluate to the target?
        if validate_expression(expr, numbers, target):
            correct += 1

    total = len(val_data)
    print(f"\n{'='*60}")
    print(f"Model: {model_path}")
    print(f"Samples: {total}, {n_shots}-shot (GSM8K-style)")
    print(f"  Extracted (####): {extracted}/{total} ({100*extracted/total:.1f}%)")
    print(f"  Parsable (AST):   {parsable}/{total} ({100*parsable/total:.1f}%)")
    print(f"  Numbers match:    {nums_match}/{total} ({100*nums_match/total:.1f}%)")
    print(f"  Correct:          {correct}/{total} ({100*correct/total:.1f}%)")
    print(f"{'='*60}")

    return {
        "model": model_path,
        "n_shots": n_shots,
        "style": "gsm8k",
        "total": total,
        "extracted": extracted,
        "parsable": parsable,
        "nums_match": nums_match,
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
