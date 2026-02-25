"""
Countdown Numbers Game: dataset loading, expression validation, and reward function.

The countdown task: given N numbers and a target, find an arithmetic expression
using +, -, *, / where each number is used exactly once that equals the target.

Expected answer format: <answer>EXPRESSION</answer>
  e.g. <answer>(25 + 3) * 7</answer>

Reward structure: {0.0, 0.1, 1.1}
  0.0 — no <answer> tag, empty content, or content fails lexical filter
  0.1 — format reward: <answer> tags present, non-empty, passes lexical filter
  1.1 — format + correctness: expression uses all numbers exactly once and evaluates to target

Data source: Jiayi-Pan/Countdown-Tasks-3to4 from HuggingFace
"""

import re
import random
from itertools import permutations, product

import datasets

# ── Constants ───────────────────────────────────────────────────────────────

DEFAULT_COUNTDOWN_SYSTEM_MSG = (
    "You are a countdown numbers game solver. Given a set of numbers and a target, "
    "find an arithmetic expression using +, -, *, / that equals the target. Each "
    "number must be used exactly once. Show your reasoning, then put your final "
    "expression inside <answer>...</answer> tags.\n\n"
    "Example: numbers [1, 3, 4, 6], target 24\n"
    "<answer>6 / (1 - 3 / 4)</answer>"
)

ANSWER_PATTERN = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)
ALLOWED_EXPR_CHARS = re.compile(r"^[\d\s()+\-*/\.]+$")


# ── Expression validation ──────────────────────────────────────────────────

def validate_expression(expr: str, numbers: list[int], target: int, tol: float = 1e-6) -> bool:
    """Validate that an arithmetic expression is a correct countdown solution.

    Checks:
    1. Only allowed characters (digits, spaces, parentheses, +-*/, decimal point)
    2. No exponentiation (**)
    3. Integers in expression match provided numbers exactly (multiset equality)
    4. Expression evaluates to target (within tolerance)
    """
    expr = expr.strip()
    if not expr:
        return False

    if not ALLOWED_EXPR_CHARS.match(expr):
        return False

    if "**" in expr:
        return False

    # Multiset equality: all numbers used exactly once
    expr_nums = sorted(int(x) for x in re.findall(r"\d+", expr))
    if expr_nums != sorted(numbers):
        return False

    # Safe eval
    try:
        result = eval(expr, {"__builtins__": {}})
    except Exception:
        return False

    return abs(result - target) < tol


# ── Reward function ─────────────────────────────────────────────────────────

def countdown_reward_fn(response: str, answer: float, prompt_data: dict) -> float:
    """Reward function for countdown problems.

    Matches the RewardFn protocol: (response, answer, prompt_data) -> float

    Format reward (0.1): <answer> tags present, non-empty content, passes lexical filter.
    Correctness reward (+1.0): expression uses all numbers exactly once, evaluates to target.

    Total reward set: {0.0, 0.1, 1.1}
    """
    numbers = prompt_data.get("numbers")
    target = int(answer)

    # Step 1: Extract answer tag
    matches = ANSWER_PATTERN.findall(response)
    if not matches:
        return 0.0

    content = matches[-1].strip()

    # Step 2: Non-empty check
    if not content:
        return 0.0

    # Step 3: Lexical filter (cheap structural check)
    if not ALLOWED_EXPR_CHARS.match(content) or "**" in content:
        return 0.0

    # Format reward: valid structure
    reward = 0.1

    # Step 4: Correctness check
    if numbers is not None and validate_expression(content, numbers, target):
        reward += 1.0

    return reward


# ── Countdown solver (brute-force for 3-4 numbers) ─────────────────────────

def _eval_safe(expr: str) -> float | None:
    """Safely evaluate an arithmetic expression."""
    try:
        return eval(expr, {"__builtins__": {}})
    except Exception:
        return None


def _generate_trees(exprs: list[str]) -> list[str]:
    """Generate all possible parenthesized expressions from a list of expression strings.

    For N expressions, tries all ways to pick 2, combine with an operator,
    and recurse. Returns all valid expression strings that evaluate without error.
    """
    if len(exprs) == 1:
        return exprs

    results = []
    for i in range(len(exprs)):
        for j in range(len(exprs)):
            if i == j:
                continue
            remaining = [exprs[k] for k in range(len(exprs)) if k != i and k != j]
            for op in ["+", "-", "*", "/"]:
                combined = f"({exprs[i]} {op} {exprs[j]})"
                for tree in _generate_trees(remaining + [combined]):
                    results.append(tree)
    return results


def solve_countdown(numbers: list[int], target: int, tol: float = 1e-6) -> str | None:
    """Find an arithmetic expression using all numbers exactly once that equals target.

    Brute-force: enumerate all binary tree structures × operator combinations.
    For 3-4 numbers this is fast (< 10K combinations).

    Returns a parenthesized expression string, or None if unsolvable.
    """
    str_nums = [str(n) for n in numbers]

    for tree_expr in _generate_trees(str_nums):
        result = _eval_safe(tree_expr)
        if result is not None and abs(result - target) < tol:
            return tree_expr

    return None


# ── Dataset generation ──────────────────────────────────────────────────────

def _format_user_prompt(numbers: list[int], target: int) -> str:
    nums_str = ", ".join(str(n) for n in numbers)
    return (
        f"Using the numbers [{nums_str}], reach the target {target}. "
        f"Each number must be used exactly once. Available operations: +, -, *, /."
    )


def generate_countdown_dataset(
    total_samples: int = 0,
    system_msg: str = DEFAULT_COUNTDOWN_SYSTEM_MSG,
    seed: int = 42,
    val_split: float = 0.05,
    test_split: float = 0.05,
) -> dict[str, list[dict]]:
    """Load countdown problems from HuggingFace and generate paired GRPO/SFT datasets.

    Uses Jiayi-Pan/Countdown-Tasks-3to4 (~490K problems, 3-4 numbers, targets 0-100).
    For SFT, solves each problem to produce a valid expression.
    Skips unsolvable problems. Verifies all solutions through the reward function.

    Flow:
    1. Load full dataset from HuggingFace
    2. Shuffle with seed
    3. Take first total_samples (0 = all)
    4. Solve each problem, skip unsolvable
    5. Split into train/val/test
    6. Verify all solutions pass reward function
    7. Convert to GRPO and SFT formats

    Args:
        total_samples: Max samples to include (0 = all available).
        system_msg: System message for the chat template.
        seed: Random seed for shuffling.
        val_split: Fraction of data for validation set.
        test_split: Fraction of data for test set.

    Returns:
        Dict with keys: grpo_train, grpo_val, grpo_test, sft_train, sft_val, sft_test
    """
    # Load from HuggingFace
    raw_dataset = datasets.load_dataset("Jiayi-Pan/Countdown-Tasks-3to4", split="train")

    # Shuffle deterministically
    raw_dataset = raw_dataset.shuffle(seed=seed)

    # Solve all problems, collect solvable ones
    solved = []  # list of (numbers, target, expression)
    skipped = 0
    limit = total_samples if total_samples > 0 else len(raw_dataset)

    for sample in raw_dataset:
        if len(solved) >= limit:
            break

        target = sample["target"]
        numbers = list(sample["nums"])

        expression = solve_countdown(numbers, target)
        if expression is None:
            skipped += 1
            continue

        solved.append((numbers, target, expression))

    if skipped > 0:
        print(f"Skipped {skipped} unsolvable problems")

    # Split into train/val/test
    n = len(solved)
    n_test = int(n * test_split)
    n_val = int(n * val_split)
    n_train = n - n_val - n_test

    train_solved = solved[:n_train]
    val_solved = solved[n_train:n_train + n_val]
    test_solved = solved[n_train + n_val:]

    def _to_formats(items):
        grpo, sft = [], []
        for numbers, target, expression in items:
            user_prompt = _format_user_prompt(numbers, target)
            sft_response = f"<answer>{expression}</answer>"

            grpo.append({
                "messages": [
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_prompt},
                ],
                "answer": target,
                "numbers": numbers,
                "problem": user_prompt,
                "operation": "countdown",
            })

            sft.append({
                "messages": [
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_prompt},
                    {"role": "assistant", "content": sft_response},
                ],
                "answer": target,
                "numbers": numbers,
            })
        return grpo, sft

    grpo_train, sft_train = _to_formats(train_solved)
    grpo_val, sft_val = _to_formats(val_solved)
    grpo_test, sft_test = _to_formats(test_solved)

    # ── Verify all solutions pass the reward function ──
    for label, grpo_list, sft_list in [
        ("train", grpo_train, sft_train),
        ("val", grpo_val, sft_val),
        ("test", grpo_test, sft_test),
    ]:
        for i in range(len(grpo_list)):
            numbers = grpo_list[i]["numbers"]
            target = grpo_list[i]["answer"]
            sft_response = sft_list[i]["messages"][2]["content"]
            r = countdown_reward_fn(sft_response, target, {"numbers": numbers})
            assert r == 1.1, (
                f"{label} sample {i} failed verification: reward={r}, "
                f"target={target}, numbers={numbers}, response={sft_response!r}"
            )

    return {
        "grpo_train": grpo_train,
        "grpo_val": grpo_val,
        "grpo_test": grpo_test,
        "sft_train": sft_train,
        "sft_val": sft_val,
        "sft_test": sft_test,
    }
