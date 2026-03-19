"""
Countdown Numbers Game: dataset loading, expression validation, and reward function.

The countdown task: given N numbers and a target, find an arithmetic expression
using +, -, *, / where each number is used exactly once that equals the target.

Expected response format (R1-style):
  <think>...step-by-step reasoning...</think>
  <answer>EXPRESSION</answer>
  e.g. <think>I can try 25 + 3 = 28, then 28 * 7 = 196...</think>
       <answer>(25 + 3) * 7</answer>

Reward structure: {0.0, format_reward, 1.0 + format_reward}
  0.0            — missing format OR missing/invalid answer
  format_reward  — </think> appears before <answer>, and <answer> tags present
  1.0 + format   — correct format AND expression evaluates to target

Data source: Jiayi-Pan/Countdown-Tasks-3to4 from HuggingFace
"""

import ast
import json
import os
import re
import random
from itertools import permutations, product

import datasets
from tqdm import tqdm

# ── Constants ───────────────────────────────────────────────────────────────

DEFAULT_COUNTDOWN_SYSTEM_MSG = (
    "You are a helpful assistant that solves arithmetic problems. "
    "You always think step by step before answering."
)

# Assistant prefix that forces the model to begin chain-of-thought
R1_ASSISTANT_PREFIX = "<think>\n"

ANSWER_PATTERN = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)
ALLOWED_EXPR_CHARS = re.compile(r"^[\d\s()+\-*/\.]+$")


# ── AST validation ────────────────────────────────────────────────────────

# Allowed AST node types for a valid countdown expression.
_ALLOWED_AST_NODES = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.USub,
)


def validate_ast(expr: str, numbers: list[int]) -> bool:
    """Validate that expr is a well-formed arithmetic expression using exactly
    the given numbers.

    Parses into a Python AST and checks:
      1. Only allowed node types (binary +,-,*,/, unary -, numeric literals, parens)
      2. Numeric literals extracted from the AST form the same multiset as `numbers`

    Numbers are extracted from AST nodes, not regex — this prevents the model
    from smuggling digits inside operator sequences or string tricks.
    """
    try:
        tree = ast.parse(expr, mode='eval')
    except SyntaxError:
        return False

    extracted = []
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_AST_NODES):
            return False
        if isinstance(node, ast.UnaryOp) and not isinstance(node.op, ast.USub):
            return False
        if isinstance(node, ast.BinOp) and not isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
            return False
        if isinstance(node, ast.Constant):
            if not isinstance(node.value, (int, float)):
                return False
            extracted.append(int(node.value))

    return sorted(extracted) == sorted(numbers)


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

def countdown_reward_fn(
    response: str, answer: float, prompt_data: dict,
) -> dict:
    """Grade a countdown response, returning structured results.

    The model must produce the full format including opening tags:

        <think>...reasoning...</think>
        <answer>(3 + 4) * 2</answer>

    Returns dict with:
      has_format:  bool — ``<think>...</think>`` before ``<answer>...</answer>``
      is_parsable: bool — valid AST using exactly the given numbers
      is_correct:  bool — is_parsable AND expression evaluates to target
    """
    numbers = prompt_data.get("numbers")
    target = int(answer)
    result = {"has_format": False, "is_parsable": False, "is_correct": False}

    # ── Format check: <think>...</think> before <answer>...</answer> ──
    resp_lower = response.lower()
    think_start = resp_lower.find("<think>")
    think_end = resp_lower.find("</think>")
    answer_start = resp_lower.find("<answer>")
    has_think = think_start != -1 and think_end != -1 and think_start < think_end
    has_answer_tag = answer_start != -1
    if has_think and has_answer_tag and think_end < answer_start:
        result["has_format"] = True

    # ── Answer validation ──
    matches = ANSWER_PATTERN.findall(response)
    if not matches:
        return result

    content = matches[-1].strip()

    if not content:
        return result

    # Lexical filter (cheap, rejects obvious junk before parsing)
    if not ALLOWED_EXPR_CHARS.match(content) or "**" in content:
        return result

    # AST validation — valid expression structure using exactly the given numbers
    if numbers is not None and not validate_ast(content, numbers):
        return result

    result["is_parsable"] = True

    # Correctness check (evaluates to target)
    if numbers is not None and validate_expression(content, numbers, target):
        result["is_correct"] = True

    return result


# ── Countdown solver (brute-force for 3-4 numbers) ─────────────────────────

def _eval_safe(expr: str) -> float | None:
    """Safely evaluate an arithmetic expression."""
    try:
        return eval(expr, {"__builtins__": {}})
    except Exception:
        return None


ALL_OPS = ("+", "-", "*", "/")
ADDITIVE_OPS = ("+", "-")


def _generate_trees(exprs: list[str], ops: tuple[str, ...] = ALL_OPS) -> list[str]:
    """Generate all possible parenthesized expressions from a list of expression strings.

    For N expressions, tries all ways to pick 2, combine with an operator
    from `ops`, and recurse. Returns all valid expression strings.
    """
    if len(exprs) == 1:
        return exprs

    results = []
    for i in range(len(exprs)):
        for j in range(len(exprs)):
            if i == j:
                continue
            remaining = [exprs[k] for k in range(len(exprs)) if k != i and k != j]
            for op in ops:
                combined = f"({exprs[i]} {op} {exprs[j]})"
                for tree in _generate_trees(remaining + [combined], ops=ops):
                    results.append(tree)
    return results


def solve_countdown(
    numbers: list[int], target: int, ops: tuple[str, ...] = ALL_OPS, tol: float = 1e-6,
) -> str | None:
    """Find an arithmetic expression using all numbers exactly once that equals target.

    Brute-force: enumerate all binary tree structures × operator combinations.
    For 3-4 numbers this is fast (< 10K combinations).

    Args:
        ops: Tuple of allowed operators. Default is all four (+, -, *, /).

    Returns a parenthesized expression string, or None if unsolvable.
    """
    str_nums = [str(n) for n in numbers]

    for tree_expr in _generate_trees(str_nums, ops=ops):
        result = _eval_safe(tree_expr)
        if result is not None and abs(result - target) < tol:
            return tree_expr

    return None


def requires_mult_or_div(numbers: list[int], target: int) -> bool:
    """Return True if the problem cannot be solved with only + and -.

    These are the 'hard' problems that require multiplication or division.
    """
    return solve_countdown(numbers, target, ops=ADDITIVE_OPS) is None


# ── Dataset generation ──────────────────────────────────────────────────────

def _format_user_prompt(numbers: list[int], target: int) -> str:
    nums_str = ", ".join(str(n) for n in numbers)
    return (
        f"Find a mathematical expression that equals {target} "
        f"using the numbers [{nums_str}].\n\n"
        f"Rules:\n"
        f"- Use each number exactly once\n"
        f"- You may use the following operations:\n"
        f"  - Addition (+)\n"
        f"  - Subtraction (-)\n"
        f"  - Multiplication (*)\n"
        f"  - Division (/)\n"
        f"  - Parentheses ( )\n"
        f"- Give your final expression in <answer>...</answer> tags\n\n"
        f"Example: <answer>(8 - 3) * (12 / 4)</answer>"
    )


def generate_countdown_dataset(
    total_samples: int = 0,
    system_msg: str = DEFAULT_COUNTDOWN_SYSTEM_MSG,
    seed: int = 42,
    val_split: float = 0.05,
    test_split: float = 0.05,
    verify: bool = True,
    grpo_only: bool = False,
) -> dict[str, list[dict]]:
    """Load countdown problems from HuggingFace and generate paired GRPO/SFT datasets.

    Uses Jiayi-Pan/Countdown-Tasks-3to4 (~490K problems, 3-4 numbers, targets 0-100).
    For SFT, solves each problem to produce a valid expression.
    Skips unsolvable problems. Verifies all solutions through the reward function.

    Args:
        total_samples: Max samples to include (0 = all available).
        system_msg: System message for the chat template.
        seed: Random seed for shuffling.
        val_split: Fraction of data for validation set.
        test_split: Fraction of data for test set.
        verify: Whether to verify all SFT solutions pass the reward function.
        grpo_only: If True, skip solving (much faster, only GRPO data).

    Returns:
        Dict with keys: grpo_train, grpo_val, grpo_test, sft_train, sft_val, sft_test
    """
    # Load from HuggingFace
    raw_dataset = datasets.load_dataset("Jiayi-Pan/Countdown-Tasks-3to4", split="train")

    # Shuffle deterministically
    raw_dataset = raw_dataset.shuffle(seed=seed)

    # Collect problems (solve for SFT ground truth unless grpo_only)
    solved = []  # list of (numbers, target, expression)
    skipped = 0
    limit = total_samples if total_samples > 0 else len(raw_dataset)

    desc = "Loading problems" if grpo_only else "Solving problems"
    pbar = tqdm(raw_dataset, total=limit, desc=desc)
    for sample in pbar:
        if len(solved) >= limit:
            break

        target = sample["target"]
        numbers = list(sample["nums"])

        if grpo_only:
            solved.append((numbers, target, None))
        else:
            expression = solve_countdown(numbers, target)
            if expression is None:
                skipped += 1
                pbar.total = min(pbar.total + 1, len(raw_dataset))
                continue
            solved.append((numbers, target, expression))
        pbar.update(0)  # refresh display
    pbar.close()

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

            if expression is not None:
                sft.append({
                    "messages": [
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": user_prompt},
                        {"role": "assistant", "content": f"<answer>{expression}</answer>"},
                    ],
                    "answer": target,
                    "numbers": numbers,
                })
        return grpo, sft

    grpo_train, sft_train = _to_formats(train_solved)
    grpo_val, sft_val = _to_formats(val_solved)
    grpo_test, sft_test = _to_formats(test_solved)

    # ── Verify all solutions pass the reward function ──
    if verify and not grpo_only:
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
                assert r["is_parsable"] and r["is_correct"], (
                    f"{label} sample {i} failed verification: {r}, "
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


# ── Synthetic countdown problem generator (backward decomposition) ────────

COUNTDOWN_SYSTEM_MSG = """A conversation between User and Assistant. The User asks a question and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. The reasoning process and answer are enclosed within <think>...</think> and <answer>...</answer> tags, respectively, that is, <think> reasoning process here </think><answer> answer here </answer>"""


def _get_factors(n: int) -> list[int]:
    """Return all factors of n in [2, n//2]."""
    if n <= 1:
        return []
    factors = []
    for f in range(2, int(n ** 0.5) + 1):
        if n % f == 0:
            factors.append(f)
            if f != n // f and n // f != n:
                factors.append(n // f)
    return factors


def _operand_pairs(v: int, op: str, rng: random.Random,
                   lo: int, hi: int, left_leaf: bool, right_leaf: bool,
                   max_val: int = 9999, n: int = 6) -> list[tuple[int, int]]:
    """Generate (a, b) pairs where a op b = v. All values are positive integers."""
    pairs = []

    if op == '+':
        # a + b = v, need v >= lo + 1 so both operands can be positive
        if v < 2:
            return []
        for _ in range(n):
            if left_leaf and right_leaf:
                a_lo, a_hi = max(lo, v - hi), min(hi, v - lo)
                if a_lo > a_hi:
                    break
                a = rng.randint(a_lo, a_hi)
            elif left_leaf:
                upper = min(hi, v - 1)
                if lo > upper:
                    break
                a = rng.randint(lo, upper)
            elif right_leaf:
                upper = min(hi, v - 1)
                if lo > upper:
                    break
                b = rng.randint(lo, upper)
                pairs.append((v - b, b))
                continue
            else:
                if v <= 3:
                    break
                a = rng.randint(max(2, v // 4), max(3, 3 * v // 4))
            b = v - a
            if b > 0 and a > 0 and a <= max_val and b <= max_val:
                pairs.append((a, b))

    elif op == '-':
        # a - b = v → a = v + b
        for _ in range(n):
            if right_leaf:
                b = rng.randint(lo, hi)
                a = v + b
                if left_leaf and a > hi:
                    continue
            elif left_leaf:
                if v >= hi:
                    break
                a = rng.randint(max(lo, v + 1), hi)
                b = a - v
            else:
                b = rng.randint(1, min(99, max_val - v))
                a = v + b
            if a > 0 and b > 0 and a <= max_val and b <= max_val:
                pairs.append((a, b))

    elif op == '*':
        # a * b = v, use factors
        factors = _get_factors(v)
        if not factors:
            return []
        rng.shuffle(factors)
        for f in factors[:n]:
            a, b = f, v // f
            if rng.random() < 0.5:
                a, b = b, a
            if left_leaf and not (lo <= a <= hi):
                continue
            if right_leaf and not (lo <= b <= hi):
                continue
            if a <= max_val and b <= max_val and a > 1 and b > 1:
                pairs.append((a, b))

    elif op == '/':
        # a / b = v → a = v * b
        for _ in range(n):
            if right_leaf:
                b = rng.randint(max(2, lo), hi)
            else:
                b = rng.randint(2, 20)
            a = v * b
            if left_leaf and not (lo <= a <= hi):
                continue
            if a <= max_val:
                pairs.append((a, b))

    return pairs


def _decompose(v: int, n_leaves: int, rng: random.Random,
               num_range: tuple[int, int] = (1, 99),
               need_mult_div: bool = True,
               op_weights: dict[str, float] | None = None) -> tuple[list[int], bool] | None:
    """Decompose value v into n_leaves numbers by building an expression tree backward.

    Args:
        op_weights: Optional dict mapping operator -> weight (higher = preferred).
            Used to bias toward underrepresented operators.

    Returns (numbers, used_mult_div) or None if decomposition fails.
    """
    lo, hi = num_range

    if n_leaves == 1:
        if lo <= v <= hi:
            return [v], False
        return None

    splits = list(range(1, n_leaves))
    rng.shuffle(splits)

    for n_left in splits:
        n_right = n_leaves - n_left

        ops = ['+', '-', '*', '/']
        if need_mult_div:
            md = [op for op in ops if op in ('*', '/')]
            ad = [op for op in ops if op in ('+', '-')]
            rng.shuffle(md)
            rng.shuffle(ad)
            ops = md + ad
        elif op_weights:
            # Sort by weight (descending) with noise for randomness
            ops.sort(key=lambda op: op_weights.get(op, 1.0) + rng.random() * 0.3, reverse=True)
        else:
            rng.shuffle(ops)

        for op in ops:
            is_md = op in ('*', '/')
            pairs = _operand_pairs(
                v, op, rng, lo, hi,
                left_leaf=(n_left == 1), right_leaf=(n_right == 1),
            )
            for a, b in pairs:
                left = _decompose(a, n_left, rng, num_range,
                                  need_mult_div and not is_md, op_weights)
                if left is None:
                    continue
                left_nums, left_md = left

                still_need = need_mult_div and not is_md and not left_md
                right = _decompose(b, n_right, rng, num_range, still_need, op_weights)
                if right is None:
                    continue
                right_nums, right_md = right

                got_md = is_md or left_md or right_md
                if need_mult_div and not got_md:
                    continue

                return left_nums + right_nums, got_md

    return None


def _generate_few_shot_examples(
    n: int, rng: random.Random,
    num_range: tuple[int, int] = (1, 99),
    target_range: tuple[int, int] = (10, 100),
    n_numbers: tuple[int, ...] = (3, 4),
) -> tuple[list[tuple[str, str]], set]:
    """Generate solved few-shot examples as (user_prompt, answer_expr) pairs.

    Returns (examples, seen_keys) so the caller can exclude these from training.
    """
    examples = []
    seen = set()

    while len(examples) < n:
        target = rng.randint(*target_range)
        k = rng.choice(n_numbers)

        result = _decompose(target, k, rng, num_range, need_mult_div=True)
        if result is None:
            continue

        numbers, _ = result
        key = (tuple(sorted(numbers)), target)
        if key in seen:
            continue
        if not requires_mult_or_div(numbers, target):
            continue

        # Solve to get the expression string
        expr = solve_countdown(numbers, target)
        if expr is None:
            continue

        seen.add(key)
        user_prompt = _format_user_prompt(numbers, target)
        examples.append((user_prompt, f"<answer>{expr}</answer>"))

    return examples, seen


def generate_synthetic_countdown(
    n_train: int,
    n_val: int = 1000,
    num_range: tuple[int, int] = (1, 99),
    target_range: tuple[int, int] = (10, 100),
    n_numbers: tuple[int, ...] = (3, 4),
    system_msg: str = COUNTDOWN_SYSTEM_MSG,
    seed: int = 42,
    few_shot: int = 0,
    think: bool = False,
    hard: bool = True,
    r1_prompt: bool = False,
) -> dict[str, list[dict]]:
    """Generate synthetic countdown problems via backward decomposition.

    Picks a target in target_range, then decomposes it into n numbers by
    building an expression tree in reverse. At least one operation is * or /,
    and division always produces integer results.

    Args:
        n_train: Number of training samples to generate.
        n_val: Number of validation samples to generate.
        num_range: (min, max) for leaf numbers.
        target_range: (min, max) for targets.
        n_numbers: Tuple of allowed operand counts (e.g. (3, 4)).
        system_msg: System message for the chat template (None = no system msg).
        seed: Random seed.
        few_shot: Number of solved ICL examples to prepend to each prompt.
            These are held out from training/val and formatted as multi-turn
            user/assistant exchanges (answer only, no thinking trace).

    Returns:
        Dict with keys: train, val (each a list of GRPO-format dicts).
    """
    rng = random.Random(seed)

    # System message: R1 prompt if --r1-prompt or --think, otherwise none
    if not think and not r1_prompt:
        system_msg = None

    # Load few-shot ICL examples from annotated dataset
    icl_messages = []
    seen = set()
    if few_shot > 0:
        icl_path = os.path.join(os.path.dirname(__file__), "data", "countdown_icl_examples.json")
        with open(icl_path) as f:
            all_icl = json.load(f)
        if few_shot > len(all_icl):
            raise ValueError(
                f"Requested {few_shot} few-shot examples but only "
                f"{len(all_icl)} available in {icl_path}"
            )
        selected_icl = all_icl[:few_shot]
        for ex in selected_icl:
            icl_messages.append({"role": "user", "content": ex["user"]})
            if think:
                # Include full <think>...<answer> response
                icl_messages.append({"role": "assistant", "content": ex["assistant"]})
            else:
                # Strip <think>/<think> tags but keep the reasoning text before <answer>
                response = ex["assistant"]
                response = re.sub(r"</?think>", "", response).strip()
                icl_messages.append({"role": "assistant", "content": response})
            # Exclude ICL problems from train/val
            key = (tuple(sorted(ex["numbers"])), ex["target"])
            seen.add(key)
        mode = "with <think> traces" if think else "answer-only"
        print(f"Loaded {few_shot} ICL examples ({mode}) from {icl_path}")

    total_needed = n_train + n_val
    results = []
    attempts = 0

    # Track operator counts for reweighing
    op_keys = ["+", "-", "*", "/"]
    op_counts = {op: 0 for op in op_keys}

    pbar = tqdm(total=total_needed, desc="Generating problems")
    while len(results) < total_needed:
        attempts += 1
        target = rng.randint(*target_range)
        k = rng.choice(n_numbers)

        # Compute weights: inverse of current counts so underrepresented ops are preferred
        total_ops = sum(op_counts.values()) + len(op_keys)  # +len to avoid div by 0
        op_weights = {op: total_ops / (op_counts[op] + 1) for op in op_keys}

        result = _decompose(target, k, rng, num_range, need_mult_div=hard, op_weights=op_weights)
        if result is None:
            continue

        numbers, _ = result

        # Deduplicate
        key = (tuple(sorted(numbers)), target)
        if key in seen:
            continue

        # Verify the problem genuinely requires * or /
        if hard and not requires_mult_or_div(numbers, target):
            continue

        # Solve to track which operators the solution actually uses
        sol = solve_countdown(numbers, target)
        if sol is not None:
            for op in op_keys:
                if f" {op} " in sol:
                    op_counts[op] += 1

        seen.add(key)
        results.append((numbers, target))
        pbar.update(1)
    pbar.close()

    total = max(len(results), 1)
    dist_str = ", ".join(f"{op}={op_counts[op]} ({100*op_counts[op]/total:.0f}%)" for op in op_keys)
    print(f"Generated {len(results)} problems in {attempts} attempts "
          f"({100 * len(results) / attempts:.1f}% acceptance rate)")
    print(f"Operator distribution: {dist_str}")

    def _to_grpo(items):
        out = []
        for numbers, target in items:
            user_prompt = _format_user_prompt(numbers, target)
            messages = []
            if system_msg:
                messages.append({"role": "system", "content": system_msg})
            messages.extend(icl_messages)
            messages.append({"role": "user", "content": user_prompt})
            out.append({
                "messages": messages,
                "answer": target,
                "numbers": numbers,
                "problem": user_prompt,
                "operation": "countdown",
            })
        return out

    return {
        "train": _to_grpo(results[n_val:]),
        "val": _to_grpo(results[:n_val]),
    }


def generate_countdown_from_hf(
    n_train: int = 50000,
    n_val: int = 1000,
    n_test: int = 0,
    seed: int = 42,
    few_shot: int = 0,
    think: bool = False,
    hard: bool = True,
    r1_prompt: bool = False,
) -> dict[str, list[dict]]:
    """Load countdown problems from Jiayi-Pan/Countdown-Tasks-3to4 and format for GRPO.

    Applies the same ICL/think/hard options as the synthetic generator.

    Args:
        n_train: Number of training samples.
        n_val: Number of validation samples.
        seed: Random seed for shuffling.
        few_shot: Number of annotated ICL examples to prepend.
        think: If True, use <think> traces in ICL and require them for reward.
        hard: If True, keep only problems requiring * or /.
        r1_prompt: If True, prepend R1 system prompt (independent of --think).
    """
    rng = random.Random(seed)
    system_msg = COUNTDOWN_SYSTEM_MSG if (think or r1_prompt) else None

    # Load ICL examples
    icl_messages = []
    icl_keys = set()
    if few_shot > 0:
        icl_path = os.path.join(os.path.dirname(__file__), "data", "countdown_icl_examples.json")
        with open(icl_path) as f:
            all_icl = json.load(f)
        if few_shot > len(all_icl):
            raise ValueError(f"Requested {few_shot} few-shot but only {len(all_icl)} available")
        for ex in all_icl[:few_shot]:
            icl_messages.append({"role": "user", "content": ex["user"]})
            if think:
                icl_messages.append({"role": "assistant", "content": ex["assistant"]})
            else:
                response = re.sub(r"</?think>", "", ex["assistant"]).strip()
                icl_messages.append({"role": "assistant", "content": response})
            icl_keys.add((tuple(sorted(ex["numbers"])), ex["target"]))
        mode = "with <think> traces" if think else "answer-only"
        print(f"Loaded {few_shot} ICL examples ({mode}) from {icl_path}")

    # Load from HuggingFace
    raw = datasets.load_dataset("Jiayi-Pan/Countdown-Tasks-3to4", split="train")
    raw = raw.shuffle(seed=seed)

    total_needed = n_train + n_val + n_test
    seen = set(icl_keys)
    results = []

    pbar = tqdm(raw, desc="Loading from HuggingFace", total=total_needed)
    for sample in pbar:
        if len(results) >= total_needed:
            break

        target = sample["target"]
        numbers = list(sample["nums"])

        key = (tuple(sorted(numbers)), target)
        if key in seen:
            continue

        if hard and not requires_mult_or_div(numbers, target):
            continue

        seen.add(key)
        results.append((numbers, target))
        pbar.update(0)
    pbar.close()

    print(f"Loaded {len(results)} problems from HuggingFace "
          f"({'hard only' if hard else 'all'})")

    rng.shuffle(results)

    def _to_grpo(items):
        out = []
        for numbers, target in items:
            user_prompt = _format_user_prompt(numbers, target)
            messages = []
            if system_msg:
                messages.append({"role": "system", "content": system_msg})
            messages.extend(icl_messages)
            messages.append({"role": "user", "content": user_prompt})
            out.append({
                "messages": messages,
                "answer": target,
                "numbers": numbers,
                "problem": user_prompt,
                "operation": "countdown",
            })
        return out

    out = {
        "train": _to_grpo(results[n_val + n_test:]),
        "val": _to_grpo(results[:n_val]),
    }
    if n_test > 0:
        out["test"] = _to_grpo(results[n_val:n_val + n_test])
    return out
