"""
Task registry for GRPO training.

Maps task names to reward functions that return structured RewardResult objects.
This provides uniform metric logging (parse_rate, format_rate, correct_rate)
regardless of task.
"""

import re
from dataclasses import dataclass
from typing import Callable

# ── Reward result ──────────────────────────────────────────────────────────

@dataclass
class RewardResult:
    """Structured reward result for uniform metric logging across tasks.

    Attributes:
        reward: Numeric reward used by the GRPO loss.
        has_format: Whether the response follows the expected format.
        is_parsable: Whether the answer content is parsable/valid.
        is_correct: Whether the answer is correct.
    """
    reward: float
    has_format: bool = False
    is_parsable: bool = False
    is_correct: bool = False


# Type alias for reward functions
RewardFn = Callable[[str, float, dict], RewardResult]

# ── GSM8K reward ───────────────────────────────────────────────────────────

_answer_pattern = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)


def parse_number(text: str) -> float:
    """Parse a number from text, handling common GSM8K formats."""
    if not text or not isinstance(text, str):
        raise ValueError(f"Empty or invalid input: {text}")
    text = text.strip()
    text = re.sub(r"[$\u20AC\u00A3\u00A5\u20B9]", "", text)
    text = text.replace("%", "").replace(",", "").strip()
    if not any(c.isdigit() for c in text):
        raise ValueError(f"No digits found: {text}")
    match = re.search(r"-?\d+\.?\d*", text)
    if not match:
        raise ValueError(f"Could not extract number: {text}")
    return float(match.group())


def gsm8k_reward_fn(response: str, answer: float, prompt_data: dict) -> RewardResult:
    """GSM8K format reward.

    Uses the LAST <answer>...</answer> tag:
      0.0  — no parsable <answer> tag
      0.1  — parsable but numerically wrong  (+0.1 format reward)
      1.1  — correct answer                  (+0.1 format + 1.0 correct)
    """
    matches = _answer_pattern.findall(response)
    if not matches:
        return RewardResult(reward=0.0)

    last_match = matches[-1]
    try:
        parsed = parse_number(last_match)
    except ValueError:
        return RewardResult(reward=0.0)

    # Has a parsable <answer> tag
    is_correct = abs(parsed - float(answer)) < 1e-6
    reward = 0.1 + (1.0 if is_correct else 0.0)

    return RewardResult(
        reward=reward,
        has_format=True,
        is_parsable=True,
        is_correct=is_correct,
    )


# ── Countdown reward (delegates to countdown_utils) ───────────────────────

def countdown_reward_fn_wrapper(response: str, answer: float, prompt_data: dict) -> RewardResult:
    """Countdown reward wrapper that returns RewardResult.

    Delegates to countdown_utils.countdown_reward_fn which returns a dict,
    then converts to RewardResult.

    Reward space: [0.0, format_reward, correct_reward]
      0.0            — missing format (no <think>...</think> before <answer>)
      format_reward   — correct format but invalid/incorrect answer
      correct_reward  — correct format AND valid correct answer
    """
    from countdown_utils import countdown_reward_fn as _raw_fn

    result = _raw_fn(response, answer, prompt_data)

    has_think_format = result.get("has_format", False)  # <think> before <answer>
    has_answer_tag = bool(re.search(r"<answer>.*?</answer>", response, re.DOTALL | re.IGNORECASE))
    is_parsable = result.get("is_parsable", False)
    is_correct = result.get("is_correct", False)

    format_reward = prompt_data.get("format_reward", 0.1)
    correct_reward = prompt_data.get("correct_reward", 1.0)
    require_think = prompt_data.get("require_think", False)

    # Format check: require <think> tags only when think mode is on,
    # otherwise just require parsable <answer> tags
    has_format = has_think_format if require_think else has_answer_tag

    if is_correct and (has_format or not require_think):
        reward = correct_reward
    elif has_format:
        reward = format_reward
    else:
        reward = 0.0

    return RewardResult(
        reward=reward,
        has_format=has_format,
        is_parsable=is_parsable,
        is_correct=is_correct,
    )


# ── Task registry ─────────────────────────────────────────────────────────

TASK_REGISTRY: dict[str, RewardFn] = {
    "gsm8k": gsm8k_reward_fn,
    "countdown": countdown_reward_fn_wrapper,
}


def get_reward_fn(task: str) -> RewardFn:
    """Get the reward function for a task by name."""
    if task not in TASK_REGISTRY:
        available = ", ".join(TASK_REGISTRY.keys())
        raise ValueError(f"Unknown task: {task!r}. Available tasks: {available}")
    return TASK_REGISTRY[task]
