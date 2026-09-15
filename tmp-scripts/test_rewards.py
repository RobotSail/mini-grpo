"""Test reward functions and task registry."""
import sys
sys.path.insert(0, ".")

from tasks import gsm8k_reward_fn, countdown_reward_fn_wrapper, get_reward_fn, RewardResult


def test_gsm8k_correct():
    r = gsm8k_reward_fn("The answer is <answer>42</answer>", 42.0, {})
    assert isinstance(r, RewardResult)
    assert r.reward == 1.1
    assert r.is_correct and r.is_parsable and r.has_format


def test_gsm8k_wrong():
    r = gsm8k_reward_fn("The answer is <answer>99</answer>", 42.0, {})
    assert r.reward == 0.1
    assert not r.is_correct and r.is_parsable and r.has_format


def test_gsm8k_no_tag():
    r = gsm8k_reward_fn("No answer tags here", 42.0, {})
    assert r.reward == 0.0
    assert not r.is_correct and not r.is_parsable and not r.has_format


def test_gsm8k_unparsable():
    r = gsm8k_reward_fn("<answer>abc</answer>", 42.0, {})
    assert r.reward == 0.0
    assert not r.is_parsable


def test_gsm8k_last_tag():
    r = gsm8k_reward_fn("<answer>99</answer> wait <answer>42</answer>", 42.0, {})
    assert r.reward == 1.1  # takes last tag


def test_gsm8k_currency():
    r = gsm8k_reward_fn("<answer>$42.00</answer>", 42.0, {})
    assert r.reward == 1.1


def test_countdown_correct():
    r = countdown_reward_fn_wrapper(
        "<think>reasoning here</think><answer>(3 + 4) * 2</answer>",
        14, {"numbers": [3, 4, 2]},
    )
    assert isinstance(r, RewardResult)
    assert r.reward == 1.1
    assert r.is_correct and r.is_parsable and r.has_format


def test_countdown_format_only():
    r = countdown_reward_fn_wrapper(
        "<think>thinking</think><answer>bad stuff</answer>",
        14, {"numbers": [3, 4, 2]},
    )
    assert r.reward == 0.1  # format reward only
    assert r.has_format and not r.is_correct


def test_countdown_no_format():
    r = countdown_reward_fn_wrapper("no tags at all", 14, {"numbers": [3, 4, 2]})
    assert r.reward == 0.0
    assert not r.has_format


def test_countdown_wrong_numbers():
    r = countdown_reward_fn_wrapper(
        "<think>hmm</think><answer>(5 + 4) * 2</answer>",
        14, {"numbers": [3, 4, 2]},
    )
    assert not r.is_parsable  # uses 5, not in numbers


def test_countdown_custom_format_reward():
    r = countdown_reward_fn_wrapper(
        "<think>reasoning</think><answer>(3 + 4) * 2</answer>",
        14, {"numbers": [3, 4, 2], "format_reward": 0.5},
    )
    assert r.reward == 1.5  # 1.0 + 0.5 format reward


def test_registry():
    fn = get_reward_fn("gsm8k")
    assert fn is gsm8k_reward_fn
    fn2 = get_reward_fn("countdown")
    assert fn2 is countdown_reward_fn_wrapper

    try:
        get_reward_fn("nonexistent")
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


def test_countdown_ast_validation():
    from countdown_utils import validate_ast
    assert validate_ast("(3 + 4) * 2", [3, 4, 2])
    assert not validate_ast("(5 + 4) * 2", [3, 4, 2])
    assert not validate_ast("3 ** 2", [3, 2])  # exponentiation not allowed
    assert validate_ast("3 - 2", [3, 2])
    assert validate_ast("3 / 2", [3, 2])
    assert not validate_ast("import os", [])  # not valid


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test_fn in tests:
        test_fn()
        print(f"  PASS: {test_fn.__name__}")
    print(f"\nAll {len(tests)} tests passed!")
