"""Rule-based math reward shared by training (verl custom_reward_function) and pass@k eval.

Score = 1.0 if the content of the last \\boxed{...} in the response is equivalent to the ground
truth under math_verify, else 0.0. No format bonus or penalty.
"""
from math_verify import ExprExtractionConfig, LatexExtractionConfig, parse, verify


def last_boxed(text):
    """Return the content of the last \\boxed{...} (or \\fbox{...}), braces balanced, or None."""
    start = max(text.rfind("\\boxed"), text.rfind("\\fbox"))
    if start < 0:
        return None
    i = text.find("{", start)
    if i < 0:
        return None
    depth = 0
    for j in range(i, len(text)):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return text[i + 1 : j]
    return None


def _parse(s):
    # Timeouts in math_verify use signal.alarm, which fails outside the main thread (verl reward
    # workers), so they are disabled here.
    return parse(
        f"${s}$",
        extraction_config=[LatexExtractionConfig(), ExprExtractionConfig()],
        parsing_timeout=None,
    )


def is_correct(response, ground_truth):
    pred = last_boxed(response)
    if pred is None or not pred.strip():
        return False
    if pred.strip() == str(ground_truth).strip():
        return True
    if len(pred) > 300:  # no timeout is available here, so skip symbolic checks on huge answers
        return False
    try:
        return bool(verify(_parse(str(ground_truth)), _parse(pred), timeout_seconds=None))
    except Exception:
        return False


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    """verl entry point: custom_reward_function.path=rlvr/reward.py, name=compute_score."""
    return 1.0 if is_correct(solution_str, ground_truth) else 0.0
