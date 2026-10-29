"""CPU checks for the reward and the pass@k estimator: python -m pytest rlvr/test_cpu.py"""
import itertools
import math

from passk_eval import pass_at_k, summarize
from reward import compute_score, last_boxed


def test_last_boxed_nested():
    assert last_boxed("a \\boxed{1} b \\boxed{\\frac{1}{2}}") == "\\frac{1}{2}"
    assert last_boxed("no answer") is None


def test_reward_equivalence():
    assert compute_score("math", "\\boxed{\\dfrac12}", "\\frac{1}{2}") == 1.0
    assert compute_score("math", "\\boxed{0.5}", "\\frac{1}{2}") == 1.0
    assert compute_score("math", "\\boxed{(3,\\pi/2)}", "\\left( 3, \\frac{\\pi}{2} \\right)") == 1.0
    assert compute_score("math", "\\boxed{5}", "4") == 0.0
    assert compute_score("math", "the answer is 4", "4") == 0.0


def test_pass_at_k_matches_brute_force():
    n = 8
    for c in range(n + 1):
        for k in range(1, n + 1):
            samples = [1] * c + [0] * (n - c)
            subsets = list(itertools.combinations(samples, k))
            exact = sum(any(s) for s in subsets) / len(subsets)
            assert math.isclose(pass_at_k(n, c, k), exact)


def test_summarize_ks():
    s = summarize([{"correct": 0}, {"correct": 4}], n=8)
    assert list(s["pass@k"]) == [1, 2, 4, 8]
    assert math.isclose(s["pass@k"][1], 0.25)
    assert s["frac_solved_by_none"] == 0.5
