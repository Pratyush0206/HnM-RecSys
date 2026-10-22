import numpy as np
import pytest

from hmrec.bootstrap import paired_bootstrap, per_customer_scores
from hmrec.metrics import map_at_k


def test_per_customer_scores_average_equals_map():
    labels = {0: [1, 2], 1: [3], 2: [4, 5]}
    lists = {0: [1, 9], 1: [8, 3], 2: []}
    scores = per_customer_scores(labels, lists)
    assert len(scores) == 3
    assert scores.mean() == pytest.approx(map_at_k(labels, lists))


def test_per_customer_scores_follow_sorted_customer_order():
    labels = {5: [1], 2: [7]}
    lists = {5: [1], 2: [9]}
    assert per_customer_scores(labels, lists).tolist() == [0.0, 1.0]


def test_identical_scores_give_zero_difference():
    scores = np.array([0.0, 0.5, 1.0, 0.25])
    result = paired_bootstrap(scores, scores, n_boot=200)
    assert result["mean_difference"] == 0.0
    assert result["ci_low"] == 0.0
    assert result["ci_high"] == 0.0


def test_clear_improvement_has_interval_above_zero():
    base = np.random.default_rng(0).random(500)
    result = paired_bootstrap(base + 0.1, base, n_boot=500)
    assert result["ci_low"] > 0
    assert result["share_positive"] == 1.0


def test_bootstrap_is_reproducible():
    rng = np.random.default_rng(1)
    a = rng.random(100)
    b = rng.random(100)
    first = paired_bootstrap(a, b, n_boot=100, seed=3)
    second = paired_bootstrap(a, b, n_boot=100, seed=3)
    assert first == second


def test_mismatched_lengths_raise():
    with pytest.raises(ValueError):
        paired_bootstrap(np.zeros(3), np.zeros(4))


def test_empty_input_raises():
    with pytest.raises(ValueError):
        paired_bootstrap(np.array([]), np.array([]))