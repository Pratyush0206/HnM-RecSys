import pytest

from hmrec.metrics import average_precision_at_k, map_at_k


def test_worked_example():
    assert average_precision_at_k([7, 9], [3, 7, 5, 9]) == pytest.approx(0.5)


def test_perfect_ranking_scores_one():
    assert average_precision_at_k([1, 2], [1, 2, 3]) == pytest.approx(1.0)


def test_no_hits_scores_zero():
    assert average_precision_at_k([1, 2], [3, 4, 5]) == 0.0


def test_empty_actual_scores_zero():
    assert average_precision_at_k([], [1, 2]) == 0.0


def test_duplicate_predictions_do_not_count_twice():
    assert average_precision_at_k([1, 2], [1, 1, 2]) == pytest.approx(5 / 6)


def test_only_first_k_predictions_count():
    predicted = list(range(100, 112)) + [1]
    assert average_precision_at_k([1], predicted, k=12) == 0.0


def test_denominator_caps_at_k():
    actual = list(range(20))
    predicted = list(range(12))
    assert average_precision_at_k(actual, predicted, k=12) == pytest.approx(1.0)


def test_map_penalises_missing_customers():
    actual = {1: [10], 2: [20]}
    predicted = {1: [10]}
    assert map_at_k(actual, predicted) == pytest.approx(0.5)