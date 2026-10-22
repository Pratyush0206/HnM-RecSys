import numpy as np
import pytest

from hmrec.ablate import compare


def make_run(customers, scores):
    return {
        "customers": np.asarray(customers),
        "ranker": np.asarray(scores, dtype="float64"),
    }


def test_compare_reports_full_minus_variant():
    scores = np.random.default_rng(0).random(300)
    full = make_run(range(300), scores + 0.05)
    variant = make_run(range(300), scores)
    result = compare(full, variant)
    assert result["mean_difference"] == pytest.approx(0.05)
    assert result["ci_low"] > 0
    assert result["map_full"] == pytest.approx(result["map_variant"] + 0.05)


def test_compare_rejects_different_customers():
    full = make_run([1, 2, 3], [0.1, 0.2, 0.3])
    variant = make_run([1, 2, 4], [0.1, 0.2, 0.3])
    with pytest.raises(ValueError):
        compare(full, variant)