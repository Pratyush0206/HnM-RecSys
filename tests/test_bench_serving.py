import numpy as np
import pytest

from hmrec.bench_serving import sample_ids, summarize
from hmrec.store import Store


def make_store(n):
    return Store(
        customer_ids=np.array([f"c{i}" for i in range(n)]),
        offsets=np.arange(n + 1),
        article_ids=np.arange(n, dtype="int64"),
        features=np.zeros((n, 1), dtype="float32"),
        feature_names=["a"],
        fallback=np.array([1], dtype="int64"),
    )


def test_summarize_reports_known_percentiles():
    result = summarize(range(1, 101))
    assert result["requests"] == 100
    assert result["p50_ms"] == pytest.approx(50.5)
    assert result["p95_ms"] == pytest.approx(95.05)
    assert result["p99_ms"] == pytest.approx(99.01)
    assert result["max_ms"] == 100
    assert result["mean_ms"] == pytest.approx(50.5)


def test_sample_ids_are_reproducible_and_come_from_the_store():
    store = make_store(50)
    first = sample_ids(store, 10, seed=1)
    assert first == sample_ids(store, 10, seed=1)
    assert len(first) == 10
    assert set(first) <= set(store.customer_ids.tolist())


def test_sample_ids_allows_more_requests_than_customers():
    assert len(sample_ids(make_store(3), 20, seed=1)) == 20