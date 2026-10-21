import numpy as np
import pandas as pd
import pytest

from hmrec import candidates
from hmrec.metrics import recall_at_k
from hmrec.splits import LeakageError, Splitter, make_cuts


def make_frame():
    rows = []
    for day, date in enumerate(pd.date_range("2020-08-20", "2020-09-22", freq="D")):
        for customer in range(6):
            rows.append((date, customer, 100 + (day + customer) % 8))
    return pd.DataFrame(rows, columns=["t_dat", "customer_idx", "article_id"])


@pytest.fixture
def setup():
    splitter = Splitter.from_frame(make_frame())
    return splitter, splitter.cuts["valid"], splitter.history("valid")


def test_generate_is_capped_and_has_no_duplicates(setup):
    _, cut, history = setup
    result = candidates.generate(history, cut, np.arange(6), max_per_customer=5)
    assert result.groupby("customer_idx").size().max() <= 5
    assert not result.duplicated(["customer_idx", "article_id"]).any()


def test_position_starts_at_one_for_each_customer(setup):
    _, cut, history = setup
    result = candidates.generate(history, cut, np.arange(6), max_per_customer=5)
    assert (result.groupby("customer_idx")["position"].min() == 1).all()


def test_sources_reject_leaky_history():
    frame = make_frame()
    cut = make_cuts("2020-09-22")["train"]
    customers = np.arange(6)
    for build in (candidates.repurchase, candidates.copurchase, candidates.popular):
        with pytest.raises(LeakageError):
            build(frame, cut, customers)


def test_repurchase_only_returns_recently_bought_items(setup):
    _, cut, history = setup
    result = candidates.repurchase(history, cut, np.arange(6), weeks=1)
    recent = history[history["t_dat"] >= cut.history_end - pd.Timedelta(days=7)]
    allowed = set(zip(recent["customer_idx"], recent["article_id"]))
    assert set(zip(result["customer_idx"], result["article_id"])) <= allowed


def test_cold_start_customer_gets_popular_items(setup):
    _, cut, history = setup
    result = candidates.generate(history, cut, np.array([0, 999]), max_per_customer=5)
    cold = result[result["customer_idx"] == 999]
    assert len(cold) == 5
    assert cold["rank_repurchase"].isna().all()
    assert cold["rank_copurchase"].isna().all()
    assert cold["rank_popular"].notna().all()


def test_to_lists_keeps_position_order(setup):
    _, cut, history = setup
    result = candidates.generate(history, cut, np.arange(6), max_per_customer=5)
    lists = candidates.to_lists(result)
    first = result[result["customer_idx"] == 0].sort_values("position")
    assert lists[0] == [int(a) for a in first["article_id"]]


def test_recall_of_full_candidate_list_is_between_zero_and_one(setup):
    splitter, cut, history = setup
    labels = splitter.labels("valid")
    result = candidates.generate(history, cut, np.array(sorted(labels)), max_per_customer=5)
    value = recall_at_k(labels, candidates.to_lists(result))
    assert 0.0 <= value <= 1.0