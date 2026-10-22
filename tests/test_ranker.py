import numpy as np
import pandas as pd
import pytest

from hmrec.ranker import group_sizes, split_customers, top_k_lists


def make_frame():
    return pd.DataFrame(
        {
            "customer_idx": [0, 0, 0, 1, 1, 2, 2, 2, 2, 3],
            "article_id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        }
    )


def test_group_sizes_counts_rows_per_customer():
    assert group_sizes(make_frame()).tolist() == [3, 2, 4, 1]


def test_group_sizes_rejects_unsorted_rows():
    with pytest.raises(ValueError):
        group_sizes(make_frame().iloc[::-1])


def test_split_customers_keeps_each_customer_whole():
    fit, early = split_customers(make_frame(), 0.5, seed=1)
    assert set(fit["customer_idx"]).isdisjoint(set(early["customer_idx"]))
    assert len(fit) + len(early) == len(make_frame())
    assert early["customer_idx"].nunique() == 2


def test_top_k_lists_orders_by_score_and_cuts_to_k():
    scores = np.array([0.1, 0.9, 0.5, 0.2, 0.3, 0.0, 0.4, 0.8, 0.6, 0.7])
    lists = top_k_lists(make_frame(), scores, k=2)
    assert lists[0] == [2, 3]
    assert lists[1] == [5, 4]
    assert lists[2] == [8, 9]
    assert lists[3] == [10]