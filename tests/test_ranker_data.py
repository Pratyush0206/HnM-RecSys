import numpy as np
import pandas as pd
import pytest

from hmrec.ranker_data import add_twotower_score, keep_trainable, label_candidates
from hmrec.twotower import Embeddings


def make_table():
    return pd.DataFrame(
        {
            "customer_idx": [0, 0, 0, 1, 1],
            "article_id": [10, 20, 30, 10, 40],
            "position": [1, 2, 3, 1, 2],
        }
    )


def test_label_marks_bought_items():
    labelled = label_candidates(make_table(), {0: [20], 1: [99]})
    assert labelled["label"].tolist() == [0, 1, 0, 0, 0]


def test_label_keeps_row_count_and_dtype():
    table = make_table()
    labelled = label_candidates(table, {0: [10, 20, 30]})
    assert len(labelled) == len(table)
    assert labelled["label"].dtype == np.int8


def test_keep_trainable_drops_customers_without_positives():
    labelled = label_candidates(make_table(), {0: [20], 1: [99]})
    kept = keep_trainable(labelled)
    assert set(kept["customer_idx"]) == {0}
    assert len(kept) == 3


def test_twotower_score_is_dot_product_and_nan_when_unknown():
    embeddings = Embeddings(
        article_ids=np.array([10, 20]),
        item_vectors=np.array([[1.0, 0.0], [0.0, 1.0]], dtype="float32"),
        customer_vectors=np.array([[1.0, 0.0], [0.0, 1.0]], dtype="float32"),
        has_history=np.array([True, False]),
        losses=[],
    )
    scored = add_twotower_score(make_table(), embeddings)
    assert scored["tt_score"].iloc[0] == pytest.approx(1.0)
    assert scored["tt_score"].iloc[1] == pytest.approx(0.0)
    assert np.isnan(scored["tt_score"].iloc[2])
    assert scored["tt_score"].iloc[3:].isna().all()