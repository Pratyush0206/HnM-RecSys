import numpy as np
import pandas as pd
import pytest

from hmrec.features import FEATURE_COLUMNS
from hmrec.store import build_store, choose_customers, load_store, save_store

NAN = np.nan
MARKER = "tt_score"


def make_featured():
    frame = pd.DataFrame(
        {
            "customer_idx": [7, 7, 7, 3, 3, 3],
            "article_id": [11, 12, 13, 21, 12, 13],
            "position": [3, 1, 2, 3, 1, 2],
            "rank_popular": [NAN, 1, 2, NAN, 1, 2],
        }
    )
    for column in FEATURE_COLUMNS:
        if column not in frame.columns:
            frame[column] = np.arange(len(frame), dtype="float64")
    return frame


def make_customer_map():
    return pd.DataFrame(
        {"customer_id": ["hash3", "hash5", "hash7"], "customer_idx": [3, 5, 7]}
    )


def make_store():
    return build_store(make_featured(), make_customer_map(), FEATURE_COLUMNS)


def test_rows_follow_candidate_position_order():
    articles, features = make_store().rows("hash7")
    assert articles.tolist() == [12, 13, 11]
    marker = features[:, FEATURE_COLUMNS.index(MARKER)]
    assert marker.tolist() == [1.0, 2.0, 0.0]


def test_customers_are_looked_up_by_original_id():
    store = make_store()
    assert "hash3" in store
    assert "hash5" not in store
    assert "nobody" not in store
    with pytest.raises(KeyError):
        store.rows("nobody")


def test_offsets_split_rows_per_customer():
    store = make_store()
    assert len(store) == 2
    assert store.offsets.tolist() == [0, 3, 6]
    assert len(store.rows("hash3")[0]) == 3
    assert len(store.rows("hash7")[0]) == 3


def test_features_are_float32_with_every_feature_column():
    features = make_store().features
    assert features.dtype == np.float32
    assert features.shape == (6, len(FEATURE_COLUMNS))


def test_fallback_is_the_popular_list_in_rank_order():
    assert make_store().fallback.tolist() == [12, 13]


def test_save_and_load_roundtrip(tmp_path):
    store = make_store()
    save_store(store, tmp_path, extra={"cut": "valid"})
    loaded = load_store(tmp_path)
    assert loaded.feature_names == FEATURE_COLUMNS
    assert loaded.fallback.tolist() == store.fallback.tolist()
    assert np.array_equal(loaded.features, store.features, equal_nan=True)
    assert loaded.rows("hash3")[0].tolist() == store.rows("hash3")[0].tolist()


def test_choose_customers_is_sorted_reproducible_and_capped(tmp_path):
    path = tmp_path / "valid_features.parquet"
    pd.DataFrame({"customer_idx": [4, 4, 1, 9, 9, 2, 8, 8]}).to_parquet(path)
    first = choose_customers(path, 3, seed=42)
    assert first.tolist() == sorted(first.tolist())
    assert len(first) == 3
    assert first.tolist() == choose_customers(path, 3, seed=42).tolist()
    assert choose_customers(path, 100, seed=42).tolist() == [1, 2, 4, 8, 9]