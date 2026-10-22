import numpy as np
import pandas as pd
import pytest

from hmrec.ranker import group_sizes, load_training, top_k_lists


def make_frame():
    return pd.DataFrame(
        {
            "customer_idx": [0, 0, 0, 1, 1, 2, 2, 2, 2, 3],
            "article_id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        }
    )


def write_cut(directory, name, customer_ids):
    frame = pd.DataFrame(
        {"customer_idx": customer_ids, "article_id": range(len(customer_ids))}
    )
    frame.to_parquet(directory / f"{name}_features.parquet", index=False)


def test_group_sizes_counts_rows_per_customer():
    assert group_sizes(make_frame()).tolist() == [3, 2, 4, 1]


def test_group_sizes_rejects_unsorted_rows():
    with pytest.raises(ValueError):
        group_sizes(make_frame().iloc[::-1])


def test_group_sizes_accepts_a_custom_key():
    frame = make_frame().assign(group=[5, 5, 5, 7, 7, 9, 9, 9, 9, 12])
    assert group_sizes(frame, "group").tolist() == [3, 2, 4, 1]


def test_load_training_keeps_weeks_of_one_customer_apart(tmp_path):
    write_cut(tmp_path, "train_1", [0, 0, 1])
    write_cut(tmp_path, "train_2", [0, 1, 1, 1])
    result = load_training(tmp_path, ["train_1", "train_2"])
    assert result["cut"].tolist() == [0, 0, 0, 1, 1, 1, 1]
    assert result["customer_idx"].nunique() == 2
    assert result["group"].nunique() == 4
    assert group_sizes(result, "group").tolist() == [2, 1, 1, 3]


def test_stacked_weeks_cannot_be_grouped_by_customer(tmp_path):
    write_cut(tmp_path, "train_1", [0, 0, 1])
    write_cut(tmp_path, "train_2", [0, 1, 1, 1])
    result = load_training(tmp_path, ["train_1", "train_2"])
    with pytest.raises(ValueError):
        group_sizes(result)


def test_top_k_lists_orders_by_score_and_cuts_to_k():
    scores = np.array([0.1, 0.9, 0.5, 0.2, 0.3, 0.0, 0.4, 0.8, 0.6, 0.7])
    lists = top_k_lists(make_frame(), scores, k=2)
    assert lists[0] == [2, 3]
    assert lists[1] == [5, 4]
    assert lists[2] == [8, 9]
    assert lists[3] == [10]