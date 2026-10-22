import numpy as np
import pandas as pd
import pytest

from hmrec.ranker import group_sizes, load_training, top_k_lists
from hmrec.ranker import GROUPS, run_tag, select_columns
from hmrec.ranker import GROUPS, PARAMS, params_for, run_tag, select_columns


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

def test_select_columns_removes_whole_groups():
    kept = select_columns(["twotower"])
    assert "tt_score" not in kept
    assert "rank_twotower" not in kept
    assert "position" in kept


def test_select_columns_without_drop_keeps_everything():
    from hmrec.features import FEATURE_COLUMNS

    assert select_columns() == FEATURE_COLUMNS


def test_select_columns_rejects_unknown_group():
    with pytest.raises(ValueError):
        select_columns(["nonsense"])


def test_every_group_column_is_a_real_feature():
    from hmrec.features import FEATURE_COLUMNS

    for columns in GROUPS.values():
        assert set(columns) <= set(FEATURE_COLUMNS)


def test_run_tag_names_the_dropped_groups():
    assert run_tag([]) == "full"
    assert run_tag(["metadata", "pair"]) == "no_metadata_pair"

def test_params_for_binary_changes_objective_and_metric():
    params = params_for("binary")
    assert params["objective"] == "binary"
    assert params["metric"] == "binary_logloss"
    assert params["num_leaves"] == PARAMS["num_leaves"]


def test_params_for_does_not_modify_the_shared_defaults():
    params_for("binary")
    assert PARAMS["objective"] == "lambdarank"
    assert PARAMS["metric"] == "map"


def test_params_for_rejects_unknown_objective():
    with pytest.raises(ValueError):
        params_for("regression")


def test_run_tag_includes_the_objective_unless_lambdarank():
    assert run_tag([], "binary") == "binary_full"
    assert run_tag(["metadata"], "binary") == "binary_no_metadata"
    assert run_tag(["metadata"]) == "no_metadata"