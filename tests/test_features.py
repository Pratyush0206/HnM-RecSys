import numpy as np
import pandas as pd
import pytest

from hmrec import features
from hmrec.splits import LeakageError, Splitter, make_cuts

CUT = make_cuts("2020-09-22")["valid"]
COLUMNS = ["t_dat", "customer_idx", "article_id", "price", "sales_channel_id"]


def make_history():
    rows = [
        ("2020-08-25", 0, 10, 0.02, 1),
        ("2020-09-01", 0, 10, 0.04, 2),
        ("2020-09-05", 0, 20, 0.06, 2),
        ("2020-09-02", 1, 20, 0.10, 1),
        ("2020-09-06", 1, 30, 0.20, 1),
    ]
    frame = pd.DataFrame(rows, columns=COLUMNS)
    frame["t_dat"] = pd.to_datetime(frame["t_dat"])
    return frame


def make_table():
    return pd.DataFrame(
        {
            "customer_idx": [0, 0, 0],
            "article_id": [10, 20, 30],
            "position": [1, 2, 3],
            "rank_repurchase": [1.0, 2.0, np.nan],
            "rank_copurchase": [np.nan, 1.0, 1.0],
            "rank_twotower": [np.nan, np.nan, np.nan],
            "rank_popular": [np.nan, np.nan, np.nan],
        }
    )


def test_customer_features():
    row = features.customer_features(make_history(), CUT).loc[0]
    assert row["c_n_all"] == 3
    assert row["c_days_since_last"] == 4
    assert row["c_days_since_first"] == 15
    assert row["c_mean_price"] == pytest.approx(0.04)
    assert row["c_online_share"] == pytest.approx(2 / 3)
    assert (row["c_n_1w"], row["c_n_2w"], row["c_n_4w"]) == (1, 2, 3)


def test_item_and_pair_features():
    result = features.build_features(make_table(), make_history(), CUT)
    first = result.iloc[0]
    assert first["i_n_all"] == 2
    assert first["i_n_1w"] == 0
    assert first["i_age_days"] == 15
    assert first["i_mean_price"] == pytest.approx(0.03)
    assert first["p_times_bought"] == 2
    assert first["p_days_since_bought"] == 8
    assert first["p_price_ratio"] == pytest.approx(0.03 / 0.04)
    assert first["p_n_sources"] == 1

    never_bought = result.iloc[2]
    assert never_bought["p_times_bought"] == 0
    assert np.isnan(never_bought["p_days_since_bought"])


def test_features_reject_leaky_history():
    future = pd.DataFrame(
        [("2020-09-10", 0, 10, 0.05, 1)], columns=COLUMNS
    )
    future["t_dat"] = pd.to_datetime(future["t_dat"])
    leaky = pd.concat([make_history(), future], ignore_index=True)
    with pytest.raises(LeakageError):
        features.build_features(make_table(), leaky, CUT)


def frame_with_future(item):
    future = pd.DataFrame(
        {
            "t_dat": pd.to_datetime(["2020-09-10", "2020-09-20", "2020-09-22"]),
            "customer_idx": [0, 1, 0],
            "article_id": [item] * 3,
            "price": [0.5] * 3,
            "sales_channel_id": [1] * 3,
        }
    )
    return pd.concat([make_history(), future], ignore_index=True)


def test_label_week_changes_do_not_affect_features():
    results = []
    for item in (99, 77):
        splitter = Splitter.from_frame(frame_with_future(item))
        history = splitter.history("valid")
        results.append(features.build_features(make_table(), history, splitter.cuts["valid"]))
    pd.testing.assert_frame_equal(results[0], results[1])

def make_customers():
    return pd.DataFrame({"customer_idx": [0, 1], "c_age": [22.0, 45.0]})


def make_articles():
    return pd.DataFrame(
        {
            "article_id": [10, 20, 30],
            "i_type": [253, 253, 306],
            "i_group": [1, 1, 2],
            "i_dept": [5, 6, 7],
            "i_colour": [9, 9, 10],
            "i_index": [1, 1, 1],
            "i_garment": [1001, 1001, 1002],
        }
    )


def test_metadata_columns_absent_without_metadata():
    result = features.build_features(make_table(), make_history(), CUT)
    assert "c_age" not in result.columns
    assert "p_type_n" not in result.columns


def test_metadata_features():
    result = features.build_features(
        make_table(), make_history(), CUT, make_customers(), make_articles()
    )
    assert result["c_age"].tolist() == [22.0, 22.0, 22.0]
    assert result["i_type"].tolist() == [253, 253, 306]
    assert result["p_type_n"].tolist() == [3, 3, 0]
    assert result["p_type_share"].tolist() == pytest.approx([1.0, 1.0, 0.0])
    assert result["p_type_days_since"].iloc[:2].tolist() == [4, 4]
    assert np.isnan(result["p_type_days_since"].iloc[2])
    assert result["p_dept_n"].tolist() == [2, 1, 0]
    assert result["p_dept_share"].tolist() == pytest.approx([2 / 3, 1 / 3, 0.0])
    assert result["p_dept_days_since"].iloc[:2].tolist() == [8, 4]
    assert result["p_age_n_1w"].tolist() == [0, 1, 0]
    assert result["p_age_n_4w"].tolist() == [2, 1, 0]


def test_label_week_changes_do_not_affect_metadata_features():
    results = []
    for item in (99, 77):
        splitter = Splitter.from_frame(frame_with_future(item))
        history = splitter.history("valid")
        results.append(
            features.build_features(
                make_table(),
                history,
                splitter.cuts["valid"],
                make_customers(),
                make_articles(),
            )
        )
    pd.testing.assert_frame_equal(results[0], results[1])