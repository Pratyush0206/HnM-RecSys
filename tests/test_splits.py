import pandas as pd
import pytest

from hmrec.baselines import recent_popular
from hmrec.splits import (
    LeakageError,
    Splitter,
    development_names,
    ensure_before_cut,
    make_cuts,
)

ALL_CUTS = ["train_1", "train_2", "train_3", "valid", "test"]


def make_frame():
    dates = pd.date_range("2020-07-01", "2020-09-22", freq="D")
    return pd.DataFrame(
        {
            "t_dat": dates,
            "customer_idx": [i % 3 for i in range(len(dates))],
            "article_id": [100 + i % 5 for i in range(len(dates))],
        }
    )


def test_cut_names_and_order():
    cuts = make_cuts("2020-09-22")
    assert list(cuts) == ALL_CUTS


def test_cut_dates():
    cuts = make_cuts("2020-09-22")
    assert cuts["train_1"].label_start == pd.Timestamp("2020-08-19")
    assert cuts["train_3"].label_end == pd.Timestamp("2020-09-09")
    assert cuts["valid"].label_start == pd.Timestamp("2020-09-09")
    assert cuts["valid"].label_end == pd.Timestamp("2020-09-16")
    assert cuts["test"].label_start == pd.Timestamp("2020-09-16")
    assert cuts["test"].label_end == pd.Timestamp("2020-09-23")


def test_history_end_equals_label_start():
    cuts = make_cuts("2020-09-22")
    for cut in cuts.values():
        assert cut.history_end == cut.label_start


def test_label_weeks_are_seven_days_and_adjacent():
    cuts = make_cuts("2020-09-22")
    for cut in cuts.values():
        assert (cut.label_end - cut.label_start).days == 7
    ordered = list(cuts.values())
    for earlier, later in zip(ordered, ordered[1:]):
        assert earlier.label_end == later.label_start


def test_number_of_train_cuts_is_configurable():
    cuts = make_cuts("2020-09-22", n_train=2)
    assert list(cuts) == ["train_1", "train_2", "valid", "test"]
    assert cuts["test"].label_end == pd.Timestamp("2020-09-23")


def test_development_names_never_include_test():
    cuts = make_cuts("2020-09-22")
    assert development_names(cuts) == ["train_1", "train_2", "train_3", "valid"]
    assert "test" not in development_names(cuts)


@pytest.mark.parametrize("cut_name", ALL_CUTS)
def test_history_ends_before_cut(cut_name):
    splitter = Splitter.from_frame(make_frame())
    history = splitter.history(cut_name)
    assert history["t_dat"].max() < splitter.cuts[cut_name].history_end


@pytest.mark.parametrize("cut_name", ALL_CUTS)
def test_history_and_labels_do_not_overlap(cut_name):
    splitter = Splitter.from_frame(make_frame())
    history = splitter.history(cut_name)
    window = splitter.label_window(cut_name)
    assert history["t_dat"].max() < window["t_dat"].min()
    assert window["t_dat"].nunique() == 7


def test_later_cuts_see_more_history():
    splitter = Splitter.from_frame(make_frame())
    sizes = [len(splitter.history(name)) for name in ALL_CUTS]
    assert sizes == sorted(sizes)
    assert len(set(sizes)) == len(sizes)


def test_ensure_before_cut_raises_on_leak():
    cuts = make_cuts("2020-09-22")
    with pytest.raises(LeakageError):
        ensure_before_cut(make_frame(), cuts["train_3"])


def test_baselines_reject_leaky_history():
    cuts = make_cuts("2020-09-22")
    with pytest.raises(LeakageError):
        recent_popular(make_frame(), cuts["train_3"])


def test_labels_remove_repeat_purchases():
    frame = pd.DataFrame(
        {
            "t_dat": pd.to_datetime(["2020-09-01", "2020-09-20", "2020-09-21"]),
            "customer_idx": [0, 0, 0],
            "article_id": [5, 7, 7],
        }
    )
    splitter = Splitter.from_frame(frame)
    assert splitter.labels("test") == {0: [7]}
    assert splitter.labels("train_3") == {0: [5]}