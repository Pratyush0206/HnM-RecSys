import pandas as pd
import pytest

from hmrec.baselines import recent_popular
from hmrec.splits import LeakageError, Splitter, ensure_before_cut, make_cuts


def make_frame():
    dates = pd.date_range("2020-08-20", "2020-09-22", freq="D")
    return pd.DataFrame(
        {
            "t_dat": dates,
            "customer_idx": [i % 3 for i in range(len(dates))],
            "article_id": [100 + i % 5 for i in range(len(dates))],
        }
    )


def test_cut_dates():
    cuts = make_cuts("2020-09-22")
    assert cuts["train"].history_end == pd.Timestamp("2020-09-09")
    assert cuts["train"].label_end == pd.Timestamp("2020-09-16")
    assert cuts["valid"].history_end == pd.Timestamp("2020-09-16")
    assert cuts["valid"].label_end == pd.Timestamp("2020-09-23")


def test_label_weeks_are_seven_days_and_adjacent():
    cuts = make_cuts("2020-09-22")
    for cut in cuts.values():
        assert (cut.label_end - cut.label_start).days == 7
    assert cuts["train"].label_end == cuts["valid"].label_start


@pytest.mark.parametrize("cut_name", ["train", "valid"])
def test_history_ends_before_cut(cut_name):
    splitter = Splitter.from_frame(make_frame())
    history = splitter.history(cut_name)
    assert history["t_dat"].max() < splitter.cuts[cut_name].history_end


@pytest.mark.parametrize("cut_name", ["train", "valid"])
def test_history_and_labels_do_not_overlap(cut_name):
    splitter = Splitter.from_frame(make_frame())
    history = splitter.history(cut_name)
    window = splitter.label_window(cut_name)
    assert history["t_dat"].max() < window["t_dat"].min()
    assert window["t_dat"].nunique() == 7


def test_ensure_before_cut_raises_on_leak():
    cuts = make_cuts("2020-09-22")
    with pytest.raises(LeakageError):
        ensure_before_cut(make_frame(), cuts["train"])


def test_baselines_reject_leaky_history():
    cuts = make_cuts("2020-09-22")
    with pytest.raises(LeakageError):
        recent_popular(make_frame(), cuts["train"])


def test_labels_remove_repeat_purchases():
    frame = pd.DataFrame(
        {
            "t_dat": pd.to_datetime(["2020-09-01", "2020-09-20", "2020-09-21"]),
            "customer_idx": [0, 0, 0],
            "article_id": [5, 7, 7],
        }
    )
    splitter = Splitter.from_frame(frame)
    assert splitter.labels("valid") == {0: [7]}