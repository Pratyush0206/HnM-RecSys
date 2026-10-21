from dataclasses import dataclass

import pandas as pd

from hmrec import config


class LeakageError(Exception):
    pass


@dataclass(frozen=True)
class Cut:
    name: str
    history_end: pd.Timestamp
    label_start: pd.Timestamp
    label_end: pd.Timestamp


def make_cuts(last_day, week_days=config.WEEK_DAYS):
    last_day = pd.Timestamp(last_day).normalize()
    valid_start = last_day - pd.Timedelta(days=week_days - 1)
    train_start = valid_start - pd.Timedelta(days=week_days)
    return {
        "train": Cut(
            name="train",
            history_end=train_start,
            label_start=train_start,
            label_end=valid_start,
        ),
        "valid": Cut(
            name="valid",
            history_end=valid_start,
            label_start=valid_start,
            label_end=last_day + pd.Timedelta(days=1),
        ),
    }


def ensure_before_cut(frame, cut, column="t_dat"):
    if len(frame) and frame[column].max() >= cut.history_end:
        raise LeakageError(
            f"cut '{cut.name}': found {column} {frame[column].max()} "
            f"on or after history end {cut.history_end}"
        )
    return frame


class Splitter:
    def __init__(self, transactions, cuts):
        self.transactions = transactions
        self.cuts = cuts

    @classmethod
    def from_frame(cls, transactions, week_days=config.WEEK_DAYS):
        cuts = make_cuts(transactions["t_dat"].max(), week_days)
        return cls(transactions, cuts)

    def history(self, cut_name):
        cut = self.cuts[cut_name]
        past = self.transactions[self.transactions["t_dat"] < cut.history_end]
        return ensure_before_cut(past, cut)

    def label_window(self, cut_name):
        cut = self.cuts[cut_name]
        dates = self.transactions["t_dat"]
        in_window = (dates >= cut.label_start) & (dates < cut.label_end)
        return self.transactions[in_window]

    def labels(self, cut_name):
        window = self.label_window(cut_name)
        pairs = window[["customer_idx", "article_id"]].drop_duplicates()
        grouped = pairs.groupby("customer_idx")["article_id"].agg(list)
        return {
            int(customer): [int(article) for article in articles]
            for customer, articles in grouped.items()
        }