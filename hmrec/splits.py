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


def make_cuts(last_day, week_days=config.WEEK_DAYS, n_train=config.N_TRAIN_CUTS):
    last_day = pd.Timestamp(last_day).normalize()
    names = [f"train_{i}" for i in range(1, n_train + 1)] + ["valid", "test"]
    first_start = last_day - pd.Timedelta(days=week_days * len(names) - 1)
    cuts = {}
    for position, name in enumerate(names):
        start = first_start + pd.Timedelta(days=week_days * position)
        cuts[name] = Cut(
            name=name,
            history_end=start,
            label_start=start,
            label_end=start + pd.Timedelta(days=week_days),
        )
    return cuts


def development_names(cuts):
    return [name for name in cuts if name != "test"]


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