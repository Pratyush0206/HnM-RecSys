import pandas as pd

from hmrec import config
from hmrec.splits import ensure_before_cut


def top_items(transactions, k=config.K):
    counts = transactions.groupby("article_id").size()
    ranked = counts.reset_index(name="count").sort_values(
        ["count", "article_id"], ascending=[False, True]
    )
    return [int(article) for article in ranked["article_id"].head(k)]


def global_popular(history, cut, k=config.K):
    ensure_before_cut(history, cut)
    return top_items(history, k)


def recent_popular(history, cut, weeks=1, k=config.K, week_days=config.WEEK_DAYS):
    ensure_before_cut(history, cut)
    window_start = cut.history_end - pd.Timedelta(days=week_days * weeks)
    recent = history[history["t_dat"] >= window_start]
    return top_items(recent, k)


def same_for_everyone(customers, items):
    return {int(customer): list(items) for customer in customers}