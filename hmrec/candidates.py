import numpy as np
import pandas as pd

from hmrec import config
from hmrec.baselines import recent_popular
from hmrec.splits import ensure_before_cut

SOURCE_ORDER = {"repurchase": 0, "copurchase": 1, "twotower": 2, "popular": 3}
COLUMNS = ["customer_idx", "article_id", "source", "source_rank"]


def recent_window(history, cut, weeks, week_days=config.WEEK_DAYS):
    ensure_before_cut(history, cut)
    start = cut.history_end - pd.Timedelta(days=week_days * weeks)
    return history[history["t_dat"] >= start]


def rank_within_customer(frame, by, max_per_customer, source):
    ranked = frame.sort_values(
        ["customer_idx", *by, "article_id"],
        ascending=[True, *([False] * len(by)), True],
        kind="stable",
    )
    ranked["source_rank"] = ranked.groupby("customer_idx").cumcount() + 1
    ranked = ranked[ranked["source_rank"] <= max_per_customer].copy()
    ranked["source"] = source
    return ranked[COLUMNS].reset_index(drop=True)


def latest_items(frame, max_per_customer):
    latest = frame.sort_values("t_dat", ascending=False, kind="stable")
    latest = latest.drop_duplicates(["customer_idx", "article_id"])
    keep = latest.groupby("customer_idx").cumcount() < max_per_customer
    return latest[keep][["customer_idx", "article_id"]]


def repurchase(history, cut, customers, weeks=4, max_per_customer=20):
    recent = recent_window(history, cut, weeks)
    recent = recent[recent["customer_idx"].isin(customers)]
    stats = (
        recent.groupby(["customer_idx", "article_id"])
        .agg(last_day=("t_dat", "max"), count=("t_dat", "size"))
        .reset_index()
    )
    return rank_within_customer(stats, ["last_day", "count"], max_per_customer, "repurchase")


def item_neighbours(recent, per_item=10, max_items_per_customer=30):
    baskets = latest_items(recent, max_items_per_customer)
    pairs = baskets.merge(baskets, on="customer_idx", suffixes=("", "_other"))
    pairs = pairs[pairs["article_id"] != pairs["article_id_other"]]
    counts = pairs.groupby(["article_id", "article_id_other"]).size().reset_index(name="pair_count")
    counts = counts.sort_values(
        ["article_id", "pair_count", "article_id_other"],
        ascending=[True, False, True],
        kind="stable",
    )
    return counts[counts.groupby("article_id").cumcount() < per_item]


def copurchase(history, cut, customers, weeks=4, seed_items=10, per_item=10, max_per_customer=40):
    recent = recent_window(history, cut, weeks)
    neighbours = item_neighbours(recent, per_item)
    seeds = latest_items(recent[recent["customer_idx"].isin(customers)], seed_items)
    joined = seeds.merge(neighbours, on="article_id")
    scores = joined.groupby(["customer_idx", "article_id_other"])["pair_count"].sum().reset_index()
    scores = scores.rename(columns={"article_id_other": "article_id", "pair_count": "score"})
    return rank_within_customer(scores, ["score"], max_per_customer, "copurchase")


def popular(history, cut, customers, weeks=1, n=50):
    items = recent_popular(history, cut, weeks=weeks, k=n)
    return pd.DataFrame(
        {
            "customer_idx": np.repeat(customers, len(items)),
            "article_id": np.tile(items, len(customers)),
            "source": "popular",
            "source_rank": np.tile(np.arange(1, len(items) + 1), len(customers)),
        }
    )[COLUMNS]


def merge_candidates(frames, max_per_customer=50):
    long = pd.concat(frames, ignore_index=True)
    long["source_order"] = long["source"].map(SOURCE_ORDER)
    long = long.sort_values(["customer_idx", "source_order", "source_rank"], kind="stable")

    ranks = long.pivot_table(
        index=["customer_idx", "article_id"],
        columns="source",
        values="source_rank",
        aggfunc="min",
    )
    ranks = ranks.reindex(columns=list(SOURCE_ORDER))
    ranks.columns = [f"rank_{name}" for name in ranks.columns]

    merged = long.drop_duplicates(["customer_idx", "article_id"])[["customer_idx", "article_id"]].copy()
    merged["position"] = merged.groupby("customer_idx").cumcount() + 1
    merged = merged[merged["position"] <= max_per_customer]
    merged = merged.merge(ranks.reset_index(), on=["customer_idx", "article_id"], how="left")
    return merged.reset_index(drop=True)


def generate(history, cut, customers, max_per_customer=50, extra_frames=()):
    customers = np.asarray(sorted(customers), dtype="int64")
    frames = [
        repurchase(history, cut, customers),
        copurchase(history, cut, customers),
        popular(history, cut, customers, n=max_per_customer),
        *extra_frames,
    ]
    return merge_candidates(frames, max_per_customer)


def to_lists(candidates):
    ordered = candidates.sort_values(["customer_idx", "position"], kind="stable")
    grouped = ordered.groupby("customer_idx")["article_id"].agg(list)
    return {int(customer): [int(a) for a in items] for customer, items in grouped.items()}