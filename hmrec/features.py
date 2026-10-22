import argparse
import json
from pathlib import Path

import pandas as pd

from hmrec import config
from hmrec.evaluate import load_transactions
from hmrec.ranker_data import RANK_COLUMNS, RANKER_DIR
from hmrec.splits import Splitter, development_names, ensure_before_cut

WINDOWS = (1, 2, 4)
AGE_BINS = [0, 20, 25, 30, 40, 50, 60, 120]

CANDIDATE = ["position", *RANK_COLUMNS, "tt_score"]
CUSTOMER = [
    "c_n_all",
    "c_n_1w",
    "c_n_2w",
    "c_n_4w",
    "c_days_since_last",
    "c_days_since_first",
    "c_mean_price",
    "c_online_share",
]
ITEM = [
    "i_n_all",
    "i_n_1w",
    "i_n_2w",
    "i_n_4w",
    "i_age_days",
    "i_days_since_last",
    "i_mean_price",
]
PAIR = ["p_times_bought", "p_days_since_bought", "p_price_ratio", "p_n_sources"]
CUSTOMER_META = ["c_age"]
ITEM_META = ["i_type", "i_group", "i_dept", "i_colour", "i_index", "i_garment"]
PAIR_META = [
    "p_type_n",
    "p_type_share",
    "p_type_days_since",
    "p_dept_n",
    "p_dept_share",
    "p_dept_days_since",
    "p_age_n_1w",
    "p_age_n_4w",
]
BASE_COLUMNS = CANDIDATE + CUSTOMER + ITEM + PAIR
META_COLUMNS = CUSTOMER_META + ITEM_META + PAIR_META
FEATURE_COLUMNS = BASE_COLUMNS + META_COLUMNS


def window_counts(history, cut, key, prefix):
    columns = {}
    for weeks in WINDOWS:
        start = cut.history_end - pd.Timedelta(days=config.WEEK_DAYS * weeks)
        recent = history[history["t_dat"] >= start]
        columns[f"{prefix}_n_{weeks}w"] = recent.groupby(key).size()
    return pd.DataFrame(columns)


def customer_features(history, cut):
    ensure_before_cut(history, cut)
    marked = history.assign(online=(history["sales_channel_id"] == 2).astype("float32"))
    grouped = marked.groupby("customer_idx")
    features = pd.DataFrame(
        {
            "c_n_all": grouped.size(),
            "c_days_since_last": (cut.history_end - grouped["t_dat"].max()).dt.days,
            "c_days_since_first": (cut.history_end - grouped["t_dat"].min()).dt.days,
            "c_mean_price": grouped["price"].mean(),
            "c_online_share": grouped["online"].mean(),
        }
    )
    features = features.join(window_counts(history, cut, "customer_idx", "c"), how="left")
    counts = [name for name in features.columns if name.startswith("c_n_")]
    features[counts] = features[counts].fillna(0)
    return features


def item_features(history, cut):
    ensure_before_cut(history, cut)
    grouped = history.groupby("article_id")
    features = pd.DataFrame(
        {
            "i_n_all": grouped.size(),
            "i_age_days": (cut.history_end - grouped["t_dat"].min()).dt.days,
            "i_days_since_last": (cut.history_end - grouped["t_dat"].max()).dt.days,
            "i_mean_price": grouped["price"].mean(),
        }
    )
    features = features.join(window_counts(history, cut, "article_id", "i"), how="left")
    counts = [name for name in features.columns if name.startswith("i_n_")]
    features[counts] = features[counts].fillna(0)
    return features


def pair_features(history, cut):
    ensure_before_cut(history, cut)
    pairs = (
        history.groupby(["customer_idx", "article_id"])
        .agg(p_times_bought=("t_dat", "size"), last_bought=("t_dat", "max"))
        .reset_index()
    )
    pairs["p_days_since_bought"] = (cut.history_end - pairs["last_bought"]).dt.days
    return pairs.drop(columns="last_bought")


def age_bucket(age):
    buckets = pd.cut(age, bins=AGE_BINS, labels=False, right=False)
    return buckets.astype("float64")


def attach_metadata(history, customers, articles):
    tagged = history.merge(
        customers[["customer_idx", "c_age"]], on="customer_idx", how="left"
    )
    tagged = tagged.merge(
        articles[["article_id", "i_type", "i_dept"]], on="article_id", how="left"
    )
    tagged["age_bucket"] = age_bucket(tagged["c_age"])
    return tagged


def category_pair_features(tagged, cut, column, prefix):
    stats = (
        tagged.groupby(["customer_idx", column])
        .agg(count=("t_dat", "size"), last=("t_dat", "max"))
        .reset_index()
    )
    stats[f"{prefix}_n"] = stats["count"]
    stats[f"{prefix}_days_since"] = (cut.history_end - stats["last"]).dt.days
    return stats[["customer_idx", column, f"{prefix}_n", f"{prefix}_days_since"]]


def age_popularity(tagged, cut):
    columns = {}
    for weeks in (1, 4):
        start = cut.history_end - pd.Timedelta(days=config.WEEK_DAYS * weeks)
        recent = tagged[(tagged["t_dat"] >= start) & tagged["age_bucket"].notna()]
        columns[f"p_age_n_{weeks}w"] = recent.groupby(["age_bucket", "article_id"]).size()
    return pd.DataFrame(columns).fillna(0).reset_index()


def add_metadata(out, history, cut, customers, articles):
    out = out.merge(customers[["customer_idx", *CUSTOMER_META]], on="customer_idx", how="left")
    out = out.merge(articles[["article_id", *ITEM_META]], on="article_id", how="left")
    tagged = attach_metadata(history, customers, articles)
    has_history = out["c_n_all"].notna()

    for column, prefix in (("i_type", "p_type"), ("i_dept", "p_dept")):
        stats = category_pair_features(tagged, cut, column, prefix)
        out = out.merge(stats, on=["customer_idx", column], how="left")
        count = out[f"{prefix}_n"].fillna(0).where(has_history)
        out[f"{prefix}_n"] = count
        out[f"{prefix}_share"] = count / out["c_n_all"]

    out["age_bucket"] = age_bucket(out["c_age"])
    out = out.merge(age_popularity(tagged, cut), on=["age_bucket", "article_id"], how="left")
    known_age = out["age_bucket"].notna()
    for name in ("p_age_n_1w", "p_age_n_4w"):
        out[name] = out[name].fillna(0).where(known_age)
    return out.drop(columns="age_bucket")


def build_features(table, history, cut, customers=None, articles=None):
    ensure_before_cut(history, cut)
    out = table.merge(
        customer_features(history, cut), left_on="customer_idx", right_index=True, how="left"
    )
    out = out.merge(
        item_features(history, cut), left_on="article_id", right_index=True, how="left"
    )
    out = out.merge(pair_features(history, cut), on=["customer_idx", "article_id"], how="left")
    out["p_times_bought"] = out["p_times_bought"].fillna(0)
    out["p_price_ratio"] = out["i_mean_price"] / out["c_mean_price"]
    out["p_n_sources"] = out[RANK_COLUMNS].notna().sum(axis=1).astype("int8")

    if customers is not None and articles is not None:
        out = add_metadata(out, history, cut, customers, articles)

    for column in out.select_dtypes("float64").columns:
        out[column] = out[column].astype("float32")
    return out.sort_values(["customer_idx", "position"], kind="stable").reset_index(drop=True)


def load_metadata(processed_dir):
    processed = Path(processed_dir)
    customers = pd.read_parquet(
        processed / config.CUSTOMERS_PARQUET, columns=["customer_idx", *CUSTOMER_META]
    )
    articles = pd.read_parquet(processed / config.ARTICLES_PARQUET)
    return customers, articles


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", type=Path, default=config.PROCESSED_DIR)
    parser.add_argument("--ranker-dir", type=Path, default=None)
    parser.add_argument("--results-dir", type=Path, default=config.RESULTS_DIR)
    parser.add_argument("--no-metadata", action="store_true")
    args = parser.parse_args()

    ranker_dir = args.ranker_dir or RANKER_DIR / Path(args.processed_dir).name
    transactions = load_transactions(args.processed_dir)
    splitter = Splitter.from_frame(transactions)

    if args.no_metadata:
        customers, articles = None, None
        columns = BASE_COLUMNS
    else:
        customers, articles = load_metadata(args.processed_dir)
        columns = FEATURE_COLUMNS

    summary = {"features": columns}
    for cut_name in development_names(splitter.cuts):
        cut = splitter.cuts[cut_name]
        table = pd.read_parquet(ranker_dir / f"{cut_name}.parquet")
        featured = build_features(
            table, splitter.history(cut_name), cut, customers, articles
        )
        featured.to_parquet(ranker_dir / f"{cut_name}_features.parquet", index=False)

        missing = featured[columns].isna().mean()
        summary[cut_name] = {
            "rows": int(len(featured)),
            "missing_share": {
                name: round(float(share), 3) for name, share in missing.items() if share > 0
            },
        }

    args.results_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.results_dir / f"features_{Path(args.processed_dir).name}.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"saved to {out_path}")


if __name__ == "__main__":
    main()