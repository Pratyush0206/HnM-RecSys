import argparse
import json
from pathlib import Path

import pandas as pd

from hmrec import config
from hmrec.evaluate import load_transactions
from hmrec.ranker_data import RANK_COLUMNS, RANKER_DIR
from hmrec.splits import Splitter, ensure_before_cut

WINDOWS = (1, 2, 4)

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
FEATURE_COLUMNS = CANDIDATE + CUSTOMER + ITEM + PAIR


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


def build_features(table, history, cut):
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

    for column in out.select_dtypes("float64").columns:
        out[column] = out[column].astype("float32")
    return out.sort_values(["customer_idx", "position"], kind="stable").reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", type=Path, default=config.PROCESSED_DIR)
    parser.add_argument("--ranker-dir", type=Path, default=None)
    parser.add_argument("--results-dir", type=Path, default=config.RESULTS_DIR)
    args = parser.parse_args()

    ranker_dir = args.ranker_dir or RANKER_DIR / Path(args.processed_dir).name
    transactions = load_transactions(args.processed_dir)
    splitter = Splitter.from_frame(transactions)

    summary = {"features": FEATURE_COLUMNS}
    for cut_name in ("train", "valid"):
        cut = splitter.cuts[cut_name]
        table = pd.read_parquet(ranker_dir / f"{cut_name}.parquet")
        featured = build_features(table, splitter.history(cut_name), cut)
        featured.to_parquet(ranker_dir / f"{cut_name}_features.parquet", index=False)

        missing = featured[FEATURE_COLUMNS].isna().mean()
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