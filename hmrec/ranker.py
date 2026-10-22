import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from hmrec import config
from hmrec.evaluate import load_transactions
from hmrec.features import FEATURE_COLUMNS
from hmrec.metrics import map_at_k
from hmrec.ranker_data import RANKER_DIR
from hmrec.splits import Splitter

PARAMS = {
    "objective": "lambdarank",
    "metric": "map",
    "n_estimators": 1000,
    "learning_rate": 0.03,
    "num_leaves": 15,
    "min_child_samples": 50,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "reg_lambda": 5.0,
    "verbose": -1,
}


def split_customers(frame, holdout_share, seed):
    customers = np.sort(frame["customer_idx"].unique())
    rng = np.random.default_rng(seed)
    holdout = rng.choice(customers, size=int(len(customers) * holdout_share), replace=False)
    in_holdout = frame["customer_idx"].isin(holdout)
    return frame[~in_holdout], frame[in_holdout]


def group_sizes(frame):
    if not frame["customer_idx"].is_monotonic_increasing:
        raise ValueError("rows must be sorted by customer_idx")
    return frame.groupby("customer_idx", sort=True).size().to_numpy()


def top_k_lists(frame, scores, k=config.K):
    ranked = frame[["customer_idx", "article_id"]].assign(score=scores)
    ranked = ranked.sort_values(
        ["customer_idx", "score"], ascending=[True, False], kind="stable"
    )
    top = ranked.groupby("customer_idx").head(k)
    grouped = top.groupby("customer_idx")["article_id"].agg(list)
    return {
        int(customer): [int(article) for article in articles]
        for customer, articles in grouped.items()
    }


def train(fit_frame, early_frame, seed):
    model = lgb.LGBMRanker(random_state=seed, **PARAMS)
    model.fit(
        fit_frame[FEATURE_COLUMNS],
        fit_frame["label"],
        group=group_sizes(fit_frame),
        eval_set=[(early_frame[FEATURE_COLUMNS], early_frame["label"])],
        eval_group=[group_sizes(early_frame)],
        eval_at=[config.K],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(50)],
    )
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", type=Path, default=config.PROCESSED_DIR)
    parser.add_argument("--ranker-dir", type=Path, default=None)
    parser.add_argument("--results-dir", type=Path, default=config.RESULTS_DIR)
    parser.add_argument("--holdout", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=config.SEED)
    args = parser.parse_args()

    ranker_dir = args.ranker_dir or RANKER_DIR / Path(args.processed_dir).name
    train_all = pd.read_parquet(ranker_dir / "train_features.parquet")
    valid = pd.read_parquet(ranker_dir / "valid_features.parquet")

    fit_frame, early_frame = split_customers(train_all, args.holdout, args.seed)
    model = train(fit_frame, early_frame, args.seed)

    transactions = load_transactions(args.processed_dir)
    labels = Splitter.from_frame(transactions).labels("valid")

    ranker_lists = top_k_lists(valid, model.predict(valid[FEATURE_COLUMNS]))
    order_lists = top_k_lists(valid, -valid["position"].to_numpy())

    gain = pd.Series(
        model.booster_.feature_importance(importance_type="gain"), index=FEATURE_COLUMNS
    )
    share = (gain / gain.sum()).sort_values(ascending=False).round(4)

    report = {
        "train_customers_fit": int(fit_frame["customer_idx"].nunique()),
        "train_customers_early_stopping": int(early_frame["customer_idx"].nunique()),
        "best_iteration": int(model.best_iteration_),
        "valid": {
            "customers_scored": len(labels),
            "map_at_12_ranker": round(map_at_k(labels, ranker_lists), 5),
            "map_at_12_candidate_order": round(map_at_k(labels, order_lists), 5),
        },
        "feature_importance_gain_share": {
            name: float(value) for name, value in share.head(15).items()
        },
    }

    args.results_dir.mkdir(parents=True, exist_ok=True)
    model.booster_.save_model(str(ranker_dir / "ranker.txt"))
    out_path = args.results_dir / f"ranker_{Path(args.processed_dir).name}.json"
    out_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"saved to {out_path}")


if __name__ == "__main__":
    main()