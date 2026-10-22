import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from hmrec import config
from hmrec.bootstrap import paired_bootstrap, per_customer_scores
from hmrec.evaluate import load_transactions
from hmrec.features import FEATURE_COLUMNS, META_COLUMNS
from hmrec.metrics import map_at_k
from hmrec.ranker_data import RANKER_DIR
from hmrec.splits import Splitter, development_names

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

OBJECTIVES = ("lambdarank", "binary")

GROUPS = {
    "metadata": META_COLUMNS,
    "twotower": ["tt_score", "rank_twotower"],
    "pair": ["p_times_bought", "p_days_since_bought", "p_price_ratio", "p_n_sources"],
    "item_recent": ["i_n_1w", "i_n_2w", "i_n_4w"],
}


def params_for(objective):
    if objective not in OBJECTIVES:
        raise ValueError(f"unknown objective: {objective}")
    params = dict(PARAMS)
    if objective == "binary":
        params["objective"] = "binary"
        params["metric"] = "binary_logloss"
    return params


def select_columns(drop=()):
    unknown = [name for name in drop if name not in GROUPS]
    if unknown:
        raise ValueError(f"unknown feature groups: {unknown}")
    removed = {column for name in drop for column in GROUPS[name]}
    return [column for column in FEATURE_COLUMNS if column not in removed]


def run_tag(drop, objective="lambdarank"):
    name = "full" if not drop else "no_" + "_".join(drop)
    if objective == "lambdarank":
        return name
    return f"{objective}_{name}"


def load_training(ranker_dir, cut_names):
    frames = []
    for number, name in enumerate(cut_names):
        frame = pd.read_parquet(Path(ranker_dir) / f"{name}_features.parquet")
        frame["cut"] = number
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    stride = int(combined["customer_idx"].max()) + 1
    cut_part = combined["cut"].astype("int64") * stride
    combined["group"] = cut_part + combined["customer_idx"].astype("int64")
    return combined


def group_sizes(frame, key="customer_idx"):
    if not frame[key].is_monotonic_increasing:
        raise ValueError(f"rows must be sorted by {key}")
    return frame.groupby(key, sort=True).size().to_numpy()


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


def train(fit_frame, early_frame, seed, columns=None, objective="lambdarank"):
    columns = list(columns or FEATURE_COLUMNS)
    params = params_for(objective)
    callbacks = [lgb.early_stopping(50, verbose=False), lgb.log_evaluation(50)]
    if objective == "binary":
        model = lgb.LGBMClassifier(random_state=seed, **params)
        model.fit(
            fit_frame[columns],
            fit_frame["label"],
            eval_set=[(early_frame[columns], early_frame["label"])],
            callbacks=callbacks,
        )
        return model
    model = lgb.LGBMRanker(random_state=seed, **params)
    model.fit(
        fit_frame[columns],
        fit_frame["label"],
        group=group_sizes(fit_frame, "group"),
        eval_set=[(early_frame[columns], early_frame["label"])],
        eval_group=[group_sizes(early_frame)],
        eval_at=[config.K],
        callbacks=callbacks,
    )
    return model


def predict_scores(model, frame, columns, objective):
    if objective == "binary":
        return model.predict_proba(frame[columns])[:, 1]
    return model.predict(frame[columns])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", type=Path, default=config.PROCESSED_DIR)
    parser.add_argument("--ranker-dir", type=Path, default=None)
    parser.add_argument("--results-dir", type=Path, default=config.RESULTS_DIR)
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument("--drop", nargs="*", default=[], choices=sorted(GROUPS))
    parser.add_argument("--objective", default="lambdarank", choices=OBJECTIVES)
    args = parser.parse_args()

    columns = select_columns(args.drop)
    tag = run_tag(args.drop, args.objective)
    data_name = Path(args.processed_dir).name

    ranker_dir = args.ranker_dir or RANKER_DIR / data_name
    splitter = Splitter.from_frame(load_transactions(args.processed_dir))
    train_names = [
        name for name in development_names(splitter.cuts) if name.startswith("train")
    ]

    fit_frame = load_training(ranker_dir, train_names)
    valid = pd.read_parquet(ranker_dir / "valid_features.parquet")
    model = train(fit_frame, valid, args.seed, columns, args.objective)

    labels = splitter.labels("valid")
    scores = predict_scores(model, valid, columns, args.objective)
    ranker_lists = top_k_lists(valid, scores)
    order_lists = top_k_lists(valid, -valid["position"].to_numpy())

    ranker_scores = per_customer_scores(labels, ranker_lists)
    order_scores = per_customer_scores(labels, order_lists)
    bootstrap = paired_bootstrap(ranker_scores, order_scores)

    gain = pd.Series(
        model.booster_.feature_importance(importance_type="gain"), index=columns
    )
    share = (gain / gain.sum()).sort_values(ascending=False).round(4)

    report = {
        "run": tag,
        "objective": args.objective,
        "dropped_groups": args.drop,
        "n_features": len(columns),
        "train_cuts": train_names,
        "train_groups": int(fit_frame["group"].nunique()),
        "train_positives": int(fit_frame["label"].sum()),
        "early_stopping_on": "valid",
        "best_iteration": int(model.best_iteration_),
        "valid": {
            "customers_scored": len(labels),
            "map_at_12_ranker": round(map_at_k(labels, ranker_lists), 5),
            "map_at_12_candidate_order": round(map_at_k(labels, order_lists), 5),
            "bootstrap_ranker_minus_candidate_order": {
                name: round(value, 5) for name, value in bootstrap.items()
            },
        },
        "feature_importance_gain_share": {
            name: float(value) for name, value in share.head(15).items()
        },
    }

    args.results_dir.mkdir(parents=True, exist_ok=True)
    suffix = "" if tag == "full" else f"_{tag}"
    model_name = "ranker.txt" if tag == "full" else f"ranker_{tag}.txt"
    model.booster_.save_model(str(ranker_dir / model_name))
    np.savez(
        args.results_dir / f"scores_{data_name}_{tag}.npz",
        customers=np.asarray(sorted(labels), dtype="int64"),
        ranker=ranker_scores,
        order=order_scores,
    )
    out_path = args.results_dir / f"ranker_{data_name}{suffix}.json"
    out_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"saved to {out_path}")


if __name__ == "__main__":
    main()