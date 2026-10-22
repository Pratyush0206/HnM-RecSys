import argparse
import hashlib
import json
from argparse import Namespace
from pathlib import Path

import lightgbm as lgb

from hmrec import candidates, config
from hmrec.bootstrap import paired_bootstrap, per_customer_scores
from hmrec.evaluate import load_transactions
from hmrec.features import FEATURE_COLUMNS, build_features, load_metadata
from hmrec.metrics import map_at_k, recall_at_k
from hmrec.ranker import top_k_lists
from hmrec.ranker_data import RANKER_DIR, build_cut_table
from hmrec.splits import Splitter


def file_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def refuse_if_already_scored(out_path):
    if Path(out_path).exists():
        raise SystemExit(
            f"{out_path} already exists: the test week has already been scored. "
            "Delete the file only if you accept that the number was seen before."
        )


def load_settings(results_dir, data_name):
    path = Path(results_dir) / f"ranker_data_{data_name}.json"
    return json.loads(path.read_text())["settings"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", type=Path, default=config.PROCESSED_DIR)
    parser.add_argument("--ranker-dir", type=Path, default=None)
    parser.add_argument("--results-dir", type=Path, default=config.RESULTS_DIR)
    args = parser.parse_args()

    data_name = Path(args.processed_dir).name
    out_path = args.results_dir / f"final_test_{data_name}.json"
    refuse_if_already_scored(out_path)

    ranker_dir = args.ranker_dir or RANKER_DIR / data_name
    model_path = ranker_dir / "ranker.txt"
    settings = load_settings(args.results_dir, data_name)

    transactions = load_transactions(args.processed_dir)
    splitter = Splitter.from_frame(transactions)
    n_customers = int(transactions["customer_idx"].max()) + 1
    cut = splitter.cuts["test"]
    history = splitter.history("test")

    table, labels = build_cut_table(splitter, "test", n_customers, Namespace(**settings))
    recall = recall_at_k(labels, candidates.to_lists(table))

    customers, articles = load_metadata(args.processed_dir)
    featured = build_features(table, history, cut, customers, articles)

    model = lgb.Booster(model_file=str(model_path))
    scores = model.predict(featured[FEATURE_COLUMNS])
    ranker_lists = top_k_lists(featured, scores)
    order_lists = top_k_lists(featured, -featured["position"].to_numpy())

    ranker_scores = per_customer_scores(labels, ranker_lists)
    order_scores = per_customer_scores(labels, order_lists)
    bootstrap = paired_bootstrap(ranker_scores, order_scores)

    report = {
        "data": data_name,
        "model_sha256": file_digest(model_path),
        "settings": settings,
        "label_start": str(cut.label_start),
        "label_end_exclusive": str(cut.label_end),
        "customers_scored": len(labels),
        "candidate_recall": round(recall, 4),
        "map_at_12_ranker": round(map_at_k(labels, ranker_lists), 5),
        "map_at_12_candidate_order": round(map_at_k(labels, order_lists), 5),
        "bootstrap_ranker_minus_candidate_order": {
            name: round(value, 5) for name, value in bootstrap.items()
        },
    }

    args.results_dir.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"saved to {out_path}")


if __name__ == "__main__":
    main()