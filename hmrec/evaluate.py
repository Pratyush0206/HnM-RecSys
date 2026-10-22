import argparse
import json
from pathlib import Path

import pandas as pd

from hmrec import baselines, config
from hmrec.metrics import map_at_k
from hmrec.splits import Splitter, development_names


def load_transactions(processed_dir):
    return pd.read_parquet(Path(processed_dir) / config.TRANSACTIONS_PARQUET)


def score_baselines(splitter, cut_name, recent_weeks=(1, 2)):
    cut = splitter.cuts[cut_name]
    history = splitter.history(cut_name)
    labels = splitter.labels(cut_name)
    customers = list(labels)

    scores = {}
    items = baselines.global_popular(history, cut)
    predictions = baselines.same_for_everyone(customers, items)
    scores["global_popular"] = round(map_at_k(labels, predictions), 5)

    for weeks in recent_weeks:
        items = baselines.recent_popular(history, cut, weeks=weeks)
        predictions = baselines.same_for_everyone(customers, items)
        scores[f"recent_popular_{weeks}w"] = round(map_at_k(labels, predictions), 5)

    return scores, len(customers)


def run(processed_dir, results_dir):
    transactions = load_transactions(processed_dir)
    splitter = Splitter.from_frame(transactions)
    names = development_names(splitter.cuts)
    train_names = [name for name in names if name.startswith("train")]

    report = {"data": str(processed_dir)}
    for cut_name in names:
        cut = splitter.cuts[cut_name]
        scores, n_customers = score_baselines(splitter, cut_name)
        report[cut_name] = {
            "label_week": f"{cut.label_start.date()} to "
            f"{(cut.label_end - pd.Timedelta(days=1)).date()}",
            "customers_scored": n_customers,
            "map_at_12": scores,
        }

    recent_names = [
        name for name in report[train_names[0]]["map_at_12"] if name.startswith("recent")
    ]
    train_mean = {
        name: sum(report[cut]["map_at_12"][name] for cut in train_names) / len(train_names)
        for name in recent_names
    }
    chosen = max(train_mean, key=train_mean.get)
    report["train_mean_map_at_12"] = {name: round(value, 5) for name, value in train_mean.items()}
    report["chosen_on_train"] = chosen
    report["valid_map_at_12_of_chosen"] = report["valid"]["map_at_12"][chosen]

    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / f"baselines_{Path(processed_dir).name}.json"
    out_path.write_text(json.dumps(report, indent=2))
    return report, out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", type=Path, default=config.PROCESSED_DIR)
    parser.add_argument("--results-dir", type=Path, default=config.RESULTS_DIR)
    args = parser.parse_args()

    report, out_path = run(args.processed_dir, args.results_dir)
    print(json.dumps(report, indent=2))
    print(f"saved to {out_path}")


if __name__ == "__main__":
    main()