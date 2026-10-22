import argparse
import json
import time
from pathlib import Path

import numpy as np

from hmrec import candidates, config
from hmrec.evaluate import load_transactions
from hmrec.metrics import map_at_k, recall_at_k
from hmrec.splits import Splitter, development_names

SOURCES = {
    "repurchase": candidates.repurchase,
    "copurchase": candidates.copurchase,
    "popular": candidates.popular,
}


def source_lists(frame):
    ordered = frame.sort_values(["customer_idx", "source_rank"], kind="stable")
    grouped = ordered.groupby("customer_idx")["article_id"].agg(list)
    return {int(customer): [int(a) for a in items] for customer, items in grouped.items()}


def average_size(frame):
    if len(frame) == 0:
        return 0.0
    return round(float(frame.groupby("customer_idx").size().mean()), 1)


def evaluate_cut(splitter, cut_name, max_per_customer):
    cut = splitter.cuts[cut_name]
    history = splitter.history(cut_name)
    labels = splitter.labels(cut_name)
    customers = np.asarray(sorted(labels), dtype="int64")

    report = {"customers_scored": len(customers), "sources": {}}
    for name, build in SOURCES.items():
        frame = build(history, cut, customers)
        lists = source_lists(frame)
        report["sources"][name] = {
            "customers_covered": len(lists),
            "avg_candidates": average_size(frame),
            "recall": round(recall_at_k(labels, lists), 4),
        }

    start = time.perf_counter()
    merged = candidates.generate(history, cut, customers, max_per_customer)
    elapsed = time.perf_counter() - start
    merged_lists = candidates.to_lists(merged)
    report["merged"] = {
        "max_per_customer": max_per_customer,
        "avg_candidates": average_size(merged),
        "recall": round(recall_at_k(labels, merged_lists), 4),
        "map_at_12_of_candidate_order": round(map_at_k(labels, merged_lists), 5),
        "seconds": round(elapsed, 2),
    }
    return report


def run(processed_dir, results_dir, max_per_customer):
    transactions = load_transactions(processed_dir)
    splitter = Splitter.from_frame(transactions)

    report = {"data": str(processed_dir)}
    for cut_name in development_names(splitter.cuts):
        report[cut_name] = evaluate_cut(splitter, cut_name, max_per_customer)

    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / f"candidates_{Path(processed_dir).name}.json"
    out_path.write_text(json.dumps(report, indent=2))
    return report, out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", type=Path, default=config.PROCESSED_DIR)
    parser.add_argument("--results-dir", type=Path, default=config.RESULTS_DIR)
    parser.add_argument("--max-per-customer", type=int, default=50)
    args = parser.parse_args()

    report, out_path = run(args.processed_dir, args.results_dir, args.max_per_customer)
    print(json.dumps(report, indent=2))
    print(f"saved to {out_path}")


if __name__ == "__main__":
    main()