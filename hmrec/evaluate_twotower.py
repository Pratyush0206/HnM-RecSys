import argparse
import json
from pathlib import Path

import numpy as np

from hmrec import candidates, config
from hmrec.ann import build_index, twotower_candidates
from hmrec.evaluate import load_transactions
from hmrec.evaluate_candidates import source_lists
from hmrec.metrics import map_at_k, recall_at_k
from hmrec.splits import Splitter, development_names
from hmrec.twotower import fit


def summarise(labels, lists):
    return {
        "recall": round(recall_at_k(labels, lists), 4),
        "map_at_12": round(map_at_k(labels, lists), 5),
    }


def evaluate_cut(splitter, cut_name, n_customers, args):
    cut = splitter.cuts[cut_name]
    history = splitter.history(cut_name)
    labels = splitter.labels(cut_name)
    customers = np.asarray(sorted(labels), dtype="int64")

    embeddings = fit(
        history,
        cut,
        n_customers,
        dim=args.dim,
        epochs=args.epochs,
        use_logq=not args.no_logq,
    )
    index = build_index(embeddings.item_vectors, args.index)
    tower = twotower_candidates(embeddings, customers, index, k=args.k)
    tower_lists = source_lists(tower)

    without = candidates.generate(history, cut, customers, args.max_per_customer)
    with_tower = candidates.generate(
        history, cut, customers, args.max_per_customer, extra_frames=[tower]
    )

    return {
        "customers_scored": len(customers),
        "customers_with_twotower": len(tower_lists),
        "twotower_alone": summarise(labels, tower_lists),
        "merged_without_twotower": summarise(labels, candidates.to_lists(without)),
        "merged_with_twotower": summarise(labels, candidates.to_lists(with_tower)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", type=Path, default=config.PROCESSED_DIR)
    parser.add_argument("--results-dir", type=Path, default=config.RESULTS_DIR)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--dim", type=int, default=32)
    parser.add_argument("--k", type=int, default=30)
    parser.add_argument("--max-per-customer", type=int, default=200)
    parser.add_argument("--index", choices=["flat", "ivf", "hnsw"], default="flat")
    parser.add_argument("--no-logq", action="store_true")
    args = parser.parse_args()

    transactions = load_transactions(args.processed_dir)
    splitter = Splitter.from_frame(transactions)
    n_customers = int(transactions["customer_idx"].max()) + 1

    report = {
        "data": str(args.processed_dir),
        "settings": {
            "epochs": args.epochs,
            "dim": args.dim,
            "k": args.k,
            "max_per_customer": args.max_per_customer,
            "index": args.index,
            "logq": not args.no_logq,
        },
    }
    for cut_name in development_names(splitter.cuts):
        report[cut_name] = evaluate_cut(splitter, cut_name, n_customers, args)

    args.results_dir.mkdir(parents=True, exist_ok=True)
    tag = "nologq" if args.no_logq else "logq"
    out_path = args.results_dir / f"twotower_{Path(args.processed_dir).name}_{tag}.json"
    out_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"saved to {out_path}")


if __name__ == "__main__":
    main()