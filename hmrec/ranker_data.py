import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from hmrec import candidates, config
from hmrec.ann import build_index, twotower_candidates
from hmrec.evaluate import load_transactions
from hmrec.metrics import recall_at_k
from hmrec.splits import Splitter
from hmrec.twotower import fit

RANKER_DIR = Path("data") / "ranker"
RANK_COLUMNS = ["rank_repurchase", "rank_copurchase", "rank_twotower", "rank_popular"]


def label_candidates(table, labels):
    bought = pd.DataFrame(
        [(customer, article) for customer, articles in labels.items() for article in articles],
        columns=["customer_idx", "article_id"],
    ).drop_duplicates()
    bought["label"] = 1
    labelled = table.merge(bought, on=["customer_idx", "article_id"], how="left")
    labelled["label"] = labelled["label"].fillna(0).astype("int8")
    return labelled


def add_twotower_score(table, embeddings):
    table = table.copy()
    row_of = pd.Series(np.arange(len(embeddings.article_ids)), index=embeddings.article_ids)
    item_row = table["article_id"].map(row_of)
    customers = table["customer_idx"].to_numpy()
    known = item_row.notna().to_numpy() & embeddings.has_history[customers]
    rows = np.flatnonzero(known)

    scores = np.full(len(table), np.nan, dtype="float32")
    customer_vecs = embeddings.customer_vectors[customers[rows]]
    item_vecs = embeddings.item_vectors[item_row.to_numpy()[rows].astype("int64")]
    scores[rows] = (customer_vecs * item_vecs).sum(axis=1)
    table["tt_score"] = scores
    return table


def keep_trainable(table):
    positives = table.groupby("customer_idx")["label"].transform("max")
    return table[positives > 0].reset_index(drop=True)


def build_cut_table(splitter, cut_name, n_customers, args):
    cut = splitter.cuts[cut_name]
    history = splitter.history(cut_name)
    labels = splitter.labels(cut_name)
    customers = np.asarray(sorted(labels), dtype="int64")

    embeddings = fit(history, cut, n_customers, dim=args.dim, epochs=args.epochs)
    index = build_index(embeddings.item_vectors, "flat")
    tower = twotower_candidates(embeddings, customers, index, k=args.k)

    table = candidates.generate(
        history, cut, customers, args.max_per_customer, extra_frames=[tower]
    )
    table = table.sort_values(["customer_idx", "position"], kind="stable").reset_index(drop=True)
    table = label_candidates(table, labels)
    table = add_twotower_score(table, embeddings)
    for column in RANK_COLUMNS:
        table[column] = table[column].astype("float32")
    return table, labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", type=Path, default=config.PROCESSED_DIR)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--results-dir", type=Path, default=config.RESULTS_DIR)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--dim", type=int, default=32)
    parser.add_argument("--k", type=int, default=30)
    parser.add_argument("--max-per-customer", type=int, default=200)
    args = parser.parse_args()

    out_dir = args.out_dir or RANKER_DIR / Path(args.processed_dir).name
    out_dir.mkdir(parents=True, exist_ok=True)

    transactions = load_transactions(args.processed_dir)
    splitter = Splitter.from_frame(transactions)
    n_customers = int(transactions["customer_idx"].max()) + 1

    summary = {
        "settings": {
            "epochs": args.epochs,
            "dim": args.dim,
            "k": args.k,
            "max_per_customer": args.max_per_customer,
        }
    }
    for cut_name in ("train", "valid"):
        table, labels = build_cut_table(splitter, cut_name, n_customers, args)
        recall = recall_at_k(labels, candidates.to_lists(table))
        if cut_name == "train":
            table = keep_trainable(table)
        table.to_parquet(out_dir / f"{cut_name}.parquet", index=False)
        summary[cut_name] = {
            "customers_scored": len(labels),
            "customers_kept": int(table["customer_idx"].nunique()),
            "rows": int(len(table)),
            "positives": int(table["label"].sum()),
            "candidate_recall": round(recall, 4),
        }

    args.results_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.results_dir / f"ranker_data_{Path(args.processed_dir).name}.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"tables in {out_dir}, summary in {out_path}")


if __name__ == "__main__":
    main()