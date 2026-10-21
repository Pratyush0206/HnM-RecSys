import argparse
import time
from pathlib import Path

import faiss
import numpy as np
import pandas as pd

from hmrec import config
from hmrec.candidates import COLUMNS
from hmrec.evaluate import load_transactions
from hmrec.splits import Splitter
from hmrec.twotower import fit

_KEEP_ALIVE = []

def build_index(vectors, kind="flat", nlist=128, nprobe=8, m=32, ef_construction=100, ef_search=64):
    vectors = np.ascontiguousarray(vectors, dtype="float32")
    dim = vectors.shape[1]
    if kind == "flat":
        index = faiss.IndexFlatIP(dim)
    elif kind == "ivf":
        quantizer = faiss.IndexFlatIP(dim)
        index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
        _KEEP_ALIVE.append(quantizer)
        index.train(vectors)
        index.nprobe = nprobe
    elif kind == "hnsw":
        index = faiss.IndexHNSWFlat(dim, m, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = ef_construction
        index.hnsw.efSearch = ef_search
    else:
        raise ValueError(f"unknown index kind: {kind}")
    index.add(vectors)
    return index


def search(index, queries, k):
    queries = np.ascontiguousarray(queries, dtype="float32")
    return index.search(queries, k)


def twotower_candidates(embeddings, customers, index, k=50):
    customers = np.asarray(customers, dtype="int64")
    customers = customers[embeddings.has_history[customers]]
    if len(customers) == 0:
        return pd.DataFrame(columns=COLUMNS)
    _, ids = search(index, embeddings.customer_vectors[customers], k)
    frame = pd.DataFrame(
        {
            "customer_idx": np.repeat(customers, k),
            "item_row": ids.reshape(-1),
            "source_rank": np.tile(np.arange(1, k + 1), len(customers)),
        }
    )
    frame = frame[frame["item_row"] >= 0].copy()
    frame["article_id"] = embeddings.article_ids[frame["item_row"].to_numpy()]
    frame["source"] = "twotower"
    return frame[COLUMNS].reset_index(drop=True)


def recall_vs_exact(approx_ids, exact_ids):
    hits = []
    for approx, exact in zip(approx_ids, exact_ids):
        found = set(approx[approx >= 0].tolist()) & set(exact.tolist())
        hits.append(len(found) / len(exact))
    return float(np.mean(hits))


def latency_ms(index, queries, k):
    queries = np.ascontiguousarray(queries, dtype="float32")
    times = []
    for row in queries:
        start = time.perf_counter()
        index.search(row.reshape(1, -1), k)
        times.append((time.perf_counter() - start) * 1000)
    return float(np.percentile(times, 50)), float(np.percentile(times, 99))


SETTINGS = [
    ("flat", "exact", {}),
    ("ivf", "nprobe=1", {"nprobe": 1}),
    ("ivf", "nprobe=4", {"nprobe": 4}),
    ("ivf", "nprobe=16", {"nprobe": 16}),
    ("hnsw", "efSearch=16", {"ef_search": 16}),
    ("hnsw", "efSearch=64", {"ef_search": 64}),
]


def benchmark(item_vectors, queries, k):
    exact_ids = search(build_index(item_vectors, "flat"), queries, k)[1]
    rows = []
    for kind, label, params in SETTINGS:
        start = time.perf_counter()
        index = build_index(item_vectors, kind, **params)
        build_seconds = time.perf_counter() - start
        approx_ids = search(index, queries, k)[1]
        p50, p99 = latency_ms(index, queries, k)
        rows.append(
            {
                "index": kind,
                "setting": label,
                "build_seconds": round(build_seconds, 3),
                "recall_vs_exact": round(recall_vs_exact(approx_ids, exact_ids), 4),
                "p50_ms": round(p50, 3),
                "p99_ms": round(p99, 3),
            }
        )
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", type=Path, default=config.PROCESSED_DIR)
    parser.add_argument("--results-dir", type=Path, default=config.RESULTS_DIR)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--k", type=int, default=50)
    parser.add_argument("--queries", type=int, default=1000)
    args = parser.parse_args()

    transactions = load_transactions(args.processed_dir)
    splitter = Splitter.from_frame(transactions)
    n_customers = int(transactions["customer_idx"].max()) + 1
    embeddings = fit(
        splitter.history("train"), splitter.cuts["train"], n_customers, epochs=args.epochs
    )

    trained = np.flatnonzero(embeddings.has_history)
    rng = np.random.default_rng(config.SEED)
    chosen = rng.choice(trained, size=min(args.queries, len(trained)), replace=False)
    queries = embeddings.customer_vectors[chosen]

    table = benchmark(embeddings.item_vectors, queries, args.k)
    print(table.to_string(index=False))

    args.results_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.results_dir / "ann_benchmark.csv"
    table.to_csv(out_path, index=False)
    print(f"saved to {out_path}")


if __name__ == "__main__":
    main()