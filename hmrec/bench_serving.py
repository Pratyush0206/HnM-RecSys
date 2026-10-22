import argparse
import json
import os
import platform
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import lightgbm as lgb
import numpy as np

from hmrec import config
from hmrec.recommender import Recommender
from hmrec.store import FEATURES_ROOT, STORE_DIR, load_store


def summarize(times_ms):
    values = np.asarray(times_ms, dtype="float64")
    return {
        "requests": int(len(values)),
        "p50_ms": round(float(np.percentile(values, 50)), 3),
        "p95_ms": round(float(np.percentile(values, 95)), 3),
        "p99_ms": round(float(np.percentile(values, 99)), 3),
        "max_ms": round(float(values.max()), 3),
        "mean_ms": round(float(values.mean()), 3),
    }


def sample_ids(store, n, seed):
    rng = np.random.default_rng(seed)
    picks = rng.choice(len(store), size=n, replace=n > len(store))
    return [str(store.customer_ids[i]) for i in picks]


def bench_inprocess(recommender, warm_ids, ids):
    for customer_id in warm_ids:
        recommender.recommend(customer_id)

    store, model = recommender.store, recommender.model
    k, threads = recommender.k, recommender.threads
    total, lookup, predict, rank = [], [], [], []
    for customer_id in ids:
        start = time.perf_counter()
        recommender.recommend(customer_id)
        total.append((time.perf_counter() - start) * 1000)

        start = time.perf_counter()
        articles, features = store.rows(customer_id)
        after_lookup = time.perf_counter()
        scores = model.predict(features, num_threads=threads)
        after_predict = time.perf_counter()
        order = np.argsort(-scores, kind="stable")[:k]
        articles[order].tolist()
        end = time.perf_counter()
        lookup.append((after_lookup - start) * 1000)
        predict.append((after_predict - after_lookup) * 1000)
        rank.append((end - after_predict) * 1000)

    return {
        "total": summarize(total),
        "store_lookup": summarize(lookup),
        "model_predict": summarize(predict),
        "sort_and_format": summarize(rank),
    }


def bench_http(url, warm_ids, ids, concurrency):
    def one_call(client, customer_id):
        start = time.perf_counter()
        response = client.get(f"{url}/recommend/{customer_id}")
        response.raise_for_status()
        return (time.perf_counter() - start) * 1000

    with httpx.Client(timeout=30.0) as client:
        for customer_id in warm_ids:
            one_call(client, customer_id)
        start = time.perf_counter()
        if concurrency == 1:
            times = [one_call(client, customer_id) for customer_id in ids]
        else:
            with ThreadPoolExecutor(max_workers=concurrency) as pool:
                times = list(pool.map(lambda c: one_call(client, c), ids))
        wall = time.perf_counter() - start

    result = summarize(times)
    result["concurrency"] = concurrency
    result["wall_seconds"] = round(wall, 3)
    result["throughput_rps"] = round(len(ids) / wall, 1)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", type=Path, default=config.PROCESSED_DIR)
    parser.add_argument("--ranker-dir", type=Path, default=None)
    parser.add_argument("--store-dir", type=Path, default=None)
    parser.add_argument("--results-dir", type=Path, default=config.RESULTS_DIR)
    parser.add_argument("--mode", choices=["inprocess", "http"], default="inprocess")
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--requests", type=int, default=2000)
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--seed", type=int, default=config.SEED)
    args = parser.parse_args()

    data_name = Path(args.processed_dir).name
    ranker_dir = args.ranker_dir or FEATURES_ROOT / data_name
    store_dir = args.store_dir or STORE_DIR / data_name
    store = load_store(store_dir)

    everything = sample_ids(store, args.warmup + args.requests, args.seed)
    warm_ids, ids = everything[: args.warmup], everything[args.warmup :]

    environment = {
        "python": platform.python_version(),
        "system": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "lightgbm": lgb.__version__,
        "candidates_per_request": int(np.diff(store.offsets).mean()),
        "features": len(store.feature_names),
    }

    if args.mode == "inprocess":
        model = lgb.Booster(model_file=str(ranker_dir / "ranker.txt"))
        environment["model_trees"] = model.num_trees()
        result = bench_inprocess(Recommender(model, store), warm_ids, ids)
        key = "inprocess"
    else:
        result = bench_http(args.url, warm_ids, ids, args.concurrency)
        key = f"http_concurrency_{args.concurrency}"

    args.results_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.results_dir / f"serving_benchmark_{data_name}.json"
    report = json.loads(out_path.read_text()) if out_path.exists() else {}
    report["environment"] = {**report.get("environment", {}), **environment}
    report[key] = result
    out_path.write_text(json.dumps(report, indent=2))
    print(json.dumps({"environment": environment, key: result}, indent=2))
    print(f"saved to {out_path}")


if __name__ == "__main__":
    main()