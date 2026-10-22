import argparse
import json
from pathlib import Path

import lightgbm as lgb
import pandas as pd

from hmrec import config
from hmrec.features import FEATURE_COLUMNS
from hmrec.ranker import top_k_lists
from hmrec.recommender import Recommender
from hmrec.store import FEATURES_ROOT, STORE_DIR, STORE_META, load_store


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", type=Path, default=config.PROCESSED_DIR)
    parser.add_argument("--ranker-dir", type=Path, default=None)
    parser.add_argument("--store-dir", type=Path, default=None)
    parser.add_argument("--results-dir", type=Path, default=config.RESULTS_DIR)
    args = parser.parse_args()

    data_name = Path(args.processed_dir).name
    ranker_dir = args.ranker_dir or FEATURES_ROOT / data_name
    store_dir = args.store_dir or STORE_DIR / data_name

    store = load_store(store_dir)
    meta = json.loads((store_dir / STORE_META).read_text())
    model = lgb.Booster(model_file=str(ranker_dir / "ranker.txt"))

    customer_map = pd.read_parquet(Path(args.processed_dir) / config.CUSTOMER_MAP_PARQUET)
    index_of = dict(zip(customer_map["customer_id"], customer_map["customer_idx"]))
    chosen = [int(index_of[customer_id]) for customer_id in store.customer_ids]

    source = ranker_dir / f"{meta['cut']}_features.parquet"
    columns = ["customer_idx", "article_id", *FEATURE_COLUMNS]
    frame = pd.read_parquet(source, columns=columns, filters=[("customer_idx", "in", chosen)])
    frame = frame.sort_values(["customer_idx", "position"], kind="stable")
    offline = top_k_lists(frame, model.predict(frame[FEATURE_COLUMNS]))

    recommender = Recommender(model, store)
    mismatches = []
    for customer_id in store.customer_ids:
        online = recommender.recommend(str(customer_id))["article_ids"]
        if online != offline[int(index_of[customer_id])]:
            mismatches.append(str(customer_id))

    report = {
        "customers_compared": len(store),
        "identical_lists": len(store) - len(mismatches),
        "mismatches": len(mismatches),
        "example_mismatches": mismatches[:5],
    }
    args.results_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.results_dir / f"parity_{data_name}.json"
    out_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"saved to {out_path}")
    if mismatches:
        raise SystemExit("online and offline recommendations differ")


if __name__ == "__main__":
    main()