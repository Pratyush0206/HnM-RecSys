import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from hmrec import config
from hmrec.store import build_store, save_store


def make_demo(out_dir, customers=20, candidates=30, n_features=5, seed=config.SEED):
    rng = np.random.default_rng(seed)
    rows = customers * candidates
    names = [f"f{i}" for i in range(n_features)]
    position = np.tile(np.arange(1, candidates + 1), customers)

    featured = pd.DataFrame(rng.normal(size=(rows, n_features)), columns=names)
    featured["customer_idx"] = np.repeat(np.arange(customers), candidates)
    featured["article_id"] = 100_000 + np.tile(np.arange(candidates), customers)
    featured["position"] = position
    featured["rank_popular"] = position.astype("float64")

    labels = (featured["f0"] + rng.normal(scale=0.5, size=rows) > 0.8).astype(int)
    dataset = lgb.Dataset(featured[names], label=labels, group=[candidates] * customers)
    params = {
        "objective": "lambdarank",
        "num_leaves": 7,
        "min_data_in_leaf": 5,
        "verbose": -1,
        "seed": seed,
    }
    booster = lgb.train(params, dataset, num_boost_round=20)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(out_dir / "ranker.txt"))
    customer_map = pd.DataFrame(
        {
            "customer_idx": np.arange(customers),
            "customer_id": [f"demo{i:04d}" for i in range(customers)],
        }
    )
    store = build_store(featured, customer_map, names)
    return save_store(store, out_dir / "store", extra={"demo": True, "seed": seed})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--customers", type=int, default=20)
    parser.add_argument("--candidates", type=int, default=30)
    parser.add_argument("--seed", type=int, default=config.SEED)
    args = parser.parse_args()
    meta = make_demo(args.out_dir, args.customers, args.candidates, seed=args.seed)
    print(json.dumps(meta, indent=2))
    print(f"demo model and store saved to {args.out_dir}")


if __name__ == "__main__":
    main()