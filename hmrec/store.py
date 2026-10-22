import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from hmrec import config

STORE_FILE = "store.npz"
STORE_META = "store_meta.json"
FEATURES_ROOT = Path("data") / "ranker"
STORE_DIR = Path("data") / "store"


class Store:
    def __init__(self, customer_ids, offsets, article_ids, features, feature_names, fallback):
        self.customer_ids = customer_ids
        self.offsets = offsets
        self.article_ids = article_ids
        self.features = features
        self.feature_names = list(feature_names)
        self.fallback = fallback
        self.position = {str(customer): i for i, customer in enumerate(customer_ids)}

    def __len__(self):
        return len(self.customer_ids)

    def __contains__(self, customer_id):
        return customer_id in self.position

    def rows(self, customer_id):
        i = self.position[customer_id]
        start, end = int(self.offsets[i]), int(self.offsets[i + 1])
        return self.article_ids[start:end], self.features[start:end]


def popular_fallback(featured, k):
    popular = featured.loc[featured["rank_popular"].notna(), ["article_id", "rank_popular"]]
    popular = popular.drop_duplicates("article_id")
    popular = popular.sort_values(["rank_popular", "article_id"], kind="stable")
    return popular["article_id"].to_numpy()[:k].astype("int64")


def build_store(featured, customer_map, feature_columns, k=config.K):
    frame = featured.sort_values(["customer_idx", "position"], kind="stable")
    sizes = frame.groupby("customer_idx", sort=True).size()
    offsets = np.concatenate([[0], np.cumsum(sizes.to_numpy())]).astype("int64")
    ids = customer_map.set_index("customer_idx")["customer_id"].loc[sizes.index]
    return Store(
        customer_ids=np.asarray(ids.tolist(), dtype=str),
        offsets=offsets,
        article_ids=frame["article_id"].to_numpy().astype("int64"),
        features=frame[list(feature_columns)].to_numpy(dtype="float32"),
        feature_names=feature_columns,
        fallback=popular_fallback(featured, k),
    )


def choose_customers(source, n_customers, seed):
    customers = pd.read_parquet(source, columns=["customer_idx"])["customer_idx"].unique()
    customers = np.sort(customers)
    if n_customers >= len(customers):
        return customers
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(customers, size=n_customers, replace=False))


def save_store(store, directory, extra=None):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    np.savez(
        directory / STORE_FILE,
        customer_ids=store.customer_ids,
        offsets=store.offsets,
        article_ids=store.article_ids,
        features=store.features,
        fallback=store.fallback,
    )
    meta = {
        "feature_names": store.feature_names,
        "customers": len(store),
        "rows": int(len(store.article_ids)),
        **(extra or {}),
    }
    (directory / STORE_META).write_text(json.dumps(meta, indent=2))
    return meta


def load_store(directory):
    directory = Path(directory)
    arrays = np.load(directory / STORE_FILE)
    meta = json.loads((directory / STORE_META).read_text())
    return Store(
        customer_ids=arrays["customer_ids"],
        offsets=arrays["offsets"],
        article_ids=arrays["article_ids"],
        features=arrays["features"],
        feature_names=meta["feature_names"],
        fallback=arrays["fallback"],
    )


def main():
    from hmrec.features import FEATURE_COLUMNS

    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", type=Path, default=config.PROCESSED_DIR)
    parser.add_argument("--ranker-dir", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--cut", default="valid")
    parser.add_argument("--customers", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=config.SEED)
    args = parser.parse_args()

    if args.cut == "test":
        raise SystemExit("serving snapshots must not use the test week")

    data_name = Path(args.processed_dir).name
    ranker_dir = args.ranker_dir or FEATURES_ROOT / data_name
    out_dir = args.out_dir or STORE_DIR / data_name
    source = ranker_dir / f"{args.cut}_features.parquet"

    chosen = choose_customers(source, args.customers, args.seed)
    columns = list(dict.fromkeys(["customer_idx", "article_id", *FEATURE_COLUMNS]))
    featured = pd.read_parquet(
        source, columns=columns, filters=[("customer_idx", "in", chosen.tolist())]
    )
    customer_map = pd.read_parquet(Path(args.processed_dir) / config.CUSTOMER_MAP_PARQUET)

    store = build_store(featured, customer_map, FEATURE_COLUMNS)
    meta = save_store(store, out_dir, extra={"cut": args.cut, "seed": args.seed})
    print(json.dumps({**meta, "fallback": store.fallback.tolist()}, indent=2))
    print(f"store saved to {out_dir}")


if __name__ == "__main__":
    main()