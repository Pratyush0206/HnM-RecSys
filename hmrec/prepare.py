import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from hmrec import config

DTYPES = {
    "t_dat": "string",
    "customer_id": "string",
    "article_id": "int64",
    "price": "float32",
    "sales_channel_id": "int8",
}


def find_last_day(csv_path, chunksize):
    last_day = None
    chunks = pd.read_csv(
        csv_path, usecols=["t_dat"], dtype={"t_dat": "string"}, chunksize=chunksize
    )
    for chunk in chunks:
        chunk_max = pd.to_datetime(chunk["t_dat"], format="%Y-%m-%d").max()
        if last_day is None or chunk_max > last_day:
            last_day = chunk_max
    return last_day


def read_recent(csv_path, start_day, chunksize):
    parts = []
    for chunk in pd.read_csv(csv_path, dtype=DTYPES, chunksize=chunksize):
        chunk["t_dat"] = pd.to_datetime(chunk["t_dat"], format="%Y-%m-%d")
        parts.append(chunk[chunk["t_dat"] >= start_day])
    return pd.concat(parts, ignore_index=True)


def sample_customers(transactions, n_customers, seed):
    customers = transactions["customer_id"].drop_duplicates().sort_values().to_numpy()
    if n_customers >= len(customers):
        return customers
    rng = np.random.default_rng(seed)
    chosen = rng.choice(customers, size=n_customers, replace=False)
    return np.sort(chosen)


def encode_customers(transactions, customers):
    customer_map = pd.DataFrame(
        {"customer_id": customers, "customer_idx": np.arange(len(customers), dtype="int32")}
    )
    lookup = pd.Series(customer_map["customer_idx"].to_numpy(), index=customer_map["customer_id"])
    encoded = transactions.copy()
    encoded["customer_idx"] = encoded["customer_id"].map(lookup).astype("int32")
    encoded = encoded.drop(columns=["customer_id"])
    return encoded, customer_map


def prepare(raw_dir, out_dir, weeks, n_customers, chunksize, seed):
    csv_path = Path(raw_dir) / config.TRANSACTIONS_CSV
    last_day = find_last_day(csv_path, chunksize)
    start_day = last_day - pd.Timedelta(days=weeks * config.WEEK_DAYS - 1)

    recent = read_recent(csv_path, start_day, chunksize)
    customers = sample_customers(recent, n_customers, seed)
    sampled = recent[recent["customer_id"].isin(customers)]
    transactions, customer_map = encode_customers(sampled, customers)
    transactions = transactions.sort_values("t_dat", kind="stable").reset_index(drop=True)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    transactions.to_parquet(out_dir / config.TRANSACTIONS_PARQUET, index=False)
    customer_map.to_parquet(out_dir / config.CUSTOMER_MAP_PARQUET, index=False)

    meta = {
        "source": str(csv_path),
        "weeks": weeks,
        "start_day": str(start_day.date()),
        "last_day": str(last_day.date()),
        "customers": int(len(customers)),
        "items": int(transactions["article_id"].nunique()),
        "rows": int(len(transactions)),
        "seed": seed,
    }
    (out_dir / config.META_JSON).write_text(json.dumps(meta, indent=2))
    return meta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=config.RAW_DIR)
    parser.add_argument("--out-dir", type=Path, default=config.PROCESSED_DIR)
    parser.add_argument("--weeks", type=int, default=config.DEFAULT_WEEKS)
    parser.add_argument("--customers", type=int, default=config.DEFAULT_CUSTOMERS)
    parser.add_argument("--chunksize", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=config.SEED)
    args = parser.parse_args()

    meta = prepare(
        args.raw_dir, args.out_dir, args.weeks, args.customers, args.chunksize, args.seed
    )
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()