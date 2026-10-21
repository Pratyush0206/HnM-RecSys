import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from hmrec import config

SYNTHETIC_DIR = Path("data") / "synthetic"


def make_transactions(
    n_customers=5_000,
    n_items=2_000,
    n_styles=20,
    weeks=12,
    rows_per_customer=8,
    style_loyalty=0.7,
    trend_days=14.0,
    last_day="2020-09-22",
    seed=config.SEED,
):
    rng = np.random.default_rng(seed)
    n_days = weeks * config.WEEK_DAYS
    first_day = pd.Timestamp(last_day) - pd.Timedelta(days=n_days - 1)

    customer_ids = np.array([rng.bytes(32).hex() for _ in range(n_customers)])
    customer_style = rng.integers(0, n_styles, size=n_customers)
    customer_activity = rng.gamma(shape=1.0, scale=1.0, size=n_customers)
    customer_prob = customer_activity / customer_activity.sum()

    article_ids = 108_775_015 + np.arange(n_items) * 17
    item_style = rng.integers(0, n_styles, size=n_items)
    item_base = rng.pareto(1.5, size=n_items) + 1.0
    item_peak = rng.uniform(0, n_days, size=n_items)
    item_price = rng.uniform(0.005, 0.1, size=n_items).round(6)

    n_rows = n_customers * rows_per_customer
    row_customer = rng.choice(n_customers, size=n_rows, p=customer_prob)
    row_style = customer_style[row_customer]
    row_day = rng.integers(0, n_days, size=n_rows)
    row_loyal = rng.random(n_rows) < style_loyalty
    row_item = np.empty(n_rows, dtype=np.int64)

    for day in range(n_days):
        weights = item_base * np.exp(-0.5 * ((day - item_peak) / trend_days) ** 2)
        on_day = row_day == day

        open_rows = np.flatnonzero(on_day & ~row_loyal)
        row_item[open_rows] = rng.choice(
            n_items, size=len(open_rows), p=weights / weights.sum()
        )

        for style in range(n_styles):
            style_rows = np.flatnonzero(on_day & row_loyal & (row_style == style))
            if len(style_rows) == 0:
                continue
            style_weights = np.where(item_style == style, weights, 0.0)
            total = style_weights.sum()
            if total == 0:
                style_weights, total = weights, weights.sum()
            row_item[style_rows] = rng.choice(
                n_items, size=len(style_rows), p=style_weights / total
            )

    dates = first_day + pd.to_timedelta(row_day, unit="D")
    frame = pd.DataFrame(
        {
            "t_dat": dates.strftime("%Y-%m-%d"),
            "customer_id": customer_ids[row_customer],
            "article_id": article_ids[row_item],
            "price": item_price[row_item],
            "sales_channel_id": rng.integers(1, 3, size=n_rows),
        }
    )
    return frame.sort_values("t_dat", kind="stable").reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--customers", type=int, default=5_000)
    parser.add_argument("--items", type=int, default=2_000)
    parser.add_argument("--weeks", type=int, default=12)
    parser.add_argument("--out-dir", type=Path, default=SYNTHETIC_DIR)
    args = parser.parse_args()

    frame = make_transactions(
        n_customers=args.customers, n_items=args.items, weeks=args.weeks
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / config.TRANSACTIONS_CSV
    frame.to_csv(out_path, index=False)
    print(f"wrote {len(frame)} rows to {out_path}")


if __name__ == "__main__":
    main()