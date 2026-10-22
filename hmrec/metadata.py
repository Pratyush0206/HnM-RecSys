import argparse
from pathlib import Path

import pandas as pd

from hmrec import config

ARTICLE_COLUMNS = [
    "article_id",
    "product_type_no",
    "product_group_name",
    "department_no",
    "colour_group_code",
    "index_group_no",
    "garment_group_no",
]
CLUB_CODES = {"ACTIVE": 0, "PRE-CREATE": 1, "LEFT CLUB": 2}
NEWS_CODES = {"NONE": 0, "Regularly": 1, "Monthly": 2}


def build_customers(raw_dir, customer_map):
    frame = pd.read_csv(
        Path(raw_dir) / config.CUSTOMERS_CSV,
        usecols=["customer_id", "club_member_status", "fashion_news_frequency", "age"],
    )
    merged = customer_map.merge(frame, on="customer_id", how="left")
    out = pd.DataFrame(
        {
            "customer_idx": merged["customer_idx"].astype("int32"),
            "c_age": merged["age"].astype("float32"),
            "c_club": merged["club_member_status"].map(CLUB_CODES).astype("float32"),
            "c_news": merged["fashion_news_frequency"].map(NEWS_CODES).astype("float32"),
        }
    )
    return out.sort_values("customer_idx").reset_index(drop=True)


def build_articles(raw_dir, article_ids):
    frame = pd.read_csv(Path(raw_dir) / config.ARTICLES_CSV, usecols=ARTICLE_COLUMNS)
    frame = frame.assign(group_code=frame["product_group_name"].astype("category").cat.codes)
    kept = frame[frame["article_id"].isin(article_ids)]
    return pd.DataFrame(
        {
            "article_id": kept["article_id"].astype("int64"),
            "i_type": kept["product_type_no"].astype("int32"),
            "i_group": kept["group_code"].astype("int32"),
            "i_dept": kept["department_no"].astype("int32"),
            "i_colour": kept["colour_group_code"].astype("int32"),
            "i_index": kept["index_group_no"].astype("int32"),
            "i_garment": kept["garment_group_no"].astype("int32"),
        }
    ).reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=config.RAW_DIR)
    parser.add_argument("--processed-dir", type=Path, default=config.PROCESSED_DIR)
    args = parser.parse_args()

    processed = Path(args.processed_dir)
    customer_map = pd.read_parquet(processed / config.CUSTOMER_MAP_PARQUET)
    transactions = pd.read_parquet(
        processed / config.TRANSACTIONS_PARQUET, columns=["article_id"]
    )

    customers = build_customers(args.raw_dir, customer_map)
    articles = build_articles(args.raw_dir, transactions["article_id"].unique())
    customers.to_parquet(processed / config.CUSTOMERS_PARQUET, index=False)
    articles.to_parquet(processed / config.ARTICLES_PARQUET, index=False)

    print(f"customers: {len(customers)} rows")
    print(customers.isna().mean().round(3).to_string())
    print(f"articles: {len(articles)} rows")
    print(f"saved to {processed}")


if __name__ == "__main__":
    main()