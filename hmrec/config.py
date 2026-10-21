from pathlib import Path

K = 12
WEEK_DAYS = 7
SEED = 42
DEFAULT_WEEKS = 10
DEFAULT_CUSTOMERS = 40_000

RAW_DIR = Path("data") / "raw"
PROCESSED_DIR = Path("data") / "processed"
RESULTS_DIR = Path("results")

TRANSACTIONS_CSV = "transactions_train.csv"
TRANSACTIONS_PARQUET = "transactions.parquet"
CUSTOMER_MAP_PARQUET = "customer_map.parquet"
META_JSON = "meta.json"

COMPETITION = "h-and-m-personalized-fashion-recommendations"