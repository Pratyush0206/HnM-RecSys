import argparse
import json
from pathlib import Path

import numpy as np

from hmrec import config
from hmrec.bootstrap import paired_bootstrap


def compare(full, variant):
    if not np.array_equal(full["customers"], variant["customers"]):
        raise ValueError("the two runs scored different customers")
    result = paired_bootstrap(full["ranker"], variant["ranker"])
    result["map_full"] = float(np.mean(full["ranker"]))
    result["map_variant"] = float(np.mean(variant["ranker"]))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", type=Path, default=config.PROCESSED_DIR)
    parser.add_argument("--results-dir", type=Path, default=config.RESULTS_DIR)
    parser.add_argument("--variants", nargs="+", required=True)
    args = parser.parse_args()

    data_name = Path(args.processed_dir).name
    full = np.load(args.results_dir / f"scores_{data_name}_full.npz")

    table = {}
    for tag in args.variants:
        variant = np.load(args.results_dir / f"scores_{data_name}_{tag}.npz")
        result = compare(full, variant)
        table[tag] = {name: round(value, 5) for name, value in result.items()}

    print(f"{'variant':<28}{'MAP':>9}{'full minus variant':>22}{'95% interval':>26}")
    for tag, row in table.items():
        interval = f"[{row['ci_low']:.5f}, {row['ci_high']:.5f}]"
        print(
            f"{tag:<28}{row['map_variant']:>9.5f}{row['mean_difference']:>22.5f}"
            f"{interval:>26}"
        )

    out_path = args.results_dir / f"ablation_{data_name}.json"
    out_path.write_text(json.dumps(table, indent=2))
    print(f"saved to {out_path}")


if __name__ == "__main__":
    main()