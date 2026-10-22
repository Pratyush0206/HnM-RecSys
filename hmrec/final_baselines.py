import argparse
import json
from pathlib import Path

from hmrec import config
from hmrec.evaluate import load_transactions, score_baselines
from hmrec.splits import Splitter


def run(processed_dir, results_dir):
    results_dir = Path(results_dir)
    name = Path(processed_dir).name
    development = json.loads((results_dir / f"baselines_{name}.json").read_text())
    chosen = development["chosen_on_train"]

    splitter = Splitter.from_frame(load_transactions(processed_dir))
    scores, n_customers = score_baselines(splitter, "test")

    report = {
        "data": name,
        "chosen_on_train": chosen,
        "customers_scored": n_customers,
        "map_at_12": scores,
        "map_at_12_of_chosen": scores[chosen],
    }
    out_path = results_dir / f"final_baselines_{name}.json"
    out_path.write_text(json.dumps(report, indent=2))
    return report, out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", type=Path, default=config.PROCESSED_DIR)
    parser.add_argument("--results-dir", type=Path, default=config.RESULTS_DIR)
    args = parser.parse_args()

    report, out_path = run(args.processed_dir, args.results_dir)
    print(json.dumps(report, indent=2))
    print(f"saved to {out_path}")


if __name__ == "__main__":
    main()