import argparse
import json
from pathlib import Path

from hmrec import config

ABLATION_LABELS = {
    "no_metadata": "drop metadata features",
    "no_twotower": "drop two-tower features",
    "no_pair": "drop customer-item pair features",
    "no_item_recent": "drop recent item popularity features",
    "binary_full": "classifier objective instead of LambdaRank",
    "nosrc_repurchase": "remove repurchase candidates",
    "nosrc_copurchase": "remove co-purchase candidates",
    "nosrc_twotower": "remove two-tower candidates",
    "nosrc_popular": "remove popularity candidates",
}


def load(path):
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else None


def table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return "\n".join(lines)


def fmt(value):
    return f"{value:.5f}"


def interval(entry):
    return f"{entry['mean_difference']:+.5f} [{entry['ci_low']:+.5f}, {entry['ci_high']:+.5f}]"


def ladder_section(baselines, ranker, ranker_data):
    valid_baselines = baselines["valid"]["map_at_12"]
    valid_ranker = ranker["valid"]
    chosen = baselines["chosen_on_train"]
    reference = valid_baselines["global_popular"]
    rows = [
        ("global popularity", valid_baselines["global_popular"]),
        (f"recent popularity ({chosen}, window chosen on training weeks)", baselines["valid_map_at_12_of_chosen"]),
        ("candidate generation order", valid_ranker["map_at_12_candidate_order"]),
        ("LightGBM LambdaRank ranker", valid_ranker["map_at_12_ranker"]),
    ]
    body = [(name, fmt(value), f"{value / reference:.1f}x") for name, value in rows]
    lines = [
        table(["Method (validation week)", "MAP@12", "vs global popularity"], body),
        f"Customers scored: {valid_ranker['customers_scored']}. Label week: {baselines['valid']['label_week']}.",
    ]
    bootstrap = valid_ranker.get("bootstrap_ranker_minus_candidate_order")
    if bootstrap:
        lines.append(f"Ranker minus candidate order: {interval(bootstrap)} (paired bootstrap over customers).")
    if ranker_data:
        lines.append(f"Candidate recall on the validation week: {ranker_data['valid']['candidate_recall']:.4f}.")
    lines.append(
        "The tree count and early stopping use the validation week, so these are development numbers, not the final test result."
    )
    return "\n\n".join(lines)


def ablation_section(ranker, ablation):
    if ablation is None:
        return "No ablation results found."
    full = ranker["valid"]["map_at_12_ranker"]
    rows = [("full model", fmt(full), "", f"{ranker['best_iteration']} trees")]
    ordered = [key for key in ABLATION_LABELS if key in ablation]
    ordered += [key for key in ablation if key not in ABLATION_LABELS]
    for key in ordered:
        entry = ablation[key]
        rows.append((ABLATION_LABELS.get(key, key), fmt(entry["map_variant"]), interval(entry), ""))
    text = table(["Variant", "MAP@12", "Full minus variant [CI]", "Note"], rows)
    return text + "\n\nPaired bootstrap over customers, 1,000 resamples. An interval that includes 0 means the difference is not reliably detectable."


def parity_section(parity):
    if parity is None:
        return "No parity check found."
    return (
        f"{parity['identical_lists']} of {parity['customers_compared']} online lists identical "
        f"to the offline lists ({parity['mismatches']} mismatches)."
    )


def latency_section(serving, ranker):
    if serving is None:
        return "No serving benchmark found."
    rows = []
    inprocess = serving.get("inprocess")
    if inprocess:
        total = inprocess["total"]
        rows.append(("in-process (no HTTP)", total["p50_ms"], total["p95_ms"], total["p99_ms"], "n/a"))
    for key in sorted(name for name in serving if name.startswith("http_concurrency_")):
        entry = serving[key]
        rows.append(
            (
                f"HTTP, {entry['concurrency']} client(s)",
                entry["p50_ms"],
                entry["p95_ms"],
                entry["p99_ms"],
                f"{entry['throughput_rps']:.0f} req/s",
            )
        )
    text = table(["Setting", "p50 ms", "p95 ms", "p99 ms", "Throughput"], rows)
    env = serving.get("environment", {})
    parts = [env.get("system"), f"{env.get('cpu_count')} logical cores", f"{ranker['best_iteration']} trees"]
    if env.get("candidates_per_request"):
        parts.append(f"{env['candidates_per_request']} candidates x {env.get('features')} features per request")
    note = ", ".join(str(part) for part in parts if part and "None" not in str(part))
    return text + f"\n\nMachine and model: {note}. Client and server on the same machine."


def build_report(name, results_dir=config.RESULTS_DIR):
    results_dir = Path(results_dir)
    baselines = load(results_dir / f"baselines_{name}.json")
    ranker = load(results_dir / f"ranker_{name}.json")
    if baselines is None or ranker is None:
        raise SystemExit(f"missing baselines or ranker results for {name}")
    ranker_data = load(results_dir / f"ranker_data_{name}.json")
    ablation = load(results_dir / f"ablation_{name}.json")
    parity = load(results_dir / f"parity_{name}.json")
    serving = load(results_dir / f"serving_benchmark_{name}.json")
    parts = [
        f"# Results for {name}",
        "## Ranking quality",
        ladder_section(baselines, ranker, ranker_data),
        "## Ablations",
        ablation_section(ranker, ablation),
        "## Online and offline parity",
        parity_section(parity),
        "## Serving latency",
        latency_section(serving, ranker),
    ]
    return "\n\n".join(parts) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", type=Path, default=config.PROCESSED_DIR)
    parser.add_argument("--results-dir", type=Path, default=config.RESULTS_DIR)
    args = parser.parse_args()
    name = Path(args.processed_dir).name
    report = build_report(name, args.results_dir)
    out_path = Path(args.results_dir) / f"report_{name}.md"
    out_path.write_text(report)
    print(report)
    print(f"report saved to {out_path}")


if __name__ == "__main__":
    main()