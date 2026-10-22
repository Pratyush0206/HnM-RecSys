import json

import pytest

from hmrec import report


def write(directory, filename, data):
    (directory / filename).write_text(json.dumps(data))


def write_required(directory, name):
    write(
        directory,
        f"baselines_{name}.json",
        {
            "valid": {"label_week": "2020-09-09 to 2020-09-15", "map_at_12": {"global_popular": 0.004}},
            "chosen_on_train": "recent_popular_1w",
            "valid_map_at_12_of_chosen": 0.006,
        },
    )
    write(
        directory,
        f"ranker_{name}.json",
        {
            "best_iteration": 100,
            "valid": {
                "customers_scored": 10,
                "map_at_12_ranker": 0.012,
                "map_at_12_candidate_order": 0.01,
                "bootstrap_ranker_minus_candidate_order": {
                    "mean_difference": 0.002,
                    "ci_low": 0.001,
                    "ci_high": 0.003,
                },
            },
        },
    )


def test_interval_is_formatted_with_signs():
    entry = {"mean_difference": 0.00166, "ci_low": 0.0009, "ci_high": 0.00246}
    assert report.interval(entry) == "+0.00166 [+0.00090, +0.00246]"


def test_ladder_reports_ratios_against_global_popularity(tmp_path):
    write_required(tmp_path, "demo")
    text = report.build_report("demo", tmp_path)
    assert "| global popularity | 0.00400 | 1.0x |" in text
    assert "| LightGBM LambdaRank ranker | 0.01200 | 3.0x |" in text
    assert "development numbers" in text


def test_missing_optional_files_are_reported_not_fatal(tmp_path):
    write_required(tmp_path, "demo")
    text = report.build_report("demo", tmp_path)
    assert "No ablation results found." in text
    assert "No parity check found." in text
    assert "No serving benchmark found." in text


def test_missing_required_files_stop_the_report(tmp_path):
    with pytest.raises(SystemExit):
        report.build_report("demo", tmp_path)