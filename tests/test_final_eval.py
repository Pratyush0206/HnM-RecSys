import json

import pytest

from hmrec.final_eval import file_digest, load_settings, refuse_if_already_scored


def test_refuses_when_the_result_file_exists(tmp_path):
    out_path = tmp_path / "final_test_x.json"
    out_path.write_text("{}")
    with pytest.raises(SystemExit):
        refuse_if_already_scored(out_path)


def test_allows_the_first_run(tmp_path):
    refuse_if_already_scored(tmp_path / "final_test_x.json")


def test_digest_changes_when_the_model_changes(tmp_path):
    model = tmp_path / "ranker.txt"
    model.write_text("tree one")
    first = file_digest(model)
    assert file_digest(model) == first
    model.write_text("tree two")
    assert file_digest(model) != first


def test_settings_come_from_the_saved_ranker_data_summary(tmp_path):
    summary = {"settings": {"epochs": 20, "dim": 32, "k": 30, "max_per_customer": 200}}
    (tmp_path / "ranker_data_demo.json").write_text(json.dumps(summary))
    assert load_settings(tmp_path, "demo") == summary["settings"]