import copy
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[2]
_spec = importlib.util.spec_from_file_location(
    "check_baseline", ROOT / "app/eval/check_baseline.py"
)
assert _spec and _spec.loader
cb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cb)

RECORDED = json.loads((ROOT / "app/eval/results/dense_baseline.json").read_text())


def _changed(**overall_changes: float) -> dict:
    new = copy.deepcopy(RECORDED)
    for path, value in overall_changes.items():
        metric, k = path.split("_at_")
        new["overall"][metric][k] = value
    return new


def test_the_recorded_baseline_compared_with_itself_is_ok():
    result = cb.compare(RECORDED, RECORDED)
    assert result.exit_code == 0 and not (result.regressions or result.improvements)
    assert cb.render(result).endswith("0 overall regression(s).")


def test_a_drop_within_the_tolerance_is_not_a_regression():
    small = RECORDED["overall"]["mrr"]["10"] - cb.TOLERANCE / 2
    assert cb.compare(_changed(mrr_at_10=small), RECORDED).exit_code == 0


def test_a_drop_past_the_tolerance_fails_and_names_the_metric():
    dropped = RECORDED["overall"]["ndcg"]["5"] - 0.05
    result = cb.compare(_changed(ndcg_at_5=dropped), RECORDED)
    assert result.exit_code == 1
    [d] = result.regressions
    assert (d.scope, d.metric, d.k) == ("overall", "ndcg", "5") and d.delta == pytest.approx(-0.05)
    assert "REGRESSIONS" in cb.render(result) and "ndcg@5" in cb.render(result)


def test_an_improvement_never_fails():
    result = cb.compare(_changed(recall_at_1=0.9), RECORDED)
    assert result.exit_code == 0 and [d.metric for d in result.improvements] == ["recall"]


def test_per_kind_moves_are_informational_and_never_fail():
    new = copy.deepcopy(RECORDED)
    new["by_kind"]["paraphrase"]["mrr"]["5"] = 0.0
    result = cb.compare(new, RECORDED)
    assert result.exit_code == 0 and [d.scope for d in result.informational] == ["paraphrase"]


@pytest.mark.parametrize(
    ("key", "value"),
    [("ks", [1, 5]), ("queries", 99), ("corpus_paragraphs", 1), ("embedding_model", "other")],
)
def test_records_that_differ_in_setup_are_not_comparable(key, value):
    new = copy.deepcopy(RECORDED)
    new[key] = value
    result = cb.compare(new, RECORDED)
    assert result.exit_code == 2 and key in cb.render(result)
    # One clear reason, not that reason plus a cascade of "missing metric" lines it caused.
    assert len(result.incomparable) == 1, result.incomparable


def test_a_run_with_fewer_k_values_gives_one_clear_reason_not_a_cascade():
    new = copy.deepcopy(RECORDED)
    new["ks"] = [1, 5]  # a realistic different run: its metrics only have those k values too
    for block in [new["overall"], *new["by_kind"].values()]:
        for metric in ("recall", "mrr", "ndcg"):
            block[metric] = {k: v for k, v in block[metric].items() if k in ("1", "5")}
    result = cb.compare(new, RECORDED)
    assert result.exit_code == 2
    assert len(result.incomparable) == 1 and "ks" in result.incomparable[0], result.incomparable


def test_a_missing_metric_or_kind_in_the_new_run_is_not_comparable_not_a_pass():
    no_k = copy.deepcopy(RECORDED)
    del no_k["overall"]["recall"]["10"]
    assert cb.compare(no_k, RECORDED).exit_code == 2
    no_kind = copy.deepcopy(RECORDED)
    del no_kind["by_kind"]["multi"]
    assert cb.compare(no_kind, RECORDED).exit_code == 2


def test_main_returns_the_exit_code(tmp_path, capsys):
    good, bad = tmp_path / "good.json", tmp_path / "bad.json"
    good.write_text(json.dumps(RECORDED))
    bad.write_text(json.dumps(_changed(mrr_at_5=0.1)))
    recorded = ROOT / "app/eval/results/dense_baseline.json"
    assert cb.main(["--new", str(good), "--recorded", str(recorded)]) == 0
    assert cb.main(["--new", str(bad), "--recorded", str(recorded)]) == 1
    assert "FAIL" in capsys.readouterr().out
