from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from src.evaluation.compare import EvaluationComparisonError, compare_reports
from src.evaluation.runner import evaluate_prediction_directory
from src.evaluation.types import EvaluationReport

from .conftest import write_baseline_prediction_dir


def _report(food_opportunities_root: Path, tmp_path: Path) -> EvaluationReport:
    return evaluate_prediction_directory(
        task="food_recommendation",
        split="test",
        prediction_dir=write_baseline_prediction_dir(tmp_path, food_opportunities_root),
        opportunities_root=food_opportunities_root,
    )


def test_compare_keeps_each_run_provenance(food_opportunities_root: Path, tmp_path: Path) -> None:
    baseline = _report(food_opportunities_root, tmp_path)
    foundation_payload = deepcopy(baseline.payload)
    foundation_payload["model_provenance"] = {"kind": "foundation", "run_id": "foundation-run-1"}

    comparison = compare_reports([baseline, EvaluationReport(payload=foundation_payload)])

    assert [row["model_provenance"]["kind"] for row in comparison["runs"]] == ["baseline", "foundation"]


@pytest.mark.parametrize(
    "field",
    [
        "evaluator_version",
        "metric_policy_version",
        "tie_break_policy",
        "timestamp_policy",
        "task",
        "split",
        "data_version",
        "candidate_manifest_sha256",
        "candidate_split_sha256",
    ],
)
def test_compare_rejects_each_incompatible_identity(
    food_opportunities_root: Path, tmp_path: Path, field: str
) -> None:
    report = _report(food_opportunities_root, tmp_path)
    incompatible = deepcopy(report.payload)
    incompatible[field] = "different-value"

    with pytest.raises(EvaluationComparisonError, match=field):
        compare_reports([report, EvaluationReport(payload=incompatible)])


def test_compare_rejects_reports_that_do_not_match_evaluation_report_contract() -> None:
    malformed = EvaluationReport(
        payload={
            "schema_version": "not_a_report",
            "evaluator_version": None,
            "metric_policy_version": None,
            "tie_break_policy": None,
            "timestamp_policy": None,
            "task": None,
            "split": None,
            "data_version": None,
            "candidate_manifest_sha256": None,
            "candidate_split_sha256": None,
            "model_provenance": {},
            "metrics": {"roc_auc": "garbage"},
        }
    )

    with pytest.raises(EvaluationComparisonError, match="schema_version"):
        compare_reports([malformed, malformed])
