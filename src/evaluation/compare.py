"""Strictly compare reports produced by the shared downstream evaluator."""

from __future__ import annotations

import json
import math
import os
from collections.abc import Mapping, Sequence
from pathlib import Path

from .policy import (
    EVALUATOR_VERSION,
    METRIC_POLICY_VERSION,
    TIE_BREAK_POLICY,
    TIMESTAMP_POLICY,
    score_semantics_for_task,
)
from .types import EvaluationReport

_IDENTITY_FIELDS = (
    "evaluator_version",
    "metric_policy_version",
    "tie_break_policy",
    "timestamp_policy",
    "task",
    "split",
    "data_version",
    "candidate_manifest_sha256",
    "candidate_split_sha256",
)


class EvaluationComparisonError(ValueError):
    """Raised when reports cannot support a fair comparison."""


def _require_text(payload: Mapping[str, object], field: str) -> str:
    value = payload.get(field)
    if not isinstance(value, str) or not value:
        raise EvaluationComparisonError(f"Evaluation report has invalid {field}")
    return value


def _validate_evaluation_report(report: EvaluationReport) -> None:
    """Reject malformed reports before they can enter a fair comparison."""

    payload = report.payload
    if not isinstance(payload, Mapping):
        raise EvaluationComparisonError("Evaluation report payload must be an object")
    if payload.get("schema_version") != "green_sm_evaluation_report_v1":
        raise EvaluationComparisonError("Evaluation report has unsupported schema_version")
    expected_policy = {
        "evaluator_version": EVALUATOR_VERSION,
        "metric_policy_version": METRIC_POLICY_VERSION,
        "tie_break_policy": TIE_BREAK_POLICY,
        "timestamp_policy": TIMESTAMP_POLICY,
    }
    for field, expected in expected_policy.items():
        if payload.get(field) != expected:
            raise EvaluationComparisonError(f"Evaluation report has unsupported {field}")
    task = _require_text(payload, "task")
    try:
        score_semantics_for_task(task)
    except ValueError as error:
        raise EvaluationComparisonError(f"Evaluation report has unsupported task {task!r}") from error
    if payload.get("split") not in {"validation", "test"}:
        raise EvaluationComparisonError("Evaluation report has unsupported split")
    _require_text(payload, "data_version")
    for field in ("candidate_manifest_sha256", "candidate_split_sha256"):
        value = _require_text(payload, field)
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value.lower()):
            raise EvaluationComparisonError(f"Evaluation report has invalid {field}")
    provenance = payload.get("model_provenance")
    if not isinstance(provenance, Mapping) or provenance.get("kind") not in {"baseline", "foundation"}:
        raise EvaluationComparisonError("Evaluation report has invalid model_provenance")
    _require_text(provenance, "run_id")
    metrics = payload.get("metrics")
    if not isinstance(metrics, Mapping) or not metrics:
        raise EvaluationComparisonError("Evaluation report has invalid metrics")
    for name, value in metrics.items():
        if not isinstance(name, str) or not name or value is not None and (
            isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value))
        ):
            raise EvaluationComparisonError("Evaluation report has invalid metrics")
    if not isinstance(payload.get("eligibility"), Mapping):
        raise EvaluationComparisonError("Evaluation report has invalid eligibility")
    undefined_metrics = payload.get("undefined_metrics")
    if not isinstance(undefined_metrics, Mapping) or any(
        not isinstance(name, str) or not isinstance(reason, str) or not reason
        for name, reason in undefined_metrics.items()
    ):
        raise EvaluationComparisonError("Evaluation report has invalid undefined_metrics")


def compare_reports(reports: Sequence[EvaluationReport]) -> dict[str, object]:
    """Return a comparison only when every immutable evaluation identity agrees."""

    if len(reports) < 2:
        raise EvaluationComparisonError("Comparison requires at least two evaluation reports")
    for report in reports:
        _validate_evaluation_report(report)
    reference = reports[0].payload
    for report in reports[1:]:
        for field in _IDENTITY_FIELDS:
            if report.payload.get(field) != reference.get(field):
                raise EvaluationComparisonError(f"Reports differ in {field}")
    identity = {field: reference[field] for field in _IDENTITY_FIELDS}
    runs: list[dict[str, object]] = []
    for report in reports:
        provenance = report.payload["model_provenance"]
        metrics = report.payload["metrics"]
        eligibility = report.payload["eligibility"]
        runs.append(
            {
                "model_provenance": dict(provenance),
                "metrics": dict(metrics),
                "eligibility": dict(eligibility) if isinstance(eligibility, Mapping) else {},
                "undefined_metrics": dict(report.payload.get("undefined_metrics", {})),
            }
        )
    return {"schema_version": "green_sm_evaluation_comparison_v1", "identity": identity, "runs": runs}


def load_evaluation_report(path: str | Path) -> EvaluationReport:
    """Read one JSON evaluator report with enough shape validation for comparison."""

    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise EvaluationComparisonError(f"Cannot read evaluation report {source}") from error
    if not isinstance(payload, dict):
        raise EvaluationComparisonError(f"Evaluation report {source} must be a JSON object")
    missing = [field for field in _IDENTITY_FIELDS if field not in payload]
    if missing:
        raise EvaluationComparisonError(f"Evaluation report {source} is missing identity fields {missing}")
    report = EvaluationReport(payload=payload)
    _validate_evaluation_report(report)
    return report


def write_comparison(comparison: Mapping[str, object], output_path: str | Path) -> Path:
    """Atomically write a comparison and reject an accidental overwrite."""

    destination = Path(output_path)
    if destination.exists():
        raise FileExistsError(f"Comparison report already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(dict(comparison), sort_keys=True, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, destination)
    return destination
