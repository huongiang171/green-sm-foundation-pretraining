"""Strictly compare reports produced by the shared downstream evaluator."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path

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


def compare_reports(reports: Sequence[EvaluationReport]) -> dict[str, object]:
    """Return a comparison only when every immutable evaluation identity agrees."""

    if len(reports) < 2:
        raise EvaluationComparisonError("Comparison requires at least two evaluation reports")
    reference = reports[0].payload
    for report in reports[1:]:
        for field in _IDENTITY_FIELDS:
            if report.payload.get(field) != reference.get(field):
                raise EvaluationComparisonError(f"Reports differ in {field}")
    identity = {field: reference[field] for field in _IDENTITY_FIELDS}
    runs: list[dict[str, object]] = []
    for report in reports:
        provenance = report.payload.get("model_provenance")
        metrics = report.payload.get("metrics")
        eligibility = report.payload.get("eligibility")
        if not isinstance(provenance, Mapping) or not isinstance(metrics, Mapping):
            raise EvaluationComparisonError("Evaluation report has invalid provenance or metrics")
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
    return EvaluationReport(payload=payload)


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
