"""Orchestration and report writing for verified downstream predictions."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Literal

import sklearn

from .artifacts import load_verified_predictions
from .metrics import ScoredLabel, evaluate_ads, evaluate_food_requests
from .policy import METRIC_POLICY
from .types import EvaluationReport, VerifiedEvaluationInput


def _scored_labels(verified: VerifiedEvaluationInput) -> list[ScoredLabel]:
    predictions = verified.predictions
    labels_by_candidate = verified.labels_by_candidate
    return [
        ScoredLabel(
            request_id=prediction.request_id,
            candidate_id=prediction.candidate_id,
            score=prediction.score,
            label=labels_by_candidate[(prediction.request_id, prediction.candidate_id)],
        )
        for prediction in predictions
    ]


def evaluate_prediction_directory(
    *,
    task: Literal["food_recommendation", "ad_optimization"],
    split: Literal["validation", "test"],
    prediction_dir: str | Path,
    opportunities_root: str | Path,
) -> EvaluationReport:
    """Evaluate one fully verified prediction directory without model access."""

    verified = load_verified_predictions(task, split, prediction_dir, opportunities_root)
    rows = _scored_labels(verified)
    result = evaluate_food_requests(rows) if task == "food_recommendation" else evaluate_ads(rows)
    source_request_count = len({row.request_id for row in rows})
    eligibility = dict(result.eligibility)
    eligibility.setdefault("source_request_count", source_request_count)
    eligibility["complete_candidate_request_count"] = source_request_count
    payload = {
        "schema_version": "green_sm_evaluation_report_v1",
        **METRIC_POLICY.identity(),
        "metric_library_version": sklearn.__version__,
        "task": verified.identity.task,
        "data_version": verified.identity.data_version,
        "split": verified.identity.split,
        "candidate_manifest_sha256": verified.identity.candidate_manifest_sha256,
        "candidate_split_sha256": verified.identity.candidate_split_sha256,
        "model_provenance": {
            "kind": verified.provenance.kind,
            "run_id": verified.provenance.run_id,
            **dict(verified.provenance.details),
        },
        "metrics": dict(result.metrics),
        "eligibility": eligibility,
        "undefined_metrics": dict(result.undefined_metrics),
    }
    return EvaluationReport(payload=payload)


def write_evaluation_report(
    report: EvaluationReport, output_path: str | Path, *, overwrite: bool = False
) -> Path:
    """Atomically persist an evaluator report without silently overwriting one."""

    destination = Path(output_path)
    if destination.exists() and not overwrite:
        raise FileExistsError(f"Evaluation report already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(dict(report.payload), sort_keys=True, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, destination)
    return destination
