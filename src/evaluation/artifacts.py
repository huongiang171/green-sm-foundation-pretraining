"""Checksum-bound loading of label-free downstream prediction artifacts."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import pyarrow.parquet as pq

from .policy import score_semantics_for_task
from .types import CanonicalPrediction, ModelProvenance, SourceIdentity, VerifiedEvaluationInput

EvaluationTask = Literal["food_recommendation", "ad_optimization"]
EvaluationSplit = Literal["validation", "test"]

_TASK_CONTRACTS = {
    "food_recommendation": {
        "manifest_schema": "green_sm_food_opportunities_v1",
        "label_column": "is_positive",
        "manifest_hash_key": "food_manifest_sha256",
    },
    "ad_optimization": {
        "manifest_schema": "green_sm_ads_opportunities_v1",
        "label_column": "clicked",
        "manifest_hash_key": "ads_manifest_sha256",
    },
}
_PREDICTION_COLUMNS = {
    "task",
    "data_version",
    "run_id",
    "split",
    "request_id",
    "user_id",
    "cutoff_timestamp",
    "candidate_id",
    "score",
}
_FORBIDDEN_LABEL_COLUMNS = {"clicked", "is_positive", "label"}


class EvaluationArtifactError(ValueError):
    """Raised when an artifact cannot be safely evaluated."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path, *, label: str) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise EvaluationArtifactError(f"Cannot read {label}: {path}") from error
    if not isinstance(payload, dict):
        raise EvaluationArtifactError(f"{label} must be a JSON object")
    return payload


def _required_text(payload: Mapping[str, object], key: str, *, label: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise EvaluationArtifactError(f"{label}.{key} must be a non-empty string")
    return value


def canonicalize_utc_instant(value: object) -> str:
    """Return one RFC-3339 UTC spelling; reject naive/non-temporal values."""

    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as error:
            raise EvaluationArtifactError(f"cutoff_timestamp is invalid: {value!r}") from error
    elif isinstance(value, datetime):
        parsed = value
    else:
        raise EvaluationArtifactError("cutoff_timestamp must be a timezone-aware datetime or ISO-8601 string")
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise EvaluationArtifactError("cutoff_timestamp must be timezone-aware")
    return parsed.astimezone(timezone.utc).isoformat()


def _opportunity_identity(
    *, task: EvaluationTask, split: EvaluationSplit, opportunities_root: Path
) -> tuple[SourceIdentity, dict[tuple[str, str], tuple[str, str]], Path, str]:
    contract = _TASK_CONTRACTS[task]
    manifest_path = opportunities_root / "manifest.json"
    manifest = _read_json(manifest_path, label="opportunity manifest")
    if manifest.get("schema_version") != contract["manifest_schema"]:
        raise EvaluationArtifactError("Opportunity manifest has unsupported schema_version")
    files = manifest.get("files")
    if not isinstance(files, Mapping) or not isinstance(files.get(split), Mapping):
        raise EvaluationArtifactError(f"Opportunity manifest has no {split!r} split")
    split_metadata = files[split]
    filename = _required_text(split_metadata, "file", label=f"manifest.files.{split}")
    expected_split_sha = _required_text(split_metadata, "sha256", label=f"manifest.files.{split}")
    split_path = opportunities_root / filename
    if _sha256(split_path) != expected_split_sha:
        raise EvaluationArtifactError("Opportunity split SHA-256 does not match manifest")
    columns = {"request_id", "candidate_id", "user_id", "cutoff_timestamp", "split"}
    table = pq.read_table(split_path, columns=sorted(columns))
    actual_columns = set(table.column_names)
    if actual_columns != columns:
        raise EvaluationArtifactError("Opportunity split is missing required candidate identity columns")
    expected: dict[tuple[str, str], tuple[str, str]] = {}
    for row in table.to_pylist():
        if row["split"] != split:
            raise EvaluationArtifactError("Opportunity row split does not match selected split")
        key = (str(row["request_id"]), str(row["candidate_id"]))
        if key in expected:
            raise EvaluationArtifactError("Opportunity split contains duplicate request/candidate pairs")
        expected[key] = (str(row["user_id"]), canonicalize_utc_instant(row["cutoff_timestamp"]))
    if not expected:
        raise EvaluationArtifactError("Opportunity split has no candidates")
    identity = SourceIdentity(
        task=task,
        data_version="",
        split=split,
        candidate_manifest_sha256=_sha256(manifest_path),
        candidate_split_sha256=expected_split_sha,
    )
    return identity, expected, split_path, str(contract["label_column"])


def _load_provenance(
    *,
    task: EvaluationTask,
    split: EvaluationSplit,
    prediction_dir: Path,
    source_identity: SourceIdentity,
    source_foundation_manifest_sha: str,
) -> tuple[ModelProvenance, str, str]:
    baseline_sidecar = prediction_dir / "evaluation_input_v1.json"
    if baseline_sidecar.is_file():
        payload = _read_json(baseline_sidecar, label="evaluation input sidecar")
        if payload.get("schema_version") != "evaluation_input_v1":
            raise EvaluationArtifactError("Baseline sidecar has unsupported schema_version")
        if payload.get("task") != task or payload.get("split") != split:
            raise EvaluationArtifactError("Baseline sidecar task or split does not match requested evaluation")
        if payload.get("candidate_manifest_sha256") != source_identity.candidate_manifest_sha256:
            raise EvaluationArtifactError("Baseline candidate_manifest_sha256 does not match source")
        if payload.get("candidate_split_sha256") != source_identity.candidate_split_sha256:
            raise EvaluationArtifactError("Baseline candidate_split_sha256 does not match source")
        if payload.get("score_semantics") != score_semantics_for_task(task):
            raise EvaluationArtifactError("Baseline score_semantics does not match task policy")
        provenance = payload.get("model_provenance")
        if not isinstance(provenance, Mapping) or provenance.get("kind") != "baseline":
            raise EvaluationArtifactError("Baseline model_provenance.kind must be 'baseline'")
        foundation_identity = provenance.get("foundation_identity")
        if foundation_identity != {"applicable": False}:
            raise EvaluationArtifactError("Baseline foundation_identity must declare applicable=false")
        return (
            ModelProvenance(kind="baseline", run_id=_required_text(payload, "run_id", label="baseline sidecar"), details=dict(provenance)),
            _required_text(payload, "data_version", label="baseline sidecar"),
            _required_text(payload, "run_id", label="baseline sidecar"),
        )
    run_manifest = _read_json(prediction_dir / f"run_manifest.{split}.json", label="Foundation run manifest")
    validation = _read_json(prediction_dir / f"validation_report.{split}.json", label="Foundation validation report")
    if validation.get("schema_version") != "output_contract_v1" or validation.get("passed") is not True:
        raise EvaluationArtifactError("Foundation validation report must be a passed OutputContractV1 report")
    if run_manifest.get("task") != task or run_manifest.get("split") != split:
        raise EvaluationArtifactError("Foundation run manifest task or split does not match requested evaluation")
    if run_manifest.get("score_semantics") != score_semantics_for_task(task):
        raise EvaluationArtifactError("Foundation score_semantics does not match task policy")
    source_checksums = run_manifest.get("source_checksums")
    if not isinstance(source_checksums, Mapping):
        raise EvaluationArtifactError("Foundation run manifest is missing source_checksums")
    manifest_hash_key = str(_TASK_CONTRACTS[task]["manifest_hash_key"])
    if source_checksums.get(manifest_hash_key) != source_identity.candidate_manifest_sha256:
        raise EvaluationArtifactError("Foundation candidate manifest binding does not match source")
    if run_manifest.get("foundation_manifest_sha256") != source_foundation_manifest_sha:
        raise EvaluationArtifactError("Foundation manifest binding does not match opportunity manifest")
    for key in ("selected_checkpoint_sha256", "vocabulary_sha256"):
        _required_text(run_manifest, key, label="Foundation run manifest")
    run_id = _required_text(run_manifest, "run_id", label="Foundation run manifest")
    return ModelProvenance(kind="foundation", run_id=run_id, details=run_manifest), _required_text(
        run_manifest, "data_version", label="Foundation run manifest"
    ), run_id


def load_verified_predictions(
    task: EvaluationTask,
    split: EvaluationSplit,
    prediction_dir: str | Path,
    opportunities_root: str | Path,
) -> VerifiedEvaluationInput:
    """Verify all bindings before exposing evaluator-only labels."""

    if task not in _TASK_CONTRACTS:
        raise EvaluationArtifactError(f"Unsupported evaluation task {task!r}")
    if split not in {"validation", "test"}:
        raise EvaluationArtifactError(f"Unsupported evaluation split {split!r}")
    prediction_path = Path(prediction_dir) / f"predictions.{split}.parquet"
    identity, expected, source_path, label_column = _opportunity_identity(
        task=task, split=split, opportunities_root=Path(opportunities_root)
    )
    source_manifest = _read_json(Path(opportunities_root) / "manifest.json", label="opportunity manifest")
    provenance, data_version, run_id = _load_provenance(
        task=task,
        split=split,
        prediction_dir=Path(prediction_dir),
        source_identity=identity,
        source_foundation_manifest_sha=_required_text(
            source_manifest, "foundation_manifest_sha256", label="opportunity manifest"
        ),
    )
    table = pq.read_table(prediction_path)
    columns = set(table.column_names)
    missing = _PREDICTION_COLUMNS - columns
    if missing:
        raise EvaluationArtifactError(f"Prediction Parquet is missing columns {sorted(missing)}")
    labels = _FORBIDDEN_LABEL_COLUMNS & columns
    if labels:
        raise EvaluationArtifactError(f"Prediction Parquet contains forbidden label columns {sorted(labels)}")
    predictions: dict[tuple[str, str], CanonicalPrediction] = {}
    for row in table.to_pylist():
        if row["task"] != task or row["split"] != split:
            raise EvaluationArtifactError("Prediction task or split does not match requested evaluation")
        if row["data_version"] != data_version or row["run_id"] != run_id:
            raise EvaluationArtifactError("Prediction data_version or run_id does not match artifact metadata")
        key = (str(row["request_id"]), str(row["candidate_id"]))
        if key in predictions:
            raise EvaluationArtifactError("Prediction Parquet contains duplicate request/candidate pairs")
        score = row["score"]
        if not isinstance(score, (int, float)) or not math.isfinite(float(score)):
            raise EvaluationArtifactError("Prediction score must be finite")
        if task == "ad_optimization" and not 0.0 <= float(score) <= 1.0:
            raise EvaluationArtifactError("Ads prediction score must be within [0, 1]")
        predictions[key] = CanonicalPrediction(
            request_id=key[0], candidate_id=key[1], user_id=str(row["user_id"]),
            cutoff_timestamp_utc=canonicalize_utc_instant(row["cutoff_timestamp"]), split=split, score=float(score)
        )
    if set(predictions) != set(expected):
        raise EvaluationArtifactError("Prediction candidates do not exactly match opportunity candidates")
    for key, prediction in predictions.items():
        expected_user, expected_cutoff = expected[key]
        if prediction.user_id != expected_user:
            raise EvaluationArtifactError("Prediction user_id does not match opportunity candidate")
        if prediction.cutoff_timestamp_utc != expected_cutoff:
            raise EvaluationArtifactError("Prediction cutoff_timestamp does not match opportunity candidate")
    label_table = pq.read_table(source_path, columns=["request_id", "candidate_id", label_column])
    labels_by_candidate: dict[tuple[str, str], int] = {}
    for row in label_table.to_pylist():
        key = (str(row["request_id"]), str(row["candidate_id"]))
        raw_label = row[label_column]
        if raw_label not in {0, 1}:
            raise EvaluationArtifactError("Opportunity labels must be binary")
        if key in labels_by_candidate:
            raise EvaluationArtifactError("Opportunity split contains duplicate request/candidate pairs")
        labels_by_candidate[key] = int(raw_label)
    if set(labels_by_candidate) != set(predictions):
        raise EvaluationArtifactError("Opportunity labels do not match verified candidate universe")
    return VerifiedEvaluationInput(
        identity=SourceIdentity(
            task=task, data_version=data_version, split=split,
            candidate_manifest_sha256=identity.candidate_manifest_sha256,
            candidate_split_sha256=identity.candidate_split_sha256,
        ),
        provenance=provenance,
        predictions=tuple(predictions[key] for key in sorted(predictions)),
        labels_by_candidate=labels_by_candidate,
    )
