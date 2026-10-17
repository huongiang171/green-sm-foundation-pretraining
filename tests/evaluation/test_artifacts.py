from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src.evaluation.artifacts import EvaluationArtifactError, load_verified_predictions

from .conftest import (
    prediction_rows,
    sha256,
    write_baseline_prediction_dir,
    write_foundation_prediction_dir,
)


def test_loads_foundation_v1_and_canonicalizes_offset_timestamp(food_opportunities_root: Path, tmp_path: Path) -> None:
    prediction_dir = write_foundation_prediction_dir(tmp_path, food_opportunities_root)

    verified = load_verified_predictions("food_recommendation", "test", prediction_dir, food_opportunities_root)

    assert verified.provenance.kind == "foundation"
    assert verified.predictions[0].cutoff_timestamp_utc == "2026-10-01T00:00:00+00:00"
    assert verified.labels_by_candidate[("food-test-1", "food-a")] == 1


def test_loads_baseline_without_fake_foundation_sha(food_opportunities_root: Path, tmp_path: Path) -> None:
    prediction_dir = write_baseline_prediction_dir(tmp_path, food_opportunities_root)

    verified = load_verified_predictions("food_recommendation", "test", prediction_dir, food_opportunities_root)

    assert verified.provenance.kind == "baseline"
    assert verified.provenance.details["foundation_identity"] == {"applicable": False}


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        ("wrong_user", "user_id"),
        ("naive_cutoff", "timezone-aware"),
        ("different_cutoff", "cutoff_timestamp"),
        ("label_column", "forbidden label"),
        ("duplicate", "duplicate"),
    ],
)
def test_rejects_prediction_identity_or_label_leakage(
    food_opportunities_root: Path, tmp_path: Path, mutator: str, message: str
) -> None:
    prediction_dir = write_baseline_prediction_dir(tmp_path, food_opportunities_root)
    rows = prediction_rows()
    if mutator == "wrong_user":
        rows[0]["user_id"] = "other-user"
    elif mutator == "naive_cutoff":
        rows[0]["cutoff_timestamp"] = "2026-10-01T00:00:00"
    elif mutator == "different_cutoff":
        rows[0]["cutoff_timestamp"] = "2026-10-01T00:00:01+00:00"
    elif mutator == "label_column":
        for row in rows:
            row["is_positive"] = 1
    elif mutator == "duplicate":
        rows[1]["candidate_id"] = "food-a"
    pq.write_table(pa.Table.from_pylist(rows), prediction_dir / "predictions.test.parquet")

    with pytest.raises(EvaluationArtifactError, match=message):
        load_verified_predictions("food_recommendation", "test", prediction_dir, food_opportunities_root)


@pytest.mark.parametrize("checksum_key", ["candidate_manifest_sha256", "candidate_split_sha256"])
def test_rejects_stale_baseline_candidate_checksum(
    food_opportunities_root: Path, tmp_path: Path, checksum_key: str
) -> None:
    prediction_dir = write_baseline_prediction_dir(tmp_path, food_opportunities_root)
    sidecar_path = prediction_dir / "evaluation_input_v1.json"
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    sidecar[checksum_key] = "0" * 64
    sidecar_path.write_text(json.dumps(sidecar), encoding="utf-8")

    with pytest.raises(EvaluationArtifactError, match=checksum_key):
        load_verified_predictions("food_recommendation", "test", prediction_dir, food_opportunities_root)


def test_rejects_baseline_that_claims_foundation_identity(food_opportunities_root: Path, tmp_path: Path) -> None:
    prediction_dir = write_baseline_prediction_dir(tmp_path, food_opportunities_root)
    sidecar_path = prediction_dir / "evaluation_input_v1.json"
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    sidecar["model_provenance"]["foundation_identity"] = {"applicable": True, "sha": "fake"}
    sidecar_path.write_text(json.dumps(sidecar), encoding="utf-8")

    with pytest.raises(EvaluationArtifactError, match="foundation_identity"):
        load_verified_predictions("food_recommendation", "test", prediction_dir, food_opportunities_root)


@pytest.mark.parametrize("fractional_label", [0.9, 1.9])
def test_rejects_non_binary_source_labels_before_integer_coercion(
    food_opportunities_root: Path, tmp_path: Path, fractional_label: float
) -> None:
    split_path = food_opportunities_root / "food_opportunities.test.parquet"
    rows = pq.read_table(split_path).to_pylist()
    rows[0]["is_positive"] = fractional_label
    pq.write_table(pa.Table.from_pylist(rows), split_path)
    manifest_path = food_opportunities_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"]["test"]["sha256"] = sha256(split_path)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    prediction_dir = write_baseline_prediction_dir(tmp_path, food_opportunities_root)

    with pytest.raises(EvaluationArtifactError, match="binary"):
        load_verified_predictions("food_recommendation", "test", prediction_dir, food_opportunities_root)
