from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@pytest.fixture
def food_opportunities_root(tmp_path: Path) -> Path:
    root = tmp_path / "food_opportunities"
    root.mkdir()
    rows = [
        {
            "request_id": "food-test-1",
            "user_id": "user-1",
            "cutoff_timestamp": datetime(2026, 10, 1, 7, 0, tzinfo=timezone(timedelta(hours=7))),
            "candidate_id": "food-a",
            "split": "test",
            "is_positive": 1,
            "evaluation_group": "warm",
        },
        {
            "request_id": "food-test-1",
            "user_id": "user-1",
            "cutoff_timestamp": datetime(2026, 10, 1, 7, 0, tzinfo=timezone(timedelta(hours=7))),
            "candidate_id": "food-b",
            "split": "test",
            "is_positive": 0,
            "evaluation_group": "warm",
        },
    ]
    split_path = root / "food_opportunities.test.parquet"
    pq.write_table(pa.Table.from_pylist(rows), split_path)
    manifest = {
        "schema_version": "green_sm_food_opportunities_v1",
        "foundation_manifest_sha256": "foundation-manifest-sha",
        "files": {"test": {"file": split_path.name, "rows": len(rows), "sha256": sha256(split_path)}},
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root


def prediction_rows(*, user_id: str = "user-1", cutoff_timestamp: str = "2026-10-01T00:00:00+00:00") -> list[dict[str, object]]:
    return [
        {
            "task": "food_recommendation",
            "data_version": "green_sm_sim_full_v1",
            "run_id": "food-run-1",
            "split": "test",
            "request_id": "food-test-1",
            "user_id": user_id,
            "cutoff_timestamp": cutoff_timestamp,
            "candidate_id": "food-a",
            "score": 0.8,
            "raw_logit": None,
        },
        {
            "task": "food_recommendation",
            "data_version": "green_sm_sim_full_v1",
            "run_id": "food-run-1",
            "split": "test",
            "request_id": "food-test-1",
            "user_id": user_id,
            "cutoff_timestamp": cutoff_timestamp,
            "candidate_id": "food-b",
            "score": 0.2,
            "raw_logit": None,
        },
    ]


def write_foundation_prediction_dir(root: Path, opportunities_root: Path) -> Path:
    directory = root / "foundation_prediction"
    directory.mkdir()
    pq.write_table(pa.Table.from_pylist(prediction_rows()), directory / "predictions.test.parquet")
    manifest_sha = sha256(opportunities_root / "manifest.json")
    run_manifest = {
        "task": "food_recommendation",
        "data_version": "green_sm_sim_full_v1",
        "split": "test",
        "run_id": "food-run-1",
        "score_semantics": "ranking",
        "selected_checkpoint_sha256": "checkpoint-sha",
        "vocabulary_sha256": "vocabulary-sha",
        "foundation_manifest_sha256": "foundation-manifest-sha",
        "food_manifest_sha256": manifest_sha,
        "source_checksums": {
            "food_manifest_sha256": manifest_sha,
            "foundation_manifest_sha256": "foundation-manifest-sha",
        },
    }
    (directory / "run_manifest.test.json").write_text(json.dumps(run_manifest), encoding="utf-8")
    (directory / "validation_report.test.json").write_text(
        json.dumps({"schema_version": "output_contract_v1", "passed": True, "split": "test"}),
        encoding="utf-8",
    )
    return directory


def write_baseline_prediction_dir(root: Path, opportunities_root: Path) -> Path:
    directory = root / "baseline_prediction"
    directory.mkdir()
    pq.write_table(pa.Table.from_pylist(prediction_rows()), directory / "predictions.test.parquet")
    manifest = json.loads((opportunities_root / "manifest.json").read_text(encoding="utf-8"))
    sidecar = {
        "schema_version": "evaluation_input_v1",
        "model_provenance": {
            "kind": "baseline",
            "baseline_id": "lightgbm-v1",
            "foundation_identity": {"applicable": False},
        },
        "task": "food_recommendation",
        "data_version": "green_sm_sim_full_v1",
        "split": "test",
        "run_id": "food-run-1",
        "candidate_manifest_sha256": sha256(opportunities_root / "manifest.json"),
        "candidate_split_sha256": manifest["files"]["test"]["sha256"],
        "score_semantics": "ranking",
    }
    (directory / "evaluation_input_v1.json").write_text(json.dumps(sidecar), encoding="utf-8")
    return directory
