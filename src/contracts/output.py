"""Checksum-bound, label-free prediction output contract."""

from __future__ import annotations

import json
import math
import os
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pyarrow as pa
import pyarrow.parquet as pq


@dataclass(frozen=True)
class ExpectedCandidate:
    request_id: str
    candidate_id: str
    user_id: str
    cutoff_timestamp: str
    split: Literal["validation", "test"]


@dataclass(frozen=True)
class PredictionRow:
    request_id: str
    candidate_id: str
    score: float
    split: Literal["validation", "test"]
    raw_logit: float | None = None


@dataclass(frozen=True)
class OutputPaths:
    predictions: Path
    run_manifest: Path
    validation_report: Path


class VerifiedOpportunityIndex:
    """The complete, immutable candidate universe permitted in one output file."""

    def __init__(
        self,
        rows: Iterable[ExpectedCandidate],
        *,
        task: str,
        data_version: str,
        source_checksums: Mapping[str, str],
    ) -> None:
        if task != "ad_optimization":
            raise ValueError("Only ad_optimization is supported by OutputContractV1")
        if not data_version or not source_checksums:
            raise ValueError("Verified opportunities require data_version and source checksums")
        values = tuple(rows)
        keys = [(row.request_id, row.candidate_id) for row in values]
        if not values or len(keys) != len(set(keys)):
            raise ValueError("Verified opportunities must be non-empty and have unique candidate pairs")
        if any(not all((row.request_id, row.candidate_id, row.user_id, row.cutoff_timestamp)) for row in values):
            raise ValueError("Verified opportunities contain an empty required field")
        self.task = task
        self.data_version = data_version
        self.source_checksums = dict(source_checksums)
        self.rows = {key: row for key, row in zip(keys, values, strict=True)}


class OutputContractV1:
    """Write one exact label-free prediction set against verified opportunities."""

    @staticmethod
    def write_predictions(
        rows: Iterable[PredictionRow],
        *,
        expected_candidates: VerifiedOpportunityIndex,
        output_dir: str | Path,
        split: Literal["validation", "test"],
        run_metadata: Mapping[str, object],
    ) -> OutputPaths:
        required = {
            "selected_checkpoint_sha256", "vocabulary_sha256", "foundation_manifest_sha256",
            "ads_manifest_sha256", "config_digest", "seed", "run_id", "score_semantics",
        }
        missing = sorted(required - set(run_metadata))
        if missing:
            raise ValueError(f"run_metadata missing required fields {missing}")
        if run_metadata["score_semantics"] != "probability":
            raise ValueError("ad_optimization output requires probability score semantics")
        submitted = tuple(rows)
        actual = {(row.request_id, row.candidate_id): row for row in submitted}
        if len(actual) != len(submitted):
            raise ValueError("Predictions contain duplicate request/candidate pairs")
        expected = {key: row for key, row in expected_candidates.rows.items() if row.split == split}
        if set(actual) != set(expected):
            raise ValueError("Predictions do not exactly match the verified candidate set")
        records: list[dict[str, object]] = []
        for key in sorted(expected):
            prediction = actual[key]
            opportunity = expected[key]
            if prediction.split != split:
                raise ValueError("Prediction split does not match requested output split")
            if not math.isfinite(prediction.score) or not 0.0 <= prediction.score <= 1.0:
                raise ValueError("Ads prediction score must be finite and within [0, 1]")
            if prediction.raw_logit is not None and not math.isfinite(prediction.raw_logit):
                raise ValueError("raw_logit must be finite when supplied")
            records.append({
                "task": expected_candidates.task, "data_version": expected_candidates.data_version,
                "run_id": str(run_metadata["run_id"]), "split": split,
                "request_id": opportunity.request_id, "user_id": opportunity.user_id,
                "cutoff_timestamp": opportunity.cutoff_timestamp, "candidate_id": opportunity.candidate_id,
                "score": float(prediction.score), "raw_logit": prediction.raw_logit,
            })
        directory = Path(output_dir)
        directory.mkdir(parents=True, exist_ok=True)
        prediction_path = directory / f"predictions.{split}.parquet"
        temporary = prediction_path.with_suffix(".parquet.tmp")
        pq.write_table(pa.Table.from_pylist(records), temporary, compression="zstd")
        os.replace(temporary, prediction_path)
        manifest_path = directory / "run_manifest.json"
        report_path = directory / "validation_report.json"
        manifest = dict(run_metadata)
        manifest.update({"task": expected_candidates.task, "data_version": expected_candidates.data_version, "split": split, "prediction_file": prediction_path.name, "source_checksums": expected_candidates.source_checksums})
        report = {"schema_version": "output_contract_v1", "passed": True, "split": split, "candidate_count": len(records)}
        for path, value in ((manifest_path, manifest), (report_path, report)):
            temporary_json = path.with_suffix(path.suffix + ".tmp")
            temporary_json.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")
            os.replace(temporary_json, path)
        return OutputPaths(prediction_path, manifest_path, report_path)
