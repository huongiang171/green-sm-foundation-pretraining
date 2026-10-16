from __future__ import annotations

import json

import pyarrow.parquet as pq
import pytest

from src.contracts.output import (
    ExpectedCandidate,
    OutputContractV1,
    PredictionRow,
    VerifiedOpportunityIndex,
)


def _index() -> VerifiedOpportunityIndex:
    return VerifiedOpportunityIndex(
        [
            ExpectedCandidate("r1", "a1", "u1", "2026-10-01T01:00:00+07:00", "validation"),
            ExpectedCandidate("r1", "a2", "u1", "2026-10-01T01:00:00+07:00", "validation"),
        ],
        task="ad_optimization", data_version="green-sm-v2", source_checksums={"ads": "abc"},
    )


def _metadata() -> dict[str, object]:
    return {
        "selected_checkpoint_sha256": "checkpoint", "vocabulary_sha256": "vocabulary",
        "foundation_manifest_sha256": "foundation", "ads_manifest_sha256": "ads",
        "config_digest": "config", "seed": 7, "run_id": "ads-v2", "score_semantics": "probability",
    }


def _food_index(split: str = "test") -> VerifiedOpportunityIndex:
    return VerifiedOpportunityIndex(
        [ExpectedCandidate("food-r1", "food-1", "u1", "2026-09-01T01:00:00+00:00", split)],
        task="food_recommendation",
        data_version="green-sm-pilot-v2",
        source_checksums={"food_manifest_sha256": "food-manifest"},
    )


def _food_metadata() -> dict[str, object]:
    return {
        "selected_checkpoint_sha256": "checkpoint",
        "vocabulary_sha256": "vocabulary",
        "foundation_manifest_sha256": "foundation",
        "food_manifest_sha256": "food-manifest",
        "config_digest": "config",
        "seed": 7,
        "run_id": "food-v1",
        "score_semantics": "ranking",
    }


def test_output_contract_writes_exact_label_free_verified_candidate_set(tmp_path) -> None:
    paths = OutputContractV1.write_predictions(
        [PredictionRow("r1", "a1", 0.25, "validation"), PredictionRow("r1", "a2", 0.75, "validation")],
        expected_candidates=_index(), output_dir=tmp_path, split="validation", run_metadata=_metadata(),
    )

    rows = pq.read_table(paths.predictions).to_pylist()
    assert len(rows) == 2
    assert "clicked" not in rows[0]
    assert rows[0]["user_id"] == "u1"
    assert json.loads(paths.validation_report.read_text(encoding="utf-8"))["passed"] is True


def test_output_contract_rejects_missing_duplicate_and_out_of_range_scores(tmp_path) -> None:
    with pytest.raises(ValueError, match="exactly match"):
        OutputContractV1.write_predictions(
            [PredictionRow("r1", "a1", 0.25, "validation")],
            expected_candidates=_index(), output_dir=tmp_path, split="validation", run_metadata=_metadata(),
        )
    with pytest.raises(ValueError, match="duplicate"):
        OutputContractV1.write_predictions(
            [PredictionRow("r1", "a1", 0.25, "validation"), PredictionRow("r1", "a1", 0.4, "validation")],
            expected_candidates=_index(), output_dir=tmp_path, split="validation", run_metadata=_metadata(),
        )
    with pytest.raises(ValueError, match="within"):
        OutputContractV1.write_predictions(
            [PredictionRow("r1", "a1", 1.1, "validation"), PredictionRow("r1", "a2", 0.4, "validation")],
            expected_candidates=_index(), output_dir=tmp_path, split="validation", run_metadata=_metadata(),
        )


def test_food_output_contract_accepts_unbounded_finite_ranking_score(tmp_path) -> None:
    paths = OutputContractV1.write_predictions(
        [PredictionRow("food-r1", "food-1", 3.75, "test")],
        expected_candidates=_food_index(),
        output_dir=tmp_path,
        split="test",
        run_metadata=_food_metadata(),
    )

    assert pq.read_table(paths.predictions).to_pylist()[0]["score"] == 3.75


def test_food_output_contract_rejects_nan_and_missing_candidate(tmp_path) -> None:
    with pytest.raises(ValueError, match="finite"):
        OutputContractV1.write_predictions(
            [PredictionRow("food-r1", "food-1", float("nan"), "test")],
            expected_candidates=_food_index(),
            output_dir=tmp_path,
            split="test",
            run_metadata=_food_metadata(),
        )
    with pytest.raises(ValueError, match="exactly match"):
        OutputContractV1.write_predictions(
            [],
            expected_candidates=_food_index(),
            output_dir=tmp_path,
            split="test",
            run_metadata=_food_metadata(),
        )


def test_output_contract_keeps_validation_and_test_metadata_separate(tmp_path) -> None:
    validation_paths = OutputContractV1.write_predictions(
        [PredictionRow("r1", "a1", 0.25, "validation"), PredictionRow("r1", "a2", 0.75, "validation")],
        expected_candidates=_index(), output_dir=tmp_path, split="validation", run_metadata=_metadata(),
    )
    test_paths = OutputContractV1.write_predictions(
        [PredictionRow("food-r1", "food-1", 3.75, "test")],
        expected_candidates=_food_index(), output_dir=tmp_path, split="test", run_metadata=_food_metadata(),
    )

    assert validation_paths.run_manifest != test_paths.run_manifest
    assert validation_paths.validation_report != test_paths.validation_report
    assert json.loads(validation_paths.run_manifest.read_text(encoding="utf-8"))["split"] == "validation"
    assert json.loads(test_paths.run_manifest.read_text(encoding="utf-8"))["split"] == "test"
