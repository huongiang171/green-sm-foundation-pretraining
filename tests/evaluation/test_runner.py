from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from src.evaluation.runner import evaluate_prediction_directory, write_evaluation_report

from .conftest import sha256, write_baseline_prediction_dir


def test_food_report_contains_policy_metrics_and_eligibility(
    food_opportunities_root: Path, tmp_path: Path
) -> None:
    prediction_dir = write_baseline_prediction_dir(tmp_path, food_opportunities_root)

    report = evaluate_prediction_directory(
        task="food_recommendation",
        split="test",
        prediction_dir=prediction_dir,
        opportunities_root=food_opportunities_root,
    )

    assert report.payload["evaluator_version"] == "green_sm_evaluator_v1"
    assert report.payload["metrics"].keys() >= {"ndcg_at_5", "ndcg_at_10", "recall_at_10", "mrr"}
    assert report.payload["eligibility"] == {
        "source_request_count": 1,
        "complete_candidate_request_count": 1,
        "ranking_eligible_request_count": 1,
        "excluded_request_counts": {},
    }


def test_ads_report_records_all_rows_and_null_metrics_after_json_roundtrip(tmp_path: Path) -> None:
    opportunities_root = tmp_path / "ads_opportunities"
    opportunities_root.mkdir()
    rows = [
        {
            "request_id": "ads-test-1",
            "user_id": "user-1",
            "cutoff_timestamp": datetime(2026, 10, 1, tzinfo=timezone.utc),
            "candidate_id": "ad-a",
            "split": "test",
            "clicked": 0,
        },
        {
            "request_id": "ads-test-1",
            "user_id": "user-1",
            "cutoff_timestamp": datetime(2026, 10, 1, tzinfo=timezone.utc),
            "candidate_id": "ad-b",
            "split": "test",
            "clicked": 0,
        },
    ]
    split_path = opportunities_root / "ads_opportunities.test.parquet"
    pq.write_table(pa.Table.from_pylist(rows), split_path)
    (opportunities_root / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "green_sm_ads_opportunities_v1",
                "foundation_manifest_sha256": "foundation-manifest-sha",
                "files": {"test": {"file": split_path.name, "rows": 2, "sha256": sha256(split_path)}},
            }
        ),
        encoding="utf-8",
    )
    prediction_dir = tmp_path / "ads_prediction"
    prediction_dir.mkdir()
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "task": "ad_optimization",
                    "data_version": "green_sm_sim_full_v1",
                    "run_id": "ads-run-1",
                    "split": "test",
                    "request_id": row["request_id"],
                    "user_id": row["user_id"],
                    "cutoff_timestamp": "2026-10-01T00:00:00+00:00",
                    "candidate_id": row["candidate_id"],
                    "score": 0.0,
                }
                for row in rows
            ]
        ),
        prediction_dir / "predictions.test.parquet",
    )
    manifest = json.loads((opportunities_root / "manifest.json").read_text(encoding="utf-8"))
    (prediction_dir / "evaluation_input_v1.json").write_text(
        json.dumps(
            {
                "schema_version": "evaluation_input_v1",
                "model_provenance": {
                    "kind": "baseline",
                    "baseline_id": "ads-baseline-v1",
                    "foundation_identity": {"applicable": False},
                },
                "task": "ad_optimization",
                "data_version": "green_sm_sim_full_v1",
                "split": "test",
                "run_id": "ads-run-1",
                "candidate_manifest_sha256": sha256(opportunities_root / "manifest.json"),
                "candidate_split_sha256": manifest["files"]["test"]["sha256"],
                "score_semantics": "probability",
            }
        ),
        encoding="utf-8",
    )

    report = evaluate_prediction_directory(
        task="ad_optimization", split="test", prediction_dir=prediction_dir, opportunities_root=opportunities_root
    )
    output_path = write_evaluation_report(report, tmp_path / "evaluation_report.ad_optimization.test.json")
    saved = json.loads(output_path.read_text(encoding="utf-8"))

    assert saved["metrics"]["roc_auc"] is None
    assert saved["undefined_metrics"] == {
        "pr_auc": "single_class_labels",
        "roc_auc": "single_class_labels",
    }
    assert saved["eligibility"]["candidate_row_count"] == 2
    assert saved["eligibility"]["zero_click_excluded_request_count"] == 1
