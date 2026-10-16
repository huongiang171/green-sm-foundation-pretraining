from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src.data.green_sm.build import build_dataset
from src.data.green_sm.config import GreenSmConfig
from src.data.green_sm.validate import _token_and_oov_metrics, validate_outputs


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _build_valid_pilot(tmp_path: Path) -> GreenSmConfig:
    config = replace(GreenSmConfig.from_json("configs/green_sm_sim_pilot.json"), output_root=tmp_path / "out")
    build_dataset(config)
    return config


def _inject_control_only_label_marker(config: GreenSmConfig) -> None:
    root = config.output_root / "green_sm_sim_control_paired_v1" / "foundation"
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    shard = manifest["shards"][0]
    path = root / shard["events_file"]
    rows = pq.read_table(path).to_pylist()
    context = json.loads(rows[0]["context_json"])
    context["label_marker"] = 1
    rows[0]["context_json"] = json.dumps(context, sort_keys=True)
    pq.write_table(pa.Table.from_pylist(rows), path, compression="zstd")
    shard["events_sha256"] = _sha256(path)
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    foundation_manifest_sha256 = _sha256(manifest_path)
    for task in ("food", "ads"):
        opportunity_manifest = root.parent / f"{task}_opportunities" / "manifest.json"
        payload = json.loads(opportunity_manifest.read_text(encoding="utf-8"))
        payload["foundation_manifest_sha256"] = foundation_manifest_sha256
        opportunity_manifest.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def test_validator_writes_complete_report_and_rejects_control_label_marker(tmp_path: Path) -> None:
    config = _build_valid_pilot(tmp_path)

    report = validate_outputs(config)

    assert report["roots"]["linked"]["readiness_passed"] is True
    assert report["roots"]["paired_control"]["readiness_passed"] is True
    assert report["roots"]["linked"]["data_geo_contract_version"] == 3
    assert isinstance(report["roots"]["linked"]["geo_catalogue_fingerprint"], str)
    _inject_control_only_label_marker(config)
    with pytest.raises(ValueError, match="paired-control conditional independence"):
        validate_outputs(config)


def test_token_threshold_counts_are_train_only() -> None:
    heldout_sample = SimpleNamespace(
        profile={"budget_bucket": "medium"},
        events=(
            SimpleNamespace(
                scenario="Food",
                event_type="food_search",
                context={"food_category": "noodles"},
                action={"item_price": 50000},
            ),
        ),
    )

    counts, _, _ = _token_and_oov_metrics(
        {"train": [], "validation": [heldout_sample], "test": []}
    )

    assert counts == {}
