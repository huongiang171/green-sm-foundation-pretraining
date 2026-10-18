from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src.data.green_sm.poc import build_food_poc_subset


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _rows(split: str, request_count: int) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for request_index in range(request_count):
        request_id = f"food:{split}:{request_index:03d}"
        category = "noodles" if request_index % 2 == 0 else "rice"
        for candidate_index in range(2):
            rows.append(
                {
                    "request_id": request_id,
                    "user_id": f"u{request_index:03d}",
                    "foundation_sample_id": f"foundation:{request_id}",
                    "cutoff_timestamp": "2026-09-01T00:00:00+00:00",
                    "candidate_id": f"item:{candidate_index}",
                    "split": split,
                    "candidate_category": category if candidate_index == 0 else "coffee",
                    "is_positive": int(candidate_index == 0),
                }
            )
    return rows


def _source_root(tmp_path: Path) -> Path:
    source = tmp_path / "full" / "green_sm_sim_linked_v1"
    opportunities = source / "food_opportunities"
    opportunities.mkdir(parents=True)
    files: dict[str, dict[str, object]] = {}
    for split, request_count in (("train", 10), ("validation", 10), ("test", 10)):
        path = opportunities / f"food_opportunities.{split}.parquet"
        pq.write_table(pa.Table.from_pylist(_rows(split, request_count)), path, compression="zstd")
        files[split] = {"file": path.name, "rows": request_count * 2, "sha256": _sha256(path)}
    (opportunities / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "green_sm_food_opportunities_v1",
                "foundation_manifest_sha256": "a" * 64,
                "files": files,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return source


def test_build_food_poc_keeps_complete_requests_and_full_test(tmp_path: Path) -> None:
    source = _source_root(tmp_path)
    destination = tmp_path / "poc"

    report = build_food_poc_subset(source, destination, fraction=0.2, seed=20261018)

    output = destination / "food_opportunities"
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    train = pq.read_table(output / "food_opportunities.train.parquet").to_pylist()
    validation = pq.read_table(output / "food_opportunities.validation.parquet").to_pylist()
    source_test = source / "food_opportunities" / "food_opportunities.test.parquet"
    output_test = output / "food_opportunities.test.parquet"
    assert len({row["request_id"] for row in train}) == 2
    assert len({row["request_id"] for row in validation}) == 2
    assert {row["request_id"] for row in train} <= {
        row["request_id"] for row in pq.read_table(source / "food_opportunities" / "food_opportunities.train.parquet").to_pylist()
    }
    assert all(sum(row["request_id"] == request_id for row in train) == 2 for request_id in {row["request_id"] for row in train})
    assert _sha256(output_test) == _sha256(source_test)
    assert manifest["files"]["test"]["sha256"] == _sha256(source_test)
    assert manifest["derivation"]["parent_opportunity_manifest_sha256"] == _sha256(
        source / "food_opportunities" / "manifest.json"
    )
    assert report["selected_request_counts"] == {"train": 2, "validation": 2, "test": 10}


def test_build_food_poc_refuses_nonempty_destination(tmp_path: Path) -> None:
    source = _source_root(tmp_path)
    destination = tmp_path / "poc"
    destination.mkdir()
    (destination / "existing.txt").write_text("keep", encoding="utf-8")

    with pytest.raises(ValueError, match="non-empty"):
        build_food_poc_subset(source, destination, fraction=0.2, seed=20261018)
