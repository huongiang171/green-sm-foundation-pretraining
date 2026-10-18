"""Build a deterministic, immutable Food PoC subset from the full linked corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

_SPLITS = ("train", "validation", "test")
_SCHEMA_VERSION = "green_sm_food_opportunities_v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read JSON artifact: {path}") from error
    if not isinstance(payload, dict):
        raise TypeError(f"JSON artifact must be an object: {path}")
    return payload


def _require_text(value: Mapping[str, Any], key: str, *, label: str) -> str:
    parsed = value.get(key)
    if not isinstance(parsed, str) or not parsed:
        raise ValueError(f"{label}.{key} must be a non-empty string")
    return parsed


def _source_files(source: Path) -> tuple[dict[str, Any], dict[str, Path]]:
    opportunities = source / "food_opportunities"
    manifest_path = opportunities / "manifest.json"
    manifest = _read_json(manifest_path)
    if manifest.get("schema_version") != _SCHEMA_VERSION:
        raise ValueError("source Food manifest has unsupported schema_version")
    files = manifest.get("files")
    if not isinstance(files, Mapping) or set(files) != set(_SPLITS):
        raise ValueError("source Food manifest must declare train, validation, and test")
    paths: dict[str, Path] = {}
    for split in _SPLITS:
        entry = files[split]
        if not isinstance(entry, Mapping):
            raise TypeError(f"source Food manifest files.{split} must be an object")
        filename = _require_text(entry, "file", label=f"source Food manifest files.{split}")
        expected = _require_text(entry, "sha256", label=f"source Food manifest files.{split}")
        path = opportunities / filename
        if not path.is_file() or _sha256(path) != expected:
            raise ValueError(f"source Food {split} parquet does not match its manifest")
        paths[split] = path
    return manifest, paths


def _request_groups(rows: Sequence[Mapping[str, Any]], *, split: str) -> dict[str, list[Mapping[str, Any]]]:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        request_id = row.get("request_id")
        if not isinstance(request_id, str) or not request_id:
            raise ValueError(f"Food {split} row has invalid request_id")
        if row.get("split") != split:
            raise ValueError(f"Food request {request_id!r} is in the wrong split")
        groups[request_id].append(row)
    if not groups:
        raise ValueError(f"Food {split} has no requests")
    return dict(groups)


def _stratum(rows: Sequence[Mapping[str, Any]]) -> tuple[int, str]:
    positives = [row for row in rows if int(row.get("is_positive", 0)) == 1]
    if not positives:
        raise ValueError(f"Food request {rows[0].get('request_id')!r} has no positive candidate")
    categories = sorted(str(row.get("candidate_category", "__MISSING__")) for row in positives)
    return len(positives), "|".join(categories)


def _stable_rank(*, seed: int, split: str, stratum: tuple[int, str], request_id: str) -> str:
    payload = f"{seed}|{split}|{stratum[0]}|{stratum[1]}|{request_id}".encode()
    return hashlib.sha256(payload).hexdigest()


def _select_request_ids(
    groups: Mapping[str, Sequence[Mapping[str, Any]]], *, split: str, fraction: float, seed: int
) -> set[str]:
    by_stratum: dict[tuple[int, str], list[str]] = defaultdict(list)
    for request_id, rows in groups.items():
        by_stratum[_stratum(rows)].append(request_id)
    target = max(1, round(sum(len(request_ids) for request_ids in by_stratum.values()) * fraction))
    ordered_by_stratum: dict[tuple[int, str], list[str]] = {}
    base_counts: dict[tuple[int, str], int] = {}
    remainders: list[tuple[float, tuple[int, str]]] = []
    for stratum, request_ids in by_stratum.items():
        ordered = sorted(
            request_ids,
            key=lambda request_id: _stable_rank(
                seed=seed, split=split, stratum=stratum, request_id=request_id
            ),
        )
        ordered_by_stratum[stratum] = ordered
        exact = len(ordered) * fraction
        base_counts[stratum] = math.floor(exact)
        remainders.append((exact - base_counts[stratum], stratum))
    if target >= len(ordered_by_stratum):
        for stratum in sorted(ordered_by_stratum):
            base_counts[stratum] = max(1, base_counts[stratum])
    allocated = sum(base_counts.values())
    if allocated > target:
        for _, stratum in sorted(remainders):
            if allocated == target:
                break
            if base_counts[stratum] > 0:
                base_counts[stratum] -= 1
                allocated -= 1
    for _, stratum in sorted(remainders, key=lambda item: (-item[0], item[1])):
        if allocated == target:
            break
        if base_counts[stratum] < len(ordered_by_stratum[stratum]):
            base_counts[stratum] += 1
            allocated += 1
    selected: set[str] = set()
    for stratum, ordered in ordered_by_stratum.items():
        selected.update(ordered[:base_counts[stratum]])
    return selected


def _write_parquet(rows: Sequence[Mapping[str, Any]], schema: pa.Schema, path: Path) -> None:
    table = pa.Table.from_pylist(list(rows), schema=schema)
    pq.write_table(table, path, compression="zstd")


def build_food_poc_subset(
    source_root: str | Path,
    destination_root: str | Path,
    *,
    fraction: float = 0.2,
    seed: int = 20261018,
) -> dict[str, Any]:
    """Derive Food train/validation subsets while retaining an unchanged full test split."""

    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must be within (0, 1]")
    source = Path(source_root)
    destination = Path(destination_root)
    if destination.exists() and any(destination.iterdir()):
        raise ValueError(f"refusing to overwrite non-empty destination: {destination}")
    manifest, source_paths = _source_files(source)
    output = destination / "food_opportunities"
    output.mkdir(parents=True, exist_ok=False)
    selected_counts: dict[str, int] = {}
    output_files: dict[str, dict[str, Any]] = {}
    try:
        for split in ("train", "validation"):
            source_table = pq.read_table(source_paths[split])
            rows = source_table.to_pylist()
            groups = _request_groups(rows, split=split)
            selected_ids = _select_request_ids(groups, split=split, fraction=fraction, seed=seed)
            selected_rows = [row for row in rows if str(row["request_id"]) in selected_ids]
            output_path = output / source_paths[split].name
            _write_parquet(selected_rows, source_table.schema, output_path)
            selected_counts[split] = len(selected_ids)
            output_files[split] = {
                "file": output_path.name,
                "rows": len(selected_rows),
                "sha256": _sha256(output_path),
            }
        source_test = source_paths["test"]
        output_test = output / source_test.name
        shutil.copy2(source_test, output_test)
        test_rows = pq.read_table(source_test, columns=["request_id"]).column("request_id").to_pylist()
        selected_counts["test"] = len({str(value) for value in test_rows})
        output_files["test"] = {
            "file": output_test.name,
            "rows": len(test_rows),
            "sha256": _sha256(output_test),
        }
        parent_manifest = source / "food_opportunities" / "manifest.json"
        derived_manifest = {
            "schema_version": _SCHEMA_VERSION,
            "data_version": "green_sm_sim_full_v1",
            "foundation_manifest_sha256": manifest["foundation_manifest_sha256"],
            "files": output_files,
            "derivation": {
                "schema_version": "green_sm_food_poc_subset_v1",
                "parent_opportunity_manifest_sha256": _sha256(parent_manifest),
                "parent_split_sha256": {
                    split: manifest["files"][split]["sha256"] for split in _SPLITS
                },
                "selection_unit": "request_id",
                "selected_splits": ["train", "validation"],
                "full_test_retained": True,
                "fraction": fraction,
                "seed": seed,
                "strata": ["positive_count", "positive_categories"],
                "selected_request_counts": selected_counts,
            },
        }
        manifest_path = output / "manifest.json"
        manifest_path.write_text(
            json.dumps(derived_manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8"
        )
    except Exception:
        # The destination was required empty, so leaving the partial artifact would make reruns unsafe.
        shutil.rmtree(destination)
        raise
    return {
        "destination_root": str(destination),
        "manifest_sha256": _sha256(output / "manifest.json"),
        "selected_request_counts": selected_counts,
        "files": output_files,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build a derived Green SM Food PoC opportunity artifact")
    parser.add_argument("--source-root", required=True, help="Full linked Green SM root")
    parser.add_argument("--destination-root", required=True, help="New empty derived-artifact root")
    parser.add_argument("--fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=20261018)
    args = parser.parse_args(argv)
    report = build_food_poc_subset(
        args.source_root, args.destination_root, fraction=args.fraction, seed=args.seed
    )
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
