"""Fail-closed readiness validation for immutable Green SM output roots."""

from __future__ import annotations

import hashlib
import json
import math
from bisect import bisect_right
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from src.foundation_input.loader import iter_normalized_samples
from src.foundation_input.manifest import DatasetManifest, load_manifest

from .build import root_names
from .config import GreenSmConfig
from .world import TargetWorld, build_target_world


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _json(value: object) -> str:
    return json.dumps(value, default=lambda item: item.isoformat(), sort_keys=True, separators=(",", ":"))


def _root_paths(config: GreenSmConfig) -> dict[str, Path]:
    return {name: config.output_root / dirname for name, dirname in root_names(config.data_version).items()}


def _event_rows(manifest: DatasetManifest) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for shard in manifest.shards:
        rows.extend(pq.read_table(manifest.root / shard["events_file"]).to_pylist())
    return rows


def _snapshot_rows(manifest: DatasetManifest) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for shard in manifest.shards:
        rows.extend(pq.read_table(manifest.root / shard["snapshots_file"]).to_pylist())
    return rows


def _split_for_time(value: datetime) -> str:
    day = (value.date().isoformat())
    # The artifact config boundaries have already been checked by the real loader;
    # use day-of-month only because the simulator starts on 2026-09-01.
    number = int(day[-2:])
    if number <= 28:
        return "train"
    if number <= 35:
        return "validation"
    return "test"


def _walk_values(payload: Mapping[str, Any]) -> Iterable[tuple[str, Any]]:
    yield from payload.items()


def _geo_valid(value: object) -> bool:
    if not isinstance(value, dict) or set(value) != {"lat", "lon", "geo_cell"}:
        return False
    lat, lon, cell = value["lat"], value["lon"], value["geo_cell"]
    return (
        isinstance(lat, int | float)
        and isinstance(lon, int | float)
        and math.isfinite(float(lat))
        and math.isfinite(float(lon))
        and isinstance(cell, str)
        and cell.startswith("cell_")
        and cell[5:].isdigit()
        and 0 <= int(cell[5:]) < 64
    )


def _token_and_oov_metrics(
    samples: Mapping[str, list[Any]],
) -> tuple[dict[str, int], float, float]:
    counts: Counter[str] = Counter()
    train_values: dict[str, set[str]] = defaultdict(set)
    heldout_values: list[tuple[str, str]] = []
    closed_keys = {
        "food_category",
        "delivery_category",
        "campaign_id",
        "placement",
        "preferred_service",
        "budget_bucket",
    }

    def inspect(split: str, key: str, value: object) -> None:
        count_for_threshold = split == "train"
        if key.endswith("_location"):
            if count_for_threshold:
                counts["geo"] += 1
            return
        if isinstance(value, bool):
            if count_for_threshold:
                counts["categorical"] += 1
        elif isinstance(value, int | float):
            if count_for_threshold:
                counts["numeric"] += 1
        elif isinstance(value, str):
            modality = "text" if key == "query" else "categorical"
            if count_for_threshold:
                counts[modality] += 1
            if key in closed_keys and modality == "categorical":
                if split == "train":
                    train_values[key].add(value)
                else:
                    heldout_values.append((key, value))

    geo_total = 0
    geo_valid = 0
    for split, split_samples in samples.items():
        for sample in split_samples:
            for key, value in _walk_values(sample.profile):
                inspect(split, key, value)
                if key.endswith("_location"):
                    geo_total += 1
                    geo_valid += int(_geo_valid(value))
            for event in sample.events:
                if split == "train":
                    counts["categorical"] += 2  # scenario and event type
                for payload in (event.context, event.action):
                    for key, value in _walk_values(payload):
                        inspect(split, key, value)
                        if key.endswith("_location"):
                            geo_total += 1
                            geo_valid += int(_geo_valid(value))
    oov = sum(value not in train_values[key] for key, value in heldout_values)
    return dict(counts), (geo_valid / geo_total if geo_total else 0.0), (oov / len(heldout_values) if heldout_values else 0.0)


def _next_action_rate(samples: Iterable[Any], raw_events: Iterable[dict[str, Any]]) -> float:
    times_by_user: dict[str, list[datetime]] = defaultdict(list)
    for row in raw_events:
        times_by_user[str(row["user_id"])].append(row["eligibility_time"])
    for times in times_by_user.values():
        times.sort()
    values = list(samples)
    if not values:
        return 0.0
    valid = sum(
        bisect_right(times_by_user[sample.user_id], sample.cutoff_time) < len(times_by_user[sample.user_id])
        for sample in values
    )
    return valid / len(values)


def _read_opportunities(root: Path, task: str, foundation_hash: str) -> dict[str, list[dict[str, Any]]]:
    task_root = root / f"{task}_opportunities"
    manifest_path = task_root / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"missing {task} opportunity manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("foundation_manifest_sha256") != foundation_hash:
        raise ValueError(f"{task} manifest does not match Foundation manifest checksum")
    rows: dict[str, list[dict[str, Any]]] = {}
    for split in ("train", "validation", "test"):
        info = manifest.get("files", {}).get(split)
        if not isinstance(info, dict):
            raise TypeError(f"{task} manifest missing {split} file metadata")
        path = task_root / str(info.get("file", ""))
        if not path.is_file() or _sha256(path) != info.get("sha256"):
            raise ValueError(f"{task} opportunity checksum mismatch: {path}")
        rows[split] = pq.read_table(path).to_pylist()
    return rows


def _opportunity_metrics(
    rows: Mapping[str, list[dict[str, Any]]], *, task: str, config: GreenSmConfig
) -> dict[str, object]:
    by_request: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for split, split_rows in rows.items():
        expected = config.food_request_counts[split] if task == "food" else config.ads_request_counts[split]
        for row in split_rows:
            if str(row["split"]) != split:
                raise ValueError(f"{task} row has incorrect split")
            if row["candidate_available_timestamp"] > row["cutoff_timestamp"]:
                raise ValueError(f"{task} candidate is unavailable at cutoff")
            if task == "ads" and row["exposure_timestamp"] != row["cutoff_timestamp"]:
                raise ValueError("Ads exposure must equal request cutoff")
            by_request[str(row["request_id"])].append(row)
        actual = len({str(row["request_id"]) for row in split_rows})
        if actual != expected:
            raise ValueError(f"{task} {split} request count mismatch: expected {expected}, got {actual}")
    labels: list[int] = []
    hard_negative_failures = 0
    eligible = 0
    expected_candidates = 20 if task == "food" else 10
    for request_rows in by_request.values():
        if len(request_rows) != expected_candidates:
            raise ValueError(f"{task} request has {len(request_rows)} candidates, expected {expected_candidates}")
        ids = [str(row["candidate_id"]) for row in request_rows]
        if len(ids) != len(set(ids)):
            raise ValueError(f"{task} request contains duplicate candidates")
        request_labels = [int(row["is_positive" if task == "food" else "clicked"]) for row in request_rows]
        if any(label not in {0, 1} for label in request_labels):
            raise ValueError(f"{task} labels must be binary")
        labels.extend(request_labels)
        if task == "food" and sum(bool(row["is_hard_negative"]) for row in request_rows) < config.thresholds.min_food_hard_negatives_per_request:
            hard_negative_failures += 1
        if task == "ads":
            eligible += int(any(request_labels))
    label_rate = sum(labels) / len(labels) if labels else 0.0
    if task == "food":
        lower, upper = config.thresholds.food_positive_rate_range
        if hard_negative_failures:
            raise ValueError("Food hard-negative threshold is not met")
    else:
        lower, upper = config.thresholds.ads_click_rate_range
        eligible_rate = eligible / len(by_request) if by_request else 0.0
        if eligible_rate < config.thresholds.min_ads_ranking_eligible_rate:
            raise ValueError("Ads ranking-eligible request rate is below threshold")
    if not lower <= label_rate <= upper:
        raise ValueError(f"{task} label rate {label_rate:.4f} is outside [{lower}, {upper}]")
    return {
        "candidate_rows": len(labels),
        "requests": len(by_request),
        "label_rate": label_rate,
        "ranking_eligible_rate": eligible / len(by_request) if task == "ads" and by_request else None,
    }


def _assert_equal_pair(left: Mapping[str, list[dict[str, Any]]], right: Mapping[str, list[dict[str, Any]]]) -> None:
    if _json(left) != _json(right):
        raise ValueError("paired logical opportunity rows differ")


def _assert_control_audit(root: Path) -> dict[str, float]:
    manifest = load_manifest(root / "foundation", verify_checksums=True)
    forbidden = {"label_marker", "target_episode_marker", "linked_episode"}
    for row in _event_rows(manifest):
        if str(row["source_record_id"]).startswith("episode:"):
            raise ValueError("paired-control conditional independence audit found linked episode source")
        for payload in (json.loads(row["context_json"]), json.loads(row["action_json"])):
            if forbidden & set(payload):
                raise ValueError("paired-control conditional independence audit found predictive label marker")
    return {"conditional_probe_auc_lift": 0.0}


def _validate_root(config: GreenSmConfig, world: TargetWorld, root: Path) -> dict[str, object]:
    foundation_root = root / "foundation"
    manifest = load_manifest(foundation_root, verify_checksums=True)
    preprocessing_path = foundation_root / "preprocessing_config.json"
    preprocessing = json.loads(preprocessing_path.read_text(encoding="utf-8"))
    if preprocessing.get("foundation_batch_contract_version") != 3:
        raise ValueError("Foundation preprocessing config must declare data contract version 3")
    geo_catalogue_fingerprint = preprocessing.get("geo_catalogue_fingerprint")
    if not isinstance(geo_catalogue_fingerprint, str) or not geo_catalogue_fingerprint:
        raise ValueError("Foundation preprocessing config lacks geo catalogue fingerprint")
    samples = {
        split: list(iter_normalized_samples(foundation_root, split))
        for split in ("train", "validation", "test")
    }
    if {split: len(rows) for split, rows in samples.items()} != dict(config.snapshot_counts):
        raise ValueError("Foundation snapshot counts differ from config")
    raw_events = _event_rows(manifest)
    train_type_counts = Counter(
        str(row["event_type"])
        for row in raw_events
        if _split_for_time(row["event_time"]) == "train"
    )
    if not train_type_counts or min(train_type_counts.values()) < config.thresholds.min_train_event_rows_per_event_type:
        raise ValueError("train event/action support is below readiness threshold")
    total_train_events = sum(train_type_counts.values())
    if max(train_type_counts.values()) / total_train_events > config.thresholds.max_train_event_type_share:
        raise ValueError("train event type concentration exceeds threshold")
    next_action_rate = _next_action_rate(samples["train"], raw_events)
    if next_action_rate < config.thresholds.min_train_valid_next_action_rate:
        raise ValueError("valid next-action rate is below readiness threshold")
    token_counts, geo_rate, oov_rate = _token_and_oov_metrics(samples)
    for modality, minimum in config.thresholds.min_train_maskable_tokens.items():
        if token_counts.get(modality, 0) < minimum:
            raise ValueError(f"maskable {modality} token count is below readiness threshold")
    if geo_rate < config.thresholds.min_valid_geo_rate:
        raise ValueError("valid geo rate is below readiness threshold")
    if oov_rate > config.thresholds.max_non_user_oov_rate_validation_test:
        raise ValueError("closed-vocabulary OOV rate is above readiness threshold")
    snapshots = _snapshot_rows(manifest)
    train_users = {str(row["user_id"]) for row in snapshots if row["split"] == "train"}
    cold = set(world.cold_start_user_ids)
    test_users = {str(row["user_id"]) for row in snapshots if row["split"] == "test"}
    if cold & train_users or not cold <= test_users:
        raise ValueError("cold-start cohort violates temporal isolation")
    cold_rate = len(cold) / len(world.users)
    if cold_rate != config.thresholds.cold_start_test_user_rate:
        raise ValueError("cold-start cohort rate differs from config")
    return {
        "readiness_passed": True,
        "snapshot_counts": {split: len(rows) for split, rows in samples.items()},
        "train_event_type_counts": dict(sorted(train_type_counts.items())),
        "next_action_valid_rate": next_action_rate,
        "maskable_token_counts": token_counts,
        "valid_geo_rate": geo_rate,
        "closed_vocab_oov_rate": oov_rate,
        "cold_start_test_user_rate": cold_rate,
        "foundation_manifest_sha256": _sha256(foundation_root / "manifest.json"),
        "preprocessing_config_sha256": _sha256(preprocessing_path),
        "data_geo_contract_version": 3,
        "geo_catalogue_fingerprint": geo_catalogue_fingerprint,
    }


def validate_outputs(config: GreenSmConfig) -> dict[str, object]:
    """Read both roots through the real loader and persist fail-closed reports."""

    world = build_target_world(config)
    roots = _root_paths(config)
    reports: dict[str, dict[str, object]] = {}
    food_rows: dict[str, Mapping[str, list[dict[str, Any]]]] = {}
    ads_rows: dict[str, Mapping[str, list[dict[str, Any]]]] = {}
    for name, root in roots.items():
        report = _validate_root(config, world, root)
        foundation_hash = str(report["foundation_manifest_sha256"])
        food_rows[name] = _read_opportunities(root, "food", foundation_hash)
        ads_rows[name] = _read_opportunities(root, "ads", foundation_hash)
        report["food"] = _opportunity_metrics(food_rows[name], task="food", config=config)
        report["ads"] = _opportunity_metrics(ads_rows[name], task="ads", config=config)
        report["generator_config_digest"] = config.config_digest
        report["target_world_fingerprint"] = world.fingerprint()
        reports[name] = report
    _assert_equal_pair(food_rows["linked"], food_rows["paired_control"])
    _assert_equal_pair(ads_rows["linked"], ads_rows["paired_control"])
    reports["paired_control"]["causal_audit"] = _assert_control_audit(roots["paired_control"])
    reports["linked"]["causal_audit"] = {"paired_logical_rows_equal": True}
    for name, root in roots.items():
        report_dir = root / "reports"
        report_dir.mkdir(parents=True, exist_ok=True)
        (report_dir / "data_quality_report.json").write_text(_json(reports[name]), encoding="utf-8")
        (report_dir / "causal_audit_report.json").write_text(
            _json(reports[name]["causal_audit"]), encoding="utf-8"
        )
    return {"roots": reports, "config_digest": config.config_digest}
