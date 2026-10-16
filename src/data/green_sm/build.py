"""Immutable Parquet emission for paired Green SM Foundation artifacts."""

from __future__ import annotations

import bisect
import hashlib
import json
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from datetime import datetime, timedelta
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from .config import GreenSmConfig
from .histories import EventGroup, HistoryView, build_history_views
from .world import (
    SIMULATION_START,
    LogicalRequest,
    SimulatedUser,
    TargetWorld,
    _rng,
    build_target_world,
)

SCHEMA_VERSION = "omnibehavior_pretraining_v2"
_DATA_VERSION = re.compile(r"^green_sm_sim_(?:pilot|full)_v(?P<version>[1-9][0-9]*)$")


def root_names(data_version: str) -> dict[str, str]:
    """Derive immutable paired root names from the declared corpus version."""

    matched = _DATA_VERSION.fullmatch(data_version)
    if matched is None:
        raise ValueError(
            "data_version must be green_sm_sim_pilot_vN or green_sm_sim_full_vN"
        )
    version = matched.group("version")
    return {
        "linked": f"green_sm_sim_linked_v{version}",
        "paired_control": f"green_sm_sim_control_paired_v{version}",
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _json(value: object) -> str:
    return json.dumps(value, default=lambda item: item.isoformat(), sort_keys=True, separators=(",", ":"))


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_json(payload), encoding="utf-8")


def _write_parquet(rows: Iterable[Mapping[str, object]], path: Path) -> None:
    materialized = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(materialized), path, compression="zstd")


def _safe_user_file(user_id: str) -> str:
    return hashlib.sha256(user_id.encode("utf-8")).hexdigest()[:24]


def _empty_or_missing(path: Path) -> bool:
    return not path.exists() or (path.is_dir() and not any(path.iterdir()))


def _split_start_day(split: str) -> tuple[int, int]:
    return {"train": (1, 28), "validation": (29, 35), "test": (36, 42)}[split]


def _geo_for_user(world: TargetWorld, cell_id: str) -> dict[str, object]:
    cell = world.catalogue.cell(cell_id)
    return {"lat": cell.lat, "lon": cell.lon, "geo_cell": cell.cell_id}


def _profile(world: TargetWorld, user: SimulatedUser, cutoff: datetime) -> dict[str, object]:
    return {
        "as_of_time": cutoff.isoformat(),
        "home_location": _geo_for_user(world, user.home_cell),
        "work_location": _geo_for_user(world, user.work_cell),
        "budget_bucket": user.budget_bucket,
        "preferred_service": user.preferred_service,
        "active_days_so_far": max(0, (cutoff.date() - SIMULATION_START.date()).days + 1 - user.first_active_day),
    }


def _event_row(group: EventGroup) -> dict[str, object]:
    return {
        "event_id": f"{group.source_record_id}:0",
        "user_id": group.user_id,
        "source_record_id": group.source_record_id,
        "source_record_index": group.source_record_index,
        "action_index": 0,
        "source_action_count": 1,
        "event_time": group.event_time,
        "time_precision": "second",
        "eligibility_time": group.event_time,
        "eligibility_inclusive": True,
        "scenario": group.scenario,
        "event_type": group.event_type,
        "context_json": _json(dict(group.context)),
        "action_json": _json(dict(group.action)),
    }


def _request_specs(world: TargetWorld) -> tuple[dict[str, list[tuple[str, str, datetime]]], dict[str, str]]:
    specs: dict[str, list[tuple[str, str, datetime]]] = {split: [] for split in ("train", "validation", "test")}
    sample_for_request: dict[str, str] = {}
    for request in sorted(world.all_requests(), key=lambda item: (item.cutoff_time, item.request_id)):
        sample_id = f"foundation:{request.request_id}"
        sample_for_request[request.request_id] = sample_id
        specs[request.split].append((sample_id, request.user_id, request.cutoff_time))
    return specs, sample_for_request


def _extra_specs(
    config: GreenSmConfig,
    world: TargetWorld,
    specs: dict[str, list[tuple[str, str, datetime]]],
) -> None:
    for split in ("train", "validation", "test"):
        target = config.snapshot_counts[split]
        remaining = target - len(specs[split])
        if remaining < 0:
            raise ValueError(f"{split} request count exceeds configured snapshot count")
        start_day, end_day = _split_start_day(split)
        days = end_day - start_day + 1
        existing_users = {user_id for _, user_id, _ in specs[split]}
        required_cold = (
            [user for user in world.users if user.user_id in world.cold_start_user_ids and user.user_id not in existing_users]
            if split == "test"
            else []
        )
        if len(required_cold) > remaining:
            raise ValueError("test snapshot budget cannot cover every cold-start user")
        for ordinal in range(remaining):
            if ordinal < len(required_cold):
                user = required_cold[ordinal]
                day = max(start_day + ordinal % days, user.first_active_day)
            else:
                day = start_day + ordinal % days
                eligible = [user for user in world.users if user.first_active_day <= day]
                user = eligible[_rng(config.seed, "extra-snapshot", split, ordinal).randrange(len(eligible))]
            # Seconds avoid accidental collision with request cutoffs while preserving
            # deterministic time order under the same daily volume at full scale.
            cutoff = SIMULATION_START + timedelta(days=day - 1, hours=12, seconds=ordinal // days)
            specs[split].append((f"snapshot:{split}:{ordinal:06d}", user.user_id, cutoff))
        specs[split].sort(key=lambda item: (item[2], item[0]))


def _estimate_tokens(profile: Mapping[str, object], groups: list[EventGroup]) -> int:
    # CLS + profile semantic tokens + (scenario/event type + fields) for each
    # group. This mirrors the published v3 semantic shape without claiming the
    # current v2 tokenizer can route geo tensors yet.
    return 1 + len(profile) + sum(2 + group.semantic_field_count for group in groups)


def _snapshot_rows(
    world: TargetWorld,
    view: HistoryView,
    user: SimulatedUser,
    specs: Iterable[tuple[str, str, datetime]],
    max_history_events: int,
) -> tuple[list[dict[str, object]], int]:
    groups = view.events_for(user.user_id)
    event_ids = [f"{group.source_record_id}:0" for group in groups]
    times = [group.event_time for group in groups]
    rows: list[dict[str, object]] = []
    maximum = 0
    for sample_id, _, cutoff in specs:
        stop = bisect.bisect_left(times, cutoff)
        start = max(0, stop - max_history_events)
        selected = list(groups[start:stop])
        profile = _profile(world, user, cutoff)
        maximum = max(maximum, _estimate_tokens(profile, selected))
        split = _split_for_cutoff(cutoff)
        rows.append(
            {
                "sample_id": sample_id,
                "user_id": user.user_id,
                "cutoff_time": cutoff,
                "cutoff_inclusive": False,
                "split": split,
                "history_start": start,
                "history_stop": stop,
                "history_event_ids": event_ids[start:stop],
                "history_order_known": True,
                "history_truncated": start > 0 and stop - start >= max_history_events,
                "history_group_truncated": False,
                "profile_json": _json(profile),
            }
        )
    return rows, maximum


def _split_for_cutoff(cutoff: datetime) -> str:
    day = (cutoff.date() - SIMULATION_START.date()).days + 1
    if day <= 28:
        return "train"
    if day <= 35:
        return "validation"
    return "test"


def _foundation_config(config: GreenSmConfig, world: TargetWorld, view_name: str) -> dict[str, object]:
    geo_catalogue = {
        cell.cell_id: {"lat": cell.lat, "lon": cell.lon}
        for cell in world.catalogue.cells
    }
    geo_catalogue_fingerprint = hashlib.sha256(
        _json(geo_catalogue).encode("utf-8")
    ).hexdigest()
    geo_fields = [
        "profile.home_location",
        "profile.work_location",
        "context.pickup_location",
        "context.dropoff_location",
        "context.merchant_location",
        "context.delivery_location",
        "context.ad_context_location",
    ]
    tokenizer_contract = {
        "profile_allowlist": [
            "home_location",
            "work_location",
            "budget_bucket",
            "preferred_service",
            "active_days_so_far",
        ],
        "context_allowlist": {
            "Ride": ["pickup_location", "dropoff_location"],
            "Food": ["merchant_location", "food_category"],
            "Delivery": ["delivery_location", "delivery_category"],
            "Ads": ["ad_context_location", "placement"],
        },
        "geo_field_registry": geo_fields,
        "geo_catalogue": geo_catalogue,
        "geo_catalogue_fingerprint": geo_catalogue_fingerprint,
        "coordinate_tolerance": 1e-6,
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "foundation_batch_contract_version": 3,
        "data_version": config.data_version,
        "source_dataset": f"green_sm_{view_name}",
        "timezone": config.timezone,
        "config_sha256": config.config_digest,
        "target_world_fingerprint": world.fingerprint(),
        "max_history_events": config.max_history_events,
        "max_tokens": config.max_tokens,
        "geo_field_registry": geo_fields,
        "geo_catalogue_fingerprint": geo_catalogue_fingerprint,
        "tokenizer_contract": tokenizer_contract,
        "splits": {
            "start_inclusive": SIMULATION_START.isoformat(),
            "train_end_exclusive": (SIMULATION_START + timedelta(days=28)).isoformat(),
            "validation_end_exclusive": (SIMULATION_START + timedelta(days=35)).isoformat(),
            "test_end_exclusive": (SIMULATION_START + timedelta(days=42)).isoformat(),
        },
    }


def _write_foundation(
    config: GreenSmConfig,
    world: TargetWorld,
    view: HistoryView,
    foundation_root: Path,
    specs: Mapping[str, list[tuple[str, str, datetime]]],
) -> tuple[Path, int]:
    by_user: dict[str, list[tuple[str, str, datetime]]] = defaultdict(list)
    for split_specs in specs.values():
        for spec in split_specs:
            by_user[spec[1]].append(spec)
    shards: list[dict[str, object]] = []
    max_estimated_tokens = 0
    for user in world.users:
        groups = view.events_for(user.user_id)
        event_rows = [_event_row(group) for group in groups]
        snapshots, user_max = _snapshot_rows(
            world, view, user, by_user[user.user_id], config.max_history_events
        )
        max_estimated_tokens = max(max_estimated_tokens, user_max)
        stem = _safe_user_file(user.user_id)
        events_relative = f"events/{stem}.parquet"
        snapshots_relative = f"snapshots/{stem}.parquet"
        events_path = foundation_root / events_relative
        snapshots_path = foundation_root / snapshots_relative
        _write_parquet(event_rows, events_path)
        _write_parquet(snapshots, snapshots_path)
        split_counts = Counter(str(row["split"]) for row in snapshots)
        shards.append(
            {
                "user_id": user.user_id,
                "events_file": events_relative,
                "snapshots_file": snapshots_relative,
                "events_sha256": _sha256(events_path),
                "snapshots_sha256": _sha256(snapshots_path),
                "event_rows": len(event_rows),
                "snapshot_rows": len(snapshots),
                "snapshots_by_split": {split: split_counts.get(split, 0) for split in ("train", "validation", "test")},
            }
        )
    preprocessing = _foundation_config(config, world, view.name)
    preprocessing_path = foundation_root / "preprocessing_config.json"
    _write_json(preprocessing_path, preprocessing)
    _write_json(
        foundation_root / "manifest.json",
        {
            "status": "complete",
            "schema_version": SCHEMA_VERSION,
            "source_dataset": f"green_sm_{view.name}",
            "config_sha256": _sha256(preprocessing_path),
            "shards": shards,
        },
    )
    return foundation_root, max_estimated_tokens


def _opportunity_rows(
    requests: Iterable[LogicalRequest], sample_for_request: Mapping[str, str], *, task: str
) -> dict[str, list[dict[str, object]]]:
    rows = {split: [] for split in ("train", "validation", "test")}
    for request in requests:
        for candidate in request.candidates:
            row: dict[str, object] = {
                "request_id": request.request_id,
                "user_id": request.user_id,
                "foundation_sample_id": sample_for_request[request.request_id],
                "cutoff_timestamp": request.cutoff_time,
                "candidate_id": candidate.candidate_id,
                "split": request.split,
                "candidate_available_timestamp": candidate.available_at,
                "candidate_category": candidate.category,
                "candidate_geo_cell": candidate.geo_cell,
                "candidate_lat": candidate.lat,
                "candidate_lon": candidate.lon,
            }
            if task == "food":
                row.update(
                    {
                        "is_positive": candidate.label,
                        "is_hard_negative": candidate.is_hard_negative,
                        "item_price": candidate.price,
                        "merchant_id": candidate.merchant_id,
                    }
                )
            else:
                row.update(
                    {
                        "clicked": candidate.label,
                        "campaign_id": candidate.candidate_id,
                        "placement": candidate.placement,
                        "exposure_timestamp": request.cutoff_time,
                    }
                )
            rows[request.split].append(row)
    return rows


def _write_opportunities(
    root: Path,
    rows: Mapping[str, list[dict[str, object]]],
    *,
    task: str,
    foundation_manifest_sha256: str,
) -> None:
    task_root = root / f"{task}_opportunities"
    files: dict[str, object] = {}
    for split in ("train", "validation", "test"):
        path = task_root / f"{task}_opportunities.{split}.parquet"
        _write_parquet(rows[split], path)
        files[split] = {"file": path.name, "sha256": _sha256(path), "rows": len(rows[split])}
    _write_json(
        task_root / "manifest.json",
        {
            "schema_version": f"green_sm_{task}_opportunities_v1",
            "foundation_manifest_sha256": foundation_manifest_sha256,
            "files": files,
        },
    )


def _logical_rows_equal(
    left: Mapping[str, list[dict[str, object]]], right: Mapping[str, list[dict[str, object]]]
) -> bool:
    return _json(left) == _json(right)


def build_dataset(config: GreenSmConfig) -> dict[str, object]:
    """Build two immutable paired roots; never overwrite a previous result."""

    if not _empty_or_missing(config.output_root):
        raise ValueError(f"refusing to overwrite non-empty output root: {config.output_root}")
    config.output_root.mkdir(parents=True, exist_ok=True)
    world = build_target_world(config)
    linked, control = build_history_views(world)
    specs, sample_for_request = _request_specs(world)
    _extra_specs(config, world, specs)
    expected = {split: config.snapshot_counts[split] for split in ("train", "validation", "test")}
    observed = {split: len(specs[split]) for split in expected}
    if observed != expected:
        raise ValueError(f"snapshot construction count mismatch: expected {expected}, got {observed}")

    food_rows = _opportunity_rows(world.food_requests, sample_for_request, task="food")
    ads_rows = _opportunity_rows(world.ads_requests, sample_for_request, task="ads")
    roots: dict[str, str] = {}
    max_tokens = 0
    names = root_names(config.data_version)
    for name, view in (("linked", linked), ("paired_control", control)):
        root = config.output_root / names[name]
        foundation_root, maximum = _write_foundation(config, world, view, root / "foundation", specs)
        max_tokens = max(max_tokens, maximum)
        manifest_sha256 = _sha256(foundation_root / "manifest.json")
        _write_opportunities(root, food_rows, task="food", foundation_manifest_sha256=manifest_sha256)
        _write_opportunities(root, ads_rows, task="ads", foundation_manifest_sha256=manifest_sha256)
        roots[name] = str(foundation_root)
    if max_tokens > config.max_tokens:
        raise ValueError(f"generated sample exceeds max_tokens={config.max_tokens}: {max_tokens}")
    return {
        "foundation_roots": roots,
        "paired_logical_rows_equal": _logical_rows_equal(food_rows, food_rows)
        and _logical_rows_equal(ads_rows, ads_rows),
        "max_estimated_token_count": max_tokens,
        "target_world_fingerprint": world.fingerprint(),
        "config_digest": config.config_digest,
    }
