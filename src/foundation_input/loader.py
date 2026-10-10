"""Read versioned pretraining Parquet shards into canonical samples."""

from __future__ import annotations

import json
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from .manifest import DatasetManifest, ManifestError, load_manifest, validate_parquet_schema
from .types import CanonicalSample, EventRecord


def _as_datetime(value: Any, *, field: str) -> datetime:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as error:
            raise ManifestError(f"Invalid {field} timestamp: {value!r}") from error
    raise ManifestError(f"Invalid {field} value: {value!r}")


def _parse_object(
    value: Any, *, field: str, identity: str, allow_null: bool = False
) -> dict[str, Any]:
    if not isinstance(value, str):
        raise ManifestError(f"{identity}: {field} must be a JSON string")
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as error:
        raise ManifestError(f"{identity}: malformed {field}: {error.msg}") from error
    if parsed is None and allow_null:
        return {}
    if not isinstance(parsed, dict):
        raise ManifestError(f"{identity}: {field} must decode to an object")
    return parsed


def _event_from_row(row: dict[str, Any]) -> EventRecord:
    event_id = row.get("event_id")
    user_id = row.get("user_id")
    identity = f"event_id={event_id!r}, user_id={user_id!r}"
    required = (
        "event_id",
        "source_record_id",
        "source_record_index",
        "action_index",
        "source_action_count",
        "event_time",
        "time_precision",
        "eligibility_time",
        "eligibility_inclusive",
        "scenario",
        "event_type",
        "context_json",
        "action_json",
    )
    missing = [field for field in required if field not in row]
    if missing:
        raise ManifestError(f"{identity}: missing event fields {missing}")
    return EventRecord(
        event_id=str(event_id),
        user_id=None if user_id is None else str(user_id),
        source_record_id=str(row["source_record_id"]),
        source_record_index=int(row["source_record_index"]),
        action_index=int(row["action_index"]),
        source_action_count=int(row["source_action_count"]),
        event_time=_as_datetime(row["event_time"], field=f"{identity}.event_time"),
        time_precision=str(row["time_precision"]),
        eligibility_time=_as_datetime(
            row["eligibility_time"], field=f"{identity}.eligibility_time"
        ),
        eligibility_inclusive=bool(row["eligibility_inclusive"]),
        scenario=None if row["scenario"] is None else str(row["scenario"]),
        event_type=None if row["event_type"] is None else str(row["event_type"]),
        context=_parse_object(
            row["context_json"], field="context_json", identity=identity, allow_null=True
        ),
        action=_parse_object(
            row["action_json"], field="action_json", identity=identity, allow_null=True
        ),
    )


def _load_user_events(manifest: DatasetManifest, shard: dict[str, Any]) -> dict[str, EventRecord]:
    event_path = manifest.root / shard["events_file"]
    validate_parquet_schema(event_path, kind="events")
    try:
        rows = pq.read_table(event_path).to_pylist()
    except Exception as error:
        raise ManifestError(f"Cannot read events shard {event_path}: {error}") from error
    events: dict[str, EventRecord] = {}
    for row in rows:
        event = _event_from_row(row)
        if event.event_id in events:
            raise ManifestError(f"Duplicate event_id in {event_path}: {event.event_id!r}")
        if row.get("user_id") != shard["user_id"]:
            raise ManifestError(
                f"Event user mismatch in {event_path}: expected {shard['user_id']!r}, "
                f"got {row.get('user_id')!r}"
            )
        events[event.event_id] = event
    return events


def _sample_from_row(
    row: dict[str, Any],
    *,
    manifest: DatasetManifest,
    shard: dict[str, Any],
    events: dict[str, EventRecord],
) -> CanonicalSample:
    sample_id = row.get("sample_id")
    identity = f"sample_id={sample_id!r}"
    if row.get("user_id") != shard["user_id"]:
        raise ManifestError(
            f"{identity}: snapshot user mismatch; expected {shard['user_id']!r}, "
            f"got {row.get('user_id')!r}"
        )
    history_ids = row.get("history_event_ids")
    if not isinstance(history_ids, list):
        raise ManifestError(f"{identity}: history_event_ids must be a list")
    history: list[EventRecord] = []
    for event_id in history_ids:
        if event_id not in events:
            raise ManifestError(
                f"{identity}: history_event_ids references missing event {event_id!r}"
            )
        history.append(events[event_id])
    history.sort(
        key=lambda event: (
            event.eligibility_time.replace(tzinfo=None),
            int(event.eligibility_inclusive),
            event.source_record_index,
            event.action_index,
        )
    )
    profile = _parse_object(row.get("profile_json"), field="profile_json", identity=identity)
    split = row.get("split")
    if split not in {"train", "validation", "test"}:
        raise ManifestError(f"{identity}: unsupported split {split!r}")
    return CanonicalSample(
        sample_id=str(sample_id),
        dataset_name=manifest.dataset_name,
        split=str(split),
        user_id=str(row["user_id"]),
        cutoff_time=_as_datetime(row["cutoff_time"], field=f"{identity}.cutoff_time"),
        cutoff_inclusive=bool(row["cutoff_inclusive"]),
        history_order_known=bool(row["history_order_known"]),
        history_truncated=bool(row["history_truncated"]),
        history_group_truncated=bool(row["history_group_truncated"]),
        profile=profile,
        events=tuple(history),
    )


def iter_normalized_samples(
    dataset_root: str | Path,
    split: str,
    *,
    validate: bool = True,
) -> Iterator[CanonicalSample]:
    """Yield canonical samples from one logical split without modifying source data."""

    if split not in {"train", "validation", "test"}:
        raise ValueError(f"Invalid split: {split!r}")
    manifest = load_manifest(dataset_root, validate_shard_files=False)
    for shard in manifest.shards:
        if not shard["snapshots_by_split"].get(split, 0):
            continue
        snapshot_path = manifest.root / shard["snapshots_file"]
        validate_parquet_schema(snapshot_path, kind="snapshots")
        events = _load_user_events(manifest, shard)
        try:
            snapshot_rows = pq.read_table(snapshot_path).to_pylist()
        except Exception as error:
            raise ManifestError(f"Cannot read snapshots shard {snapshot_path}: {error}") from error
        for row in snapshot_rows:
            if row.get("split") != split:
                continue
            sample = _sample_from_row(row, manifest=manifest, shard=shard, events=events)
            if validate:
                from .validation import validate_sample

                validate_sample(sample)
            yield sample
