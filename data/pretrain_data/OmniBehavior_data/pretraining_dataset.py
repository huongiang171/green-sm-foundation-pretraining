"""Build and read the versioned OmniBehavior pretraining dataset.

Events are observations, including negative action states. Eligibility order is
an indexing convention; interval metadata preserves uncertain temporal order.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from collections import Counter, deque
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from itertools import groupby
from pathlib import Path
from time import perf_counter
from typing import Iterator

import pyarrow as pa
import pyarrow.parquet as pq


SCHEMA_VERSION = "omnibehavior_pretraining_v2"
MAX_EVENTS = 30
BATCH_SIZE = 4096
SPLITS = ("train", "validation", "test")
START = datetime(2025, 9, 1)
TRAIN_END = datetime(2025, 11, 1)
VALIDATION_END = datetime(2025, 11, 16)
TEST_END = datetime(2025, 12, 1)
CONTEXT_ALLOWLIST = {
    "Advertisement": (
        "product", "industry_main", "industry_sub", "caption", "ocr_text", "asr_text",
    ),
    "Live Streaming": (
        "live_category", "live_title", "is_shop_live", "live_cover_content",
    ),
    "Video Browsing": ("caption", "photo_type", "duration", "ocr", "asr"),
    "E-commerce": (
        "product_source", "product_id", "item_title", "item_desc", "item_price",
        "external_item_title", "category_level1_name", "category_level2_name",
        "category_level3_name", "industry_level1_name", "industry_level2_name",
        "industry_level3_name",
    ),
    "Customer Service": (
        "ticket_category", "product_name", "item_price", "product_category_info",
    ),
    "Search Behavior": (),
}
SCENARIO_PROFILE_FIELDS = {
    "Advertisement": "advertisement_event_count_30d",
    "Live Streaming": "live_streaming_event_count_30d",
    "Video Browsing": "video_browsing_event_count_30d",
    "E-commerce": "e_commerce_event_count_30d",
    "Customer Service": "customer_service_event_count_30d",
    "Search Behavior": "search_behavior_event_count_30d",
}

_METADATA = {b"schema_version": SCHEMA_VERSION.encode()}
EVENT_SCHEMA = pa.schema([
    ("row_index", pa.int32()), ("event_id", pa.string()), ("user_id", pa.string()),
    ("source_record_id", pa.string()), ("source_record_index", pa.int32()),
    ("action_index", pa.int32()), ("source_action_count", pa.int32()),
    ("source_timestamp", pa.string()), ("event_time", pa.string()),
    ("time_precision", pa.string()), ("time_lower_bound", pa.timestamp("us")),
    ("time_upper_bound", pa.timestamp("us")), ("upper_bound_exclusive", pa.bool_()),
    ("eligibility_time", pa.timestamp("us")), ("eligibility_inclusive", pa.bool_()),
    ("scenario", pa.string()), ("event_type", pa.string()),
    ("context_json", pa.large_string()), ("action_json", pa.large_string()),
], metadata=_METADATA)
SNAPSHOT_SCHEMA = pa.schema([
    ("sample_id", pa.string()), ("user_id", pa.string()),
    ("cutoff_time", pa.timestamp("us")), ("cutoff_inclusive", pa.bool_()),
    ("split", pa.string()), ("history_start", pa.int32()), ("history_stop", pa.int32()),
    ("history_event_ids", pa.list_(pa.field("element", pa.string()))), ("history_order_known", pa.bool_()),
    ("history_truncated", pa.bool_()), ("history_group_truncated", pa.bool_()),
    ("profile_json", pa.string()),
], metadata=_METADATA)


def json_text(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False)


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)
                    + "\n", encoding="utf-8")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def iso(value: datetime) -> str:
    return value.isoformat(timespec="seconds")


def parse_event_time(value: str) -> tuple:
    if not isinstance(value, str) or not re.fullmatch(
        r"\d{4}-\d{2}-\d{2}(?:[ T]\d{2}:\d{2}:\d{2})?", value
    ):
        raise ValueError(f"Unsupported timestamp (timezone must remain unknown): {value!r}")
    parsed = datetime.fromisoformat(value)
    if len(value) == 10:
        end = parsed + timedelta(days=1)
        return value, "day", parsed, end, True, end, False
    return iso(parsed), "second", parsed, parsed, False, parsed, True


@dataclass(slots=True)
class Event:
    event_id: str
    user_id: str
    source_record_id: str
    source_record_index: int
    action_index: int
    source_action_count: int
    source_timestamp: str
    event_time: str
    time_precision: str
    time_lower_bound: datetime
    time_upper_bound: datetime
    upper_bound_exclusive: bool
    eligibility_time: datetime
    eligibility_inclusive: bool
    scenario: str
    event_type: str
    context_json: str
    action_json: str

    @property
    def coordinate(self) -> tuple[datetime, bool]:
        return self.eligibility_time, self.eligibility_inclusive

    def row(self, row_index: int) -> dict:
        return {"row_index": row_index, **{
            name: getattr(self, name) for name in self.__dataclass_fields__
        }}


class DatasetStats:
    def __init__(self):
        self.raw_records = 0
        self.date_only_records = 0
        self.events = 0
        self.date_only_events = 0
        self.scenario_records = Counter()
        self.scenario_events = Counter()
        self.action_types = Counter()
        self.context_fields = {}
        self.action_fields = {}
        self.false_flags = Counter()
        self.snapshots = Counter({split: 0 for split in SPLITS})
        self.users_by_split = {split: set() for split in SPLITS}
        self.history_lengths = Counter()
        self.uncertain_histories = 0
        self.truncated_histories = 0
        self.partial_groups = 0
        self.multi_event_eligibility_groups = 0
        self.skipped_snapshot_groups = 0

    @staticmethod
    def count_fields(store: dict, kind: str, values: dict) -> None:
        group = store.setdefault(kind, {})
        for name, value in values.items():
            item = group.setdefault(name, {"present": 0, "non_null": 0})
            item["present"] += 1
            item["non_null"] += value is not None

    def snapshot(self, row: dict) -> None:
        split = row["split"]
        self.snapshots[split] += 1
        self.users_by_split[split].add(row["user_id"])
        self.history_lengths[row["history_stop"] - row["history_start"]] += 1
        self.uncertain_histories += not row["history_order_known"]
        self.truncated_histories += row["history_truncated"]
        self.partial_groups += row["history_group_truncated"]

    def report(self) -> dict:
        context_coverage = {}
        for scenario, fields in self.context_fields.items():
            total = self.scenario_records[scenario]
            context_coverage[scenario] = {
                key: {**counts, "missing": total - counts["present"],
                      "null": counts["present"] - counts["non_null"],
                      "used_in_input": key in CONTEXT_ALLOWLIST[scenario]}
                for key, counts in sorted(fields.items())
            }
        action_coverage = {}
        for action_type, fields in self.action_fields.items():
            total = self.action_types[action_type]
            action_coverage[action_type] = {
                key: {**counts, "missing": total - counts["present"],
                      "null": counts["present"] - counts["non_null"]}
                for key, counts in sorted(fields.items())
            }
        return {
            "raw_records": self.raw_records, "action_events": self.events,
            "date_only_records": self.date_only_records,
            "date_only_action_events": self.date_only_events,
            "raw_records_by_scenario": dict(self.scenario_records),
            "action_events_by_scenario": dict(self.scenario_events),
            "action_events_by_type": dict(self.action_types),
            "context_field_coverage_per_raw_record": context_coverage,
            "action_field_coverage_per_action_event": action_coverage,
            "preserved_false_action_flags": dict(self.false_flags),
            "snapshots_by_split": dict(self.snapshots),
            "users_by_split": {key: len(value) for key, value in self.users_by_split.items()},
            "total_snapshots": sum(self.snapshots.values()),
            "history_length_distribution": dict(sorted(self.history_lengths.items())),
            "histories_with_uncertain_order": self.uncertain_histories,
            "histories_truncated_to_30": self.truncated_histories,
            "histories_with_partial_eligibility_group": self.partial_groups,
            "multi_event_eligibility_groups": self.multi_event_eligibility_groups,
            "snapshot_groups_outside_split_period": self.skipped_snapshot_groups,
            "history_empty": 0,
            "objectives": {"targets_created": False, "mask_created": False,
                           "next_action_label_validity": "not_assessed"},
        }


def normalize_user(raw: dict, user_id: str, stats: DatasetStats) -> list[Event]:
    if not re.fullmatch(r"user_\d+", user_id):
        raise ValueError(f"Invalid user ID: {user_id!r}")
    if not isinstance(raw, dict) or set(raw) != {user_id}:
        raise ValueError(f"{user_id}: expected exactly one matching user key")
    user = raw[user_id]
    if not isinstance(user, dict) or not isinstance(user.get("action_history"), list):
        raise ValueError(f"{user_id}: action_history must be a list")
    events = []
    for index, record in enumerate(user["action_history"]):
        prefix = f"{user_id}:{index}"
        if not isinstance(record, dict):
            raise ValueError(f"{prefix}: record must be an object")
        scenario = record.get("type")
        if scenario not in CONTEXT_ALLOWLIST:
            raise ValueError(f"{prefix}: unsupported scenario {scenario!r}")
        context = record.get("context")
        if context is not None and not isinstance(context, dict):
            raise ValueError(f"{prefix}: context must be an object or null")
        actions = record.get("action")
        if not isinstance(actions, list) or not actions:
            raise ValueError(f"{prefix}: action must be a nonempty list")
        try:
            timing = parse_event_time(record.get("timestamp"))
            context_json = json_text(context)
        except (ValueError, TypeError) as error:
            raise ValueError(f"{prefix}: {error}") from error
        stats.raw_records += 1
        stats.date_only_records += timing[1] == "day"
        stats.scenario_records[scenario] += 1
        stats.count_fields(stats.context_fields, scenario, context or {})
        for action_index, action in enumerate(actions):
            if not isinstance(action, dict) or not isinstance(action.get("type"), str) or not action["type"]:
                raise ValueError(f"{prefix}:{action_index}: action.type must be a nonempty string")
            action_json = json_text(action)
            events.append(Event(
                f"{prefix}:{action_index}", user_id, prefix, index, action_index,
                len(actions), record["timestamp"], *timing, scenario, action["type"],
                context_json, action_json,
            ))
            stats.events += 1
            stats.date_only_events += timing[1] == "day"
            stats.scenario_events[scenario] += 1
            stats.action_types[action["type"]] += 1
            stats.count_fields(stats.action_fields, action["type"], action)
            for key, value in action.items():
                if value is False:
                    stats.false_flags[f"{action['type']}.{key}"] += 1
    events.sort(key=lambda event: (*event.coordinate, event.source_record_index, event.action_index))
    return events


def split_for_cutoff(cutoff: datetime, inclusive: bool) -> str | None:
    coordinate = cutoff, inclusive
    if coordinate < (START, True) or coordinate > (TEST_END, False):
        return None
    if coordinate <= (TRAIN_END, False):
        return "train"
    if coordinate <= (VALIDATION_END, False):
        return "validation"
    return "test"


def order_is_known(events: list[Event]) -> bool:
    if not events:
        return True
    upper = events[0].time_upper_bound
    exclusive = events[0].upper_bound_exclusive
    for event in events[1:]:
        lower = event.time_lower_bound
        if upper > lower or (upper == lower and not exclusive):
            return False
        if event.time_upper_bound > upper:
            upper, exclusive = event.time_upper_bound, event.upper_bound_exclusive
        elif event.time_upper_bound == upper:
            exclusive = exclusive and event.upper_bound_exclusive
    return True


def iter_snapshot_rows(events: list[Event], stats: DatasetStats) -> Iterator[dict]:
    recent = deque()
    scenarios = Counter()
    days = Counter()
    stop = 0
    for coordinate, group in groupby(events, key=lambda event: event.coordinate):
        group_size = 0
        for event in group:
            recent.append(event)
            scenarios[event.scenario] += 1
            days[event.time_lower_bound.date()] += 1
            stop += 1
            group_size += 1
        stats.multi_event_eligibility_groups += group_size > 1
        cutoff, inclusive = coordinate
        lower = cutoff - timedelta(days=30), inclusive
        while recent and recent[0].coordinate <= lower:
            old = recent.popleft()
            scenarios[old.scenario] -= 1
            day = old.time_lower_bound.date()
            days[day] -= 1
            if days[day] == 0:
                del days[day]
        split = split_for_cutoff(cutoff, inclusive)
        if split is None:
            stats.skipped_snapshot_groups += 1
            continue
        start = max(0, stop - MAX_EVENTS)
        history = events[start:stop]
        profile = {
            "as_of_time": iso(cutoff), "event_count_30d": len(recent),
            "active_days_30d": len(days),
            **{field: scenarios[scenario] for scenario, field in SCENARIO_PROFILE_FIELDS.items()},
        }
        row = {
            "sample_id": f"{history[-1].user_id}:{iso(cutoff)}:{'closed' if inclusive else 'open'}",
            "user_id": history[-1].user_id, "cutoff_time": cutoff,
            "cutoff_inclusive": inclusive, "split": split,
            "history_start": start, "history_stop": stop,
            "history_event_ids": [event.event_id for event in history],
            "history_order_known": order_is_known(history),
            "history_truncated": stop > MAX_EVENTS,
            "history_group_truncated": start > 0 and events[start - 1].coordinate == history[0].coordinate,
            "profile_json": json_text(profile),
        }
        stats.snapshot(row)
        yield row


def write_parquet(path: Path, rows: Iterator[dict], schema: pa.Schema) -> int:
    count = 0
    pending = []
    with pq.ParquetWriter(path, schema, compression="zstd", compression_level=3) as writer:
        for row in rows:
            pending.append(row)
            if len(pending) == BATCH_SIZE:
                writer.write_table(pa.Table.from_pylist(pending, schema=schema))
                count += len(pending)
                pending.clear()
        if pending:
            writer.write_table(pa.Table.from_pylist(pending, schema=schema))
            count += len(pending)
    return count


def input_event(row: dict, allowlist: dict) -> dict:
    context = json.loads(row["context_json"])
    filtered = None if context is None else {
        key: context[key] for key in allowlist[row["scenario"]] if key in context
    }
    return {
        "event_id": row["event_id"], "source_record_id": row["source_record_id"],
        "source_record_index": row["source_record_index"], "action_index": row["action_index"],
        "source_action_count": row["source_action_count"], "event_time": row["event_time"],
        "event_type": row["event_type"], "scenario": row["scenario"],
        "time_precision": row["time_precision"],
        "time_lower_bound": iso(row["time_lower_bound"]),
        "time_upper_bound": iso(row["time_upper_bound"]),
        "upper_bound_exclusive": row["upper_bound_exclusive"],
        "eligibility_time": iso(row["eligibility_time"]),
        "eligibility_inclusive": row["eligibility_inclusive"],
        "context": filtered, "action": json.loads(row["action_json"]),
    }


def make_sample(snapshot: dict, event_getter) -> dict:
    return {
        "sample_id": snapshot["sample_id"],
        "input": {
            "user_id": snapshot["user_id"], "cutoff_time": iso(snapshot["cutoff_time"]),
            "cutoff_inclusive": snapshot["cutoff_inclusive"], "timezone": None,
            "history_order_known": snapshot["history_order_known"],
            "history_truncated": snapshot["history_truncated"],
            "history_group_truncated": snapshot["history_group_truncated"],
            "profile": json.loads(snapshot["profile_json"]),
            "history": [copy.deepcopy(event_getter(index)) for index in
                        range(snapshot["history_start"], snapshot["history_stop"])],
        },
    }


def safe_path(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"Path escapes dataset directory: {relative}")
    return path


def _iter_user_samples(root: Path, shard: dict, allowlist: dict, split: str | None) -> Iterator[dict]:
    table = pq.read_table(safe_path(root, shard["events_file"]))
    if not table.schema.equals(EVENT_SCHEMA, check_metadata=True):
        raise ValueError(f"Invalid events schema for {shard['user_id']}")

    @lru_cache(maxsize=128)
    def get_event(index: int) -> dict:
        row = {name: table[name][index].as_py() for name in table.column_names}
        return input_event(row, allowlist)

    with pq.ParquetFile(safe_path(root, shard["snapshots_file"])) as parquet:
        if not parquet.schema_arrow.equals(SNAPSHOT_SCHEMA, check_metadata=True):
            raise ValueError(f"Invalid snapshots schema for {shard['user_id']}")
        for batch in parquet.iter_batches(batch_size=256):
            for snapshot in batch.to_pylist():
                if split is None or snapshot["split"] == split:
                    yield make_sample(snapshot, get_event)


def iter_pretraining_samples(dataset_dir: str | Path, split: str | None = None) -> Iterator[dict]:
    """Yield independent, unmasked records; split is train/validation/test or None.

    Histories carry uncertain-order metadata. Modifying one yielded record does
    not modify another snapshot. No targets or profile prose are returned.
    """
    if split is not None and split not in SPLITS:
        raise ValueError(f"Invalid split: {split!r}; expected one of {SPLITS}")
    root = Path(dataset_dir).resolve()
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "complete" or manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Dataset must have a complete manifest with the supported schema version")
    config_path = root / "preprocessing_config.json"
    if file_sha256(config_path) != manifest["config_sha256"]:
        raise ValueError("Preprocessing config checksum mismatch")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    for shard in manifest["shards"]:
        if split is None or shard["snapshots_by_split"].get(split, 0):
            yield from _iter_user_samples(root, shard, config["context_allowlist"], split)


def verify_sources(input_dir: Path) -> tuple[dict, list[dict], list[str]]:
    root = input_dir.parent.parent.resolve()
    if input_dir.name != "en" or input_dir.parent.name != "raw_user_data":
        raise ValueError("--input must be the raw_user_data/en directory of a pinned snapshot")
    source_path = root / "SOURCE.json"
    source = json.loads(source_path.read_text(encoding="utf-8"))
    prefix = "raw_user_data/en/"
    expected = {entry["rfilename"]: entry for entry in source["files"]
                if entry["rfilename"].startswith(prefix) and entry["rfilename"].endswith(".json")}
    files = sorted(input_dir.glob("*.json"))
    if not files:
        raise ValueError("No English user JSON files found")
    verified = []
    for number, path in enumerate(files, 1):
        relative = path.relative_to(root).as_posix()
        if relative not in expected:
            raise ValueError(f"File absent from source manifest: {relative}")
        if path.resolve() != safe_path(root, relative):
            raise ValueError(f"Unexpected source path: {relative}")
        entry = expected[relative]
        size = path.stat().st_size
        if size != entry["size"]:
            raise ValueError(f"Source size mismatch: {relative}")
        sha256 = hashlib.sha256()
        blob = hashlib.sha1(f"blob {size}\0".encode("ascii"))
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                sha256.update(chunk)
                blob.update(chunk)
        actual = sha256.hexdigest() if entry.get("lfs", {}).get("sha256") else blob.hexdigest()
        wanted = entry.get("lfs", {}).get("sha256") or entry.get("blobId")
        if actual != wanted:
            raise ValueError(f"Source checksum mismatch: {relative}")
        verified.append({"path": relative, "user_id": path.stem,
                         "bytes": size, "sha256": sha256.hexdigest()})
        if number % 25 == 0 or number == len(files):
            print(f"Verified source {number}/{len(files)}", flush=True)
    missing = sorted(Path(relative).stem for relative in expected
                     if not safe_path(root, relative).is_file())
    return source, verified, missing


def verify_user_shard(root: Path, shard: dict, events: list[Event]) -> dict:
    """Read back all written rows and check storage, references and profile cutoffs."""
    event_file = safe_path(root, shard["events_file"])
    if len({event.event_id for event in events}) != len(events):
        raise ValueError(f"Duplicate event IDs: {shard['user_id']}")
    offset = 0
    with pq.ParquetFile(event_file) as parquet:
        if not parquet.schema_arrow.equals(EVENT_SCHEMA, check_metadata=True):
            raise ValueError(f"Events schema mismatch: {event_file}")
        for batch in parquet.iter_batches(batch_size=BATCH_SIZE):
            for row in batch.to_pylist():
                if offset >= len(events) or row != events[offset].row(offset):
                    raise ValueError(f"Event round-trip mismatch: {event_file}, row {offset}")
                offset += 1
    if offset != len(events):
        raise ValueError(f"Event count mismatch: {event_file}")
    snapshot_file = safe_path(root, shard["snapshots_file"])
    with pq.ParquetFile(snapshot_file) as parquet:
        if not parquet.schema_arrow.equals(SNAPSHOT_SCHEMA, check_metadata=True):
            raise ValueError(f"Snapshot schema mismatch: {snapshot_file}")
        result = _verify_snapshot_batches(parquet.iter_batches(batch_size=BATCH_SIZE), shard, events)
    return {"event_rows": offset, **result}


def _verify_snapshot_batches(batches, shard: dict, events: list[Event]) -> dict:
    counts = Counter({split: 0 for split in SPLITS})
    seen = set()
    previous = None
    left = right = 0
    scenarios = Counter()
    days = Counter()
    for batch in batches:
        for row in batch.to_pylist():
            start, stop = row["history_start"], row["history_stop"]
            coordinate = row["cutoff_time"], row["cutoff_inclusive"]
            history = events[start:stop]
            if not (0 <= start < stop <= len(events)) or start != max(0, stop - MAX_EVENTS):
                raise ValueError(f"Invalid history bounds: {row['sample_id']}")
            if row["history_event_ids"] != [event.event_id for event in history]:
                raise ValueError(f"Invalid event references: {row['sample_id']}")
            if row["user_id"] != shard["user_id"] or row["sample_id"] in seen:
                raise ValueError(f"Duplicate sample or wrong user: {row['sample_id']}")
            seen.add(row["sample_id"])
            if previous is not None and coordinate <= previous:
                raise ValueError(f"Snapshots not strictly ordered: {row['sample_id']}")
            previous = coordinate
            if history[-1].coordinate != coordinate or any(event.coordinate > coordinate for event in history):
                raise ValueError(f"History violates cutoff: {row['sample_id']}")
            if stop < len(events) and events[stop].coordinate <= coordinate:
                raise ValueError(f"Incomplete cutoff group: {row['sample_id']}")
            if row["split"] != split_for_cutoff(*coordinate):
                raise ValueError(f"Invalid time split: {row['sample_id']}")
            if row["history_order_known"] != order_is_known(history):
                raise ValueError(f"Invalid uncertainty flag: {row['sample_id']}")
            while right < stop:
                event = events[right]
                scenarios[event.scenario] += 1
                days[event.time_lower_bound.date()] += 1
                right += 1
            lower = row["cutoff_time"] - timedelta(days=30), row["cutoff_inclusive"]
            while left < right and events[left].coordinate <= lower:
                event = events[left]
                scenarios[event.scenario] -= 1
                day = event.time_lower_bound.date()
                days[day] -= 1
                if not days[day]:
                    del days[day]
                left += 1
            expected_profile = {
                "as_of_time": iso(row["cutoff_time"]), "event_count_30d": right - left,
                "active_days_30d": len(days),
                **{field: scenarios[scenario] for scenario, field in SCENARIO_PROFILE_FIELDS.items()},
            }
            if json.loads(row["profile_json"]) != expected_profile:
                raise ValueError(f"Profile violates 30-day window: {row['sample_id']}")
            counts[row["split"]] += 1
    if dict(counts) != shard["snapshots_by_split"]:
        raise ValueError(f"Snapshot count mismatch: {shard['snapshots_file']}")
    return {"snapshot_rows": sum(counts.values()),
            "snapshots_by_split": dict(counts)}


def preprocessing_config() -> dict:
    return {
        "schema_version": SCHEMA_VERSION, "language": "en", "timezone": None,
        "max_history_events": MAX_EVENTS, "event_unit": "one raw action observation",
        "profile_text_used": False, "context_allowlist": CONTEXT_ALLOWLIST,
        "scenario_profile_fields": SCENARIO_PROFILE_FIELDS,
        "profile_window": "(cutoff - 30 days, cutoff] in (eligibility_time, inclusive) coordinates",
        "active_day_basis": "calendar date of source timestamp; counts observations including negative states",
        "history_selection": "last 30 sorted by eligibility_time, eligibility_inclusive, source_record_index, action_index",
        "snapshot_cadence": "one per eligibility coordinate; day closing precedes an exact midnight observation",
        "splits": {"start_inclusive": iso(START), "train_end_exclusive": iso(TRAIN_END),
                   "validation_end_exclusive": iso(VALIDATION_END), "test_end_exclusive": iso(TEST_END),
                   "day_closing_at_boundary": "belongs to preceding split"},
        "compression": {"codec": "zstd", "level": 3, "rows_per_write_batch": BATCH_SIZE},
        "training_targets_stored": False, "mask_policy": None, "vocabulary_fit": False,
        "metadata_not_business_features": [
            "sample_id", "cutoff_time", "cutoff_inclusive", "timezone", "as_of_time",
            "history_order_known", "history_truncated", "history_group_truncated",
            "event_id", "source_record_id", "source_record_index", "action_index",
            "source_action_count", "eligibility_time", "eligibility_inclusive",
            "time_lower_bound", "time_upper_bound", "upper_bound_exclusive",
        ],
        "temporal_metadata": ["event_time", "time_precision", "time_lower_bound",
                              "time_upper_bound", "upper_bound_exclusive"],
        "limitations": [
            "Source timezone and actual log availability are unknown.",
            "Date-only records are eligible only when the source day has ended.",
            "Eligibility ordering does not establish actual order within overlapping intervals.",
            "The last 30 observations may include a partial eligibility or source-record group.",
            "Actions include negative and unknown states; action.type is not a success label.",
            "Shared context and related fields require coordinated masking across source_record_id.",
            "Next-action targets must be strictly ordered, excluded from input and constrained to their split.",
            "No CVR definition or conversion attribution has been validated.",
        ],
    }


def build_dataset(input_dir: str | Path, output_dir: str | Path) -> dict:
    input_dir, output_dir = Path(input_dir).resolve(), Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists; refusing to overwrite: {output_dir}")
    if output_dir.is_relative_to(input_dir.parent.parent):
        raise ValueError("Output must not be inside the raw snapshot directory")
    started = perf_counter()
    source, verified, missing = verify_sources(input_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    manifest = {
        "schema_version": SCHEMA_VERSION, "status": "building",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_repository": source["repository"], "source_revision": source["revision"],
        "source_url": source.get("source_url"), "source_language": "en",
        "license": source.get("license"), "source_files": verified,
        "source_manifest_sha256": file_sha256(input_dir.parent.parent / "SOURCE.json"),
        "expected_users": len(verified) + len(missing), "processed_users": len(verified),
        "missing_users": missing, "shards": [], "pyarrow_version": pa.__version__,
        "generator_sha256": file_sha256(Path(__file__)),
    }
    write_json(output_dir / "manifest.json", manifest)
    stats = DatasetStats()
    previews = {}
    fallbacks = []
    try:
        for name in ("events", "snapshots"):
            (output_dir / name).mkdir()
        config = preprocessing_config()
        write_json(output_dir / "preprocessing_config.json", config)
        manifest["config_sha256"] = file_sha256(output_dir / "preprocessing_config.json")
        license_path = input_dir.parent.parent / "LICENSE"
        if license_path.is_file():
            (output_dir / "LICENSE").write_bytes(license_path.read_bytes())
        for number, proof in enumerate(verified, 1):
            path = safe_path(input_dir.parent.parent, proof["path"])
            # Check again after preflight to reject changes between validation and parsing.
            payload = path.read_bytes()
            if hashlib.sha256(payload).hexdigest() != proof["sha256"]:
                raise ValueError(f"Source changed after verification: {proof['path']}")
            try:
                raw = json.loads(payload)
                events = normalize_user(raw, proof["user_id"], stats)
            except (ValueError, TypeError, KeyError) as error:
                raise ValueError(f"{proof['path']}: {error}") from error
            del payload, raw
            shard = {"user_id": proof["user_id"],
                     "events_file": f"events/{proof['user_id']}.parquet",
                     "snapshots_file": f"snapshots/{proof['user_id']}.parquet"}
            shard["event_rows"] = write_parquet(
                output_dir / shard["events_file"],
                (event.row(index) for index, event in enumerate(events)), EVENT_SCHEMA,
            )
            counts = Counter({split: 0 for split in SPLITS})
            selected_rows = []

            def snapshot_rows():
                for row in iter_snapshot_rows(events, stats):
                    counts[row["split"]] += 1
                    history = events[row["history_start"]:row["history_stop"]]
                    criteria = {f"scenario:{event.scenario}" for event in history}
                    criteria.add(f"split:{row['split']}")
                    if any(event.event_type == "purchase" and any(
                        value is False for value in json.loads(event.action_json).values()
                    ) for event in history):
                        criteria.add("negative_purchase")
                    new = criteria - previews.keys()
                    if new or len(fallbacks) < 10:
                        selected_rows.append(row)
                        for criterion in new:
                            previews[criterion] = row["sample_id"]
                        if len(fallbacks) < 10:
                            fallbacks.append(row["sample_id"])
                    yield row

            shard["snapshot_rows"] = write_parquet(
                output_dir / shard["snapshots_file"], snapshot_rows(), SNAPSHOT_SCHEMA,
            )
            shard["snapshots_by_split"] = dict(counts)
            verify_user_shard(output_dir, shard, events)
            # Read selected previews from the actual event Parquet, not raw input.
            if selected_rows:
                table = pq.read_table(output_dir / shard["events_file"])

                @lru_cache(maxsize=128)
                def get_event(index):
                    row = {name: table[name][index].as_py() for name in table.column_names}
                    return input_event(row, CONTEXT_ALLOWLIST)

                for row in selected_rows:
                    previews.setdefault("samples", {})[row["sample_id"]] = make_sample(row, get_event)
                get_event.cache_clear()
                del table
            shard["events_bytes"] = (output_dir / shard["events_file"]).stat().st_size
            shard["snapshots_bytes"] = (output_dir / shard["snapshots_file"]).stat().st_size
            shard["events_sha256"] = file_sha256(output_dir / shard["events_file"])
            shard["snapshots_sha256"] = file_sha256(output_dir / shard["snapshots_file"])
            manifest["shards"].append(shard)
            del events
            if number % 10 == 0 or number == len(verified):
                print(f"Built and verified {number}/{len(verified)} users; "
                      f"{stats.events:,} events; {sum(stats.snapshots.values()):,} snapshots", flush=True)
        chosen = []
        for criterion in [*(f"scenario:{name}" for name in CONTEXT_ALLOWLIST),
                          *(f"split:{name}" for name in SPLITS), "negative_purchase"]:
            sample_id = previews.get(criterion)
            if sample_id is not None and sample_id not in chosen:
                chosen.append(sample_id)
        for sample_id in fallbacks:
            if sample_id not in chosen and len(chosen) < 10:
                chosen.append(sample_id)
        chosen = chosen[:10]
        write_json(output_dir / "preview.json", [previews["samples"][key] for key in chosen])
        report = stats.report()
        report.update({"schema_version": SCHEMA_VERSION, "users": len(verified),
                       "expected_users": len(verified) + len(missing), "missing_users": missing,
                       "timezone": None, "raw_profile_text_used": False,
                       "verification": {"source_checksums": "passed", "parquet_round_trip": "passed",
                                        "index_references": "passed", "cutoffs_and_profiles": "passed"},
                       "preview_samples": len(chosen),
                       "preview_scenarios": sorted({event["scenario"] for key in chosen
                           for event in previews["samples"][key]["input"]["history"]}),
                       "events_parquet_bytes": sum(s["events_bytes"] for s in manifest["shards"]),
                       "snapshots_parquet_bytes": sum(s["snapshots_bytes"] for s in manifest["shards"]),
                       "elapsed_seconds": round(perf_counter() - started, 3),
                       "limitations": config["limitations"]})
        if sum(s["event_rows"] for s in manifest["shards"]) != stats.events:
            raise ValueError("Global event count mismatch")
        if sum(s["snapshot_rows"] for s in manifest["shards"]) != report["total_snapshots"]:
            raise ValueError("Global snapshot count mismatch")
        manifest.update({"status": "complete", "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                         "total_events": stats.events, "total_snapshots": report["total_snapshots"]})
        manifest["preview_sha256"] = file_sha256(output_dir / "preview.json")
        manifest["report_sha256"] = "0" * 64
        base_bytes = sum(p.stat().st_size for p in output_dir.rglob("*")
                         if p.is_file() and p.name not in {"manifest.json", "data_report.json"})
        total_bytes = 0
        for _ in range(8):
            report["total_output_bytes"] = total_bytes
            report_bytes = (json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
            manifest["report_sha256"] = hashlib.sha256(report_bytes).hexdigest()
            manifest_bytes = (json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
            measured = base_bytes + len(report_bytes) + len(manifest_bytes)
            if measured == total_bytes:
                break
            total_bytes = measured
        else:
            raise ValueError("Output size accounting did not converge")
        (output_dir / "data_report.json").write_bytes(report_bytes)
        # Publish complete only after all data, previews and the report are final.
        (output_dir / "manifest.json").write_bytes(manifest_bytes)
        return report
    except Exception as error:
        manifest.update({"status": "failed", "error": str(error)})
        write_json(output_dir / "manifest.json", manifest)
        raise
