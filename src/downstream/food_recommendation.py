"""Strict Green SM Food-ranking data handoff and candidate batches.

This module deliberately owns only the data boundary shared by Food training,
prediction, and the demo.  It contains no train loop: later modules consume
the request-major :class:`FoodBatch` values built here.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
import re
import sys
import tempfile
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pyarrow.parquet as pq
import torch
import yaml
from torch import nn

from src.contracts.output import (
    ExpectedCandidate,
    OutputContractV1,
    OutputPaths,
    PredictionRow,
    VerifiedOpportunityIndex,
)
from src.foundation_input.dataset import DatasetConfig, encode_sample, foundation_collate
from src.foundation_input.loader import iter_normalized_samples
from src.foundation_input.types import EncodedSample, FoundationBatch
from src.foundation_input.vocab import VocabularyArtifacts
from src.pretraining.green_sm_contract import load_green_sm_tokenizer_config
from src.training.checkpoint import load_foundation_checkpoint

FoodSplit = Literal["train", "validation", "test"]
_SPLITS = frozenset({"train", "validation", "test"})
_PAD = "<PAD>"
_UNK = "<UNK>"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class FoodConfigurationError(ValueError):
    """The Food configuration or immutable upstream handoff is invalid."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise FoodConfigurationError(f"Cannot read {label}: {path}") from error
    if not isinstance(value, dict):
        raise FoodConfigurationError(f"{label} must be a JSON object")
    return value


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise FoodConfigurationError(f"{label} must be a mapping")
    return value


def _config_text(mapping: Mapping[str, Any], key: str, *, label: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise FoodConfigurationError(f"{label}.{key} must be a non-empty string")
    return value.strip()


def _verify_sha256(path: Path, expected: str, *, label: str) -> None:
    if not _SHA256.fullmatch(expected):
        raise FoodConfigurationError(f"{label} expected SHA-256 must be 64 lowercase hexadecimal characters")
    if not path.is_file():
        raise FoodConfigurationError(f"{label} does not exist: {path}")
    if _sha256(path) != expected:
        raise FoodConfigurationError(f"{label} SHA-256 mismatch")


def load_food_recommendation_config(path: str | Path) -> dict[str, Any]:
    """Load configuration only; validation has no training side effects."""

    source = Path(path)
    try:
        config = yaml.safe_load(source.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise FoodConfigurationError(f"Cannot read Food config: {source}") from error
    if not isinstance(config, dict):
        raise FoodConfigurationError("Food config root must be a mapping")
    return config


def validate_food_data_handoff(config: Mapping[str, Any]) -> dict[str, Any]:
    """Verify immutable Food/Foundation manifests and all split parquet checksums."""

    if config.get("task") != "food_recommendation":
        raise FoodConfigurationError("config.task must be 'food_recommendation'")
    if config.get("freeze_foundation") is not True:
        raise FoodConfigurationError("config.freeze_foundation must be true")
    data = _mapping(config.get("data"), label="config.data")
    opportunities_root = Path(_config_text(data, "opportunities_root", label="config.data"))
    food_manifest_path = Path(_config_text(data, "manifest_path", label="config.data"))
    foundation_manifest_path = Path(
        _config_text(data, "foundation_manifest_path", label="config.data")
    )
    food_sha = _config_text(data, "expected_food_manifest_sha256", label="config.data")
    foundation_sha = _config_text(
        data, "expected_foundation_manifest_sha256", label="config.data"
    )
    _verify_sha256(food_manifest_path, food_sha, label="Food manifest")
    _verify_sha256(foundation_manifest_path, foundation_sha, label="Foundation manifest")
    food_manifest = _read_json(food_manifest_path, label="Food manifest")
    foundation_manifest = _read_json(foundation_manifest_path, label="Foundation manifest")
    if food_manifest.get("schema_version") != "green_sm_food_opportunities_v1":
        raise FoodConfigurationError("Food manifest has unsupported schema_version")
    if food_manifest.get("foundation_manifest_sha256") != foundation_sha:
        raise FoodConfigurationError("Food manifest is not linked to the configured Foundation manifest")
    if foundation_manifest.get("status") not in {None, "complete"}:
        raise FoodConfigurationError("Foundation manifest status must be complete")
    files = _mapping(food_manifest.get("files"), label="Food manifest files")
    if set(files) != _SPLITS:
        raise FoodConfigurationError("Food manifest must declare exactly train, validation, and test")
    for split in sorted(_SPLITS):
        entry = _mapping(files[split], label=f"Food manifest files.{split}")
        _verify_sha256(
            opportunities_root / _config_text(entry, "file", label=f"Food manifest files.{split}"),
            _config_text(entry, "sha256", label=f"Food manifest files.{split}"),
            label=f"Food {split} parquet",
        )
    return {
        "food_manifest_sha256": food_sha,
        "foundation_manifest_sha256": foundation_sha,
        "data_version": str(data.get("data_version", "unknown")),
        "splits": sorted(_SPLITS),
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Run a checksum-bound Food validation, training, or prediction lifecycle."""

    parser = argparse.ArgumentParser(description="Green SM frozen-Foundation Food ranking")
    parser.add_argument("--config", required=True, help="Food ranking YAML configuration")
    parser.add_argument(
        "--mode", required=True, choices=("validate", "train", "predict", "demo"),
        help="validate verifies immutable inputs; other modes require the run lifecycle.",
    )
    parser.add_argument(
        "--split", choices=("validation", "test"),
        help="prediction split; required when --mode predict",
    )
    args = parser.parse_args(argv)
    try:
        config = load_food_recommendation_config(args.config)
        report = validate_food_data_handoff(config)
    except FoodConfigurationError as error:
        print(f"Food configuration error: {error}", file=sys.stderr)
        return 2
    if args.mode == "validate":
        print(json.dumps({"status": "data_handoff_valid", **report}, sort_keys=True))
        return 0
    try:
        if args.mode == "train":
            result = run_food_training(config)
            print(json.dumps({"status": "food_training_complete", **result}, sort_keys=True))
            return 0
        if args.mode == "predict":
            if args.split is None:
                parser.error("--split is required when --mode predict")
            result = run_food_prediction(config, split=args.split)
            print(json.dumps({"status": "food_prediction_exported", "predictions": str(result.predictions)}, sort_keys=True))
            return 0
    except FoodConfigurationError as error:
        print(f"Food configuration error: {error}", file=sys.stderr)
        return 2
    print("Food demo is not implemented; use --mode predict --split test for export.", file=sys.stderr)
    return 3


def _text(value: Any, *, field: str, request_id: str = "<unknown>") -> str:
    if value is None or not str(value).strip():
        raise ValueError(f"request_id={request_id!r}: {field} must be non-empty")
    return str(value).strip()


def _finite_float(value: Any, *, field: str, request_id: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"request_id={request_id!r}: {field} must be numeric") from error
    if not math.isfinite(result):
        raise ValueError(f"request_id={request_id!r}: {field} must be finite")
    return result


def _timestamp(value: Any, *, field: str, request_id: str) -> int:
    """Convert only timezone-aware Arrow datetimes to UTC epoch seconds."""

    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(
            f"request_id={request_id!r}: {field} must be a timezone-aware datetime"
        )
    return int(value.timestamp())


def _optional_binary(value: Any, *, field: str, request_id: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or value not in (0, 1):
        raise ValueError(f"request_id={request_id!r}: {field} must be binary when present")
    return int(value)


@dataclass(frozen=True)
class FoodOpportunity:
    """One Food candidate from the versioned Green SM opportunity artifact."""

    request_id: str
    user_id: str
    foundation_sample_id: str
    cutoff_timestamp: int
    candidate_id: str
    split: FoodSplit
    candidate_available_timestamp: int
    item_id: str
    merchant_id: str
    category: str
    geo_cell: str
    latitude: float
    longitude: float
    item_price: float
    is_positive: int | None
    is_hard_negative: bool

    @classmethod
    def from_mapping(cls, row: Mapping[str, Any]) -> FoodOpportunity:
        request_id = _text(row.get("request_id"), field="request_id")
        split = _text(row.get("split"), field="split", request_id=request_id)
        if split not in _SPLITS:
            raise ValueError(f"request_id={request_id!r}: unsupported split {split!r}")
        return cls(
            request_id=request_id,
            user_id=_text(row.get("user_id"), field="user_id", request_id=request_id),
            foundation_sample_id=_text(
                row.get("foundation_sample_id"),
                field="foundation_sample_id",
                request_id=request_id,
            ),
            cutoff_timestamp=_timestamp(
                row.get("cutoff_timestamp"), field="cutoff_timestamp", request_id=request_id
            ),
            candidate_id=_text(
                row.get("candidate_id"), field="candidate_id", request_id=request_id
            ),
            split=split,  # type: ignore[arg-type]
            candidate_available_timestamp=_timestamp(
                row.get("candidate_available_timestamp"),
                field="candidate_available_timestamp",
                request_id=request_id,
            ),
            item_id=_text(row.get("candidate_id"), field="candidate_id", request_id=request_id),
            merchant_id=_text(
                row.get("merchant_id"), field="merchant_id", request_id=request_id
            ),
            category=_text(
                row.get("candidate_category"),
                field="candidate_category",
                request_id=request_id,
            ),
            geo_cell=_text(
                row.get("candidate_geo_cell"),
                field="candidate_geo_cell",
                request_id=request_id,
            ),
            latitude=_finite_float(
                row.get("candidate_lat"), field="candidate_lat", request_id=request_id
            ),
            longitude=_finite_float(
                row.get("candidate_lon"), field="candidate_lon", request_id=request_id
            ),
            item_price=_finite_float(
                row.get("item_price"), field="item_price", request_id=request_id
            ),
            is_positive=_optional_binary(
                row.get("is_positive"), field="is_positive", request_id=request_id
            ),
            is_hard_negative=bool(row.get("is_hard_negative", False)),
        )


def validate_food_opportunities(
    rows: Sequence[FoodOpportunity],
    expected_split: FoodSplit,
    *,
    require_labels: bool,
) -> tuple[FoodOpportunity, ...]:
    """Validate request-level integrity without inspecting labels for prediction."""

    if expected_split not in _SPLITS:
        raise ValueError(f"unsupported expected_split {expected_split!r}")
    if not rows:
        raise ValueError("Food opportunities must not be empty")
    grouped: dict[str, list[FoodOpportunity]] = defaultdict(list)
    for row in rows:
        if row.split != expected_split:
            raise ValueError(
                f"request_id={row.request_id!r}: split {row.split!r} does not match {expected_split!r}"
            )
        if row.candidate_available_timestamp > row.cutoff_timestamp:
            raise ValueError(
                f"request_id={row.request_id!r}: candidate_available_timestamp must be "
                "at or before cutoff_timestamp"
            )
        if not all(math.isfinite(value) for value in (row.latitude, row.longitude, row.item_price)):
            raise ValueError(f"request_id={row.request_id!r}: candidate numeric fields must be finite")
        grouped[row.request_id].append(row)

    for request_id, candidates in grouped.items():
        user_ids = {row.user_id for row in candidates}
        sample_ids = {row.foundation_sample_id for row in candidates}
        cutoffs = {row.cutoff_timestamp for row in candidates}
        candidate_ids = [row.candidate_id for row in candidates]
        if len(user_ids) != 1 or len(sample_ids) != 1 or len(cutoffs) != 1:
            raise ValueError(
                f"request_id={request_id!r}: candidates must share one user, foundation sample, and cutoff"
            )
        if len(set(candidate_ids)) != len(candidate_ids):
            raise ValueError(f"duplicate candidate in request-{request_id}")
        if require_labels:
            labels = [row.is_positive for row in candidates]
            if any(label is None for label in labels):
                raise ValueError(f"request_id={request_id!r}: is_positive is required")
            if 1 not in labels or 0 not in labels:
                raise ValueError(
                    f"request_id={request_id!r}: labelled data needs a positive and a negative candidate"
                )
    return tuple(rows)


def load_green_sm_food_opportunities(
    root: str | Path, *, split: FoodSplit
) -> tuple[FoodOpportunity, ...]:
    """Load an immutable Food opportunity split using its published field names."""

    if split not in _SPLITS:
        raise ValueError(f"unsupported split {split!r}")
    path = Path(root) / f"food_opportunities.{split}.parquet"
    if not path.is_file():
        raise FileNotFoundError(f"Food opportunity parquet not found: {path}")
    rows = tuple(FoodOpportunity.from_mapping(row) for row in pq.read_table(path).to_pylist())
    return validate_food_opportunities(rows, split, require_labels=split != "test")


def resolve_green_sm_food_samples(
    rows: Sequence[FoodOpportunity],
    *,
    foundation_root: str | Path,
    vocabulary_path: str | Path,
    split: FoodSplit,
    max_tokens: int,
) -> dict[str, EncodedSample]:
    """Resolve Food requests to exactly their causal Foundation snapshots."""

    validated = validate_food_opportunities(rows, split, require_labels=split != "test")
    by_sample: dict[str, FoodOpportunity] = {}
    for row in validated:
        existing = by_sample.setdefault(row.foundation_sample_id, row)
        if (
            existing.user_id != row.user_id
            or existing.cutoff_timestamp != row.cutoff_timestamp
            or existing.split != row.split
        ):
            raise ValueError(
                f"foundation_sample_id={row.foundation_sample_id!r}: inconsistent Food rows"
            )

    artifacts = VocabularyArtifacts.load(vocabulary_path)
    tokenizer = load_green_sm_tokenizer_config(foundation_root)
    dataset_config = DatasetConfig(max_tokens=max_tokens)
    found: dict[str, EncodedSample] = {}
    requested_users = {opportunity.user_id for opportunity in by_sample.values()}
    for canonical in iter_normalized_samples(
        foundation_root, split, validate=True, user_ids=requested_users
    ):
        opportunity = by_sample.get(canonical.sample_id)
        if opportunity is None:
            continue
        if canonical.user_id != opportunity.user_id:
            raise ValueError(f"{canonical.sample_id!r}: user_id does not match Food opportunity")
        if int(canonical.cutoff_time.timestamp()) != opportunity.cutoff_timestamp:
            raise ValueError(
                f"{canonical.sample_id!r}: cutoff_timestamp does not match Food opportunity"
            )
        if canonical.split != opportunity.split:
            raise ValueError(f"{canonical.sample_id!r}: split does not match Food opportunity")
        if canonical.sample_id in found:
            raise ValueError(f"duplicate canonical foundation_sample_id {canonical.sample_id!r}")
        found[canonical.sample_id] = encode_sample(
            canonical, artifacts, tokenizer, dataset_config
        )
    missing = sorted(set(by_sample) - set(found))
    if missing:
        raise ValueError(f"foundation_sample_id not found in split {split!r}: {missing[0]!r}")
    return found


def stable_food_hash_id(field_name: str, value: str, bucket_count: int) -> int:
    """Stable Food-local ID with zero reserved for padded candidate slots."""

    if bucket_count < 2:
        raise ValueError("bucket_count must be at least 2 so zero remains padding")
    payload = f"green-sm-food-v1\0{field_name}\0{value}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (bucket_count - 1) + 1


def _build_vocabulary(values: Sequence[str]) -> dict[str, int]:
    vocabulary = {_PAD: 0, _UNK: 1}
    for value in sorted(set(values)):
        vocabulary[value] = len(vocabulary)
    return vocabulary


def _mean_and_scale(values: Sequence[float]) -> tuple[float, float]:
    if not values:
        raise ValueError("cannot fit a numeric normalizer without train values")
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    scale = math.sqrt(variance)
    return mean, scale if scale > 0.0 else 1.0


@dataclass(frozen=True)
class FoodCandidateFeatures:
    category_ids: tuple[int, ...]
    geo_cell_ids: tuple[int, ...]
    price_values: tuple[float, ...]
    geo_values: tuple[tuple[float, float], ...]


@dataclass(frozen=True)
class FoodFeatureArtifacts:
    """Train-only candidate vocabularies and continuous-feature normalizers."""

    category_vocabulary: dict[str, int]
    geo_cell_vocabulary: dict[str, int]
    price_mean: float
    price_scale: float
    latitude_mean: float
    latitude_scale: float
    longitude_mean: float
    longitude_scale: float

    @classmethod
    def fit(cls, rows: Sequence[FoodOpportunity]) -> FoodFeatureArtifacts:
        validate_food_opportunities(rows, "train", require_labels=True)
        return cls(
            category_vocabulary=_build_vocabulary([row.category for row in rows]),
            geo_cell_vocabulary=_build_vocabulary([row.geo_cell for row in rows]),
            price_mean=_mean_and_scale([row.item_price for row in rows])[0],
            price_scale=_mean_and_scale([row.item_price for row in rows])[1],
            latitude_mean=_mean_and_scale([row.latitude for row in rows])[0],
            latitude_scale=_mean_and_scale([row.latitude for row in rows])[1],
            longitude_mean=_mean_and_scale([row.longitude for row in rows])[0],
            longitude_scale=_mean_and_scale([row.longitude for row in rows])[1],
        )

    def encode(self, rows: Sequence[FoodOpportunity]) -> FoodCandidateFeatures:
        return FoodCandidateFeatures(
            category_ids=tuple(self.category_vocabulary.get(row.category, 1) for row in rows),
            geo_cell_ids=tuple(self.geo_cell_vocabulary.get(row.geo_cell, 1) for row in rows),
            price_values=tuple((row.item_price - self.price_mean) / self.price_scale for row in rows),
            geo_values=tuple(
                (
                    (row.latitude - self.latitude_mean) / self.latitude_scale,
                    (row.longitude - self.longitude_mean) / self.longitude_scale,
                )
                for row in rows
            ),
        )

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "category_vocabulary": self.category_vocabulary,
            "geo_cell_vocabulary": self.geo_cell_vocabulary,
            "price_mean": self.price_mean,
            "price_scale": self.price_scale,
            "latitude_mean": self.latitude_mean,
            "latitude_scale": self.latitude_scale,
            "longitude_mean": self.longitude_mean,
            "longitude_scale": self.longitude_scale,
        }
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=destination.parent, delete=False
        ) as handle:
            json.dump(payload, handle, sort_keys=True, separators=(",", ":"))
            temporary = Path(handle.name)
        os.replace(temporary, destination)

    @classmethod
    def load(cls, path: str | Path) -> FoodFeatureArtifacts:
        source = Path(path)
        try:
            payload = json.loads(source.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"invalid Food feature artifact: {source}") from error
        if not isinstance(payload, dict):
            raise TypeError("Food feature artifact must be a JSON object")
        try:
            category_vocabulary = {str(key): int(value) for key, value in payload["category_vocabulary"].items()}
            geo_cell_vocabulary = {str(key): int(value) for key, value in payload["geo_cell_vocabulary"].items()}
            return cls(
                category_vocabulary=category_vocabulary,
                geo_cell_vocabulary=geo_cell_vocabulary,
                price_mean=float(payload["price_mean"]),
                price_scale=float(payload["price_scale"]),
                latitude_mean=float(payload["latitude_mean"]),
                latitude_scale=float(payload["latitude_scale"]),
                longitude_mean=float(payload["longitude_mean"]),
                longitude_scale=float(payload["longitude_scale"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("Food feature artifact has an invalid schema") from error


@dataclass
class FoodBatch:
    foundation_batch: FoundationBatch
    item_hash_ids: torch.LongTensor
    merchant_hash_ids: torch.LongTensor
    category_ids: torch.LongTensor
    geo_cell_ids: torch.LongTensor
    price_values: torch.FloatTensor
    geo_values: torch.FloatTensor
    positive_mask: torch.BoolTensor
    candidate_mask: torch.BoolTensor
    opportunities: tuple[tuple[FoodOpportunity, ...], ...]


@dataclass(frozen=True)
class CachedFoodBatch:
    """One request-major Food batch with detached, CPU-resident user embeddings."""

    long_term: torch.FloatTensor
    short_term: torch.FloatTensor
    item_hash_ids: torch.LongTensor
    merchant_hash_ids: torch.LongTensor
    category_ids: torch.LongTensor
    geo_cell_ids: torch.LongTensor
    price_values: torch.FloatTensor
    geo_values: torch.FloatTensor
    positive_mask: torch.BoolTensor
    candidate_mask: torch.BoolTensor


def _group_requests(rows: Sequence[FoodOpportunity]) -> tuple[tuple[FoodOpportunity, ...], ...]:
    grouped: dict[str, list[FoodOpportunity]] = defaultdict(list)
    for row in rows:
        grouped[row.request_id].append(row)
    return tuple(
        tuple(sorted(grouped[request_id], key=lambda row: row.candidate_id))
        for request_id in sorted(grouped)
    )


def build_food_batches(
    rows: Sequence[FoodOpportunity],
    *,
    encoded_samples_by_id: Mapping[str, EncodedSample],
    feature_artifacts: FoodFeatureArtifacts,
    batch_size: int,
    item_hash_buckets: int,
    merchant_hash_buckets: int,
    require_labels: bool,
) -> tuple[FoodBatch, ...]:
    """Build deterministic request-major batches; padded columns are never candidates."""

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if not rows:
        return ()
    split = rows[0].split
    validate_food_opportunities(rows, split, require_labels=require_labels)
    requests = _group_requests(rows)
    batches: list[FoodBatch] = []
    for start in range(0, len(requests), batch_size):
        chunk = requests[start : start + batch_size]
        samples: list[EncodedSample] = []
        for request in chunk:
            sample_id = request[0].foundation_sample_id
            try:
                sample = encoded_samples_by_id[sample_id]
            except KeyError as error:
                raise ValueError(f"request_id={request[0].request_id!r}: missing Foundation sample {sample_id!r}") from error
            if sample.user_id != request[0].user_id:
                raise ValueError(f"request_id={request[0].request_id!r}: Foundation sample user does not match")
            samples.append(sample)

        max_candidates = max(len(request) for request in chunk)
        item_ids = torch.zeros((len(chunk), max_candidates), dtype=torch.long)
        merchant_ids = torch.zeros_like(item_ids)
        category_ids = torch.zeros_like(item_ids)
        geo_cell_ids = torch.zeros_like(item_ids)
        price_values = torch.zeros((len(chunk), max_candidates), dtype=torch.float32)
        geo_values = torch.zeros((len(chunk), max_candidates, 2), dtype=torch.float32)
        positive_mask = torch.zeros((len(chunk), max_candidates), dtype=torch.bool)
        candidate_mask = torch.zeros((len(chunk), max_candidates), dtype=torch.bool)

        for request_index, request in enumerate(chunk):
            features = feature_artifacts.encode(request)
            for candidate_index, row in enumerate(request):
                item_ids[request_index, candidate_index] = stable_food_hash_id(
                    "item_id", row.item_id, item_hash_buckets
                )
                merchant_ids[request_index, candidate_index] = stable_food_hash_id(
                    "merchant_id", row.merchant_id, merchant_hash_buckets
                )
                category_ids[request_index, candidate_index] = features.category_ids[candidate_index]
                geo_cell_ids[request_index, candidate_index] = features.geo_cell_ids[candidate_index]
                price_values[request_index, candidate_index] = features.price_values[candidate_index]
                geo_values[request_index, candidate_index] = torch.tensor(
                    features.geo_values[candidate_index], dtype=torch.float32
                )
                candidate_mask[request_index, candidate_index] = True
                if require_labels:
                    positive_mask[request_index, candidate_index] = row.is_positive == 1

        if torch.any(positive_mask & ~candidate_mask):
            raise AssertionError("Food positive_mask must be a subset of candidate_mask")
        batches.append(
            FoodBatch(
                foundation_batch=foundation_collate(samples),
                item_hash_ids=item_ids,
                merchant_hash_ids=merchant_ids,
                category_ids=category_ids,
                geo_cell_ids=geo_cell_ids,
                price_values=price_values,
                geo_values=geo_values,
                positive_mask=positive_mask,
                candidate_mask=candidate_mask,
                opportunities=tuple(chunk),
            )
        )
    return tuple(batches)


class FoodRanker(nn.Module):
    """Candidate ranker that consumes frozen long- and short-term user states."""

    def __init__(
        self,
        *,
        d_model: int,
        item_hash_buckets: int,
        merchant_hash_buckets: int,
        num_categories: int,
        num_geo_cells: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if d_model < 1:
            raise ValueError("d_model must be positive")
        for name, size in (
            ("item_hash_buckets", item_hash_buckets),
            ("merchant_hash_buckets", merchant_hash_buckets),
            ("num_categories", num_categories),
            ("num_geo_cells", num_geo_cells),
        ):
            if size < 2:
                raise ValueError(f"{name} must reserve padding ID 0 and contain at least two entries")
        self.user_projection = nn.Sequential(
            nn.Linear(2 * d_model, d_model), nn.GELU(), nn.LayerNorm(d_model)
        )
        self.item_embedding = nn.Embedding(item_hash_buckets, d_model, padding_idx=0)
        self.merchant_embedding = nn.Embedding(merchant_hash_buckets, d_model, padding_idx=0)
        self.category_embedding = nn.Embedding(num_categories, d_model, padding_idx=0)
        self.geo_cell_embedding = nn.Embedding(num_geo_cells, d_model, padding_idx=0)
        self.price_projection = nn.Linear(1, d_model)
        self.coordinate_projection = nn.Linear(2, d_model)
        self.candidate_projection = nn.Sequential(
            nn.Linear(6 * d_model, d_model), nn.GELU(), nn.LayerNorm(d_model)
        )
        self.scorer = nn.Sequential(
            nn.Linear(2 * d_model, d_model), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model, 1)
        )
        self.config = {
            "d_model": d_model,
            "item_hash_buckets": item_hash_buckets,
            "merchant_hash_buckets": merchant_hash_buckets,
            "num_categories": num_categories,
            "num_geo_cells": num_geo_cells,
            "dropout": dropout,
        }

    def forward(
        self,
        long_term: torch.Tensor,
        short_term: torch.Tensor,
        *,
        item_hash_ids: torch.LongTensor,
        merchant_hash_ids: torch.LongTensor,
        category_ids: torch.LongTensor,
        geo_cell_ids: torch.LongTensor,
        price_values: torch.FloatTensor,
        geo_values: torch.FloatTensor,
    ) -> torch.Tensor:
        if long_term.shape != short_term.shape or long_term.ndim != 2:
            raise ValueError("long_term and short_term must be two-dimensional tensors of equal shape")
        batch_size, _ = long_term.shape
        candidate_shape = item_hash_ids.shape
        if len(candidate_shape) != 2 or candidate_shape[0] != batch_size:
            raise ValueError("candidate IDs must be [batch, candidates] and align with user embeddings")
        if any(tensor.shape != candidate_shape for tensor in (
            merchant_hash_ids, category_ids, geo_cell_ids, price_values
        )):
            raise ValueError("scalar candidate features must share the item_hash_ids shape")
        if geo_values.shape != (*candidate_shape, 2):
            raise ValueError("geo_values must be [batch, candidates, 2]")
        user = self.user_projection(torch.cat((long_term, short_term), dim=-1))
        candidate = torch.cat(
            (
                self.item_embedding(item_hash_ids),
                self.merchant_embedding(merchant_hash_ids),
                self.category_embedding(category_ids),
                self.geo_cell_embedding(geo_cell_ids),
                self.price_projection(price_values.unsqueeze(-1)),
                self.coordinate_projection(geo_values),
            ),
            dim=-1,
        )
        candidate = self.candidate_projection(candidate)
        users = user.unsqueeze(1).expand(-1, candidate.shape[1], -1)
        return self.scorer(torch.cat((users, candidate), dim=-1)).squeeze(-1)


def masked_multi_positive_listwise_loss(
    scores: torch.Tensor,
    positive_mask: torch.BoolTensor,
    candidate_mask: torch.BoolTensor,
) -> tuple[torch.Tensor, int]:
    """Mean listwise loss for request rows with one or more positive candidates."""

    if scores.shape != positive_mask.shape or scores.shape != candidate_mask.shape:
        raise ValueError("scores, positive_mask, and candidate_mask must share shape [batch, candidates]")
    if scores.ndim != 2:
        raise ValueError("listwise Food scores must be two-dimensional")
    if torch.any(positive_mask & ~candidate_mask):
        raise ValueError("positive_mask must be a subset of candidate_mask")
    losses: list[torch.Tensor] = []
    for row_index in range(scores.shape[0]):
        valid = candidate_mask[row_index]
        positive = positive_mask[row_index]
        valid_count = int(valid.sum().item())
        positive_count = int(positive.sum().item())
        if positive_count == 0 or positive_count == valid_count:
            raise ValueError("each request requires at least one valid positive and one valid negative")
        losses.append(torch.logsumexp(scores[row_index, valid], dim=0) - torch.logsumexp(scores[row_index, positive], dim=0))
    if not losses:
        raise ValueError("listwise Food loss requires at least one request")
    return torch.stack(losses).mean(), len(losses)


def frozen_foundation_food_logits(
    embedder: Any, ranker: FoodRanker, batch: FoodBatch
) -> torch.Tensor:
    """Embed users without an autograd graph, then retain ranker gradients."""

    with torch.no_grad():
        embeddings = embedder.embed_users(batch.foundation_batch)
    return ranker(
        embeddings["long_term"],
        embeddings["short_term"],
        item_hash_ids=batch.item_hash_ids,
        merchant_hash_ids=batch.merchant_hash_ids,
        category_ids=batch.category_ids,
        geo_cell_ids=batch.geo_cell_ids,
        price_values=batch.price_values,
        geo_values=batch.geo_values,
    )


def _to_device_food_batch(batch: FoodBatch, device: torch.device) -> FoodBatch:
    copied = copy.copy(batch)
    foundation = copy.copy(batch.foundation_batch)
    for name, value in vars(foundation).items():
        if isinstance(value, torch.Tensor):
            setattr(foundation, name, value.to(device))
        elif name == "targets":
            foundation.targets = {key: tensor.to(device) for key, tensor in value.items()}
    copied.foundation_batch = foundation
    for name in (
        "item_hash_ids", "merchant_hash_ids", "category_ids", "geo_cell_ids",
        "price_values", "geo_values", "positive_mask", "candidate_mask",
    ):
        setattr(copied, name, getattr(batch, name).to(device))
    return copied


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _atomic_torch_save(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(value), temporary)
    os.replace(temporary, path)


def _atomic_text(path: Path, value: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def _cache_food_embeddings(
    embedder: Any, batches: Sequence[FoodBatch], device: torch.device
) -> tuple[CachedFoodBatch, ...]:
    """Run the frozen Foundation exactly once for each downstream batch."""

    embedder.to(device)
    embedder.eval()
    for parameter in embedder.parameters():
        parameter.requires_grad_(False)
    cached: list[CachedFoodBatch] = []
    with torch.no_grad():
        for batch in batches:
            moved = _to_device_food_batch(batch, device)
            embeddings = embedder.embed_users(moved.foundation_batch)
            cached.append(
                CachedFoodBatch(
                    long_term=embeddings["long_term"].detach().cpu(),
                    short_term=embeddings["short_term"].detach().cpu(),
                    item_hash_ids=batch.item_hash_ids,
                    merchant_hash_ids=batch.merchant_hash_ids,
                    category_ids=batch.category_ids,
                    geo_cell_ids=batch.geo_cell_ids,
                    price_values=batch.price_values,
                    geo_values=batch.geo_values,
                    positive_mask=batch.positive_mask,
                    candidate_mask=batch.candidate_mask,
                )
            )
    return tuple(cached)


def _cached_food_logits(
    ranker: FoodRanker, batch: CachedFoodBatch, device: torch.device
) -> torch.Tensor:
    return ranker(
        batch.long_term.to(device),
        batch.short_term.to(device),
        item_hash_ids=batch.item_hash_ids.to(device),
        merchant_hash_ids=batch.merchant_hash_ids.to(device),
        category_ids=batch.category_ids.to(device),
        geo_cell_ids=batch.geo_cell_ids.to(device),
        price_values=batch.price_values.to(device),
        geo_values=batch.geo_values.to(device),
    )


def _mean_cached_food_loss(
    ranker: FoodRanker, batches: Sequence[CachedFoodBatch], device: torch.device
) -> float:
    ranker.eval()
    losses: list[float] = []
    with torch.no_grad():
        for batch in batches:
            scores = _cached_food_logits(ranker, batch, device)
            loss, _ = masked_multi_positive_listwise_loss(
                scores, batch.positive_mask.to(device), batch.candidate_mask.to(device)
            )
            losses.append(float(loss.item()))
    if not losses:
        raise ValueError("Food validation requires at least one batch")
    return sum(losses) / len(losses)


def train_food_ranker(
    *,
    embedder: Any,
    ranker: FoodRanker,
    train_batches: Sequence[FoodBatch],
    validation_batches: Sequence[FoodBatch],
    feature_artifacts: FoodFeatureArtifacts,
    run_dir: str | Path,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    device: str | torch.device,
    run_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    """Train only FoodRanker parameters and write reloadable run artifacts."""

    if epochs < 1 or learning_rate <= 0.0 or weight_decay < 0.0:
        raise ValueError("epochs and learning_rate must be positive; weight_decay cannot be negative")
    if not train_batches or not validation_batches:
        raise ValueError("Food training and validation batches must both be non-empty")
    destination = Path(run_dir)
    if destination.exists() and any(destination.iterdir()):
        raise ValueError(f"Food run directory must be new and empty: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    resolved_device = torch.device(device)
    ranker.to(resolved_device)
    cached_train_batches = _cache_food_embeddings(embedder, train_batches, resolved_device)
    cached_validation_batches = _cache_food_embeddings(embedder, validation_batches, resolved_device)
    optimizer = torch.optim.AdamW(ranker.parameters(), lr=learning_rate, weight_decay=weight_decay)
    feature_path = destination / "feature_artifacts.json"
    feature_artifacts.save(feature_path)
    metrics: list[dict[str, float | int]] = []
    best_validation = math.inf
    best_epoch = 0
    for epoch in range(1, epochs + 1):
        ranker.train()
        train_losses: list[float] = []
        for batch in cached_train_batches:
            optimizer.zero_grad(set_to_none=True)
            scores = _cached_food_logits(ranker, batch, resolved_device)
            loss, _ = masked_multi_positive_listwise_loss(
                scores, batch.positive_mask.to(resolved_device), batch.candidate_mask.to(resolved_device)
            )
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach().item()))
        validation_loss = _mean_cached_food_loss(ranker, cached_validation_batches, resolved_device)
        metrics.append({
            "epoch": epoch,
            "train_listwise_loss": sum(train_losses) / len(train_losses),
            "validation_listwise_loss": validation_loss,
        })
        payload = {"ranker_config": ranker.config, "ranker_state_dict": ranker.state_dict(), "epoch": epoch}
        _atomic_torch_save(destination / "last.pt", payload)
        if validation_loss < best_validation:
            best_validation = validation_loss
            best_epoch = epoch
            _atomic_torch_save(destination / "best.pt", payload)
        metrics_path = destination / "train_metrics.jsonl"
        _atomic_text(metrics_path, "".join(json.dumps(row, sort_keys=True) + "\n" for row in metrics))
        print(
            f"Food epoch {epoch}/{epochs} | train_listwise_loss={metrics[-1]['train_listwise_loss']:.6f} "
            f"| validation_listwise_loss={validation_loss:.6f}",
            flush=True,
        )
    metrics_path = destination / "train_metrics.jsonl"
    manifest = {
        **dict(run_metadata),
        "schema_version": "green_sm_food_ranker_v1",
        "feature_artifacts": feature_path.name,
        "feature_artifacts_sha256": _sha256(feature_path),
        "best_checkpoint": "best.pt",
        "best_checkpoint_sha256": _sha256(destination / "best.pt"),
        "last_checkpoint": "last.pt",
        "last_checkpoint_sha256": _sha256(destination / "last.pt"),
        "ranker_config": ranker.config,
        "best_epoch": best_epoch,
        "best_validation_loss": best_validation,
        "metrics": metrics_path.name,
    }
    _atomic_json(destination / "run_manifest.json", manifest)
    return {"best_validation_loss": best_validation, "best_epoch": best_epoch, "run_dir": str(destination)}


def _config_digest(config: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(dict(config), sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _seed_food_training(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_verified_foundation(config: Mapping[str, Any], *, device: str | torch.device) -> tuple[Any, dict[str, Any]]:
    checkpoint = _mapping(config.get("checkpoint"), label="config.checkpoint")
    checkpoint_path = Path(_config_text(checkpoint, "path", label="config.checkpoint"))
    manifest_path = Path(_config_text(checkpoint, "manifest_path", label="config.checkpoint"))
    expected_sha = _config_text(checkpoint, "expected_sha256", label="config.checkpoint")
    _verify_sha256(checkpoint_path, expected_sha, label="Foundation checkpoint")
    embedder = load_foundation_checkpoint(checkpoint_path, manifest_path, map_location=device)
    return embedder, _read_json(manifest_path, label="Foundation checkpoint manifest")


def verify_foundation_checkpoint_binding(
    *,
    data: Mapping[str, Any],
    handoff: Mapping[str, Any],
    foundation_checkpoint_manifest: Mapping[str, Any],
) -> None:
    """Require the selected Foundation checkpoint to originate from this corpus."""

    foundation_corpus = _config_text(data, "foundation_corpus", label="config.data")
    if foundation_corpus not in {"linked", "control"}:
        raise FoodConfigurationError("config.data.foundation_corpus must be 'linked' or 'control'")
    checkpoint_manifest_key = f"{foundation_corpus}_manifest_sha256"
    if foundation_checkpoint_manifest.get(checkpoint_manifest_key) != handoff.get(
        "foundation_manifest_sha256"
    ):
        raise FoodConfigurationError(
            "Foundation checkpoint is not bound to the configured Food foundation corpus"
        )


def _food_model_settings(config: Mapping[str, Any]) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    model = _mapping(config.get("model"), label="config.model")
    training = _mapping(config.get("training"), label="config.training")
    for key in ("item_hash_buckets", "merchant_hash_buckets"):
        if not isinstance(model.get(key), int) or int(model[key]) < 2:
            raise FoodConfigurationError(f"config.model.{key} must be an integer >= 2")
    if not isinstance(model.get("dropout"), (int, float)) or not 0.0 <= float(model["dropout"]) < 1.0:
        raise FoodConfigurationError("config.model.dropout must be in [0, 1)")
    for key in ("batch_size", "epochs"):
        if not isinstance(training.get(key), int) or int(training[key]) < 1:
            raise FoodConfigurationError(f"config.training.{key} must be a positive integer")
    return model, training


def run_food_training(config: Mapping[str, Any]) -> dict[str, Any]:
    """Run a checksum-gated Food train/validation lifecycle from one YAML config."""

    handoff = validate_food_data_handoff(config)
    model_settings, training = _food_model_settings(config)
    seed = config.get("seed")
    if not isinstance(seed, int):
        raise FoodConfigurationError("config.seed must be an integer")
    _seed_food_training(seed)
    device = str(training.get("device", "cpu"))
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise FoodConfigurationError("CUDA was requested for Food training but is unavailable")
    embedder, foundation_checkpoint_manifest = _load_verified_foundation(config, device=device)
    data = _mapping(config.get("data"), label="config.data")
    verify_foundation_checkpoint_binding(
        data=data,
        handoff=handoff,
        foundation_checkpoint_manifest=foundation_checkpoint_manifest,
    )
    food_root = Path(_config_text(data, "opportunities_root", label="config.data"))
    foundation_root = Path(_config_text(data, "foundation_root", label="config.data"))
    vocabulary_path = Path(_config_text(data, "vocabulary_path", label="config.data"))
    max_tokens = data.get("max_tokens")
    if not isinstance(max_tokens, int) or max_tokens < 3:
        raise FoodConfigurationError("config.data.max_tokens must be an integer >= 3")
    training_splits: tuple[FoodSplit, FoodSplit] = ("train", "validation")
    rows = {
        split: load_green_sm_food_opportunities(food_root, split=split)
        for split in training_splits
    }
    samples = {
        split: resolve_green_sm_food_samples(
            rows[split], foundation_root=foundation_root, vocabulary_path=vocabulary_path,
            split=split, max_tokens=max_tokens,
        )
        for split in training_splits
    }
    features = FoodFeatureArtifacts.fit(rows["train"])
    batch_size = int(training["batch_size"])
    item_buckets = int(model_settings["item_hash_buckets"])
    merchant_buckets = int(model_settings["merchant_hash_buckets"])
    train_batches = build_food_batches(
        rows["train"], encoded_samples_by_id=samples["train"], feature_artifacts=features,
        batch_size=batch_size, item_hash_buckets=item_buckets, merchant_hash_buckets=merchant_buckets,
        require_labels=True,
    )
    validation_batches = build_food_batches(
        rows["validation"], encoded_samples_by_id=samples["validation"], feature_artifacts=features,
        batch_size=batch_size, item_hash_buckets=item_buckets, merchant_hash_buckets=merchant_buckets,
        require_labels=True,
    )
    ranker = FoodRanker(
        d_model=int(embedder.model.config.d_model), item_hash_buckets=item_buckets,
        merchant_hash_buckets=merchant_buckets, num_categories=len(features.category_vocabulary),
        num_geo_cells=len(features.geo_cell_vocabulary), dropout=float(model_settings["dropout"]),
    )
    ranker_config = _mapping(config.get("ranker"), label="config.ranker")
    run_dir = Path(_config_text(ranker_config, "run_dir", label="config.ranker"))
    metadata = {
        **handoff,
        "run_id": run_dir.name,
        "seed": seed,
        "config_digest": _config_digest(config),
        "selected_checkpoint_sha256": _config_text(config["checkpoint"], "expected_sha256", label="config.checkpoint"),
        "vocabulary_sha256": _sha256(vocabulary_path),
        "foundation_checkpoint_identity": foundation_checkpoint_manifest,
        "train_request_count": len(_group_requests(rows["train"])),
        "validation_request_count": len(_group_requests(rows["validation"])),
    }
    return train_food_ranker(
        embedder=embedder, ranker=ranker, train_batches=train_batches, validation_batches=validation_batches,
        feature_artifacts=features, run_dir=run_dir, epochs=int(training["epochs"]),
        learning_rate=float(training["learning_rate"]), weight_decay=float(training["weight_decay"]),
        device=device, run_metadata=metadata,
    )


def _load_food_ranker(run_dir: str | Path, *, expected_manifest_sha256: str | None, device: str | torch.device) -> tuple[FoodRanker, FoodFeatureArtifacts, dict[str, Any]]:
    directory = Path(run_dir)
    manifest_path = directory / "run_manifest.json"
    if expected_manifest_sha256 is not None:
        _verify_sha256(manifest_path, expected_manifest_sha256, label="Food run manifest")
    manifest = _read_json(manifest_path, label="Food run manifest")
    if manifest.get("schema_version") != "green_sm_food_ranker_v1":
        raise FoodConfigurationError("Food run manifest has unsupported schema_version")
    checkpoint_path = directory / str(manifest.get("best_checkpoint"))
    _verify_sha256(checkpoint_path, str(manifest.get("best_checkpoint_sha256")), label="Food best checkpoint")
    feature_path = directory / str(manifest.get("feature_artifacts"))
    _verify_sha256(feature_path, str(manifest.get("feature_artifacts_sha256")), label="Food feature artifacts")
    ranker_config = _mapping(manifest.get("ranker_config"), label="Food run manifest ranker_config")
    ranker = FoodRanker(**dict(ranker_config))
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    ranker.load_state_dict(payload["ranker_state_dict"])
    ranker.to(device).eval()
    return ranker, FoodFeatureArtifacts.load(feature_path), manifest


def verify_food_prediction_identity(
    *,
    food_run_manifest: Mapping[str, Any],
    handoff: Mapping[str, Any],
    expected_foundation_checkpoint_sha: str,
) -> None:
    """Reject a Food ranker trained from a different immutable handoff."""

    if food_run_manifest.get("selected_checkpoint_sha256") != expected_foundation_checkpoint_sha:
        raise FoodConfigurationError(
            "Food run was trained with a different Foundation checkpoint than this config"
        )
    for field, label in (
        ("food_manifest_sha256", "Food manifest"),
        ("foundation_manifest_sha256", "Foundation manifest"),
    ):
        if food_run_manifest.get(field) != handoff.get(field):
            raise FoodConfigurationError(f"Food run was trained with a different {label}")


def export_food_predictions(
    *,
    embedder: Any,
    ranker: FoodRanker,
    batches: Sequence[FoodBatch],
    rows: Sequence[FoodOpportunity],
    split: Literal["validation", "test"],
    output_dir: str | Path,
    device: str | torch.device,
    data_version: str,
    source_checksums: Mapping[str, str],
    run_metadata: Mapping[str, Any],
) -> OutputPaths:
    """Score an exact Food candidate universe and export label-free predictions."""

    if not batches:
        raise ValueError("Food prediction requires at least one batch")
    destination = Path(output_dir)
    if destination.exists() and any(destination.iterdir()):
        raise ValueError(f"Food prediction directory must be new and empty: {destination}")
    resolved_device = torch.device(device)
    embedder.to(resolved_device)
    embedder.eval()
    ranker.to(resolved_device)
    ranker.eval()
    predictions: list[PredictionRow] = []
    with torch.no_grad():
        for batch in batches:
            moved = _to_device_food_batch(batch, resolved_device)
            scores = frozen_foundation_food_logits(embedder, ranker, moved)
            predictions.extend(food_prediction_rows_from_batch(moved, scores.cpu(), split=split))
    return OutputContractV1.write_predictions(
        predictions,
        expected_candidates=expected_food_candidates(
            rows, data_version=data_version, source_checksums=source_checksums
        ),
        output_dir=destination,
        split=split,
        run_metadata=run_metadata,
    )


def run_food_prediction(
    config: Mapping[str, Any], *, split: Literal["validation", "test"]
) -> OutputPaths:
    """Load verified Food/Foundation artifacts and export one immutable split."""

    handoff = validate_food_data_handoff(config)
    model_settings, training = _food_model_settings(config)
    seed = config.get("seed")
    if not isinstance(seed, int):
        raise FoodConfigurationError("config.seed must be an integer")
    device = str(training.get("device", "cpu"))
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise FoodConfigurationError("CUDA was requested for Food prediction but is unavailable")
    embedder, foundation_checkpoint_manifest = _load_verified_foundation(config, device=device)
    data = _mapping(config.get("data"), label="config.data")
    verify_foundation_checkpoint_binding(
        data=data,
        handoff=handoff,
        foundation_checkpoint_manifest=foundation_checkpoint_manifest,
    )
    ranker_settings = _mapping(config.get("ranker"), label="config.ranker")
    run_dir = Path(_config_text(ranker_settings, "run_dir", label="config.ranker"))
    expected_run_manifest_sha256 = _config_text(
        ranker_settings, "expected_run_manifest_sha256", label="config.ranker"
    )
    ranker, features, food_run_manifest = _load_food_ranker(
        run_dir,
        expected_manifest_sha256=expected_run_manifest_sha256,
        device=device,
    )
    expected_foundation_checkpoint_sha = _config_text(
        _mapping(config.get("checkpoint"), label="config.checkpoint"),
        "expected_sha256",
        label="config.checkpoint",
    )
    verify_food_prediction_identity(
        food_run_manifest=food_run_manifest,
        handoff=handoff,
        expected_foundation_checkpoint_sha=expected_foundation_checkpoint_sha,
    )
    if ranker.config != {
        "d_model": int(embedder.model.config.d_model),
        "item_hash_buckets": int(model_settings["item_hash_buckets"]),
        "merchant_hash_buckets": int(model_settings["merchant_hash_buckets"]),
        "num_categories": len(features.category_vocabulary),
        "num_geo_cells": len(features.geo_cell_vocabulary),
        "dropout": float(model_settings["dropout"]),
    }:
        raise FoodConfigurationError("Food model config does not match the checksum-bound Food run")
    food_root = Path(_config_text(data, "opportunities_root", label="config.data"))
    foundation_root = Path(_config_text(data, "foundation_root", label="config.data"))
    vocabulary_path = Path(_config_text(data, "vocabulary_path", label="config.data"))
    max_tokens = data.get("max_tokens")
    if not isinstance(max_tokens, int) or max_tokens < 3:
        raise FoodConfigurationError("config.data.max_tokens must be an integer >= 3")
    rows = load_green_sm_food_opportunities(food_root, split=split)
    samples = resolve_green_sm_food_samples(
        rows,
        foundation_root=foundation_root,
        vocabulary_path=vocabulary_path,
        split=split,
        max_tokens=max_tokens,
    )
    batches = build_food_batches(
        rows,
        encoded_samples_by_id=samples,
        feature_artifacts=features,
        batch_size=int(training["batch_size"]),
        item_hash_buckets=int(model_settings["item_hash_buckets"]),
        merchant_hash_buckets=int(model_settings["merchant_hash_buckets"]),
        require_labels=split == "validation",
    )
    return export_food_predictions(
        embedder=embedder,
        ranker=ranker,
        batches=batches,
        rows=rows,
        split=split,
        output_dir=run_dir / "predictions" / split,
        device=device,
        data_version=handoff["data_version"],
        source_checksums={
            "food_manifest_sha256": handoff["food_manifest_sha256"],
            "foundation_manifest_sha256": handoff["foundation_manifest_sha256"],
        },
        run_metadata={
            "selected_checkpoint_sha256": expected_foundation_checkpoint_sha,
            "vocabulary_sha256": _sha256(vocabulary_path),
            "foundation_manifest_sha256": handoff["foundation_manifest_sha256"],
            "food_manifest_sha256": handoff["food_manifest_sha256"],
            "config_digest": str(food_run_manifest["config_digest"]),
            "seed": seed,
            "run_id": str(food_run_manifest["run_id"]),
            "score_semantics": "ranking",
        },
    )


def expected_food_candidates(
    rows: Sequence[FoodOpportunity],
    *,
    data_version: str,
    source_checksums: Mapping[str, str],
) -> VerifiedOpportunityIndex:
    """Build the immutable candidate universe permitted in Food prediction output."""

    expected = [
        ExpectedCandidate(
            request_id=row.request_id,
            candidate_id=row.candidate_id,
            user_id=row.user_id,
            cutoff_timestamp=datetime.fromtimestamp(row.cutoff_timestamp, tz=timezone.utc).isoformat(),
            split=row.split,  # type: ignore[arg-type]
        )
        for row in rows
        if row.split in {"validation", "test"}
    ]
    return VerifiedOpportunityIndex(
        expected,
        task="food_recommendation",
        data_version=data_version,
        source_checksums=source_checksums,
    )


def food_prediction_rows_from_batch(
    batch: FoodBatch, scores: torch.Tensor, *, split: Literal["validation", "test"]
) -> list[PredictionRow]:
    """Flatten original candidate rows only; padding and labels cannot leak into export."""

    if scores.shape != batch.candidate_mask.shape:
        raise ValueError("Food scores must align with request-major candidate_mask")
    rows: list[PredictionRow] = []
    for request_index, opportunities in enumerate(batch.opportunities):
        for candidate_index, opportunity in enumerate(opportunities):
            if not bool(batch.candidate_mask[request_index, candidate_index]):
                raise AssertionError("original Food opportunity was unexpectedly masked as padding")
            rows.append(
                PredictionRow(
                    request_id=opportunity.request_id,
                    candidate_id=opportunity.candidate_id,
                    score=float(scores[request_index, candidate_index]),
                    split=split,
                )
            )
    return rows


if __name__ == "__main__":  # pragma: no cover - exercised by the module CLI
    raise SystemExit(main())
