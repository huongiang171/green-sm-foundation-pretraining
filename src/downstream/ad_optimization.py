"""Frozen-foundation CTR ranking utilities for the Taobao Ads artifact."""

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
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import pyarrow.parquet as pq
import torch
import yaml
from torch import nn
from torch.nn import functional as F

from src.contracts.output import OutputContractV1, OutputPaths
from src.foundation_input.dataset import DatasetConfig, FoundationPretrainDataset, encode_sample
from src.foundation_input.loader import iter_normalized_samples
from src.foundation_input.tokenizer import TokenizerConfig
from src.foundation_input.types import EncodedSample, FoundationBatch
from src.foundation_input.vocab import VocabularyArtifacts
from src.training.checkpoint import load_foundation_checkpoint

SPLITS = frozenset({"train", "validation", "test"})
VOCABULARY_FIELDS = frozenset({"cate_id", "pid"})
OOV_FIELDS = ("candidate_id", "campaign_id", "cate_id", "user_id")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class AdConfigurationError(ValueError):
    """The Member 4 config or the Data-team handoff is invalid."""


class UpstreamHandoffError(RuntimeError):
    """A required Member 3/contract handoff has not been delivered yet."""


@dataclass(frozen=True)
class SelectedFoundationHandoff:
    """The concrete Foundation files selected by a published checkpoint manifest."""

    checkpoint_path: Path
    selection_manifest_path: Path
    vocabulary_path: Path
    selected_kind: str
    manifest: Mapping[str, Any]


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise AdConfigurationError(f"{label} not found: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise AdConfigurationError(f"{label} is not valid JSON: {path}") from error
    if not isinstance(value, dict):
        raise AdConfigurationError(f"{label} must contain a JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_selected_foundation_handoff(package_root: str | Path) -> SelectedFoundationHandoff:
    """Resolve the published Foundation selection without assuming `best.pt` exists."""

    root = Path(package_root)
    selection_manifest_path = root / "checkpoint_manifest.json"
    manifest = _read_json(selection_manifest_path, label="Foundation checkpoint manifest")
    if manifest.get("schema_version") != "green_sm_foundation_checkpoint_v1":
        raise AdConfigurationError("Foundation checkpoint manifest has unsupported schema_version")
    if manifest.get("checkpoint_format_version") != 1:
        raise AdConfigurationError("Foundation checkpoint manifest has unsupported format version")
    selected_kind = manifest.get("selected_kind")
    if selected_kind not in {"best", "last"}:
        raise AdConfigurationError("Foundation checkpoint manifest selected_kind must be 'best' or 'last'")
    checkpoint_name = manifest.get("checkpoint")
    checkpoint_sha = manifest.get("checkpoint_sha256")
    if not isinstance(checkpoint_name, str) or Path(checkpoint_name).name != checkpoint_name:
        raise AdConfigurationError("Foundation checkpoint manifest checkpoint must be a filename")
    if not isinstance(checkpoint_sha, str) or not _SHA256_RE.fullmatch(checkpoint_sha):
        raise AdConfigurationError("Foundation checkpoint manifest has invalid checkpoint_sha256")
    checkpoint_path = root / checkpoint_name
    _verify_sha256(checkpoint_path, checkpoint_sha, label="Selected Foundation checkpoint")
    vocabulary_path = root / "vocabulary.json"
    vocabulary_sha = manifest.get("vocabulary_sha256")
    if not isinstance(vocabulary_sha, str) or not _SHA256_RE.fullmatch(vocabulary_sha):
        raise AdConfigurationError("Foundation checkpoint manifest has invalid vocabulary_sha256")
    _verify_sha256(vocabulary_path, vocabulary_sha, label="Foundation vocabulary")
    if manifest.get("corpus") != "linked":
        raise AdConfigurationError("Ads requires a linked Foundation checkpoint")
    return SelectedFoundationHandoff(
        checkpoint_path=checkpoint_path,
        selection_manifest_path=selection_manifest_path,
        vocabulary_path=vocabulary_path,
        selected_kind=selected_kind,
        manifest=manifest,
    )


def _require_mapping(config: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = config.get(key)
    if not isinstance(value, Mapping):
        raise AdConfigurationError(f"config.{key} must be a mapping")
    return value


def _require_text(config: Mapping[str, Any], key: str, *, prefix: str = "config") -> str:
    value = config.get(key)
    if not isinstance(value, str) or not value.strip():
        raise AdConfigurationError(f"{prefix}.{key} must be a non-empty string")
    return value.strip()


def _verify_sha256(path: Path, expected: str, *, label: str) -> None:
    if not _SHA256_RE.fullmatch(expected):
        raise AdConfigurationError(f"{label} expected SHA-256 must be 64 lowercase hexadecimal characters")
    if not path.is_file():
        raise AdConfigurationError(f"{label} not found: {path}")
    actual = _sha256(path)
    if actual != expected:
        raise AdConfigurationError(f"{label} SHA-256 mismatch: expected {expected}, got {actual}")


def load_ad_optimization_config(path: str | Path) -> dict[str, Any]:
    """Load the explicit Member 4 YAML configuration without starting training."""

    config_path = Path(path)
    if not config_path.is_file():
        raise AdConfigurationError(f"config file not found: {config_path}")
    try:
        value = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        raise AdConfigurationError(f"config file is not valid YAML: {config_path}") from error
    if not isinstance(value, dict):
        raise AdConfigurationError("config root must be a mapping")
    return value


def validate_ads_data_handoff(config: Mapping[str, Any]) -> dict[str, Any]:
    """Verify the immutable Data-team artifacts before any model is constructed.

    This deliberately checks checksums and the cross-manifest link, so a Kaggle
    mount with stale or mixed artifacts is rejected before training.
    """

    if config.get("task") != "ad_optimization":
        raise AdConfigurationError("config.task must be 'ad_optimization'")
    data = _require_mapping(config, "data")
    ads_root = Path(_require_text(data, "opportunities_root", prefix="config.data"))
    foundation_root = Path(_require_text(data, "foundation_root", prefix="config.data"))
    ads_manifest_path = Path(_require_text(data, "manifest_path", prefix="config.data"))
    foundation_manifest_path = Path(_require_text(data, "foundation_manifest_path", prefix="config.data"))
    expected_ads_sha = _require_text(data, "expected_ads_manifest_sha256", prefix="config.data")
    expected_foundation_sha = _require_text(data, "expected_foundation_manifest_sha256", prefix="config.data")

    _verify_sha256(ads_manifest_path, expected_ads_sha, label="Ads manifest")
    _verify_sha256(foundation_manifest_path, expected_foundation_sha, label="Foundation manifest")
    ads_manifest = _read_json(ads_manifest_path, label="Ads manifest")
    foundation_manifest = _read_json(foundation_manifest_path, label="Foundation manifest")

    if ads_manifest.get("schema_version") != "taobao_ads_opportunities_v1":
        raise AdConfigurationError("Ads manifest has unsupported schema_version")
    if ads_manifest.get("data_version") != _require_text(data, "expected_data_version", prefix="config.data"):
        raise AdConfigurationError("Ads manifest data_version does not match config.data.expected_data_version")
    if ads_manifest.get("foundation_manifest_sha256") != expected_foundation_sha:
        raise AdConfigurationError("Ads manifest is not linked to the configured Foundation manifest")
    if foundation_manifest.get("schema_version") != "omnibehavior_pretraining_v2":
        raise AdConfigurationError("Foundation manifest has unsupported schema_version")
    if foundation_manifest.get("status") != "complete":
        raise AdConfigurationError("Foundation manifest status must be 'complete'")

    files = ads_manifest.get("files")
    if not isinstance(files, Mapping) or set(files) != SPLITS:
        raise AdConfigurationError("Ads manifest must contain exactly train, validation, and test files")
    for split in sorted(SPLITS):
        entry = files[split]
        if not isinstance(entry, Mapping):
            raise AdConfigurationError(f"Ads manifest files.{split} must be a mapping")
        filename = _require_text(entry, "file", prefix=f"Ads manifest files.{split}")
        checksum = _require_text(entry, "sha256", prefix=f"Ads manifest files.{split}")
        _verify_sha256(ads_root / filename, checksum, label=f"Ads {split} parquet")

    if not ads_root.is_dir() or not foundation_root.is_dir():
        raise AdConfigurationError("configured opportunities_root and foundation_root must be directories")
    return {
        "ads_manifest_sha256": expected_ads_sha,
        "foundation_manifest_sha256": expected_foundation_sha,
        "data_version": ads_manifest["data_version"],
        "splits": sorted(SPLITS),
    }


def validate_green_sm_ads_data_handoff(config: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the immutable full linked Green SM Ads handoff before training."""

    if config.get("task") != "ad_optimization" or config.get("freeze_foundation") is not True:
        raise AdConfigurationError("Green SM Ads requires task='ad_optimization' and freeze_foundation=true")
    data = _require_mapping(config, "data")
    if _require_text(data, "data_version", prefix="config.data") != "green_sm_sim_full_v1":
        raise AdConfigurationError("Green SM Ads requires data_version='green_sm_sim_full_v1'")
    if _require_text(data, "corpus", prefix="config.data") != "linked":
        raise AdConfigurationError("Green SM Ads requires data.corpus='linked'")
    ads_root = Path(_require_text(data, "opportunities_root", prefix="config.data"))
    foundation_root = Path(_require_text(data, "foundation_root", prefix="config.data"))
    ads_manifest_path = Path(_require_text(data, "manifest_path", prefix="config.data"))
    foundation_manifest_path = Path(_require_text(data, "foundation_manifest_path", prefix="config.data"))
    report_path = Path(_require_text(data, "quality_report_path", prefix="config.data"))
    expected_ads_sha = _require_text(data, "expected_ads_manifest_sha256", prefix="config.data")
    expected_foundation_sha = _require_text(data, "expected_foundation_manifest_sha256", prefix="config.data")
    _verify_sha256(ads_manifest_path, expected_ads_sha, label="Ads manifest")
    _verify_sha256(foundation_manifest_path, expected_foundation_sha, label="Foundation manifest")
    ads_manifest = _read_json(ads_manifest_path, label="Ads manifest")
    report = _read_json(report_path, label="Green SM data quality report")
    if report.get("readiness_passed") is not True:
        raise AdConfigurationError("Green SM quality report must have readiness_passed=true")
    if report.get("foundation_manifest_sha256") != expected_foundation_sha:
        raise AdConfigurationError("Green SM quality report is not bound to the configured Foundation manifest")
    if ads_manifest.get("schema_version") != "green_sm_ads_opportunities_v1":
        raise AdConfigurationError("Ads manifest has unsupported schema_version")
    if ads_manifest.get("data_version") != "green_sm_sim_full_v1":
        raise AdConfigurationError("Ads manifest data_version must be green_sm_sim_full_v1")
    if ads_manifest.get("foundation_manifest_sha256") != expected_foundation_sha:
        raise AdConfigurationError("Ads manifest is not linked to the configured Foundation manifest")
    files = ads_manifest.get("files")
    if not isinstance(files, Mapping) or set(files) != SPLITS:
        raise AdConfigurationError("Ads manifest must contain exactly train, validation, and test files")
    for split in sorted(SPLITS):
        entry = files[split]
        if not isinstance(entry, Mapping):
            raise AdConfigurationError(f"Ads manifest files.{split} must be a mapping")
        filename = _require_text(entry, "file", prefix=f"Ads manifest files.{split}")
        if Path(filename).name != filename:
            raise AdConfigurationError(f"Ads manifest files.{split}.file must be a filename")
        checksum = _require_text(entry, "sha256", prefix=f"Ads manifest files.{split}")
        _verify_sha256(ads_root / filename, checksum, label=f"Ads {split} parquet")
    if not ads_root.is_dir() or not foundation_root.is_dir():
        raise AdConfigurationError("configured opportunities_root and foundation_root must be directories")
    checkpoint = _require_mapping(config, "checkpoint")
    expected_selected_sha = _require_text(
        checkpoint, "expected_selected_checkpoint_sha256", prefix="config.checkpoint"
    )
    if not _SHA256_RE.fullmatch(expected_selected_sha):
        raise AdConfigurationError("config.checkpoint.expected_selected_checkpoint_sha256 must be a SHA-256")
    selected = resolve_selected_foundation_handoff(
        _require_text(checkpoint, "package_root", prefix="config.checkpoint")
    )
    if selected.manifest.get("checkpoint_sha256") != expected_selected_sha:
        raise AdConfigurationError("selected Foundation checkpoint SHA does not match config")
    if selected.manifest.get("linked_manifest_sha256") != expected_foundation_sha:
        raise AdConfigurationError("selected Foundation checkpoint is not bound to configured Foundation manifest")
    return {
        "data_version": "green_sm_sim_full_v1",
        "ads_manifest_sha256": expected_ads_sha,
        "foundation_manifest_sha256": expected_foundation_sha,
        "selected_kind": selected.selected_kind,
        "selected_checkpoint_sha256": expected_selected_sha,
        "vocabulary_sha256": selected.manifest["vocabulary_sha256"],
    }


def validate_member3_handoff(config: Mapping[str, Any]) -> None:
    """Load only the public checksum/identity-verified frozen Foundation artifact."""

    if config.get("freeze_foundation") is not True:
        raise AdConfigurationError("config.freeze_foundation must be true for the frozen-foundation Ads task")
    checkpoint = _require_mapping(config, "checkpoint")
    expected_sha = _require_text(checkpoint, "expected_sha256", prefix="config.checkpoint")
    if expected_sha.startswith("supplied-by-member-3"):
        raise UpstreamHandoffError(
            "Member 3 checkpoint manifest has not been supplied: replace "
            "checkpoint.expected_sha256 and add the checkpoint artifact before training."
        )
    checkpoint_path = Path(_require_text(checkpoint, "path", prefix="config.checkpoint"))
    _verify_sha256(checkpoint_path, expected_sha, label="Foundation checkpoint")
    manifest_path = Path(_require_text(checkpoint, "manifest_path", prefix="config.checkpoint"))
    try:
        from src.contracts.output import OutputContractV1  # noqa: F401
        from src.training.checkpoint import load_foundation_checkpoint

        load_foundation_checkpoint(checkpoint_path, manifest_path)
    except (ModuleNotFoundError, ValueError) as error:
        raise UpstreamHandoffError(f"Member 3 checkpoint handoff is invalid: {error}") from error


def main(argv: Sequence[str] | None = None) -> int:
    """Run the Green SM Ads validation, training, or prediction lifecycle."""

    parser = argparse.ArgumentParser(description="Green SM frozen-Foundation Ads CTR ranking")
    parser.add_argument("--config", required=True, help="Ads YAML configuration")
    parser.add_argument("--mode", choices=("validate", "train", "predict"))
    parser.add_argument("--split", choices=("validation", "test"))
    parser.add_argument("--validate-data", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--preflight", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.validate_data or args.preflight:
        try:
            config = load_ad_optimization_config(args.config)
            report = validate_ads_data_handoff(config)
            if args.validate_data:
                print(json.dumps({"status": "data_handoff_valid", **report}, sort_keys=True))
                return 0
            validate_member3_handoff(config)
        except UpstreamHandoffError as error:
            print(f"upstream handoff blocked: {error}", file=sys.stderr)
            return 3
        except AdConfigurationError as error:
            print(f"configuration error: {error}", file=sys.stderr)
            return 2
        print(json.dumps({"status": "preflight_valid", **report}, sort_keys=True))
        return 0
    if args.mode is None:
        parser.error("--mode is required")
    try:
        config = load_ad_optimization_config(args.config)
        report = validate_green_sm_ads_data_handoff(config)
    except AdConfigurationError as error:
        print(f"Ads configuration error: {error}", file=sys.stderr)
        return 2
    if args.mode == "validate":
        print(json.dumps({"status": "ads_data_handoff_valid", **report}, sort_keys=True))
        return 0
    try:
        if args.mode == "train":
            result = run_ads_training(config)
            print(json.dumps({"status": "ads_training_complete", **result}, sort_keys=True))
            return 0
        if args.split is None:
            parser.error("--split is required when --mode predict")
        result = run_ads_prediction(config, split=args.split)
        print(json.dumps({"status": "ads_prediction_exported", "predictions": str(result.predictions)}))
        return 0
    except (AdConfigurationError, ValueError) as error:
        print(f"Ads configuration error: {error}", file=sys.stderr)
        return 2


def _text(value: Any, *, field: str, request_id: str = "<unknown>") -> str:
    if value is None or not str(value).strip():
        raise ValueError(f"request_id={request_id!r}: {field} must be non-empty")
    return str(value).strip()


def _integer(value: Any, *, field: str, request_id: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"request_id={request_id!r}: {field} must be an integer")
    try:
        return int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"request_id={request_id!r}: {field} must be an integer") from error


def _timestamp(value: Any, *, field: str, request_id: str) -> int:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            raise ValueError(f"request_id={request_id!r}: {field} must include a timezone")
        return int(value.timestamp())
    return _integer(value, field=field, request_id=request_id)


def _finite_float(value: Any, *, field: str, request_id: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"request_id={request_id!r}: {field} must be finite")
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"request_id={request_id!r}: {field} must be finite") from error
    if not math.isfinite(number):
        raise ValueError(f"request_id={request_id!r}: {field} must be finite")
    return number


@dataclass(frozen=True)
class GreenSmAdsOpportunity:
    """One native Green SM Ads candidate with optional training label."""

    request_id: str
    user_id: str
    foundation_sample_id: str
    cutoff_timestamp: int
    candidate_id: str
    campaign_id: str
    candidate_category: str
    placement: str
    candidate_geo_cell: str
    candidate_lat: float
    candidate_lon: float
    candidate_available_timestamp: int
    exposure_timestamp: int
    split: str
    clicked: int | None

    @classmethod
    def from_mapping(
        cls, row: Mapping[str, Any], *, require_label: bool
    ) -> GreenSmAdsOpportunity:
        request_id = _text(row.get("request_id"), field="request_id")
        fields = (
            "user_id", "foundation_sample_id", "candidate_id", "campaign_id",
            "candidate_category", "placement", "candidate_geo_cell", "split",
        )
        values = {
            field: _text(row.get(field), field=field, request_id=request_id)
            for field in fields
        }
        if values["split"] not in SPLITS:
            raise ValueError(f"request_id={request_id!r}: unsupported split {values['split']!r}")
        cutoff = _timestamp(row.get("cutoff_timestamp"), field="cutoff_timestamp", request_id=request_id)
        available = _timestamp(
            row.get("candidate_available_timestamp"),
            field="candidate_available_timestamp", request_id=request_id,
        )
        exposure = _timestamp(
            row.get("exposure_timestamp"), field="exposure_timestamp", request_id=request_id
        )
        if available > cutoff or exposure > cutoff:
            raise ValueError(f"request_id={request_id!r}: candidate timestamps exceed cutoff_timestamp")
        clicked: int | None = None
        if require_label:
            clicked = _integer(row.get("clicked"), field="clicked", request_id=request_id)
            if clicked not in {0, 1}:
                raise ValueError(f"request_id={request_id!r}: clicked must be 0 or 1")
        return cls(
            request_id=request_id,
            user_id=values["user_id"],
            foundation_sample_id=values["foundation_sample_id"],
            cutoff_timestamp=cutoff,
            candidate_id=values["candidate_id"],
            campaign_id=values["campaign_id"],
            candidate_category=values["candidate_category"],
            placement=values["placement"],
            candidate_geo_cell=values["candidate_geo_cell"],
            candidate_lat=_finite_float(row.get("candidate_lat"), field="candidate_lat", request_id=request_id),
            candidate_lon=_finite_float(row.get("candidate_lon"), field="candidate_lon", request_id=request_id),
            candidate_available_timestamp=available,
            exposure_timestamp=exposure,
            split=values["split"],
            clicked=clicked,
        )


@dataclass(frozen=True)
class AdOpportunity:
    """One exposed candidate from the immutable Taobao Ads handoff."""

    request_id: str
    user_id: str
    foundation_sample_id: str
    cutoff_timestamp: int
    candidate_id: str
    clicked: int
    split: str
    exposure_timestamp: int
    candidate_available_timestamp: int
    pid: str
    cate_id: str
    campaign_id: str

    @classmethod
    def from_mapping(cls, row: Mapping[str, Any]) -> AdOpportunity:
        request_id = _text(row.get("request_id"), field="request_id")
        fields = (
            "user_id",
            "foundation_sample_id",
            "candidate_id",
            "split",
            "pid",
            "cate_id",
            "campaign_id",
        )
        values = {
            field: _text(row.get(field), field=field, request_id=request_id) for field in fields
        }
        split = values["split"]
        if split not in SPLITS:
            raise ValueError(f"request_id={request_id!r}: unsupported split {split!r}")
        cutoff = _integer(row.get("cutoff_timestamp"), field="cutoff_timestamp", request_id=request_id)
        exposure = _integer(
            row.get("exposure_timestamp"), field="exposure_timestamp", request_id=request_id
        )
        available = _integer(
            row.get("candidate_available_timestamp"),
            field="candidate_available_timestamp",
            request_id=request_id,
        )
        clicked = _integer(row.get("clicked"), field="clicked", request_id=request_id)
        if clicked not in {0, 1}:
            raise ValueError(f"request_id={request_id!r}: clicked must be 0 or 1")
        if exposure > cutoff:
            raise ValueError(
                f"request_id={request_id!r}: exposure_timestamp exceeds cutoff_timestamp"
            )
        if available > cutoff:
            raise ValueError(
                f"request_id={request_id!r}: candidate_available_timestamp exceeds cutoff_timestamp"
            )
        return cls(
            request_id=request_id,
            user_id=values["user_id"],
            foundation_sample_id=values["foundation_sample_id"],
            cutoff_timestamp=cutoff,
            candidate_id=values["candidate_id"],
            clicked=clicked,
            split=split,
            exposure_timestamp=exposure,
            candidate_available_timestamp=available,
            pid=values["pid"],
            cate_id=values["cate_id"],
            campaign_id=values["campaign_id"],
        )


def validate_opportunities(
    rows: Iterable[AdOpportunity], *, expected_split: str | None = None
) -> tuple[AdOpportunity, ...]:
    """Validate candidate-set integrity without silently changing the handoff."""

    validated = tuple(rows)
    if expected_split is not None and expected_split not in SPLITS:
        raise ValueError(f"expected_split must be one of {sorted(SPLITS)}, got {expected_split!r}")
    per_request: dict[str, AdOpportunity] = {}
    seen_pairs: set[tuple[str, str]] = set()
    for row in validated:
        if expected_split is not None and row.split != expected_split:
            raise ValueError(
                f"request_id={row.request_id!r}: split {row.split!r} does not match {expected_split!r}"
            )
        previous = per_request.setdefault(row.request_id, row)
        for field in ("user_id", "foundation_sample_id", "cutoff_timestamp", "split"):
            if getattr(previous, field) != getattr(row, field):
                raise ValueError(f"request_id={row.request_id!r}: inconsistent {field}")
        pair = (row.request_id, row.candidate_id)
        if pair in seen_pairs:
            raise ValueError(
                f"duplicate candidate in request_id={row.request_id!r}: {row.candidate_id!r}"
            )
        seen_pairs.add(pair)
    return tuple(sorted(validated, key=lambda row: (row.request_id, row.candidate_id)))


def load_ad_opportunities(path: str | Path, *, expected_split: str) -> tuple[AdOpportunity, ...]:
    """Load a single split of contract-valid opportunity rows from Parquet."""

    parquet_path = Path(path)
    if not parquet_path.is_file():
        raise ValueError(f"Ads opportunities file does not exist: {parquet_path}")
    return validate_opportunities(
        (AdOpportunity.from_mapping(row) for row in pq.read_table(parquet_path).to_pylist()),
        expected_split=expected_split,
    )


def fit_train_vocabulary(
    train_rows: Sequence[AdOpportunity], *, field: Literal["cate_id", "pid"]
) -> dict[str, int]:
    """Fit low-cardinality categorical IDs strictly on the train split."""

    if field not in VOCABULARY_FIELDS:
        raise ValueError(f"Unsupported train-only vocabulary field: {field!r}")
    validated = validate_opportunities(train_rows, expected_split="train")
    values = sorted({str(getattr(row, field)) for row in validated})
    return {"<PAD>": 0, "<UNK>": 1, **{value: index + 2 for index, value in enumerate(values)}}


def encode_train_category(value: str, vocabulary: Mapping[str, int]) -> int:
    """Encode a train-fitted category or placement with deterministic UNK fallback."""

    try:
        unknown = int(vocabulary["<UNK>"])
    except KeyError as error:
        raise ValueError("Vocabulary lacks required <UNK> entry") from error
    return int(vocabulary.get(value, unknown))


def stable_hash_id(field_name: str, value: str, *, bucket_count: int) -> int:
    """Return a deterministic non-padding embedding ID without fitting on future data."""

    if bucket_count < 3:
        raise ValueError("bucket_count must reserve PAD=0 and UNK=1")
    if not value:
        return 1
    digest = hashlib.sha256(f"taobao-ads-v1\0{field_name}\0{value}".encode()).digest()
    return 2 + int.from_bytes(digest[:8], "big") % (bucket_count - 2)


def summarize_temporal_oov(
    train_rows: Sequence[AdOpportunity], eval_rows: Sequence[AdOpportunity]
) -> dict[str, dict[str, int | float]]:
    """Describe future-catalog OOV without mutating rows or fitting evaluation values."""

    train = validate_opportunities(train_rows, expected_split="train")
    evaluation = validate_opportunities(eval_rows)
    result: dict[str, dict[str, int | float]] = {}
    for field in OOV_FIELDS:
        known = {str(getattr(row, field)) for row in train}
        values = [str(getattr(row, field)) for row in evaluation]
        unseen = [value for value in values if value not in known]
        result[field] = {
            "rows": len(values),
            "unseen_rows": len(unseen),
            "unseen_rate": 0.0 if not values else len(unseen) / len(values),
            "distinct_unseen": len(set(unseen)),
        }
    return result


@dataclass
class AdBatch:
    """A request-major batch with padded candidate fields and a Foundation batch."""

    foundation_batch: FoundationBatch
    candidate_hash_ids: torch.LongTensor
    campaign_hash_ids: torch.LongTensor
    category_ids: torch.LongTensor
    placement_ids: torch.LongTensor
    clicked: torch.FloatTensor
    candidate_mask: torch.BoolTensor
    opportunities: tuple[tuple[AdOpportunity, ...], ...]


def _group_requests(rows: Sequence[AdOpportunity]) -> tuple[tuple[AdOpportunity, ...], ...]:
    grouped: dict[str, list[AdOpportunity]] = defaultdict(list)
    for row in rows:
        grouped[row.request_id].append(row)
    return tuple(tuple(grouped[key]) for key in sorted(grouped))


def build_ad_batches(
    rows: Sequence[AdOpportunity],
    *,
    encoded_samples_by_id: Mapping[str, EncodedSample],
    category_vocabulary: Mapping[str, int],
    placement_vocabulary: Mapping[str, int],
    candidate_hash_buckets: int,
    campaign_hash_buckets: int,
    batch_size: int,
) -> tuple[AdBatch, ...]:
    """Create request-major batches without sampling or dropping candidates."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    validated = validate_opportunities(rows)
    grouped = _group_requests(validated)
    batches: list[AdBatch] = []
    for offset in range(0, len(grouped), batch_size):
        requests = grouped[offset : offset + batch_size]
        samples: list[EncodedSample] = []
        for request in requests:
            reference = request[0]
            if not request:
                raise ValueError("Cannot construct a batch with an empty request")
            sample = encoded_samples_by_id.get(reference.foundation_sample_id)
            if sample is None:
                raise ValueError(
                    f"request_id={reference.request_id!r}: missing foundation_sample_id "
                    f"{reference.foundation_sample_id!r}"
                )
            samples.append(sample)
        from src.foundation_input.dataset import foundation_collate

        foundation_batch = foundation_collate(samples)
        width = max(len(request) for request in requests)
        shape = (len(requests), width)
        candidate_hash_ids = torch.zeros(shape, dtype=torch.long)
        campaign_hash_ids = torch.zeros(shape, dtype=torch.long)
        category_ids = torch.zeros(shape, dtype=torch.long)
        placement_ids = torch.zeros(shape, dtype=torch.long)
        clicked = torch.zeros(shape, dtype=torch.float32)
        candidate_mask = torch.zeros(shape, dtype=torch.bool)
        for request_index, request in enumerate(requests):
            for candidate_index, row in enumerate(request):
                candidate_hash_ids[request_index, candidate_index] = stable_hash_id(
                    "candidate_id", row.candidate_id, bucket_count=candidate_hash_buckets
                )
                campaign_hash_ids[request_index, candidate_index] = stable_hash_id(
                    "campaign_id", row.campaign_id, bucket_count=campaign_hash_buckets
                )
                category_ids[request_index, candidate_index] = encode_train_category(
                    row.cate_id, category_vocabulary
                )
                placement_ids[request_index, candidate_index] = encode_train_category(
                    row.pid, placement_vocabulary
                )
                clicked[request_index, candidate_index] = float(row.clicked)
                candidate_mask[request_index, candidate_index] = True
        batches.append(
            AdBatch(
                foundation_batch=foundation_batch,
                candidate_hash_ids=candidate_hash_ids,
                campaign_hash_ids=campaign_hash_ids,
                category_ids=category_ids,
                placement_ids=placement_ids,
                clicked=clicked,
                candidate_mask=candidate_mask,
                opportunities=requests,
            )
        )
    return tuple(batches)


def _validate_feature_ids(name: str, value: torch.Tensor, *, size: int) -> None:
    if value.ndim != 2 or value.dtype != torch.long:
        raise ValueError(f"{name} must be a torch.long tensor with shape [B, C]")
    if value.numel() and (value.min().item() < 0 or value.max().item() >= size):
        raise ValueError(f"{name} contains IDs outside [0, {size})")


class AdRanker(nn.Module):
    """Candidate scorer consuming only the public Foundation user embeddings."""

    def __init__(
        self,
        *,
        candidate_buckets: int,
        campaign_buckets: int,
        num_categories: int,
        num_placements: int,
        d_model: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if min(candidate_buckets, campaign_buckets, num_categories, num_placements, d_model) <= 0:
            raise ValueError("Embedding cardinalities and d_model must be positive")
        if candidate_buckets < 3 or campaign_buckets < 3:
            raise ValueError("Hash buckets must reserve PAD=0 and UNK=1")
        self.candidate_buckets = candidate_buckets
        self.campaign_buckets = campaign_buckets
        self.num_categories = num_categories
        self.num_placements = num_placements
        self.d_model = d_model
        self.candidate_embedding = nn.Embedding(candidate_buckets, d_model, padding_idx=0)
        self.campaign_embedding = nn.Embedding(campaign_buckets, d_model, padding_idx=0)
        self.category_embedding = nn.Embedding(num_categories, d_model, padding_idx=0)
        self.placement_embedding = nn.Embedding(num_placements, d_model, padding_idx=0)
        self.user_fusion = nn.Sequential(
            nn.LayerNorm(d_model * 2), nn.Linear(d_model * 2, d_model), nn.GELU()
        )
        self.candidate_fusion = nn.Sequential(
            nn.LayerNorm(d_model * 4), nn.Linear(d_model * 4, d_model), nn.GELU()
        )
        self.scorer = nn.Sequential(
            nn.LayerNorm(d_model * 2),
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(
        self,
        long_term: torch.Tensor,
        short_term: torch.Tensor,
        *,
        candidate_hash_ids: torch.LongTensor,
        campaign_hash_ids: torch.LongTensor,
        category_ids: torch.LongTensor,
        placement_ids: torch.LongTensor,
    ) -> torch.Tensor:
        if (
            long_term.ndim != 2
            or long_term.shape != short_term.shape
            or long_term.shape[1] != self.d_model
            or not torch.is_floating_point(long_term)
            or not torch.is_floating_point(short_term)
            or not torch.isfinite(long_term).all()
            or not torch.isfinite(short_term).all()
        ):
            raise ValueError("long_term and short_term must be finite floating [B, d_model] tensors")
        expected = candidate_hash_ids.shape
        for name, tensor, cardinality in (
            ("candidate_hash_ids", candidate_hash_ids, self.candidate_buckets),
            ("campaign_hash_ids", campaign_hash_ids, self.campaign_buckets),
            ("category_ids", category_ids, self.num_categories),
            ("placement_ids", placement_ids, self.num_placements),
        ):
            _validate_feature_ids(name, tensor, size=cardinality)
            if tensor.shape != expected:
                raise ValueError("All candidate feature tensors must share shape [B, C]")
        if expected[0] != long_term.shape[0]:
            raise ValueError("Candidate batch dimension must match Foundation embeddings")
        user = self.user_fusion(torch.cat((long_term, short_term), dim=-1))
        candidate = self.candidate_fusion(
            torch.cat(
                (
                    self.candidate_embedding(candidate_hash_ids),
                    self.campaign_embedding(campaign_hash_ids),
                    self.category_embedding(category_ids),
                    self.placement_embedding(placement_ids),
                ),
                dim=-1,
            )
        )
        logits = self.scorer(
            torch.cat((user.unsqueeze(1).expand(-1, expected[1], -1), candidate), dim=-1)
        ).squeeze(-1)
        if not torch.isfinite(logits).all():
            raise ValueError("AdRanker produced non-finite logits")
        return logits


def masked_binary_cross_entropy(
    logits: torch.Tensor, clicked: torch.Tensor, candidate_mask: torch.BoolTensor
) -> tuple[torch.Tensor, int]:
    """Binary CTR loss over only handed-off candidates, including no/multi-click requests."""

    if logits.ndim != 2 or clicked.shape != logits.shape or candidate_mask.shape != logits.shape:
        raise ValueError("logits, clicked, and candidate_mask must share shape [B, C]")
    if clicked.dtype != torch.float32 or candidate_mask.dtype != torch.bool:
        raise ValueError("clicked must be float32 and candidate_mask must be bool")
    if not torch.all((clicked == 0.0) | (clicked == 1.0)):
        raise ValueError("clicked must contain only 0.0 or 1.0")
    count = int(candidate_mask.sum().item())
    if count == 0:
        return logits.sum() * 0.0, 0
    return F.binary_cross_entropy_with_logits(logits[candidate_mask], clicked[candidate_mask]), count


def score_probabilities(logits: torch.Tensor, candidate_mask: torch.BoolTensor) -> torch.Tensor:
    """Convert finite raw logits to probabilities while preserving padded-cell semantics."""

    if logits.ndim != 2 or candidate_mask.shape != logits.shape or candidate_mask.dtype != torch.bool:
        raise ValueError("logits and candidate_mask must share shape [B, C], with bool mask")
    if not torch.isfinite(logits[candidate_mask]).all():
        raise ValueError("Real candidate logits must be finite")
    probabilities = torch.sigmoid(logits)
    if not torch.all((probabilities[candidate_mask] >= 0.0) & (probabilities[candidate_mask] <= 1.0)):
        raise ValueError("Real candidate probabilities must lie in [0, 1]")
    return probabilities.masked_fill(~candidate_mask, 0.0)


def load_encoded_foundation_samples_for_ads(
    rows: Sequence[AdOpportunity],
    *,
    foundation_root: str | Path,
    vocabulary_path: str | Path,
    split: str,
    max_tokens: int,
    tokenizer_config: TokenizerConfig | None = None,
) -> dict[str, EncodedSample]:
    """Resolve Ads rows to aligned, vocabulary-bound Foundation encoded samples."""

    validated = validate_opportunities(rows, expected_split=split)
    by_sample: dict[str, AdOpportunity] = {}
    for row in validated:
        existing = by_sample.setdefault(row.foundation_sample_id, row)
        if (
            existing.user_id != row.user_id
            or existing.cutoff_timestamp != row.cutoff_timestamp
            or existing.split != row.split
        ):
            raise ValueError(f"foundation_sample_id={row.foundation_sample_id!r}: inconsistent Ads rows")
    artifacts = VocabularyArtifacts.load(vocabulary_path)
    dataset = FoundationPretrainDataset(
        str(foundation_root), split, artifacts, tokenizer_config=tokenizer_config,
        dataset_config=DatasetConfig(max_tokens=max_tokens)
    )
    found: dict[str, EncodedSample] = {}
    for canonical in iter_normalized_samples(foundation_root, split, validate=True):
        opportunity = by_sample.get(canonical.sample_id)
        if opportunity is None:
            continue
        if canonical.user_id != opportunity.user_id:
            raise ValueError(f"{canonical.sample_id!r}: user_id does not match Ads opportunity")
        if int(canonical.cutoff_time.timestamp()) != opportunity.cutoff_timestamp:
            raise ValueError(f"{canonical.sample_id!r}: cutoff_timestamp does not match Ads opportunity")
        if canonical.split != opportunity.split:
            raise ValueError(f"{canonical.sample_id!r}: split does not match Ads opportunity")
        if canonical.sample_id in found:
            raise ValueError(f"Duplicate canonical foundation_sample_id {canonical.sample_id!r}")
        found[canonical.sample_id] = encode_sample(
            canonical, artifacts, dataset.tokenizer_config, dataset.dataset_config
        )
    missing = sorted(set(by_sample) - set(found))
    if missing:
        raise ValueError(f"foundation_sample_id not found in split {split!r}: {missing[0]!r}")
    return found


def _to_device_ad_batch(batch: AdBatch, device: torch.device) -> AdBatch:
    copied = copy.copy(batch)
    foundation = copy.copy(batch.foundation_batch)
    for name, value in vars(foundation).items():
        if isinstance(value, torch.Tensor):
            setattr(foundation, name, value.to(device))
        elif name == "targets" and isinstance(value, dict):
            foundation.targets = {key: tensor.to(device) for key, tensor in value.items()}
    copied.foundation_batch = foundation
    for name in ("candidate_hash_ids", "campaign_hash_ids", "category_ids", "placement_ids", "clicked", "candidate_mask"):
        setattr(copied, name, getattr(batch, name).to(device))
    return copied


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        json.dump(value, handle, sort_keys=True, indent=2)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _atomic_torch_save(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
    torch.save(dict(value), temporary)
    os.replace(temporary, path)


def _ads_settings(config: Mapping[str, Any]) -> tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]:
    model = _require_mapping(config, "model")
    training = _require_mapping(config, "training")
    ranker = _require_mapping(config, "ranker")
    for key in ("candidate_hash_buckets", "campaign_hash_buckets"):
        if not isinstance(model.get(key), int) or int(model[key]) < 3:
            raise AdConfigurationError(f"config.model.{key} must be an integer >= 3")
    for key in ("batch_size", "epochs"):
        if not isinstance(training.get(key), int) or int(training[key]) < 1:
            raise AdConfigurationError(f"config.training.{key} must be a positive integer")
    if not isinstance(training.get("learning_rate"), (int, float)) or float(training["learning_rate"]) <= 0:
        raise AdConfigurationError("config.training.learning_rate must be positive")
    return model, training, ranker


def _load_ads_foundation(config: Mapping[str, Any], *, device: str) -> tuple[Any, SelectedFoundationHandoff]:
    checkpoint = _require_mapping(config, "checkpoint")
    handoff = resolve_selected_foundation_handoff(_require_text(checkpoint, "package_root", prefix="config.checkpoint"))
    expected = _require_text(checkpoint, "expected_selected_checkpoint_sha256", prefix="config.checkpoint")
    if handoff.manifest["checkpoint_sha256"] != expected:
        raise AdConfigurationError("selected Foundation checkpoint SHA does not match config")
    return load_foundation_checkpoint(handoff.checkpoint_path, handoff.selection_manifest_path, map_location=device), handoff


def _load_ads_rows_and_batches(config: Mapping[str, Any], *, split: str, features: Mapping[str, Mapping[str, int]] | None = None) -> tuple[tuple[AdOpportunity, ...], tuple[AdBatch, ...], dict[str, dict[str, int]]]:
    from src.downstream.green_sm_ad_optimization import (
        load_green_sm_ads_opportunities,
        resolve_green_sm_ads_samples,
    )

    data = _require_mapping(config, "data")
    model, training, _ = _ads_settings(config)
    rows = load_green_sm_ads_opportunities(_require_text(data, "opportunities_root", prefix="config.data"), split=split)
    vocabulary_path = resolve_selected_foundation_handoff(_require_text(_require_mapping(config, "checkpoint"), "package_root", prefix="config.checkpoint")).vocabulary_path
    samples = resolve_green_sm_ads_samples(rows, foundation_root=_require_text(data, "foundation_root", prefix="config.data"), vocabulary_path=vocabulary_path, split=split, max_tokens=int(data.get("max_tokens", 256)))
    fitted = dict(features or {})
    if not fitted:
        if split != "train":
            raise AdConfigurationError("Ads feature artifacts must be supplied for evaluation")
        fitted = {"category": fit_train_vocabulary(rows, field="cate_id"), "placement": fit_train_vocabulary(rows, field="pid")}
    batches = build_ad_batches(rows, encoded_samples_by_id=samples, category_vocabulary=fitted["category"], placement_vocabulary=fitted["placement"], candidate_hash_buckets=int(model["candidate_hash_buckets"]), campaign_hash_buckets=int(model["campaign_hash_buckets"]), batch_size=int(training["batch_size"]))
    return rows, batches, fitted


def _ads_logits(embedder: Any, ranker: AdRanker, batch: AdBatch) -> torch.Tensor:
    with torch.no_grad():
        embeddings = embedder.embed_users(batch.foundation_batch)
    return ranker(embeddings["long_term"], embeddings["short_term"], candidate_hash_ids=batch.candidate_hash_ids, campaign_hash_ids=batch.campaign_hash_ids, category_ids=batch.category_ids, placement_ids=batch.placement_ids)


def _mean_ads_loss(embedder: Any, ranker: AdRanker, batches: Sequence[AdBatch], device: torch.device) -> float:
    ranker.eval(); embedder.eval(); weighted = 0.0; count = 0
    with torch.no_grad():
        for batch in batches:
            moved = _to_device_ad_batch(batch, device)
            loss, rows = masked_binary_cross_entropy(_ads_logits(embedder, ranker, moved), moved.clicked, moved.candidate_mask)
            weighted += float(loss.item()) * rows; count += rows
    return weighted / count


def run_ads_training(config: Mapping[str, Any]) -> dict[str, Any]:
    handoff = validate_green_sm_ads_data_handoff(config)
    model, training, ranker_settings = _ads_settings(config)
    seed = config.get("seed")
    if not isinstance(seed, int): raise AdConfigurationError("config.seed must be an integer")
    random.seed(seed); torch.manual_seed(seed)
    device = str(training.get("device", "cpu"))
    if device.startswith("cuda") and not torch.cuda.is_available(): raise AdConfigurationError("CUDA was requested but is unavailable")
    embedder, foundation = _load_ads_foundation(config, device=device)
    train_rows, train_batches, features = _load_ads_rows_and_batches(config, split="train")
    validation_rows, validation_batches, _ = _load_ads_rows_and_batches(config, split="validation", features=features)
    run_dir = Path(_require_text(ranker_settings, "run_dir", prefix="config.ranker"))
    if run_dir.exists() and any(run_dir.iterdir()): raise AdConfigurationError(f"Ads run directory must be new and empty: {run_dir}")
    d_model = int(embedder.model.config.d_model)
    ranker = AdRanker(candidate_buckets=int(model["candidate_hash_buckets"]), campaign_buckets=int(model["campaign_hash_buckets"]), num_categories=len(features["category"]), num_placements=len(features["placement"]), d_model=d_model, dropout=float(model.get("dropout", 0.0))).to(device)
    embedder.to(device).eval()
    for parameter in embedder.parameters(): parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(ranker.parameters(), lr=float(training["learning_rate"]), weight_decay=float(training.get("weight_decay", 0.0)))
    _atomic_json(run_dir / "feature_artifacts.json", features)
    metrics: list[dict[str, float | int]] = []; best = float("inf"); best_epoch = 0
    for epoch in range(1, int(training["epochs"]) + 1):
        ranker.train(); losses: list[float] = []
        for batch in train_batches:
            moved = _to_device_ad_batch(batch, torch.device(device)); optimizer.zero_grad(set_to_none=True)
            loss, _ = masked_binary_cross_entropy(_ads_logits(embedder, ranker, moved), moved.clicked, moved.candidate_mask)
            loss.backward(); optimizer.step(); losses.append(float(loss.item()))
        validation = _mean_ads_loss(embedder, ranker, validation_batches, torch.device(device))
        metrics.append({"epoch": epoch, "train_bce": sum(losses) / len(losses), "validation_bce": validation})
        payload = {"ranker_config": {"candidate_buckets": int(model["candidate_hash_buckets"]), "campaign_buckets": int(model["campaign_hash_buckets"]), "num_categories": len(features["category"]), "num_placements": len(features["placement"]), "d_model": d_model, "dropout": float(model.get("dropout", 0.0))}, "ranker_state_dict": ranker.state_dict(), "epoch": epoch}
        _atomic_torch_save(run_dir / "last.pt", payload)
        if validation < best: best = validation; best_epoch = epoch; _atomic_torch_save(run_dir / "best.pt", payload)
    (run_dir / "train_metrics.jsonl").write_text("".join(json.dumps(row) + "\n" for row in metrics), encoding="utf-8")
    manifest = {**handoff, "schema_version": "green_sm_ads_ranker_v1", "run_id": run_dir.name, "seed": seed, "selected_checkpoint_sha256": foundation.manifest["checkpoint_sha256"], "vocabulary_sha256": foundation.manifest["vocabulary_sha256"], "feature_artifacts": "feature_artifacts.json", "feature_artifacts_sha256": _sha256(run_dir / "feature_artifacts.json"), "best_checkpoint": "best.pt", "best_checkpoint_sha256": _sha256(run_dir / "best.pt"), "last_checkpoint": "last.pt", "last_checkpoint_sha256": _sha256(run_dir / "last.pt"), "ranker_config": payload["ranker_config"], "best_epoch": best_epoch, "best_validation_bce": best}
    _atomic_json(run_dir / "run_manifest.json", manifest)
    return {"run_dir": str(run_dir), "best_epoch": best_epoch, "best_validation_bce": best, "train_requests": len(_group_requests(train_rows)), "validation_requests": len(_group_requests(validation_rows))}


def run_ads_prediction(config: Mapping[str, Any], *, split: Literal["validation", "test"]) -> OutputPaths:
    handoff = validate_green_sm_ads_data_handoff(config)
    _, training, ranker_settings = _ads_settings(config); device = str(training.get("device", "cpu"))
    embedder, foundation = _load_ads_foundation(config, device=device)
    run_dir = Path(_require_text(ranker_settings, "run_dir", prefix="config.ranker")); manifest = _read_json(run_dir / "run_manifest.json", label="Ads run manifest")
    expected = _require_text(ranker_settings, "expected_run_manifest_sha256", prefix="config.ranker")
    _verify_sha256(run_dir / "run_manifest.json", expected, label="Ads run manifest")
    if manifest.get("selected_checkpoint_sha256") != foundation.manifest["checkpoint_sha256"]: raise AdConfigurationError("Ads run used a different Foundation checkpoint")
    features = _read_json(run_dir / "feature_artifacts.json", label="Ads feature artifacts")
    rows, batches, _ = _load_ads_rows_and_batches(config, split=split, features=features)
    payload = torch.load(run_dir / "best.pt", map_location=device, weights_only=False); ranker = AdRanker(**payload["ranker_config"]).to(device); ranker.load_state_dict(payload["ranker_state_dict"]); ranker.eval()
    from src.downstream.green_sm_ad_optimization import (
        expected_green_sm_candidates,
        prediction_rows_from_batch,
    )
    predictions = []
    with torch.no_grad():
        for batch in batches:
            moved = _to_device_ad_batch(batch, torch.device(device)); logits = _ads_logits(embedder, ranker, moved); predictions.extend(prediction_rows_from_batch(moved, score_probabilities(logits, moved.candidate_mask).cpu(), split=split))
    return OutputContractV1.write_predictions(predictions, expected_candidates=expected_green_sm_candidates(rows, data_version=handoff["data_version"], source_checksums={"ads_manifest_sha256": handoff["ads_manifest_sha256"], "foundation_manifest_sha256": handoff["foundation_manifest_sha256"]}), output_dir=run_dir / "predictions" / split, split=split, run_metadata={"selected_checkpoint_sha256": foundation.manifest["checkpoint_sha256"], "vocabulary_sha256": foundation.manifest["vocabulary_sha256"], "foundation_manifest_sha256": handoff["foundation_manifest_sha256"], "ads_manifest_sha256": handoff["ads_manifest_sha256"], "config_digest": "runtime", "seed": config["seed"], "run_id": manifest["run_id"], "score_semantics": "probability"})


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
