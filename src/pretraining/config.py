"""Strict, dependency-free configuration for Foundation pretraining."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "green_sm_pretraining_v1"


def _mapping(value: object, *, name: str, keys: set[str]) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be an object")
    actual = set(value)
    if actual != keys:
        missing = sorted(keys - actual)
        unknown = sorted(actual - keys)
        raise ValueError(f"{name} keys mismatch: missing={missing}, unknown={unknown}")
    return value


def _positive_int(value: object, *, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _nonnegative_float(value: object, *, name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return float(value)


def _unit_interval(value: object, *, name: str) -> float:
    parsed = _nonnegative_float(value, name=name)
    if parsed > 1.0:
        raise ValueError(f"{name} must be in [0, 1]")
    return parsed


@dataclass(frozen=True)
class RoutingIds:
    """Vocabulary-derived modality/source IDs; no constant is universal."""

    sequence: int
    numeric: int
    geo: int

    def __post_init__(self) -> None:
        for name in ("sequence", "numeric", "geo"):
            _positive_int(getattr(self, name), name=f"routing.{name}")


@dataclass(frozen=True)
class ModelSettings:
    d_model: int
    num_heads: int
    num_layers: int
    dropout: float
    max_length: int
    num_user_buckets: int
    categorical_pair_buckets: int

    def __post_init__(self) -> None:
        for name in (
            "d_model",
            "num_heads",
            "num_layers",
            "max_length",
            "num_user_buckets",
            "categorical_pair_buckets",
        ):
            _positive_int(getattr(self, name), name=f"model.{name}")
        if self.d_model % self.num_heads:
            raise ValueError("model.d_model must be divisible by model.num_heads")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("model.dropout must be in [0, 1)")


@dataclass(frozen=True)
class DatasetSettings:
    max_tokens: int
    max_event_count: int

    def __post_init__(self) -> None:
        _positive_int(self.max_tokens, name="dataset.max_tokens")
        _positive_int(self.max_event_count, name="dataset.max_event_count")


@dataclass(frozen=True)
class TrainingSettings:
    batch_size: int
    max_steps: int
    eval_every_steps: int
    checkpoint_every_steps: int
    learning_rate: float
    weight_decay: float
    shuffle_buffer_size: int
    validation_max_batches: int = 100

    def __post_init__(self) -> None:
        for name in (
            "batch_size",
            "max_steps",
            "eval_every_steps",
            "checkpoint_every_steps",
            "shuffle_buffer_size",
            "validation_max_batches",
        ):
            _positive_int(getattr(self, name), name=f"training.{name}")
        if _nonnegative_float(self.learning_rate, name="training.learning_rate") == 0.0:
            raise ValueError("training.learning_rate must be positive")
        _nonnegative_float(self.weight_decay, name="training.weight_decay")


@dataclass(frozen=True)
class ObjectiveWeights:
    categorical: float
    numeric_value: float
    numeric_missing: float
    geo_coordinate: float
    geo_missing: float
    next_action: float
    next_semantic: float = 0.0

    def __post_init__(self) -> None:
        for name in (
            "categorical",
            "numeric_value",
            "numeric_missing",
            "geo_coordinate",
            "geo_missing",
            "next_action",
            "next_semantic",
        ):
            _nonnegative_float(getattr(self, name), name=f"objectives.{name}_weight")


@dataclass(frozen=True)
class PretrainingConfig:
    linked_root: Path
    control_root: Path
    output_dir: Path
    seed: int
    model: ModelSettings
    dataset: DatasetSettings
    training: TrainingSettings
    masking_probability: float
    objectives: ObjectiveWeights
    config_digest: str
    run_id: str = "default"

    def __post_init__(self) -> None:
        _positive_int(self.seed, name="seed")
        _unit_interval(self.masking_probability, name="masking.probability")
        if self.model.max_length != self.dataset.max_tokens:
            raise ValueError(
                "dataset.max_tokens must equal model.max_length to prevent silent truncation"
            )
        if self.model.max_length != 256:
            raise ValueError("Green SM pretraining requires model.max_length=256")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", self.run_id):
            raise ValueError("run_id must be a non-empty slug of at most 64 characters")


def _model(value: object) -> ModelSettings:
    fields = _mapping(
        value,
        name="model",
        keys={
            "d_model",
            "num_heads",
            "num_layers",
            "dropout",
            "max_length",
            "num_user_buckets",
            "categorical_pair_buckets",
        },
    )
    return ModelSettings(
        d_model=_positive_int(fields["d_model"], name="model.d_model"),
        num_heads=_positive_int(fields["num_heads"], name="model.num_heads"),
        num_layers=_positive_int(fields["num_layers"], name="model.num_layers"),
        dropout=_nonnegative_float(fields["dropout"], name="model.dropout"),
        max_length=_positive_int(fields["max_length"], name="model.max_length"),
        num_user_buckets=_positive_int(fields["num_user_buckets"], name="model.num_user_buckets"),
        categorical_pair_buckets=_positive_int(
            fields["categorical_pair_buckets"], name="model.categorical_pair_buckets"
        ),
    )


def _dataset(value: object) -> DatasetSettings:
    fields = _mapping(value, name="dataset", keys={"max_tokens", "max_event_count"})
    return DatasetSettings(
        max_tokens=_positive_int(fields["max_tokens"], name="dataset.max_tokens"),
        max_event_count=_positive_int(fields["max_event_count"], name="dataset.max_event_count"),
    )


def _training(value: object) -> TrainingSettings:
    fields = _mapping(
        value,
        name="training",
        keys={
            "batch_size",
            "max_steps",
            "eval_every_steps",
            "checkpoint_every_steps",
            "learning_rate",
            "weight_decay",
            "shuffle_buffer_size",
            "validation_max_batches",
        },
    )
    return TrainingSettings(
        batch_size=_positive_int(fields["batch_size"], name="training.batch_size"),
        max_steps=_positive_int(fields["max_steps"], name="training.max_steps"),
        eval_every_steps=_positive_int(fields["eval_every_steps"], name="training.eval_every_steps"),
        checkpoint_every_steps=_positive_int(
            fields["checkpoint_every_steps"], name="training.checkpoint_every_steps"
        ),
        learning_rate=_nonnegative_float(fields["learning_rate"], name="training.learning_rate"),
        weight_decay=_nonnegative_float(fields["weight_decay"], name="training.weight_decay"),
        shuffle_buffer_size=_positive_int(
            fields["shuffle_buffer_size"], name="training.shuffle_buffer_size"
        ),
        validation_max_batches=_positive_int(
            fields["validation_max_batches"], name="training.validation_max_batches"
        ),
    )


def _objectives(value: object) -> ObjectiveWeights:
    if not isinstance(value, Mapping):
        raise TypeError("objectives must be an object")
    required = {
        "categorical_weight",
        "numeric_value_weight",
        "numeric_missing_weight",
        "geo_coordinate_weight",
        "geo_missing_weight",
        "next_action_weight",
    }
    allowed = required | {"next_semantic_weight"}
    if set(value) - allowed or required - set(value):
        missing = sorted(required - set(value))
        unknown = sorted(set(value) - allowed)
        raise ValueError(f"objectives keys mismatch: missing={missing}, unknown={unknown}")
    fields = value
    return ObjectiveWeights(
        categorical=_nonnegative_float(fields["categorical_weight"], name="objectives.categorical_weight"),
        numeric_value=_nonnegative_float(
            fields["numeric_value_weight"], name="objectives.numeric_value_weight"
        ),
        numeric_missing=_nonnegative_float(
            fields["numeric_missing_weight"], name="objectives.numeric_missing_weight"
        ),
        geo_coordinate=_nonnegative_float(
            fields["geo_coordinate_weight"], name="objectives.geo_coordinate_weight"
        ),
        geo_missing=_nonnegative_float(
            fields["geo_missing_weight"], name="objectives.geo_missing_weight"
        ),
        next_action=_nonnegative_float(
            fields["next_action_weight"], name="objectives.next_action_weight"
        ),
        next_semantic=_nonnegative_float(
            fields.get("next_semantic_weight", 0.0), name="objectives.next_semantic_weight"
        ),
    )


def load_pretraining_config(path: str | Path) -> PretrainingConfig:
    """Load a reproducible pretraining JSON recipe and reject ambiguous input."""

    config_path = Path(path)
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot read pretraining config {config_path}: {error}") from error
    fields = _mapping(
        payload,
        name="pretraining config",
        keys={
            "schema_version",
            "linked_root",
            "control_root",
            "output_dir",
            "seed",
            "model",
            "dataset",
            "training",
            "masking",
            "objectives",
            "run_id",
        },
    )
    if fields["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {SCHEMA_VERSION!r}")
    masking = _mapping(fields["masking"], name="masking", keys={"probability"})
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return PretrainingConfig(
        linked_root=Path(str(fields["linked_root"])),
        control_root=Path(str(fields["control_root"])),
        output_dir=Path(str(fields["output_dir"])),
        seed=_positive_int(fields["seed"], name="seed"),
        model=_model(fields["model"]),
        dataset=_dataset(fields["dataset"]),
        training=_training(fields["training"]),
        masking_probability=_unit_interval(masking["probability"], name="masking.probability"),
        objectives=_objectives(fields["objectives"]),
        config_digest=hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        run_id=str(fields["run_id"]),
    )
