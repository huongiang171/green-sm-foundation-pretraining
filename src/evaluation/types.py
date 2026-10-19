"""Typed boundaries shared by downstream evaluation modules."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal

EvaluationSplit = Literal["validation", "test"]
ModelKind = Literal["foundation", "baseline"]


@dataclass(frozen=True)
class CanonicalPrediction:
    request_id: str
    candidate_id: str
    user_id: str
    cutoff_timestamp_utc: str
    split: EvaluationSplit
    score: float


@dataclass(frozen=True)
class SourceIdentity:
    task: str
    data_version: str
    split: EvaluationSplit
    candidate_manifest_sha256: str
    candidate_split_sha256: str


@dataclass(frozen=True)
class ModelProvenance:
    kind: ModelKind
    run_id: str
    details: Mapping[str, object]


@dataclass(frozen=True)
class VerifiedEvaluationInput:
    identity: SourceIdentity
    provenance: ModelProvenance
    predictions: tuple[CanonicalPrediction, ...]
    labels_by_candidate: Mapping[tuple[str, str], int]
    evaluation_groups_by_request: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class EvaluationReport:
    payload: Mapping[str, object]
