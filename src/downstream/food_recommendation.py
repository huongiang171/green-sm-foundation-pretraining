"""Strict Green SM Food-ranking data handoff and candidate batches.

This module deliberately owns only the data boundary shared by Food training,
prediction, and the demo.  It contains no train loop: later modules consume
the request-major :class:`FoodBatch` values built here.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import pyarrow.parquet as pq
import torch
from torch import nn

from src.foundation_input.dataset import DatasetConfig, encode_sample, foundation_collate
from src.foundation_input.loader import iter_normalized_samples
from src.foundation_input.types import EncodedSample, FoundationBatch
from src.foundation_input.vocab import VocabularyArtifacts
from src.pretraining.green_sm_contract import load_green_sm_tokenizer_config

FoodSplit = Literal["train", "validation", "test"]
_SPLITS = frozenset({"train", "validation", "test"})
_PAD = "<PAD>"
_UNK = "<UNK>"


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
    for canonical in iter_normalized_samples(foundation_root, split, validate=True):
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
