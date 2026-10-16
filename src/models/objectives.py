"""Heads and losses used to pretrain the shared user Foundation model."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from src.foundation_input.types import FoundationBatch
from src.pretraining.config import ObjectiveWeights, RoutingIds
from src.pretraining.masking import CorruptedBatch

from .encoder import FoundationOutput


@dataclass(frozen=True)
class MaskedPredictions:
    categorical_logits: torch.Tensor
    numeric_values: torch.Tensor
    numeric_missing_logits: torch.Tensor
    geo_values: torch.Tensor
    geo_missing_logits: torch.Tensor


@dataclass(frozen=True)
class LossBreakdown:
    total: torch.Tensor
    masked_categorical: torch.Tensor
    numeric_value: torch.Tensor
    numeric_missing: torch.Tensor
    geo_coordinate: torch.Tensor
    geo_missing: torch.Tensor
    next_action: torch.Tensor
    counts: Mapping[str, int]


class UserEmbeddingFusion(nn.Module):
    """Fuse independently learned long- and short-term user representations."""

    def __init__(self, *, d_model: int) -> None:
        super().__init__()
        if d_model <= 0:
            raise ValueError("d_model must be positive")
        self.projection = nn.Linear(2 * d_model, d_model)
        self.normalization = nn.LayerNorm(d_model)

    def forward(self, long_term: torch.Tensor, short_term: torch.Tensor) -> torch.Tensor:
        if long_term.ndim != 2 or short_term.shape != long_term.shape:
            raise ValueError("long_term and short_term must share shape [B, D]")
        return self.normalization(self.projection(torch.cat((long_term, short_term), dim=-1)))


class PretrainingHeads(nn.Module):
    """Token-level reconstruction and user-level next-action prediction heads."""

    def __init__(self, *, d_model: int, value_vocab_size: int) -> None:
        super().__init__()
        if d_model <= 0 or value_vocab_size <= 1:
            raise ValueError("d_model must be positive and value_vocab_size must exceed one")
        self.categorical = nn.Linear(d_model, value_vocab_size)
        self.numeric_value = nn.Linear(d_model, 1)
        self.numeric_missing = nn.Linear(d_model, 1)
        self.geo_value = nn.Linear(d_model, 2)
        self.geo_missing = nn.Linear(d_model, 1)
        self.next_action = nn.Linear(d_model, value_vocab_size)

    def forward_masked(self, hidden_states: torch.Tensor) -> MaskedPredictions:
        if hidden_states.ndim != 3:
            raise ValueError("hidden_states must have shape [B, L, D]")
        return MaskedPredictions(
            categorical_logits=self.categorical(hidden_states),
            numeric_values=self.numeric_value(hidden_states).squeeze(-1),
            numeric_missing_logits=self.numeric_missing(hidden_states).squeeze(-1),
            geo_values=self.geo_value(hidden_states),
            geo_missing_logits=self.geo_missing(hidden_states).squeeze(-1),
        )

    def forward_next_action(self, z_user: torch.Tensor) -> torch.Tensor:
        if z_user.ndim != 2:
            raise ValueError("z_user must have shape [B, D]")
        return self.next_action(z_user)


def _zero(reference: torch.Tensor) -> torch.Tensor:
    return reference.sum() * 0.0


def _cross_entropy(logits: torch.Tensor, labels: torch.Tensor, mask: torch.BoolTensor) -> tuple[torch.Tensor, int]:
    count = int(mask.sum().item())
    if not count:
        return _zero(logits), count
    return F.cross_entropy(logits[mask], labels[mask]), count


def _smooth_l1(predictions: torch.Tensor, targets: torch.Tensor, mask: torch.BoolTensor) -> tuple[torch.Tensor, int]:
    count = int(mask.sum().item())
    if not count:
        return _zero(predictions), count
    return F.smooth_l1_loss(predictions[mask], targets[mask]), count


def _binary_cross_entropy(
    logits: torch.Tensor, targets: torch.BoolTensor, mask: torch.BoolTensor
) -> tuple[torch.Tensor, int]:
    count = int(mask.sum().item())
    if not count:
        return _zero(logits), count
    return F.binary_cross_entropy_with_logits(logits[mask], targets[mask].to(torch.float32)), count


def compute_pretraining_loss(
    *,
    masked_output: FoundationOutput,
    next_action_output: FoundationOutput,
    corruption: CorruptedBatch,
    original_batch: FoundationBatch,
    heads: PretrainingHeads,
    fusion: UserEmbeddingFusion,
    routing: RoutingIds,
    weights: ObjectiveWeights,
) -> LossBreakdown:
    """Compute only losses with labels made eligible by the batch contract."""

    targets = corruption.targets
    predictions = heads.forward_masked(masked_output.hidden_states)
    selected = targets.selected_mask
    if selected.shape != original_batch.input_ids.shape or selected.dtype != torch.bool:
        raise ValueError("selected_mask must be bool with shape [B, L]")
    numeric = original_batch.modality_ids.eq(routing.numeric)
    geo = original_batch.modality_ids.eq(routing.geo)
    categorical_mask = selected & ~numeric & ~geo
    numeric_mask = selected & numeric
    geo_mask = selected & geo

    categorical_loss, categorical_count = _cross_entropy(
        predictions.categorical_logits, targets.original_input_ids, categorical_mask
    )
    numeric_value_loss, numeric_value_count = _smooth_l1(
        predictions.numeric_values,
        targets.original_numeric_values,
        numeric_mask & ~targets.original_numeric_missing,
    )
    numeric_missing_loss, numeric_missing_count = _binary_cross_entropy(
        predictions.numeric_missing_logits, targets.original_numeric_missing, numeric_mask
    )
    geo_coordinate_loss, geo_coordinate_count = _smooth_l1(
        predictions.geo_values,
        targets.original_geo_values,
        (geo_mask & ~targets.original_geo_missing).unsqueeze(-1).expand_as(predictions.geo_values),
    )
    geo_missing_loss, geo_missing_count = _binary_cross_entropy(
        predictions.geo_missing_logits, targets.original_geo_missing, geo_mask
    )

    valid_next = original_batch.targets.get("next_action_valid_mask")
    labels = original_batch.targets.get("next_action_type_ids")
    if valid_next is None or labels is None:
        raise ValueError("next_action_valid_mask and next_action_type_ids are required")
    if valid_next.dtype != torch.bool or valid_next.shape != labels.shape:
        raise ValueError("next-action targets must be batch-aligned with a bool valid mask")
    z_user = fusion(next_action_output.long_term, next_action_output.short_term)
    next_action_logits = heads.forward_next_action(z_user)
    next_action_loss, next_action_count = _cross_entropy(next_action_logits, labels, valid_next)

    total = (
        weights.categorical * categorical_loss
        + weights.numeric_value * numeric_value_loss
        + weights.numeric_missing * numeric_missing_loss
        + weights.geo_coordinate * geo_coordinate_loss
        + weights.geo_missing * geo_missing_loss
        + weights.next_action * next_action_loss
    )
    return LossBreakdown(
        total=total,
        masked_categorical=categorical_loss,
        numeric_value=numeric_value_loss,
        numeric_missing=numeric_missing_loss,
        geo_coordinate=geo_coordinate_loss,
        geo_missing=geo_missing_loss,
        next_action=next_action_loss,
        counts={
            "categorical": categorical_count,
            "numeric_value": numeric_value_count,
            "numeric_missing": numeric_missing_count,
            "geo_coordinate": geo_coordinate_count,
            "geo_missing": geo_missing_count,
            "next_action": next_action_count,
        },
    )
