"""Deterministic copy-on-write corruption for masked field reconstruction."""

from __future__ import annotations

import copy
from dataclasses import dataclass

import torch

from src.foundation_input.types import FoundationBatch, SPECIAL_TOKEN_IDS

from .config import RoutingIds


@dataclass(frozen=True)
class MaskedFieldTargets:
    """Uncorrupted field values, retained outside the model input."""

    selected_mask: torch.BoolTensor
    original_input_ids: torch.LongTensor
    original_numeric_values: torch.FloatTensor
    original_numeric_missing: torch.BoolTensor
    original_geo_values: torch.FloatTensor
    original_geo_missing: torch.BoolTensor


@dataclass(frozen=True)
class CorruptedBatch:
    batch: FoundationBatch
    targets: MaskedFieldTargets


def _selection_seed(seed: int, global_step: int) -> int:
    return (seed * 1_000_003 + global_step * 9_176 + 19) % (2**63 - 1)


def _require_v3(batch: FoundationBatch) -> None:
    if not hasattr(batch, "geo_values") or not hasattr(batch, "geo_missing_mask"):
        raise RuntimeError("masked reconstruction requires FoundationBatch v3 geo_values and geo_missing_mask")


def corrupt_for_masked_reconstruction(
    batch: FoundationBatch,
    *,
    routing: RoutingIds,
    probability: float,
    seed: int,
    global_step: int,
) -> CorruptedBatch:
    """Return a new batch with selected values hidden but labels preserved separately."""

    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be in [0, 1]")
    _require_v3(batch)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(_selection_seed(seed, global_step))
    random_values = torch.rand(batch.input_ids.shape, generator=generator, device="cpu")
    random_values = random_values.to(batch.input_ids.device)
    special = torch.zeros_like(batch.attention_mask)
    for token_id in (SPECIAL_TOKEN_IDS["PAD"], SPECIAL_TOKEN_IDS["CLS"], SPECIAL_TOKEN_IDS["SEP"]):
        special |= batch.input_ids.eq(token_id)
    selected = batch.attention_mask & ~special & random_values.lt(probability)

    targets = MaskedFieldTargets(
        selected_mask=selected.clone(),
        original_input_ids=batch.input_ids.clone(),
        original_numeric_values=batch.numeric_values.clone(),
        original_numeric_missing=batch.numeric_missing_mask.clone(),
        original_geo_values=batch.geo_values.clone(),
        original_geo_missing=batch.geo_missing_mask.clone(),
    )
    corrupted = copy.copy(batch)
    corrupted.input_ids = batch.input_ids.clone()
    corrupted.numeric_values = batch.numeric_values.clone()
    corrupted.numeric_missing_mask = batch.numeric_missing_mask.clone()
    corrupted.geo_values = batch.geo_values.clone()
    corrupted.geo_missing_mask = batch.geo_missing_mask.clone()
    corrupted.input_ids[selected] = SPECIAL_TOKEN_IDS["MASK"]

    numeric_selected = selected & batch.modality_ids.eq(routing.numeric)
    corrupted.numeric_values[numeric_selected] = 0.0
    corrupted.numeric_missing_mask[numeric_selected] = True
    geo_selected = selected & batch.modality_ids.eq(routing.geo)
    corrupted.geo_values[geo_selected] = 0.0
    corrupted.geo_missing_mask[geo_selected] = True
    return CorruptedBatch(batch=corrupted, targets=targets)
