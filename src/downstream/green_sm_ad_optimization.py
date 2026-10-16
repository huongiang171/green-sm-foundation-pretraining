"""Green SM Ads bridge; deliberately separate from the Taobao-specific handoff."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path

import pyarrow.parquet as pq
import torch

from src.contracts.output import ExpectedCandidate, PredictionRow, VerifiedOpportunityIndex
from src.foundation_input.tokenizer import TokenizerConfig
from src.pretraining.artifacts import FrozenFoundationEmbedder
from src.pretraining.green_sm_contract import load_green_sm_tokenizer_config

from .ad_optimization import (
    AdBatch,
    AdOpportunity,
    AdRanker,
    build_ad_batches,
    load_encoded_foundation_samples_for_ads,
)


def _timestamp(value: object, *, field: str) -> int:
    if not isinstance(value, datetime):
        raise ValueError(f"Green SM Ads {field} must be a timestamp")
    return int(value.timestamp())


def load_green_sm_ads_opportunities(
    root: str | Path, *, split: str
) -> tuple[AdOpportunity, ...]:
    """Adapt only the documented Green SM Ads opportunity schema to the common ranker input."""

    if split not in {"train", "validation", "test"}:
        raise ValueError("split must be train, validation, or test")
    path = Path(root) / f"ads_opportunities.{split}.parquet"
    if not path.is_file():
        raise ValueError(f"Green SM Ads opportunity file does not exist: {path}")
    rows: list[AdOpportunity] = []
    required = {
        "request_id", "user_id", "foundation_sample_id", "cutoff_timestamp", "candidate_id",
        "split", "candidate_available_timestamp", "candidate_category", "clicked", "campaign_id",
        "placement", "exposure_timestamp",
    }
    for raw in pq.read_table(path).to_pylist():
        missing = required - set(raw)
        if missing:
            raise ValueError(f"Green SM Ads opportunity row lacks fields {sorted(missing)}")
        if raw["split"] != split:
            raise ValueError("Green SM Ads parquet split does not match requested split")
        rows.append(AdOpportunity.from_mapping({
            "request_id": raw["request_id"], "user_id": raw["user_id"],
            "foundation_sample_id": raw["foundation_sample_id"],
            "cutoff_timestamp": _timestamp(raw["cutoff_timestamp"], field="cutoff_timestamp"),
            "candidate_id": raw["candidate_id"], "clicked": raw["clicked"], "split": raw["split"],
            "candidate_available_timestamp": _timestamp(raw["candidate_available_timestamp"], field="candidate_available_timestamp"),
            "cate_id": raw["candidate_category"], "campaign_id": raw["campaign_id"],
            "pid": raw["placement"],
            "exposure_timestamp": _timestamp(raw["exposure_timestamp"], field="exposure_timestamp"),
        }))
    return tuple(rows)


def resolve_green_sm_ads_samples(
    rows: Sequence[AdOpportunity],
    *,
    foundation_root: str | Path,
    vocabulary_path: str | Path,
    split: str,
    max_tokens: int,
) -> dict[str, object]:
    """Resolve causal Ads snapshots using the exact Green SM tokenizer policy."""

    tokenizer: TokenizerConfig = load_green_sm_tokenizer_config(foundation_root)
    return load_encoded_foundation_samples_for_ads(
        rows,
        foundation_root=foundation_root,
        vocabulary_path=vocabulary_path,
        split=split,
        max_tokens=max_tokens,
        tokenizer_config=tokenizer,
    )


@torch.no_grad()
def frozen_foundation_logits(
    embedder: FrozenFoundationEmbedder, ranker: AdRanker, batch: AdBatch
) -> torch.Tensor:
    """Score a candidate batch while guaranteeing Foundation weights receive no gradient."""

    embeddings = embedder.embed_users(batch.foundation_batch)
    return ranker(
        embeddings["long_term"], embeddings["short_term"],
        candidate_hash_ids=batch.candidate_hash_ids, campaign_hash_ids=batch.campaign_hash_ids,
        category_ids=batch.category_ids, placement_ids=batch.placement_ids,
    )


def prediction_rows_from_batch(
    batch: AdBatch, probabilities: torch.Tensor, *, split: str
) -> list[PredictionRow]:
    """Flatten exactly the real candidates, never padded cells or labels, for OutputContractV1."""

    if probabilities.shape != batch.candidate_mask.shape:
        raise ValueError("probabilities must align with the request-major candidate mask")
    rows: list[PredictionRow] = []
    for row_index, opportunity_group in enumerate(batch.opportunities):
        for candidate_index, opportunity in enumerate(opportunity_group):
            rows.append(PredictionRow(
                opportunity.request_id, opportunity.candidate_id,
                float(probabilities[row_index, candidate_index]), split,
            ))
    return rows


def expected_green_sm_candidates(
    rows: Sequence[AdOpportunity], *, data_version: str, source_checksums: Mapping[str, str]
) -> VerifiedOpportunityIndex:
    """Construct the candidate index required before validation/test prediction export."""

    candidates = [
        ExpectedCandidate(
            row.request_id, row.candidate_id, row.user_id,
            datetime.fromtimestamp(row.cutoff_timestamp).isoformat(), row.split,  # type: ignore[arg-type]
        )
        for row in rows if row.split in {"validation", "test"}
    ]
    return VerifiedOpportunityIndex(
        candidates, task="ad_optimization", data_version=data_version, source_checksums=source_checksums
    )
