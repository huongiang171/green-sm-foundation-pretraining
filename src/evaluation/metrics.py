"""Pure, deterministic metrics for Green SM downstream evaluation."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TypeAlias

from sklearn.metrics import average_precision_score, roc_auc_score

MetricValue: TypeAlias = float | None


@dataclass(frozen=True)
class ScoredLabel:
    """One verified candidate paired with its evaluator-only binary label."""

    request_id: str
    candidate_id: str
    score: float
    label: int


@dataclass(frozen=True)
class MetricResult:
    metrics: Mapping[str, MetricValue]
    eligibility: Mapping[str, object]
    undefined_metrics: Mapping[str, str]


def rank_candidates(rows: Sequence[ScoredLabel]) -> list[ScoredLabel]:
    """Order one request by policy: score descending, candidate ID ascending."""

    return sorted(rows, key=lambda row: (-row.score, row.candidate_id))


def _validate_binary_rows(rows: Sequence[ScoredLabel], *, probability: bool) -> None:
    if not rows:
        raise ValueError("Cannot evaluate an empty candidate collection")
    for row in rows:
        if row.label not in {0, 1}:
            raise ValueError(f"Candidate {row.candidate_id!r} has non-binary label {row.label!r}")
        if not math.isfinite(row.score):
            raise ValueError(f"Candidate {row.candidate_id!r} has non-finite score")
        if probability and not 0.0 <= row.score <= 1.0:
            raise ValueError(f"Candidate {row.candidate_id!r} has probability outside [0, 1]")


def _group_requests(rows: Sequence[ScoredLabel]) -> dict[str, list[ScoredLabel]]:
    groups: dict[str, list[ScoredLabel]] = defaultdict(list)
    for row in rows:
        groups[row.request_id].append(row)
    return groups


def _recall_at_k(ranked: Sequence[ScoredLabel], k: int) -> float:
    positives = sum(row.label for row in ranked)
    if positives == 0:
        raise ValueError("Recall is undefined for a request without positives")
    return sum(row.label for row in ranked[:k]) / positives


def _ndcg_at_k(ranked: Sequence[ScoredLabel], k: int) -> float:
    def dcg(labels: Sequence[int]) -> float:
        return sum((2**label - 1) / math.log2(rank + 1) for rank, label in enumerate(labels, start=1))

    actual = dcg([row.label for row in ranked[:k]])
    ideal = dcg(sorted((row.label for row in ranked), reverse=True)[:k])
    if ideal == 0.0:
        raise ValueError("NDCG is undefined for a request without positives")
    return actual / ideal


def _mean_or_none(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def evaluate_food_requests(rows: Sequence[ScoredLabel]) -> MetricResult:
    """Evaluate multi-positive Food ranking requests under the fixed policy."""

    _validate_binary_rows(rows, probability=False)
    groups = _group_requests(rows)
    ndcg_at_5: list[float] = []
    ndcg_at_10: list[float] = []
    recall_at_3: list[float] = []
    recall_at_10: list[float] = []
    hit_rate_at_1: list[float] = []
    reciprocal_ranks: list[float] = []
    excluded_no_positive = 0
    for request_rows in groups.values():
        ranked = rank_candidates(request_rows)
        if not any(row.label for row in ranked):
            excluded_no_positive += 1
            continue
        ndcg_at_5.append(_ndcg_at_k(ranked, 5))
        ndcg_at_10.append(_ndcg_at_k(ranked, 10))
        recall_at_3.append(_recall_at_k(ranked, 3))
        recall_at_10.append(_recall_at_k(ranked, 10))
        hit_rate_at_1.append(float(ranked[0].label == 1))
        first_positive_rank = next(index for index, row in enumerate(ranked, start=1) if row.label == 1)
        reciprocal_ranks.append(1.0 / first_positive_rank)
    return MetricResult(
        metrics={
            "ndcg_at_5": _mean_or_none(ndcg_at_5),
            "ndcg_at_10": _mean_or_none(ndcg_at_10),
            "recall_at_3": _mean_or_none(recall_at_3),
            "recall_at_10": _mean_or_none(recall_at_10),
            "hit_rate_at_1": _mean_or_none(hit_rate_at_1),
            "mrr": _mean_or_none(reciprocal_ranks),
        },
        eligibility={
            "source_request_count": len(groups),
            "ranking_eligible_request_count": len(groups) - excluded_no_positive,
            "excluded_request_counts": {"no_positive": excluded_no_positive} if excluded_no_positive else {},
        },
        undefined_metrics={},
    )


def evaluate_ads(rows: Sequence[ScoredLabel]) -> MetricResult:
    """Evaluate all Ads candidates pointwise and positive-click requests by rank."""

    _validate_binary_rows(rows, probability=True)
    groups = _group_requests(rows)
    labels = [row.label for row in rows]
    scores = [row.score for row in rows]
    unique_labels = set(labels)
    undefined_metrics: dict[str, str] = {}
    if len(unique_labels) < 2:
        roc_auc: MetricValue = None
        pr_auc: MetricValue = None
        undefined_metrics = {"pr_auc": "single_class_labels", "roc_auc": "single_class_labels"}
    else:
        roc_auc = float(roc_auc_score(labels, scores))
        pr_auc = float(average_precision_score(labels, scores))
    epsilon = 1e-15
    clipped = [min(max(score, epsilon), 1.0 - epsilon) for score in scores]
    log_loss = -sum(
        label * math.log(score) + (1 - label) * math.log(1.0 - score)
        for label, score in zip(labels, clipped, strict=True)
    ) / len(labels)
    brier = sum((score - label) ** 2 for label, score in zip(labels, scores, strict=True)) / len(labels)
    ndcg_at_3: list[float] = []
    recall_at_3: list[float] = []
    zero_click_excluded = 0
    for request_rows in groups.values():
        ranked = rank_candidates(request_rows)
        if not any(row.label for row in ranked):
            zero_click_excluded += 1
            continue
        ndcg_at_3.append(_ndcg_at_k(ranked, 3))
        recall_at_3.append(_recall_at_k(ranked, 3))
    return MetricResult(
        metrics={
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "log_loss": log_loss,
            "brier": brier,
            "ndcg_at_3": _mean_or_none(ndcg_at_3),
            "recall_at_3": _mean_or_none(recall_at_3),
        },
        eligibility={
            "candidate_row_count": len(rows),
            "source_request_count": len(groups),
            "ranking_eligible_request_count": len(groups) - zero_click_excluded,
            "zero_click_excluded_request_count": zero_click_excluded,
        },
        undefined_metrics=undefined_metrics,
    )
