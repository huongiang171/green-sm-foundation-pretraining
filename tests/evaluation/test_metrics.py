from __future__ import annotations

import math

import pytest

from src.evaluation.metrics import (
    ScoredLabel,
    evaluate_ads,
    evaluate_food_requests,
    rank_candidates,
)


def test_food_metrics_use_every_positive_and_the_first_ranked_positive() -> None:
    result = evaluate_food_requests(
        [
            ScoredLabel("request-1", "food-a", 0.9, 1),
            ScoredLabel("request-1", "food-b", 0.8, 0),
            ScoredLabel("request-1", "food-c", 0.7, 1),
        ]
    )

    assert result.metrics["recall_at_3"] == pytest.approx(1.0)
    assert result.metrics["recall_at_10"] == pytest.approx(1.0)
    assert result.metrics["hit_rate_at_1"] == pytest.approx(1.0)
    assert result.metrics["mrr"] == pytest.approx(1.0)
    expected_ndcg = (1.0 + 1.0 / math.log2(4)) / (1.0 + 1.0 / math.log2(3))
    assert result.metrics["ndcg_at_5"] == pytest.approx(expected_ndcg)
    assert result.eligibility["ranking_eligible_request_count"] == 1


def test_rank_candidates_breaks_equal_scores_by_candidate_id() -> None:
    ranked = rank_candidates(
        [
            ScoredLabel("request-1", "z-item", 0.5, 0),
            ScoredLabel("request-1", "a-item", 0.5, 1),
        ]
    )

    assert [row.candidate_id for row in ranked] == ["a-item", "z-item"]


def test_food_counts_request_with_no_positive_as_ineligible() -> None:
    result = evaluate_food_requests([ScoredLabel("request-1", "food-a", 0.5, 0)])

    assert result.metrics["mrr"] is None
    assert result.eligibility["ranking_eligible_request_count"] == 0
    assert result.eligibility["excluded_request_counts"] == {"no_positive": 1}


def test_ads_keeps_zero_click_requests_in_pointwise_metrics_but_excludes_them_from_ranking() -> None:
    result = evaluate_ads(
        [
            ScoredLabel("clicked-request", "ad-a", 0.9, 1),
            ScoredLabel("clicked-request", "ad-b", 0.1, 0),
            ScoredLabel("zero-click-request", "ad-c", 0.2, 0),
            ScoredLabel("zero-click-request", "ad-d", 0.1, 0),
        ]
    )

    assert result.metrics["roc_auc"] == pytest.approx(1.0)
    assert result.metrics["pr_auc"] == pytest.approx(1.0)
    assert result.metrics["log_loss"] > 0.0
    assert result.metrics["brier"] > 0.0
    assert result.eligibility["candidate_row_count"] == 4
    assert result.eligibility["ranking_eligible_request_count"] == 1
    assert result.eligibility["zero_click_excluded_request_count"] == 1


def test_ads_single_class_writes_null_auc_metrics_but_keeps_losses() -> None:
    result = evaluate_ads(
        [
            ScoredLabel("request-1", "ad-a", 0.0, 0),
            ScoredLabel("request-1", "ad-b", 1.0, 0),
        ]
    )

    assert result.metrics["roc_auc"] is None
    assert result.metrics["pr_auc"] is None
    assert result.undefined_metrics == {
        "pr_auc": "single_class_labels",
        "roc_auc": "single_class_labels",
    }
    assert result.metrics["log_loss"] > 0.0
    assert result.metrics["brier"] > 0.0
