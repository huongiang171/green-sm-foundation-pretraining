"""Versioned policy constants for Green SM downstream evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

EvaluationTask = Literal["food_recommendation", "ad_optimization"]
ScoreSemantics = Literal["ranking", "probability"]

EVALUATOR_VERSION = "green_sm_evaluator_v1"
METRIC_POLICY_VERSION = "green_sm_metric_policy_v1"
TIE_BREAK_POLICY = "score_desc__candidate_id_asc"
TIMESTAMP_POLICY = "parse_to_utc_instant"

_SCORE_SEMANTICS: dict[str, ScoreSemantics] = {
    "food_recommendation": "ranking",
    "ad_optimization": "probability",
}


@dataclass(frozen=True)
class MetricPolicy:
    """Stable evaluator policy identity persisted with every report."""

    evaluator_version: str = EVALUATOR_VERSION
    metric_policy_version: str = METRIC_POLICY_VERSION
    tie_break_policy: str = TIE_BREAK_POLICY
    timestamp_policy: str = TIMESTAMP_POLICY

    def identity(self) -> dict[str, str]:
        return {
            "evaluator_version": self.evaluator_version,
            "metric_policy_version": self.metric_policy_version,
            "tie_break_policy": self.tie_break_policy,
            "timestamp_policy": self.timestamp_policy,
        }


METRIC_POLICY = MetricPolicy()


def score_semantics_for_task(task: str) -> ScoreSemantics:
    """Return the only prediction-score semantics allowed for ``task``."""

    try:
        return _SCORE_SEMANTICS[task]
    except KeyError as error:
        raise ValueError(f"Unsupported evaluation task {task!r}") from error
