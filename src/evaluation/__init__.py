"""Shared, versioned evaluation for Green SM downstream tasks."""

from .policy import METRIC_POLICY, score_semantics_for_task

__all__ = ["METRIC_POLICY", "score_semantics_for_task"]
