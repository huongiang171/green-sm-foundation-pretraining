"""Deterministic paired Green SM synthetic corpus tooling."""

from .build import build_dataset
from .config import GreenSmConfig, ReadinessThresholds

__all__ = ["GreenSmConfig", "ReadinessThresholds", "build_dataset"]
