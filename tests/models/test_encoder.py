from __future__ import annotations

import pytest

from src.models.encoder import FoundationModelConfig


def test_config_rejects_non_divisible_attention_dimension() -> None:
    with pytest.raises(ValueError, match="d_model=.*num_heads"):
        FoundationModelConfig(d_model=30, num_heads=8)


def test_config_rejects_unknown_default_attention_mode() -> None:
    with pytest.raises(ValueError, match="default_attention_mode"):
        FoundationModelConfig(default_attention_mode="causal")
