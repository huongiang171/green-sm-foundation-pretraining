"""Transformer encoder and public embedding API for foundation batches."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class FoundationModelConfig:
    """Static dimensions and mask policy used to construct a foundation model."""

    d_model: int = 128
    num_heads: int = 4
    num_layers: int = 4
    dropout: float = 0.1
    max_length: int = 256
    value_vocab_size: int = 50_007
    key_vocab_size: int = 1
    modality_vocab_size: int = 1
    source_vocab_size: int = 1
    time_gap_vocab_size: int = 8
    num_user_buckets: int = 50_000
    categorical_pair_buckets: int = 100_003
    sequence_source_id: int = 2
    default_attention_mode: str = "bidirectional"

    def __post_init__(self) -> None:
        if self.d_model <= 0 or self.d_model % self.num_heads != 0:
            raise ValueError(
                f"d_model={self.d_model} must be positive and divisible by "
                f"num_heads={self.num_heads}"
            )
        if self.num_layers <= 0 or self.max_length <= 0:
            raise ValueError("num_layers and max_length must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0.0, 1.0)")
        if self.default_attention_mode not in {"bidirectional", "next_action"}:
            raise ValueError("default_attention_mode must be bidirectional or next_action")
        for name in (
            "value_vocab_size",
            "key_vocab_size",
            "modality_vocab_size",
            "source_vocab_size",
            "time_gap_vocab_size",
            "num_user_buckets",
            "categorical_pair_buckets",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")


@dataclass
class FoundationOutput:
    """Encoder output consumed by objectives and the public embedding API."""

    hidden_states: torch.Tensor
    long_term: torch.Tensor
    short_term: torch.Tensor
    padding_mask: torch.BoolTensor
