from __future__ import annotations

from types import SimpleNamespace

import torch

from src.models.encoder import FoundationModel, FoundationModelConfig
from src.pretraining.config import ObjectiveWeights, RoutingIds
from src.pretraining.masking import corrupt_for_masked_reconstruction

ROUTING = RoutingIds(sequence=1, numeric=2, geo=3)
WEIGHTS = ObjectiveWeights(
    categorical=1.0,
    numeric_value=1.0,
    numeric_missing=1.0,
    geo_coordinate=1.0,
    geo_missing=1.0,
    next_action=1.0,
)


def _batch(valid_next_action: list[bool]) -> SimpleNamespace:
    return SimpleNamespace(
        input_ids=torch.tensor([[1, 11, 12, 2, 13], [1, 14, 15, 2, 16]], dtype=torch.long),
        key_ids=torch.tensor([[0, 1, 2, 0, 3], [0, 1, 2, 0, 3]], dtype=torch.long),
        modality_ids=torch.tensor([[0, 1, 2, 0, 3], [0, 1, 2, 0, 3]], dtype=torch.long),
        source_ids=torch.tensor([[0, 1, 1, 0, 1], [0, 1, 1, 0, 1]], dtype=torch.long),
        position_ids=torch.tensor([[0, 1, 2, 0, 3], [0, 1, 2, 0, 3]], dtype=torch.long),
        time_gap_ids=torch.zeros((2, 5), dtype=torch.long),
        attention_mask=torch.ones((2, 5), dtype=torch.bool),
        numeric_values=torch.tensor(
            [[0.0, 0.0, 1.25, 0.0, 0.0], [0.0, 0.0, -0.75, 0.0, 0.0]],
            dtype=torch.float32,
        ),
        numeric_missing_mask=torch.zeros((2, 5), dtype=torch.bool),
        geo_values=torch.tensor(
            [
                [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.5, -0.25]],
                [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [-0.5, 0.75]],
            ],
            dtype=torch.float32,
        ),
        geo_missing_mask=torch.zeros((2, 5), dtype=torch.bool),
        user_ids=torch.tensor([5, 7], dtype=torch.long),
        targets={
            "next_action_type_ids": torch.tensor([11, 14], dtype=torch.long),
            "next_action_valid_mask": torch.tensor(valid_next_action, dtype=torch.bool),
        },
    )


def _model() -> FoundationModel:
    return FoundationModel(
        FoundationModelConfig(
            d_model=8,
            num_heads=2,
            num_layers=1,
            dropout=0.0,
            max_length=8,
            value_vocab_size=32,
            key_vocab_size=8,
            modality_vocab_size=5,
            source_vocab_size=4,
            time_gap_vocab_size=4,
            num_user_buckets=17,
            categorical_pair_buckets=31,
            sequence_source_id=1,
            numeric_modality_id=2,
        )
    )


def _loss(model: FoundationModel, batch: SimpleNamespace):
    from src.models.objectives import (
        PretrainingHeads,
        UserEmbeddingFusion,
        compute_pretraining_loss,
    )

    heads = PretrainingHeads(d_model=8, value_vocab_size=32)
    fusion = UserEmbeddingFusion(d_model=8)
    corrupted = corrupt_for_masked_reconstruction(
        batch, routing=ROUTING, probability=1.0, seed=5, global_step=0
    )
    return compute_pretraining_loss(
        masked_output=model(corrupted.batch, attention_mode="bidirectional"),
        next_action_output=model(batch, attention_mode="next_action"),
        corruption=corrupted,
        original_batch=batch,
        heads=heads,
        fusion=fusion,
        routing=ROUTING,
        weights=WEIGHTS,
    )


def test_next_action_loss_ignores_invalid_rows() -> None:
    """Removing the valid mask must allow targetless rows to affect next-action CE."""
    model = _model()

    loss = _loss(model, _batch([True, False]))

    assert loss.counts["next_action"] == 1
    assert torch.isfinite(loss.total)


def test_next_action_backpropagates_to_user_id_adapter() -> None:
    """Removing z_user from next-action prediction must leave UserIdAdapter without gradient."""
    model = _model()

    loss = _loss(model, _batch([True, True]))
    loss.total.backward()

    gradient = model.user_adapter.embedding.weight.grad
    assert gradient is not None
    assert gradient.abs().sum().item() > 0.0


def test_zero_valid_next_action_rows_keep_total_loss_finite() -> None:
    """An all-invalid causal target batch must still train masked reconstruction safely."""
    model = _model()

    loss = _loss(model, _batch([False, False]))

    assert loss.counts["next_action"] == 0
    assert torch.isfinite(loss.total)


def test_next_semantic_loss_uses_only_declared_semantic_rows() -> None:
    model = _model()
    batch = _batch([True, True])
    batch.targets["next_semantic_value_ids"] = torch.tensor([12, 0], dtype=torch.long)
    batch.targets["next_semantic_valid_mask"] = torch.tensor([True, False], dtype=torch.bool)

    loss = _loss(model, batch)

    assert loss.counts["next_semantic"] == 1
    assert torch.isfinite(loss.next_semantic)
