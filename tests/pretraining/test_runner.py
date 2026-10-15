from __future__ import annotations

from types import SimpleNamespace

import torch
import pytest

from src.pretraining.config import ObjectiveWeights, RoutingIds


def _batch() -> SimpleNamespace:
    return SimpleNamespace(
        input_ids=torch.tensor([[1, 11, 12, 2, 13]], dtype=torch.long),
        key_ids=torch.tensor([[0, 1, 2, 0, 3]], dtype=torch.long),
        modality_ids=torch.tensor([[0, 1, 2, 0, 3]], dtype=torch.long),
        source_ids=torch.tensor([[0, 1, 1, 0, 1]], dtype=torch.long),
        position_ids=torch.tensor([[0, 1, 2, 0, 3]], dtype=torch.long),
        time_gap_ids=torch.zeros((1, 5), dtype=torch.long),
        attention_mask=torch.ones((1, 5), dtype=torch.bool),
        numeric_values=torch.tensor([[0.0, 0.0, 1.25, 0.0, 0.0]], dtype=torch.float32),
        numeric_missing_mask=torch.zeros((1, 5), dtype=torch.bool),
        geo_values=torch.tensor(
            [[[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.5, -0.25]]],
            dtype=torch.float32,
        ),
        geo_missing_mask=torch.zeros((1, 5), dtype=torch.bool),
        user_ids=torch.tensor([5], dtype=torch.long),
        targets={
            "next_action_type_ids": torch.tensor([11], dtype=torch.long),
            "next_action_valid_mask": torch.tensor([True], dtype=torch.bool),
        },
    )


def test_train_one_step_updates_foundation_parameters(monkeypatch) -> None:
    """Removing optimizer.step must leave the Foundation model unchanged after a valid batch."""
    from src.pretraining import runner

    state = runner.make_training_state(
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
        routing=RoutingIds(sequence=1, numeric=2, geo=3),
        learning_rate=0.001,
        weight_decay=0.0,
        objective_weights=ObjectiveWeights(1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
    )
    monkeypatch.setattr(runner, "foundation_collate", lambda _: _batch())
    before = state.model.user_adapter.embedding.weight.detach().clone()

    losses = runner.train_one_step(state, [object()], global_step=0)

    assert torch.isfinite(losses.total)
    assert losses.counts["next_action"] == 1
    assert not torch.equal(before, state.model.user_adapter.embedding.weight)


def test_train_one_step_rejects_current_v2_collation(monkeypatch) -> None:
    """Removing the v3 batch gate must let the known incomplete collator train silently."""
    from src.pretraining import runner
    from src.foundation_input.types import FoundationBatch

    state = runner.make_training_state(
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
        routing=RoutingIds(sequence=1, numeric=2, geo=3),
        learning_rate=0.001,
        weight_decay=0.0,
        objective_weights=ObjectiveWeights(1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
    )
    v2 = FoundationBatch(
        input_ids=torch.tensor([[1, 2]], dtype=torch.long), key_ids=torch.zeros((1, 2), dtype=torch.long),
        modality_ids=torch.zeros((1, 2), dtype=torch.long), source_ids=torch.zeros((1, 2), dtype=torch.long),
        position_ids=torch.zeros((1, 2), dtype=torch.long), time_gap_ids=torch.zeros((1, 2), dtype=torch.long),
        attention_mask=torch.ones((1, 2), dtype=torch.bool), numeric_values=torch.zeros((1, 2)),
        numeric_missing_mask=torch.zeros((1, 2), dtype=torch.bool), user_ids=torch.tensor([1]), targets={
            "next_action_type_ids": torch.tensor([0]), "next_action_valid_mask": torch.tensor([False])
        },
    )
    monkeypatch.setattr(runner, "foundation_collate", lambda _: v2)

    with pytest.raises(RuntimeError, match="geo_values"):
        runner.train_one_step(state, [object()], global_step=0)


def test_train_one_step_uses_state_masking_probability(monkeypatch) -> None:
    """Hard-coding 0.15 must not mask fields when a run explicitly configures zero masking."""
    from src.pretraining import runner

    state = runner.make_training_state(
        d_model=8, num_heads=2, num_layers=1, dropout=0.0, max_length=8,
        value_vocab_size=32, key_vocab_size=8, modality_vocab_size=5, source_vocab_size=4,
        time_gap_vocab_size=4, num_user_buckets=17, categorical_pair_buckets=31,
        routing=RoutingIds(sequence=1, numeric=2, geo=3), learning_rate=0.001,
        weight_decay=0.0, objective_weights=ObjectiveWeights(1, 1, 1, 1, 1, 1),
        masking_probability=0.0,
    )
    monkeypatch.setattr(runner, "foundation_collate", lambda _: _batch())

    losses = runner.train_one_step(state, [object()], global_step=0)

    assert losses.counts["categorical"] == 0
    assert losses.counts["numeric_value"] == 0
    assert losses.counts["geo_coordinate"] == 0
