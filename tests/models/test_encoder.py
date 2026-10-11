from __future__ import annotations

import pytest
import torch

from src.foundation_input.dataset import foundation_collate
from src.foundation_input.types import EncodedSample, KeyValueToken, TokenRecord
from src.models.encoder import FoundationModel, FoundationModelConfig, build_attention_mask


def test_config_rejects_non_divisible_attention_dimension() -> None:
    with pytest.raises(ValueError, match="d_model=.*num_heads"):
        FoundationModelConfig(d_model=30, num_heads=8)


def test_config_rejects_unknown_default_attention_mode() -> None:
    with pytest.raises(ValueError, match="default_attention_mode"):
        FoundationModelConfig(default_attention_mode="causal")


def test_config_rejects_zero_attention_heads(model_config) -> None:
    with pytest.raises(ValueError, match="num_heads"):
        model_config(num_heads=0)


def test_config_rejects_sequence_source_outside_source_vocab(model_config) -> None:
    with pytest.raises(ValueError, match="sequence_source_id"):
        model_config(source_vocab_size=2, sequence_source_id=2)


def test_config_requires_real_input_cardinalities() -> None:
    with pytest.raises(ValueError, match="value_vocab_size"):
        FoundationModelConfig()


def test_next_action_blocks_future_events_but_allows_profile_context(batch_factory) -> None:
    batch = batch_factory()

    blocked = build_attention_mask(batch, "next_action", sequence_source_id=1)

    assert blocked[0, 3, 4].item() is True
    assert blocked[0, 3, 1].item() is False
    assert blocked[0, 1, 3].item() is True


def test_bidirectional_does_not_add_causal_blocks(batch_factory) -> None:
    batch = batch_factory()

    blocked = build_attention_mask(batch, "bidirectional", sequence_source_id=1)

    assert blocked[0, 3, 4].item() is False


def test_forward_returns_contract_shapes_and_finite_values(batch_factory, model_config) -> None:
    model = FoundationModel(model_config())

    output = model(batch_factory(), attention_mode="bidirectional")

    assert output.hidden_states.shape == (2, 7, 16)
    assert output.long_term.shape == (2, 16)
    assert output.short_term.shape == (2, 16)
    assert output.padding_mask.dtype is torch.bool
    assert torch.isfinite(output.hidden_states[output.padding_mask]).all()


def test_embed_users_returns_only_the_two_public_embeddings(batch_factory, model_config) -> None:
    model = FoundationModel(model_config()).eval()

    result = model.embed_users(batch_factory())

    assert set(result) == {"long_term", "short_term"}
    assert result["long_term"].shape == (2, 16)


def test_eval_encoding_is_deterministic(batch_factory, model_config) -> None:
    model = FoundationModel(model_config(dropout=0.2)).eval()
    batch = batch_factory()

    first = model.encode(batch).short_term
    second = model.encode(batch).short_term

    torch.testing.assert_close(first, second, rtol=0.0, atol=1e-6)


def test_changing_padding_ids_does_not_change_valid_outputs(batch_factory, model_config) -> None:
    model = FoundationModel(model_config()).eval()
    original = batch_factory()
    changed = batch_factory()
    changed.input_ids[0, 5:] = torch.tensor([31, 47])
    changed.key_ids[0, 5:] = torch.tensor([9, 10])

    first = model.encode(original).hidden_states[0, :5]
    second = model.encode(changed).hidden_states[0, :5]

    torch.testing.assert_close(first, second, rtol=0.0, atol=1e-6)


def test_permuting_tabular_columns_preserves_short_term_in_eval(batch_factory, model_config) -> None:
    model = FoundationModel(model_config()).eval()
    first = batch_factory(two_tabular_tokens=True)
    second = batch_factory(two_tabular_tokens=True)
    second.input_ids[0, 1:3] = second.input_ids[0, [2, 1]]
    second.key_ids[0, 1:3] = second.key_ids[0, [2, 1]]

    torch.testing.assert_close(
        model.encode(first).short_term, model.encode(second).short_term, rtol=0.0, atol=1e-6
    )


def test_future_event_cannot_change_past_event_hidden_state(batch_factory, model_config) -> None:
    model = FoundationModel(model_config()).eval()
    first = batch_factory()
    changed = batch_factory()
    changed.input_ids[0, 4] = 37

    before = model(first, attention_mode="next_action").hidden_states[0, 3]
    after = model(changed, attention_mode="next_action").hidden_states[0, 3]

    torch.testing.assert_close(before, after, rtol=0.0, atol=1e-6)


def test_malformed_batches_raise_contextual_errors(batch_factory, model_config) -> None:
    model = FoundationModel(model_config())
    negative = batch_factory()
    negative.input_ids[0, 1] = -1
    with pytest.raises(ValueError, match="input_ids"):
        model.encode(negative)

    mismatched = batch_factory()
    mismatched.key_ids = mismatched.key_ids[:, :-1]
    with pytest.raises(ValueError, match="key_ids"):
        model.encode(mismatched)

    with pytest.raises(ValueError, match="attention_mode"):
        model(batch_factory(), attention_mode="causal")

    invalid_cls = batch_factory()
    invalid_cls.attention_mask[1, 0] = False
    with pytest.raises(ValueError, match="CLS"):
        model.encode(invalid_cls)


def test_foundation_input_smoke_uses_public_collator(model_config) -> None:
    records = (
        TokenRecord(
            token=KeyValueToken("profile.city", "Hanoi", "categorical", "tabular", None),
            value_id=7,
            key_id=3,
            modality_id=1,
            source_id=2,
            position_id=0,
            time_gap_id=0,
        ),
        TokenRecord(
            token=KeyValueToken(
                "event.event_type", "view", "categorical", "sequence", None, "event-1"
            ),
            value_id=9,
            key_id=4,
            modality_id=1,
            source_id=1,
            position_id=1,
            time_gap_id=0,
        ),
    )
    batch = foundation_collate(
        [
            EncodedSample("sample-1", "fixture-v1", "user-1", records, user_id_value_id=6),
            EncodedSample("sample-2", "fixture-v1", "user-2", records, user_id_value_id=8),
        ]
    )
    config = model_config()

    output = FoundationModel(config).encode(batch)

    assert output.hidden_states.shape[0] == 2
    assert output.hidden_states.shape[1] == batch.input_ids.shape[1]
    assert output.long_term.shape == output.short_term.shape == (2, config.d_model)
    assert torch.isfinite(output.long_term).all()
    assert torch.isfinite(output.short_term).all()
