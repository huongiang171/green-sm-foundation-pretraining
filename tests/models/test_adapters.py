from __future__ import annotations

import pytest
import torch

from src.models.adapters import AdapterRegistry, GeoAdapter, UserIdAdapter


def test_pair_keying_distinguishes_reused_value_ids(batch_factory, model_config) -> None:
    batch = batch_factory()
    batch.input_ids[0, 1] = 9
    batch.input_ids[0, 3] = 9
    batch.key_ids[0, 1] = 3
    batch.key_ids[0, 3] = 4

    representations = AdapterRegistry(model_config())(batch)

    assert not torch.equal(representations[0, 1], representations[0, 3])


def test_large_hashed_user_id_is_mapped_inside_embedding_range() -> None:
    adapter = UserIdAdapter(num_user_buckets=17, d_model=8)

    output = adapter(torch.tensor([2**62 + 123, 5], dtype=torch.long))

    assert output.shape == (2, 8)
    assert torch.isfinite(output).all()


def test_position_and_time_gaps_apply_only_to_sequence_tokens(batch_factory, model_config) -> None:
    base = batch_factory()
    changed = batch_factory()
    changed.position_ids[0, 1] = 5
    changed.time_gap_ids[0, 1] = 3
    changed.position_ids[0, 3] = 5
    changed.time_gap_ids[0, 3] = 3
    registry = AdapterRegistry(model_config()).eval()

    base_features = registry(base)
    changed_features = registry(changed)

    torch.testing.assert_close(base_features[0, 1], changed_features[0, 1])
    assert not torch.equal(base_features[0, 3], changed_features[0, 3])


def test_padding_rows_are_zero_and_negative_ids_fail(batch_factory, model_config) -> None:
    registry = AdapterRegistry(model_config())
    batch = batch_factory()

    features = registry(batch)
    assert torch.equal(features[0, 5:], torch.zeros_like(features[0, 5:]))

    batch.input_ids[0, 1] = -1
    with pytest.raises(ValueError, match="input_ids"):
        registry(batch)


def test_geo_adapter_returns_zero_for_missing_coordinates() -> None:
    adapter = GeoAdapter(d_model=8)
    coordinates = torch.tensor([[[0.1, 0.2], [0.3, 0.4]]], dtype=torch.float32)
    missing = torch.tensor([[False, True]])

    output = adapter(coordinates, missing)

    assert output.shape == (1, 2, 8)
    assert torch.isfinite(output).all()
    assert torch.equal(output[:, 1], torch.zeros_like(output[:, 1]))
