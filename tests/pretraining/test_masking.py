from __future__ import annotations

from types import SimpleNamespace

import torch

from src.pretraining.config import RoutingIds


ROUTING = RoutingIds(sequence=1, numeric=2, geo=3)


def _v3_batch() -> SimpleNamespace:
    return SimpleNamespace(
        input_ids=torch.tensor([[1, 11, 12, 2, 13, 0]], dtype=torch.long),
        key_ids=torch.zeros((1, 6), dtype=torch.long),
        modality_ids=torch.tensor([[0, 1, 2, 0, 3, 0]], dtype=torch.long),
        source_ids=torch.tensor([[0, 1, 1, 0, 1, 0]], dtype=torch.long),
        position_ids=torch.zeros((1, 6), dtype=torch.long),
        time_gap_ids=torch.zeros((1, 6), dtype=torch.long),
        attention_mask=torch.tensor([[True, True, True, True, True, False]], dtype=torch.bool),
        numeric_values=torch.tensor([[0.0, 0.0, 1.25, 0.0, 0.0, 0.0]], dtype=torch.float32),
        numeric_missing_mask=torch.zeros((1, 6), dtype=torch.bool),
        geo_values=torch.tensor(
            [[[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.25, -0.75], [0.0, 0.0]]],
            dtype=torch.float32,
        ),
        geo_missing_mask=torch.zeros((1, 6), dtype=torch.bool),
        user_ids=torch.tensor([7], dtype=torch.long),
        targets={},
    )


def test_masking_neutralizes_continuous_targets_without_mutating_source_batch() -> None:
    """Deleting copy-on-write masking must expose numeric/geo labels back to the model."""
    from src.pretraining.masking import corrupt_for_masked_reconstruction

    batch = _v3_batch()
    original_ids = batch.input_ids.clone()
    corrupted = corrupt_for_masked_reconstruction(
        batch, routing=ROUTING, probability=1.0, seed=7, global_step=3
    )

    assert not corrupted.targets.selected_mask[0, 0]  # CLS
    assert not corrupted.targets.selected_mask[0, 3]  # SEP
    assert not corrupted.targets.selected_mask[0, 5]  # PAD
    assert corrupted.batch.input_ids[0, 1].item() == 3  # MASK
    assert corrupted.batch.numeric_values[0, 2].item() == 0.0
    assert corrupted.batch.numeric_missing_mask[0, 2].item() is True
    torch.testing.assert_close(corrupted.batch.geo_values[0, 4], torch.zeros(2))
    assert corrupted.batch.geo_missing_mask[0, 4].item() is True
    torch.testing.assert_close(corrupted.targets.original_numeric_values, batch.numeric_values)
    torch.testing.assert_close(corrupted.targets.original_geo_values, batch.geo_values)
    torch.testing.assert_close(batch.input_ids, original_ids)
    assert batch.numeric_values[0, 2].item() == 1.25


def test_masking_is_deterministic_for_same_seed_and_step() -> None:
    """Changing random generator state outside the function must not change its corruption."""
    from src.pretraining.masking import corrupt_for_masked_reconstruction

    first = corrupt_for_masked_reconstruction(
        _v3_batch(), routing=ROUTING, probability=0.5, seed=11, global_step=5
    )
    second = corrupt_for_masked_reconstruction(
        _v3_batch(), routing=ROUTING, probability=0.5, seed=11, global_step=5
    )

    torch.testing.assert_close(first.targets.selected_mask, second.targets.selected_mask)
    torch.testing.assert_close(first.batch.input_ids, second.batch.input_ids)
