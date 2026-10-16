from __future__ import annotations

import json
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from src.pretraining.config import ObjectiveWeights, RoutingIds


def _state():
    from src.pretraining.runner import make_training_state

    return make_training_state(
        d_model=8, num_heads=2, num_layers=1, dropout=0.0, max_length=8,
        value_vocab_size=32, key_vocab_size=8, modality_vocab_size=5, source_vocab_size=4,
        time_gap_vocab_size=4, num_user_buckets=17, categorical_pair_buckets=31,
        routing=RoutingIds(sequence=1, numeric=2, geo=3), learning_rate=0.001,
        weight_decay=0.0, objective_weights=ObjectiveWeights(1, 1, 1, 1, 1, 1),
    )


def _identity():
    from src.pretraining.artifacts import CheckpointIdentity

    return CheckpointIdentity(
        config_digest="config", linked_manifest_sha256="linked", control_manifest_sha256="control",
        vocabulary_sha256="vocab", corpus="linked",
    )


def _v3_batch() -> SimpleNamespace:
    return SimpleNamespace(
        input_ids=torch.tensor([[1, 11, 12, 2]], dtype=torch.long),
        key_ids=torch.tensor([[0, 1, 2, 0]], dtype=torch.long),
        modality_ids=torch.tensor([[0, 1, 2, 0]], dtype=torch.long),
        source_ids=torch.tensor([[0, 1, 1, 0]], dtype=torch.long),
        position_ids=torch.tensor([[0, 1, 2, 0]], dtype=torch.long),
        time_gap_ids=torch.zeros((1, 4), dtype=torch.long), attention_mask=torch.ones((1, 4), dtype=torch.bool),
        numeric_values=torch.zeros((1, 4), dtype=torch.float32),
        numeric_missing_mask=torch.zeros((1, 4), dtype=torch.bool),
        geo_values=torch.zeros((1, 4, 2), dtype=torch.float32),
        geo_missing_mask=torch.zeros((1, 4), dtype=torch.bool), user_ids=torch.tensor([5]), targets={},
    )


def test_load_frozen_embedder_rejects_manifest_mismatch(tmp_path) -> None:
    """Removing checkpoint identity validation must allow a control checkpoint into linked downstream."""
    from src.pretraining.artifacts import CheckpointIdentity, load_frozen_embedder, save_checkpoint

    checkpoint = save_checkpoint(_state(), kind="best", output_dir=tmp_path, identity=_identity(), global_step=1)

    with pytest.raises(ValueError, match="linked_manifest_sha256"):
        load_frozen_embedder(
            checkpoint,
            expected=CheckpointIdentity("config", "wrong", "control", "vocab", "linked"),
        )


def test_loaded_embedder_is_frozen_and_returns_z_user(tmp_path) -> None:
    """Removing fusion or parameter freezing must break the shared downstream embedding contract."""
    from src.pretraining.artifacts import load_frozen_embedder, save_checkpoint

    checkpoint = save_checkpoint(_state(), kind="best", output_dir=tmp_path, identity=_identity(), global_step=1)
    embedder = load_frozen_embedder(checkpoint, expected=_identity())

    output = embedder.embed_users(_v3_batch())

    assert set(output) == {"long_term", "short_term", "z_user"}
    assert output["z_user"].shape == (1, 8)
    assert not any(parameter.requires_grad for parameter in embedder.parameters())


def test_resume_training_state_restores_weights_and_step(tmp_path) -> None:
    """Removing resume restoration must restart a requested checkpoint from random weights."""
    from src.pretraining.artifacts import resume_training_state, save_checkpoint

    original = _state()
    checkpoint = save_checkpoint(original, kind="last", output_dir=tmp_path, identity=_identity(), global_step=7)
    resumed = _state()

    step = resume_training_state(resumed, checkpoint, expected=_identity())

    assert step == 7
    for expected, actual in zip(original.model.parameters(), resumed.model.parameters(), strict=True):
        torch.testing.assert_close(actual, expected)


def test_checkpoint_keeps_per_kind_manifests_and_round_trips_rng(tmp_path) -> None:
    """A resumable run needs independent last/best identities and all RNG streams."""
    from src.pretraining.artifacts import resume_training_state, save_checkpoint

    random.seed(7)
    np.random.seed(7)
    torch.manual_seed(7)
    state = _state()
    checkpoint = save_checkpoint(state, kind="last", output_dir=tmp_path, identity=_identity(), global_step=2)
    expected = (random.random(), float(np.random.random()), float(torch.rand(())))
    random.seed(99)
    np.random.seed(99)
    torch.manual_seed(99)
    resume_training_state(_state(), checkpoint, expected=_identity())

    save_checkpoint(state, kind="best", output_dir=tmp_path, identity=_identity(), global_step=2)
    selected = json.loads((tmp_path / "checkpoint_manifest.json").read_text(encoding="utf-8"))

    assert (tmp_path / "last_manifest.json").is_file()
    assert (tmp_path / "best_manifest.json").is_file()
    assert selected["selected_kind"] == "best"
    assert (random.random(), float(np.random.random()), float(torch.rand(()))) == expected


def test_public_checkpoint_loader_verifies_selected_manifest(tmp_path) -> None:
    from src.pretraining.artifacts import save_checkpoint
    from src.training.checkpoint import load_foundation_checkpoint

    checkpoint = save_checkpoint(_state(), kind="best", output_dir=tmp_path, identity=_identity(), global_step=1)
    embedder = load_foundation_checkpoint(checkpoint, tmp_path / "best_manifest.json")

    assert not any(parameter.requires_grad for parameter in embedder.parameters())
    with pytest.raises(ValueError, match="SHA-256"):
        checkpoint.write_bytes(b"tampered")
        load_foundation_checkpoint(checkpoint, tmp_path / "best_manifest.json")
