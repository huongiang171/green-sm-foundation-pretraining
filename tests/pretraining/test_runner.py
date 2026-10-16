from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from src.data.green_sm.build import build_dataset, root_names
from src.data.green_sm.config import GreenSmConfig
from src.foundation_input.dataset import foundation_collate
from src.models.encoder import FoundationModel
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
    v2 = SimpleNamespace(
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


def test_build_corpus_inputs_uses_declared_green_sm_geo_contract(tmp_path) -> None:
    """The runner must fit and encode using the corpus's immutable tokenizer policy."""
    from src.pretraining.config import (
        DatasetSettings,
        ModelSettings,
        PretrainingConfig,
        TrainingSettings,
    )
    from src.pretraining.gates import PreflightResult, validate_v3_batch_contract
    from src.pretraining.runner import build_corpus_run_inputs

    data_config = replace(
        GreenSmConfig.from_json("configs/green_sm_sim_pilot.json"),
        data_version="green_sm_sim_pilot_v2",
        output_root=tmp_path / "green-sm-pilot-v2",
    )
    build_dataset(data_config)
    names = root_names(data_config.data_version)
    config = PretrainingConfig(
        linked_root=data_config.output_root / names["linked"] / "foundation",
        control_root=data_config.output_root / names["paired_control"] / "foundation",
        output_dir=tmp_path / "artifacts",
        seed=7,
        model=ModelSettings(8, 2, 1, 0.0, 256, 17, 97),
        dataset=DatasetSettings(256, 30),
        training=TrainingSettings(2, 1, 1, 1, 0.001, 0.0, 8),
        masking_probability=0.15,
        objectives=ObjectiveWeights(1, 1, 1, 1, 1, 1),
        config_digest="fixture-config",
    )

    inputs = build_corpus_run_inputs(
        config,
        PreflightResult("linked-manifest", "control-manifest", "generator", "world"),
        corpus="linked",
    )

    assert inputs.vocabulary.modality_to_id["geo"] == inputs.model_config.geo_modality_id
    assert inputs.vocabulary.geo_normalization is not None
    assert inputs.vocabulary.tokenizer_contract["profile_allowlist"] == [
        "active_days_so_far",
        "budget_bucket",
        "home_location",
        "preferred_service",
        "work_location",
    ]
    batch = foundation_collate([next(iter(inputs.train))])
    validate_v3_batch_contract(batch, RoutingIds(
        sequence=inputs.vocabulary.source_to_id["sequence"],
        numeric=inputs.vocabulary.modality_to_id["numeric"],
        geo=inputs.vocabulary.modality_to_id["geo"],
    ))
    with torch.no_grad():
        output = FoundationModel(inputs.model_config).encode(batch)
    assert output.short_term.shape == (1, 8)


def test_run_corpus_uses_one_based_eval_checkpoint_schedule_and_immutable_run_dir(
    monkeypatch, tmp_path: Path
) -> None:
    """Periodic evaluation must select the lowest-loss checkpoint without overwrite."""
    from src.pretraining import artifacts, runner
    from src.pretraining.config import (
        DatasetSettings,
        ModelSettings,
        PretrainingConfig,
        TrainingSettings,
    )
    from src.pretraining.gates import PreflightResult

    class Vocabulary:
        def __init__(self) -> None:
            self.source_to_id = {"sequence": 1}
            self.modality_to_id = {"numeric": 2, "geo": 3}

        def save(self, path: Path) -> None:
            path.write_text("{}\n", encoding="utf-8")

        def to_metadata(self) -> dict[str, object]:
            return {}

    model = SimpleNamespace(
        d_model=8, num_heads=2, num_layers=1, dropout=0.0, max_length=256,
        value_vocab_size=32, key_vocab_size=8, modality_vocab_size=5, source_vocab_size=4,
        time_gap_vocab_size=4, num_user_buckets=17, categorical_pair_buckets=31,
    )
    inputs = runner.CorpusRunInputs(Vocabulary(), model, range(3), [object()])
    config = PretrainingConfig(
        linked_root=tmp_path / "linked", control_root=tmp_path / "control", output_dir=tmp_path / "out",
        seed=7, run_id="schedule", model=ModelSettings(8, 2, 1, 0.0, 256, 17, 31),
        dataset=DatasetSettings(256, 30), training=TrainingSettings(1, 3, 1, 2, 0.001, 0.0, 8, 1),
        masking_probability=0.15, objectives=ObjectiveWeights(1, 1, 1, 1, 1, 1), config_digest="digest",
    )
    state = runner.make_training_state(
        d_model=8, num_heads=2, num_layers=1, dropout=0.0, max_length=256,
        value_vocab_size=32, key_vocab_size=8, modality_vocab_size=5, source_vocab_size=4,
        time_gap_vocab_size=4, num_user_buckets=17, categorical_pair_buckets=31,
        routing=RoutingIds(1, 2, 3), learning_rate=0.001, weight_decay=0.0,
        objective_weights=ObjectiveWeights(1, 1, 1, 1, 1, 1),
    )
    evaluations = iter((3.0, 1.0, 2.0))
    writes: list[tuple[str, int]] = []
    monkeypatch.setattr(runner, "build_corpus_run_inputs", lambda *args, **kwargs: inputs)
    monkeypatch.setattr(runner, "make_training_state", lambda **kwargs: state)
    monkeypatch.setattr(
        runner,
        "train_one_step",
        lambda _state, _samples, *, global_step: SimpleNamespace(
            total=torch.tensor(float(global_step + 1)), counts={}
        ),
    )
    monkeypatch.setattr(runner, "evaluate_split", lambda *args, **kwargs: {"total": next(evaluations)})
    monkeypatch.setattr(
        artifacts,
        "save_checkpoint",
        lambda _state, *, kind, output_dir, identity, global_step: (
            writes.append((kind, global_step)) or Path(output_dir) / f"{kind}.pt"
        ),
    )
    preflight = PreflightResult("linked", "control", "generator", "world")

    result = runner.run_corpus(config, preflight, corpus="linked", output_dir=config.output_dir, device="cpu")

    run_dir = config.output_dir / config.run_id / "linked"
    manifest = __import__("json").loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert writes == [("best", 1), ("best", 2), ("last", 2), ("last", 3)]
    assert [entry["step"] for entry in manifest["evaluations"]] == [1, 2, 3]
    assert result["checkpoint"] == str(run_dir / "last.pt")
    with pytest.raises(ValueError, match="refuses non-empty"):
        runner.run_corpus(config, preflight, corpus="linked", output_dir=config.output_dir, device="cpu")
