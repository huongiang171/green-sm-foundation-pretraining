"""Streaming training primitives for the Green SM Foundation model."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import random
import sys
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import asdict, dataclass
from itertools import chain, islice
from pathlib import Path
from typing import Literal

import numpy as np
import torch

from src.foundation_input.dataset import (
    DatasetConfig,
    FoundationPretrainDataset,
    foundation_collate,
)
from src.foundation_input.loader import iter_normalized_samples
from src.foundation_input.vocab import (
    VocabularyArtifacts,
    VocabularyConfig,
    fit_vocabulary_from_samples,
)
from src.models.encoder import FoundationModel, FoundationModelConfig
from src.models.objectives import (
    LossBreakdown,
    PretrainingHeads,
    UserEmbeddingFusion,
    compute_pretraining_loss,
)

from .config import ObjectiveWeights, PretrainingConfig, RoutingIds
from .gates import PreflightResult, validate_v3_batch_contract
from .green_sm_contract import load_green_sm_tokenizer_config
from .masking import corrupt_for_masked_reconstruction


@dataclass(frozen=True)
class CorpusRunInputs:
    vocabulary: VocabularyArtifacts
    model_config: FoundationModelConfig
    train: Iterable
    validation: Iterable


@dataclass
class TrainingState:
    model: FoundationModel
    fusion: UserEmbeddingFusion
    heads: PretrainingHeads
    optimizer: torch.optim.Optimizer
    routing: RoutingIds
    objective_weights: ObjectiveWeights
    masking_probability: float
    device: torch.device

    @property
    def modules(self) -> tuple[torch.nn.Module, ...]:
        return self.model, self.fusion, self.heads


def _routing_from_vocab(vocabulary: VocabularyArtifacts) -> RoutingIds:
    try:
        return RoutingIds(
            sequence=vocabulary.source_to_id["sequence"],
            numeric=vocabulary.modality_to_id["numeric"],
            geo=vocabulary.modality_to_id["geo"],
        )
    except KeyError as error:
        raise ValueError(f"Vocabulary lacks required Green SM routing ID: {error.args[0]}") from error


def _model_config(config: PretrainingConfig, vocabulary: VocabularyArtifacts, routing: RoutingIds) -> FoundationModelConfig:
    cardinalities = vocabulary.model_cardinalities()
    kwargs = {
        "d_model": config.model.d_model, "num_heads": config.model.num_heads,
        "num_layers": config.model.num_layers, "dropout": config.model.dropout,
        "max_length": config.model.max_length, "value_vocab_size": cardinalities["value_vocab_size"],
        "key_vocab_size": cardinalities["key_vocab_size"], "modality_vocab_size": cardinalities["modality_vocab_size"],
        "source_vocab_size": cardinalities["source_vocab_size"], "time_gap_vocab_size": cardinalities["time_gap_vocab_size"],
        "num_user_buckets": config.model.num_user_buckets,
        "categorical_pair_buckets": config.model.categorical_pair_buckets,
        "sequence_source_id": routing.sequence, "numeric_modality_id": routing.numeric,
    }
    kwargs["geo_modality_id"] = routing.geo
    return FoundationModelConfig(**kwargs)


def build_corpus_run_inputs(
    config: PretrainingConfig,
    preflight: PreflightResult,
    *,
    corpus: Literal["linked", "control"],
) -> CorpusRunInputs:
    """Fit one corpus's preprocessing metadata strictly from its train stream."""

    del preflight  # Identity is checked before any data is consumed by this layer.
    root = config.linked_root if corpus == "linked" else config.control_root
    train_samples = iter_normalized_samples(root, "train", validate=True)
    try:
        first = next(train_samples)
    except StopIteration as error:
        raise ValueError(f"{corpus} corpus has no train samples") from error
    tokenizer = load_green_sm_tokenizer_config(root)
    samples = chain((first,), train_samples)
    vocabulary = fit_vocabulary_from_samples(
        samples,
        VocabularyConfig(dataset_name=first.dataset_name),
        tokenizer,
    )
    routing = _routing_from_vocab(vocabulary)
    model_config = _model_config(config, vocabulary, routing)
    dataset_config = DatasetConfig(max_event_count=config.dataset.max_event_count, max_tokens=config.dataset.max_tokens, next_action_prediction=True)
    return CorpusRunInputs(
        vocabulary=vocabulary,
        model_config=model_config,
        train=FoundationPretrainDataset(root, "train", vocabulary, tokenizer, dataset_config),
        validation=FoundationPretrainDataset(root, "validation", vocabulary, tokenizer, dataset_config),
    )


def make_training_state(
    *,
    d_model: int,
    num_heads: int,
    num_layers: int,
    dropout: float,
    max_length: int,
    value_vocab_size: int,
    key_vocab_size: int,
    modality_vocab_size: int,
    source_vocab_size: int,
    time_gap_vocab_size: int,
    num_user_buckets: int,
    categorical_pair_buckets: int,
    routing: RoutingIds,
    learning_rate: float,
    weight_decay: float,
    objective_weights: ObjectiveWeights,
    masking_probability: float = 0.15,
    device: str | torch.device = "cpu",
) -> TrainingState:
    """Construct all trainable modules from persisted vocabulary metadata."""

    model_kwargs = {
        "d_model": d_model, "num_heads": num_heads, "num_layers": num_layers, "dropout": dropout,
        "max_length": max_length, "value_vocab_size": value_vocab_size, "key_vocab_size": key_vocab_size,
        "modality_vocab_size": modality_vocab_size, "source_vocab_size": source_vocab_size,
        "time_gap_vocab_size": time_gap_vocab_size, "num_user_buckets": num_user_buckets,
        "categorical_pair_buckets": categorical_pair_buckets, "sequence_source_id": routing.sequence,
        "numeric_modality_id": routing.numeric,
    }
    model_kwargs["geo_modality_id"] = routing.geo
    model = FoundationModel(FoundationModelConfig(**model_kwargs))
    fusion = UserEmbeddingFusion(d_model=d_model)
    heads = PretrainingHeads(d_model=d_model, value_vocab_size=value_vocab_size)
    resolved_device = torch.device(device)
    for module in (model, fusion, heads):
        module.to(resolved_device)
    optimizer = torch.optim.AdamW(
        chain(model.parameters(), fusion.parameters(), heads.parameters()),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    if not 0.0 <= masking_probability <= 1.0:
        raise ValueError("masking_probability must be in [0, 1]")
    return TrainingState(
        model, fusion, heads, optimizer, routing, objective_weights, masking_probability, resolved_device
    )


def _to_device(batch, device: torch.device):
    copied = copy.copy(batch)
    for name, value in vars(batch).items():
        if isinstance(value, torch.Tensor):
            setattr(copied, name, value.to(device))
        elif name == "targets":
            copied.targets = {key: tensor.to(device) for key, tensor in value.items()}
    return copied


def _finite_gradients(state: TrainingState) -> bool:
    return all(
        parameter.grad is None or torch.isfinite(parameter.grad).all().item()
        for module in state.modules
        for parameter in module.parameters()
    )


def train_one_step(state: TrainingState, samples: list, *, global_step: int) -> LossBreakdown:
    """Run both visibility modes and update only when all gradients are finite."""

    batch = _to_device(foundation_collate(samples), state.device)
    validate_v3_batch_contract(batch, state.routing)
    state.model.train()
    state.fusion.train()
    state.heads.train()
    state.optimizer.zero_grad(set_to_none=True)
    corrupted = corrupt_for_masked_reconstruction(
        batch,
        routing=state.routing,
        probability=state.masking_probability,
        seed=global_step,
        global_step=global_step,
    )
    losses = compute_pretraining_loss(
        masked_output=state.model(corrupted.batch, attention_mode="bidirectional"),
        next_action_output=state.model(batch, attention_mode="next_action"),
        corruption=corrupted,
        original_batch=batch,
        heads=state.heads,
        fusion=state.fusion,
        routing=state.routing,
        weights=state.objective_weights,
    )
    if not torch.isfinite(losses.total):
        raise FloatingPointError("Pretraining loss is non-finite")
    losses.total.backward()
    if not _finite_gradients(state):
        raise FloatingPointError("Pretraining gradients are non-finite")
    state.optimizer.step()
    return losses


def batches(samples: Iterable, *, batch_size: int) -> Iterator[list]:
    """Yield bounded batches without materializing a streaming dataset."""

    iterator = iter(samples)
    while chunk := list(islice(iterator, batch_size)):
        yield chunk


def repeating_training_batches(
    samples: Iterable,
    *,
    batch_size: int,
    start_step: int,
    max_steps: int,
) -> Iterator[tuple[int, list]]:
    """Repeat a re-iterable finite corpus until its global step budget is met.

    The global batch index is the training schedule.  Replaying the prefix on
    resume lets a checkpoint at ``start_step`` continue at the same position in
    the deterministic epoch order without materializing the corpus in memory.
    """

    if start_step < 0:
        raise ValueError("start_step must be non-negative")
    if max_steps < start_step:
        raise ValueError("max_steps must be at least start_step")

    global_step = 0
    while global_step < max_steps:
        saw_batch = False
        for sample_batch in batches(samples, batch_size=batch_size):
            saw_batch = True
            if global_step >= start_step:
                yield global_step, sample_batch
            global_step += 1
            if global_step >= max_steps:
                return
        if not saw_batch:
            raise ValueError("Cannot train from an empty training split")


def _atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _seed_runtime(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _read_json_list(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise ValueError(f"Run trace must be a list of objects: {path}")
    return list(value)


def _effective_config_record(config: PretrainingConfig) -> dict[str, object]:
    """Serialize the exact training recipe used by a run, excluding no material setting."""

    return {
        "schema_version": "green_sm_pretraining_effective_config_v1",
        "config_digest": config.config_digest,
        "run_id": config.run_id,
        "linked_root": str(config.linked_root),
        "control_root": str(config.control_root),
        "output_dir": str(config.output_dir),
        "seed": config.seed,
        "model": asdict(config.model),
        "dataset": asdict(config.dataset),
        "training": asdict(config.training),
        "masking_probability": config.masking_probability,
        "objectives": asdict(config.objectives),
    }


@torch.no_grad()
def evaluate_split(
    state: TrainingState,
    samples: Iterable,
    *,
    batch_size: int = 1,
    max_batches: int | None = None,
) -> Mapping[str, float]:
    """Evaluate masked/causal losses without mutating parameters or optimizer state."""

    state.model.eval()
    state.fusion.eval()
    state.heads.eval()
    totals: dict[str, float] = {"total": 0.0, "next_action": 0.0, "next_semantic": 0.0}
    seen = 0
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    for index, sample_batch in enumerate(batches(samples, batch_size=batch_size)):
        if max_batches is not None and index >= max_batches:
            break
        batch = _to_device(foundation_collate(sample_batch), state.device)
        validate_v3_batch_contract(batch, state.routing)
        corrupted = corrupt_for_masked_reconstruction(
            batch,
            routing=state.routing,
            probability=state.masking_probability,
            seed=0,
            global_step=index,
        )
        loss = compute_pretraining_loss(
            masked_output=state.model(corrupted.batch, attention_mode="bidirectional"),
            next_action_output=state.model(batch, attention_mode="next_action"),
            corruption=corrupted,
            original_batch=batch,
            heads=state.heads,
            fusion=state.fusion,
            routing=state.routing,
            weights=state.objective_weights,
        )
        totals["total"] += float(loss.total.item())
        totals["next_action"] += float(loss.next_action.item())
        totals["next_semantic"] += float(loss.next_semantic.item())
        seen += 1
    if not seen:
        raise ValueError("Cannot evaluate an empty split")
    return {key: value / seen for key, value in totals.items()}


def run_corpus(
    config: PretrainingConfig,
    preflight: PreflightResult,
    *,
    corpus: Literal["linked", "control"],
    output_dir: Path,
    device: str | torch.device,
    max_steps: int | None = None,
    resume: Path | None = None,
) -> Mapping[str, object]:
    """Train one corpus and emit a checkpoint whose identity is auditable downstream."""

    from .artifacts import CheckpointIdentity, resume_training_state, save_checkpoint

    _seed_runtime(config.seed)
    inputs = build_corpus_run_inputs(config, preflight, corpus=corpus)
    corpus_dir = output_dir / config.run_id / corpus
    if resume is None:
        if corpus_dir.exists() and any(corpus_dir.iterdir()):
            raise ValueError(f"Fresh run refuses non-empty run directory: {corpus_dir}")
        corpus_dir.mkdir(parents=True, exist_ok=True)
    elif not corpus_dir.is_dir():
        raise ValueError(f"Resume requires the existing configured run directory: {corpus_dir}")
    effective_config_path = corpus_dir / "effective_config.json"
    effective_config = _effective_config_record(config)
    if resume is None:
        _atomic_json(effective_config_path, effective_config)
    elif not effective_config_path.is_file() or json.loads(
        effective_config_path.read_text(encoding="utf-8")
    ) != effective_config:
        raise ValueError("Resume requires the exact effective configuration from the original run")
    vocabulary_path = corpus_dir / "vocabulary.json"
    candidate_vocabulary = inputs.vocabulary
    if resume is None:
        candidate_vocabulary.save(vocabulary_path)
    elif not vocabulary_path.is_file():
        raise ValueError(f"Resume run is missing its immutable vocabulary: {vocabulary_path}")
    vocabulary_sha256 = hashlib.sha256(vocabulary_path.read_bytes()).hexdigest()
    if resume is not None:
        candidate_payload = json.dumps(
            candidate_vocabulary.to_metadata(), ensure_ascii=False, sort_keys=True, indent=2
        ) + "\n"
        candidate_sha256 = hashlib.sha256(candidate_payload.encode("utf-8")).hexdigest()
        if candidate_sha256 != vocabulary_sha256:
            raise ValueError("Resume vocabulary differs from the configured immutable run vocabulary")
    identity = CheckpointIdentity(
        config_digest=config.config_digest,
        linked_manifest_sha256=preflight.linked_manifest_sha256,
        control_manifest_sha256=preflight.control_manifest_sha256,
        vocabulary_sha256=vocabulary_sha256,
        corpus=corpus,
    )
    trace_path = corpus_dir / "loss_trace.json"
    run_manifest_path = corpus_dir / "run_manifest.json"
    evaluations: list[dict[str, object]] = []
    trace: list[dict[str, object]] = []
    best_validation: float | None = None
    if resume is not None:
        if Path(resume).resolve().parent != corpus_dir.resolve():
            raise ValueError("Resume checkpoint must belong to the configured corpus run directory")
        if not run_manifest_path.is_file():
            raise ValueError(f"Resume run is missing its run manifest: {run_manifest_path}")
        existing_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
        if (
            existing_manifest.get("run_id") != config.run_id
            or existing_manifest.get("config_digest") != config.config_digest
            or existing_manifest.get("identity") != asdict(identity)
        ):
            raise ValueError("Resume run manifest is incompatible with the requested config or identity")
        trace = _read_json_list(trace_path)
        existing_evaluations = existing_manifest.get("evaluations", [])
        if not isinstance(existing_evaluations, list):
            raise ValueError("Resume run manifest has malformed evaluations")
        evaluations = list(existing_evaluations)
        validation_values = [float(item["validation_total"]) for item in evaluations]
        best_validation = min(validation_values) if validation_values else None
    state = make_training_state(
        d_model=inputs.model_config.d_model,
        num_heads=inputs.model_config.num_heads,
        num_layers=inputs.model_config.num_layers,
        dropout=inputs.model_config.dropout,
        max_length=inputs.model_config.max_length,
        value_vocab_size=inputs.model_config.value_vocab_size,
        key_vocab_size=inputs.model_config.key_vocab_size,
        modality_vocab_size=inputs.model_config.modality_vocab_size,
        source_vocab_size=inputs.model_config.source_vocab_size,
        time_gap_vocab_size=inputs.model_config.time_gap_vocab_size,
        num_user_buckets=inputs.model_config.num_user_buckets,
        categorical_pair_buckets=inputs.model_config.categorical_pair_buckets,
        routing=_routing_from_vocab(inputs.vocabulary),
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        objective_weights=config.objectives,
        masking_probability=config.masking_probability,
        device=device,
    )
    requested_steps = max_steps if max_steps is not None else config.training.max_steps
    if requested_steps <= 0:
        raise ValueError("max_steps must be positive")
    resumed_step = 0
    if resume is not None:
        resumed_step = resume_training_state(state, resume, expected=identity, map_location=device)
        if resumed_step > requested_steps:
            raise ValueError("resume checkpoint step exceeds requested max_steps")
        if len(trace) != resumed_step:
            raise ValueError("Resume checkpoint step must equal the persisted loss trace length")
    for step, sample_batch in repeating_training_batches(
        inputs.train,
        batch_size=config.training.batch_size,
        start_step=resumed_step,
        max_steps=requested_steps,
    ):
        loss = train_one_step(state, sample_batch, global_step=step)
        completed_step = step + 1
        trace.append(
            {
                "step": completed_step,
                "total": float(loss.total.item()),
                    "next_action": float(getattr(loss, "next_action", torch.tensor(0.0)).item()),
                    "next_semantic": float(getattr(loss, "next_semantic", torch.tensor(0.0)).item()),
                "counts": dict(loss.counts),
            }
        )
        _atomic_json(trace_path, trace)
        evaluate_now = (
            completed_step % config.training.eval_every_steps == 0
            or completed_step == requested_steps
        )
        improved = False
        if evaluate_now:
            validation = evaluate_split(
                state,
                inputs.validation,
                batch_size=config.training.batch_size,
                max_batches=config.training.validation_max_batches,
            )
            validation_total = float(validation["total"])
            evaluations.append(
                {
                    "step": completed_step,
                    "validation_total": validation_total,
                    "validation_next_action": float(validation.get("next_action", 0.0)),
                    "validation_next_semantic": float(validation.get("next_semantic", 0.0)),
                }
            )
            improved = best_validation is None or validation_total < best_validation
            if improved:
                best_validation = validation_total
                save_checkpoint(
                    state, kind="best", output_dir=corpus_dir, identity=identity, global_step=completed_step
                )
        if (
            completed_step % config.training.checkpoint_every_steps == 0
            or completed_step == requested_steps
        ):
            save_checkpoint(
                state, kind="last", output_dir=corpus_dir, identity=identity, global_step=completed_step
            )
        _atomic_json(
            run_manifest_path,
            {
                "schema_version": "green_sm_pretraining_run_v2",
                "run_id": config.run_id,
                "config_digest": config.config_digest,
                "corpus": corpus,
                "identity": asdict(identity),
                "vocabulary": vocabulary_path.name,
                "vocabulary_sha256": vocabulary_sha256,
                "loss_trace": trace_path.name,
                "evaluations": evaluations,
                "selected_checkpoint_manifest": "checkpoint_manifest.json",
                "device": str(state.device),
                "effective_step_count": completed_step,
                "reproducibility": {
                    "seed": config.seed,
                    "cuda_available": torch.cuda.is_available(),
                    "python": sys.version.split()[0],
                    "torch": torch.__version__,
                    "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                    "cudnn_benchmark": torch.backends.cudnn.benchmark,
                    "cudnn_deterministic": torch.backends.cudnn.deterministic,
                },
            },
        )
    if len(trace) != requested_steps:
        raise ValueError(
            f"{corpus} train stream ended after {resumed_step + len(trace)} steps; "
            f"expected {requested_steps}"
        )
    checkpoint = corpus_dir / "last.pt"
    return {
        "corpus": corpus,
        "steps": requested_steps,
        "resumed_from_step": resumed_step,
        "checkpoint": str(checkpoint),
        "vocabulary_sha256": vocabulary_sha256,
        "final_loss": trace[-1]["total"],
    }
