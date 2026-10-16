"""Streaming training primitives for the Green SM Foundation model."""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from itertools import chain, islice
from pathlib import Path
from typing import Literal, Mapping

import torch

from src.foundation_input.dataset import DatasetConfig, FoundationPretrainDataset, foundation_collate
from src.foundation_input.loader import iter_normalized_samples
from src.foundation_input.tokenizer import TokenizerConfig, semantic_tokens
from src.foundation_input.vocab import VocabularyArtifacts, VocabularyConfig, fit_vocabulary
from src.models.encoder import FoundationModel, FoundationModelConfig
from src.models.objectives import LossBreakdown, PretrainingHeads, UserEmbeddingFusion, compute_pretraining_loss

from .config import ObjectiveWeights, PretrainingConfig, RoutingIds
from .gates import PreflightResult, validate_v3_batch_contract
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
    tokenizer = TokenizerConfig()
    samples = chain((first,), train_samples)
    vocabulary = fit_vocabulary(
        (token for sample in samples for token in semantic_tokens(sample, tokenizer)),
        VocabularyConfig(dataset_name=first.dataset_name),
        numeric_field_registry=tokenizer.numeric_fields,
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


@torch.no_grad()
def evaluate_split(state: TrainingState, samples: Iterable, *, max_batches: int | None = None) -> Mapping[str, float]:
    """Evaluate masked/causal losses without mutating parameters or optimizer state."""

    state.model.eval()
    state.fusion.eval()
    state.heads.eval()
    totals: dict[str, float] = {"total": 0.0}
    seen = 0
    for index, sample_batch in enumerate(batches(samples, batch_size=1)):
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

    inputs = build_corpus_run_inputs(config, preflight, corpus=corpus)
    corpus_dir = output_dir / corpus
    corpus_dir.mkdir(parents=True, exist_ok=True)
    vocabulary_path = corpus_dir / "vocabulary.json"
    inputs.vocabulary.save(vocabulary_path)
    vocabulary_sha256 = hashlib.sha256(vocabulary_path.read_bytes()).hexdigest()
    identity = CheckpointIdentity(
        config_digest=config.config_digest,
        linked_manifest_sha256=preflight.linked_manifest_sha256,
        control_manifest_sha256=preflight.control_manifest_sha256,
        vocabulary_sha256=vocabulary_sha256,
        corpus=corpus,
    )
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
    trace: list[dict[str, object]] = []
    for step, sample_batch in enumerate(batches(inputs.train, batch_size=config.training.batch_size)):
        if step < resumed_step:
            continue
        if step >= requested_steps:
            break
        loss = train_one_step(state, sample_batch, global_step=step)
        trace.append({"step": step, "total": float(loss.total.item()), "counts": dict(loss.counts)})
    if len(trace) != requested_steps - resumed_step:
        raise ValueError(
            f"{corpus} train stream ended after {resumed_step + len(trace)} steps; "
            f"expected {requested_steps}"
        )
    trace_path = corpus_dir / "loss_trace.json"
    trace_path.write_text(json.dumps(trace, indent=2) + "\n", encoding="utf-8")
    checkpoint = save_checkpoint(
        state, kind="last", output_dir=corpus_dir, identity=identity, global_step=requested_steps
    )
    return {
        "corpus": corpus,
        "steps": requested_steps,
        "resumed_from_step": resumed_step,
        "checkpoint": str(checkpoint),
        "vocabulary_sha256": vocabulary_sha256,
        "final_loss": trace[-1]["total"],
    }
