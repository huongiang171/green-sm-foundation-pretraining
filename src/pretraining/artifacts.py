"""Auditable, identity-checked Foundation checkpoint artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import torch
import numpy as np
from torch import nn

from src.foundation_input.types import FoundationBatch
from src.models.encoder import FoundationModel, FoundationModelConfig
from src.models.objectives import UserEmbeddingFusion

from .runner import TrainingState


@dataclass(frozen=True)
class CheckpointIdentity:
    config_digest: str
    linked_manifest_sha256: str
    control_manifest_sha256: str
    vocabulary_sha256: str
    corpus: Literal["linked", "control"]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _capture_rng_state() -> dict[str, object]:
    """Capture every random source that can change a resumed training trajectory."""

    return {
        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_states": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "torch_deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
    }


def _restore_rng_state(payload: dict) -> None:
    try:
        random.setstate(payload["python_rng_state"])
        np.random.set_state(payload["numpy_rng_state"])
        torch.set_rng_state(payload["torch_rng_state"])
        if torch.cuda.is_available() and payload.get("cuda_rng_states") is not None:
            torch.cuda.set_rng_state_all(payload["cuda_rng_states"])
        torch.use_deterministic_algorithms(bool(payload["torch_deterministic_algorithms"]))
        torch.backends.cudnn.benchmark = bool(payload["cudnn_benchmark"])
        torch.backends.cudnn.deterministic = bool(payload["cudnn_deterministic"])
    except (KeyError, TypeError, RuntimeError, ValueError) as error:
        raise ValueError(f"Checkpoint RNG payload is invalid: {error}") from error


class FrozenFoundationEmbedder(nn.Module):
    """The reusable, frozen embedding component consumed by downstream tasks."""

    def __init__(self, model: FoundationModel, fusion: UserEmbeddingFusion) -> None:
        super().__init__()
        self.model = model
        self.fusion = fusion
        self.eval()
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def embed_users(self, batch: FoundationBatch) -> dict[str, torch.Tensor]:
        output = self.model.encode(batch)
        return {
            "long_term": output.long_term,
            "short_term": output.short_term,
            "z_user": self.fusion(output.long_term, output.short_term),
        }


def save_checkpoint(
    state: TrainingState,
    *,
    kind: Literal["last", "best"],
    output_dir: str | Path,
    identity: CheckpointIdentity,
    global_step: int,
) -> Path:
    """Persist all state needed to resume training and audit downstream reuse."""

    if global_step < 0:
        raise ValueError("global_step must be non-negative")
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{kind}.pt"
    temporary = path.with_suffix(".pt.tmp")
    payload = {
        "identity": asdict(identity),
        "global_step": global_step,
        "model_config": asdict(state.model.config),
        "foundation_state_dict": state.model.state_dict(),
        "fusion_state_dict": state.fusion.state_dict(),
        "heads_state_dict": state.heads.state_dict(),
        "optimizer_state_dict": state.optimizer.state_dict(),
        "rng_state": _capture_rng_state(),
    }
    torch.save(payload, temporary)
    os.replace(temporary, path)
    manifest = {
        "schema_version": "green_sm_foundation_checkpoint_v1",
        "checkpoint_format_version": 1,
        "checkpoint": path.name,
        "checkpoint_sha256": _sha256(path),
        "kind": kind,
        "global_step": global_step,
        **asdict(identity),
        "model": asdict(state.model.config),
    }
    _atomic_json(directory / f"{kind}_manifest.json", manifest)
    selected_kind = "best" if (directory / "best_manifest.json").is_file() else "last"
    selected = json.loads((directory / f"{selected_kind}_manifest.json").read_text(encoding="utf-8"))
    selected["selected_kind"] = selected_kind
    _atomic_json(directory / "checkpoint_manifest.json", selected)
    return path


def load_frozen_embedder(
    checkpoint_path: str | Path,
    *,
    expected: CheckpointIdentity,
    map_location: str | torch.device = "cpu",
) -> FrozenFoundationEmbedder:
    """Load only an exact known Foundation checkpoint and freeze its parameters."""

    payload = _load_payload(checkpoint_path, expected=expected, map_location=map_location)
    try:
        config = FoundationModelConfig(**payload["model_config"])
        model = FoundationModel(config)
        fusion = UserEmbeddingFusion(d_model=config.d_model)
        model.load_state_dict(payload["foundation_state_dict"])
        fusion.load_state_dict(payload["fusion_state_dict"])
    except (KeyError, RuntimeError, TypeError) as error:
        raise ValueError(f"Checkpoint model payload is invalid: {error}") from error
    return FrozenFoundationEmbedder(model, fusion)


def _load_payload(
    checkpoint_path: str | Path,
    *,
    expected: CheckpointIdentity,
    map_location: str | torch.device,
) -> dict:
    path = Path(checkpoint_path)
    try:
        payload = torch.load(path, map_location=map_location, weights_only=False)
    except (OSError, RuntimeError) as error:
        raise ValueError(f"Cannot load Foundation checkpoint {path}: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError("Checkpoint payload must be an object")
    actual = payload.get("identity")
    if not isinstance(actual, dict):
        raise ValueError("Checkpoint is missing identity metadata")
    for name, value in asdict(expected).items():
        if actual.get(name) != value:
            raise ValueError(f"Checkpoint {name} mismatch: expected {value!r}, got {actual.get(name)!r}")
    return payload


def resume_training_state(
    state: TrainingState,
    checkpoint_path: str | Path,
    *,
    expected: CheckpointIdentity,
    map_location: str | torch.device = "cpu",
) -> int:
    """Restore a compatible in-progress run, including optimizer and RNG state."""

    payload = _load_payload(checkpoint_path, expected=expected, map_location=map_location)
    try:
        state.model.load_state_dict(payload["foundation_state_dict"])
        state.fusion.load_state_dict(payload["fusion_state_dict"])
        state.heads.load_state_dict(payload["heads_state_dict"])
        state.optimizer.load_state_dict(payload["optimizer_state_dict"])
        _restore_rng_state(payload["rng_state"])
        step = int(payload["global_step"])
    except (KeyError, RuntimeError, TypeError) as error:
        raise ValueError(f"Checkpoint training payload is invalid: {error}") from error
    if step < 0:
        raise ValueError("Checkpoint global_step must be non-negative")
    return step
