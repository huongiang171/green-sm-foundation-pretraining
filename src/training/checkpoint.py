"""Verified public loader for frozen Foundation checkpoints."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

from src.pretraining.artifacts import (
    CheckpointIdentity,
    FrozenFoundationEmbedder,
    load_frozen_embedder,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_foundation_checkpoint(
    checkpoint_path: str | Path,
    manifest_path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> FrozenFoundationEmbedder:
    """Load a frozen model only after validating its selected artifact manifest."""

    checkpoint = Path(checkpoint_path)
    manifest_file = Path(manifest_path)
    try:
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot read Foundation checkpoint manifest {manifest_file}: {error}") from error
    if not isinstance(manifest, dict):
        raise TypeError("Foundation checkpoint manifest must be an object")
    if manifest.get("schema_version") != "green_sm_foundation_checkpoint_v1":
        raise ValueError("Foundation checkpoint manifest has unsupported schema_version")
    if manifest.get("checkpoint_format_version") != 1:
        raise ValueError("Foundation checkpoint manifest has unsupported format version")
    if manifest.get("checkpoint") != checkpoint.name:
        raise ValueError("Foundation checkpoint filename does not match its manifest")
    if not checkpoint.is_file():
        raise ValueError(f"Foundation checkpoint not found: {checkpoint}")
    if manifest.get("checkpoint_sha256") != _sha256(checkpoint):
        raise ValueError("Foundation checkpoint SHA-256 does not match its manifest")
    try:
        identity = CheckpointIdentity(
            config_digest=str(manifest["config_digest"]),
            linked_manifest_sha256=str(manifest["linked_manifest_sha256"]),
            control_manifest_sha256=str(manifest["control_manifest_sha256"]),
            vocabulary_sha256=str(manifest["vocabulary_sha256"]),
            corpus=str(manifest["corpus"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Foundation checkpoint manifest has invalid identity") from error
    return load_frozen_embedder(checkpoint, expected=identity, map_location=map_location)
