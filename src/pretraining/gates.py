"""Fail-closed data and runtime gates for Green SM pretraining."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from src.foundation_input.manifest import load_manifest
from src.foundation_input.types import FoundationBatch
from src.models.encoder import FoundationModelConfig

from .config import PretrainingConfig, RoutingIds


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True)
class PreflightResult:
    linked_manifest_sha256: str
    control_manifest_sha256: str
    shared_generator_config_digest: str
    shared_target_world_fingerprint: str


def _report(root: Path) -> dict[str, Any]:
    path = root.parent / "reports" / "data_quality_report.json"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot read Green SM readiness report {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"Green SM readiness report must be an object: {path}")
    required = {
        "readiness_passed",
        "upstream_v3_geo_contract_ready",
        "generator_config_digest",
        "target_world_fingerprint",
        "foundation_manifest_sha256",
    }
    missing = sorted(required - value.keys())
    if missing:
        raise ValueError(f"Green SM readiness report missing fields {missing}: {path}")
    if value["readiness_passed"] is not True:
        raise ValueError(f"Green SM readiness report is not ready: {path}")
    if value["upstream_v3_geo_contract_ready"] is not True:
        raise RuntimeError(f"Green SM report declares v3 geo contract unavailable: {path}")
    return value


def preflight_green_sm(config: PretrainingConfig) -> PreflightResult:
    """Prove paired roots and current runtime can support a Green SM training run."""

    linked_manifest = load_manifest(config.linked_root, verify_checksums=False)
    control_manifest = load_manifest(config.control_root, verify_checksums=False)
    linked_sha = _sha256(linked_manifest.root / "manifest.json")
    control_sha = _sha256(control_manifest.root / "manifest.json")
    linked = _report(linked_manifest.root)
    control = _report(control_manifest.root)
    if linked["foundation_manifest_sha256"] != linked_sha:
        raise ValueError("linked readiness report has a different foundation manifest checksum")
    if control["foundation_manifest_sha256"] != control_sha:
        raise ValueError("control readiness report has a different foundation manifest checksum")
    if linked["generator_config_digest"] != control["generator_config_digest"]:
        raise ValueError("linked/control generator config digest differs")
    if linked["target_world_fingerprint"] != control["target_world_fingerprint"]:
        raise ValueError("linked/control target-world fingerprint differs")
    fields = FoundationBatch.__dataclass_fields__
    if "geo_values" not in fields or "geo_missing_mask" not in fields:
        raise RuntimeError(
            "Runtime FoundationBatch lacks geo_values and geo_missing_mask; "
            "merge the v3 input contract before pretraining"
        )
    if "geo_modality_id" not in FoundationModelConfig.__dataclass_fields__:
        raise RuntimeError(
            "Runtime FoundationModelConfig lacks geo_modality_id; "
            "merge the v3 adapter routing contract before pretraining"
        )
    return PreflightResult(
        linked_manifest_sha256=linked_sha,
        control_manifest_sha256=control_sha,
        shared_generator_config_digest=str(linked["generator_config_digest"]),
        shared_target_world_fingerprint=str(linked["target_world_fingerprint"]),
    )


def validate_v3_batch_contract(batch: FoundationBatch, routing: RoutingIds) -> None:
    """Reject missing or malformed v3 tensors before they enter a model."""

    del routing  # Routing is validated by the caller from the persisted vocabulary.
    if not hasattr(batch, "geo_values") or not hasattr(batch, "geo_missing_mask"):
        raise RuntimeError(
            "FoundationBatch requires geo_values and geo_missing_mask under the v3 contract"
        )
    if batch.input_ids.ndim != 2 or batch.input_ids.dtype != torch.long:
        raise ValueError("input_ids must be torch.long with shape [B, L]")
    expected = (*batch.input_ids.shape, 2)
    if batch.geo_values.shape != expected or batch.geo_values.dtype != torch.float32:
        raise ValueError(
            "geo_values must be torch.float32 with shape "
            f"{expected}, got shape={tuple(batch.geo_values.shape)}, dtype={batch.geo_values.dtype}"
        )
    if batch.geo_missing_mask.shape != batch.input_ids.shape or batch.geo_missing_mask.dtype != torch.bool:
        raise ValueError(
            "geo_missing_mask must be torch.bool with shape "
            f"{tuple(batch.input_ids.shape)}"
        )
    if not torch.isfinite(batch.geo_values).all():
        raise ValueError("geo_values must be finite")
