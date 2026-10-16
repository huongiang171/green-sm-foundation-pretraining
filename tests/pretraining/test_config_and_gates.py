from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from src.foundation_input.types import FoundationBatch


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_foundation_root(
    parent: Path,
    *,
    generator_digest: str = "same-generator",
    world_fingerprint: str = "same-world",
    readiness: bool = True,
    geo_contract_version: int = 3,
    profile_field: str = "home_location",
) -> Path:
    root = parent / "foundation"
    root.mkdir(parents=True)
    config = root / "preprocessing_config.json"
    config.write_text(
        json.dumps(
            {
                "schema_version": "omnibehavior_pretraining_v2",
                "foundation_batch_contract_version": 3,
                "tokenizer_contract": {
                    "profile_allowlist": [profile_field],
                    "context_allowlist": {"Food": ["merchant_location"]},
                    "geo_field_registry": [f"profile.{profile_field}"],
                    "geo_catalogue": {"cell_00": {"lat": 21.0, "lon": 105.0}},
                    "geo_catalogue_fingerprint": "e02f9edfb6e81bf0dd61a77b5db5f5cafe9ddbf22b9423671939f236df0166d6",
                    "coordinate_tolerance": 1e-6,
                },
            }
        ),
        encoding="utf-8",
    )
    manifest = {
        "status": "complete",
        "schema_version": "omnibehavior_pretraining_v2",
        "config_sha256": _sha256(config),
        "shards": [],
    }
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    report_dir = parent / "reports"
    report_dir.mkdir()
    report_dir.joinpath("data_quality_report.json").write_text(
        json.dumps(
            {
                "readiness_passed": readiness,
                "data_geo_contract_version": geo_contract_version,
                "generator_config_digest": generator_digest,
                "target_world_fingerprint": world_fingerprint,
                "foundation_manifest_sha256": _sha256(manifest_path),
            }
        ),
        encoding="utf-8",
    )
    return root


def _config_payload(linked_root: Path, control_root: Path, output_dir: Path) -> dict[str, object]:
    return {
        "schema_version": "green_sm_pretraining_v1",
        "linked_root": str(linked_root),
        "control_root": str(control_root),
        "output_dir": str(output_dir),
        "seed": 42,
        "model": {
            "d_model": 16,
            "num_heads": 4,
            "num_layers": 2,
            "dropout": 0.0,
            "max_length": 256,
            "num_user_buckets": 17,
            "categorical_pair_buckets": 97,
        },
        "dataset": {"max_tokens": 256, "max_event_count": 30},
        "training": {
            "batch_size": 2,
            "max_steps": 2,
            "eval_every_steps": 1,
            "checkpoint_every_steps": 1,
            "learning_rate": 0.001,
            "weight_decay": 0.0,
            "shuffle_buffer_size": 8,
        },
        "masking": {"probability": 0.15},
        "objectives": {
            "categorical_weight": 1.0,
            "numeric_value_weight": 1.0,
            "numeric_missing_weight": 1.0,
            "geo_coordinate_weight": 1.0,
            "geo_missing_weight": 1.0,
            "next_action_weight": 1.0,
        },
    }


def _write_config(tmp_path: Path, **overrides: object) -> Path:
    linked = _write_foundation_root(tmp_path / "linked")
    control = _write_foundation_root(tmp_path / "control")
    payload = _config_payload(linked, control, tmp_path / "output")
    for section, values in overrides.items():
        if isinstance(values, dict):
            payload[section].update(values)  # type: ignore[index,union-attr]
        else:
            payload[section] = values
    path = tmp_path / "config.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _v2_batch() -> FoundationBatch:
    return SimpleNamespace(
        input_ids=torch.tensor([[1, 2]], dtype=torch.long),
        key_ids=torch.zeros((1, 2), dtype=torch.long),
        modality_ids=torch.zeros((1, 2), dtype=torch.long),
        source_ids=torch.zeros((1, 2), dtype=torch.long),
        position_ids=torch.zeros((1, 2), dtype=torch.long),
        time_gap_ids=torch.zeros((1, 2), dtype=torch.long),
        attention_mask=torch.ones((1, 2), dtype=torch.bool),
        numeric_values=torch.zeros((1, 2), dtype=torch.float32),
        numeric_missing_mask=torch.zeros((1, 2), dtype=torch.bool),
        user_ids=torch.tensor([1], dtype=torch.long),
        targets={},
    )


def test_config_rejects_dataset_and_model_length_mismatch(tmp_path: Path) -> None:
    """Removing the length guard must reject a truncating training configuration."""
    from src.pretraining.config import load_pretraining_config

    path = _write_config(tmp_path, model={"max_length": 128})

    with pytest.raises(ValueError, match="max_tokens.*max_length"):
        load_pretraining_config(path)


def test_preflight_rejects_paired_roots_with_different_target_world(tmp_path: Path) -> None:
    """A linked/control comparison must stop when its paired target worlds differ."""
    from src.pretraining.config import load_pretraining_config
    from src.pretraining.gates import preflight_green_sm

    linked = _write_foundation_root(tmp_path / "linked", world_fingerprint="world-a")
    control = _write_foundation_root(tmp_path / "control", world_fingerprint="world-b")
    path = tmp_path / "config.json"
    path.write_text(json.dumps(_config_payload(linked, control, tmp_path / "output")), encoding="utf-8")

    with pytest.raises(ValueError, match="target-world fingerprint"):
        preflight_green_sm(load_pretraining_config(path))


def test_preflight_rejects_paired_roots_with_different_tokenizer_contracts(tmp_path: Path) -> None:
    """Paired comparisons must share their published tokenizer/geo policy exactly."""
    from src.pretraining.config import load_pretraining_config
    from src.pretraining.gates import preflight_green_sm

    linked = _write_foundation_root(tmp_path / "linked", profile_field="home_location")
    control = _write_foundation_root(tmp_path / "control", profile_field="work_location")
    path = tmp_path / "config.json"
    path.write_text(json.dumps(_config_payload(linked, control, tmp_path / "output")), encoding="utf-8")

    with pytest.raises(ValueError, match="tokenizer contracts differ"):
        preflight_green_sm(load_pretraining_config(path))


def test_v3_contract_rejects_batch_without_geo_tensors() -> None:
    """Dropping v3 geo tensors must fail before a runner can silently omit geo loss."""
    from src.pretraining.config import RoutingIds
    from src.pretraining.gates import validate_v3_batch_contract

    with pytest.raises(RuntimeError, match="geo_values.*geo_missing_mask"):
        validate_v3_batch_contract(_v2_batch(), RoutingIds(sequence=1, numeric=2, geo=3))


def test_v3_contract_accepts_aligned_geo_tensors() -> None:
    """A valid v3 batch exposes finite [B, L, 2] coordinates and a bool missing mask."""
    from src.pretraining.config import RoutingIds
    from src.pretraining.gates import validate_v3_batch_contract

    batch = SimpleNamespace(
        input_ids=torch.tensor([[1, 5]], dtype=torch.long),
        modality_ids=torch.tensor([[0, 3]], dtype=torch.long),
        geo_values=torch.zeros((1, 2, 2), dtype=torch.float32),
        geo_missing_mask=torch.zeros((1, 2), dtype=torch.bool),
    )

    validate_v3_batch_contract(batch, RoutingIds(sequence=1, numeric=2, geo=3))


def test_v3_contract_rejects_geo_missing_flag_on_non_geo_position() -> None:
    """A padding/categorical token must not masquerade as a missing geo token."""
    from src.pretraining.config import RoutingIds
    from src.pretraining.gates import validate_v3_batch_contract

    batch = SimpleNamespace(
        input_ids=torch.tensor([[1, 5]], dtype=torch.long),
        modality_ids=torch.tensor([[0, 1]], dtype=torch.long),
        geo_values=torch.zeros((1, 2, 2), dtype=torch.float32),
        geo_missing_mask=torch.tensor([[False, True]], dtype=torch.bool),
    )

    with pytest.raises(ValueError, match="geo_missing_mask"):
        validate_v3_batch_contract(batch, RoutingIds(sequence=1, numeric=2, geo=3))
