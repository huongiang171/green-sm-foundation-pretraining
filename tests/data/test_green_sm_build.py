from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from src.data.green_sm.build import build_dataset
from src.data.green_sm.config import GreenSmConfig
from src.data.green_sm.world import build_target_world
from src.foundation_input.loader import iter_normalized_samples


def _pilot_config(tmp_path: Path) -> GreenSmConfig:
    return replace(GreenSmConfig.from_json("configs/green_sm_sim_pilot.json"), output_root=tmp_path / "out")


def test_pilot_build_loads_every_split_and_preserves_paired_requests(tmp_path: Path) -> None:
    result = build_dataset(_pilot_config(tmp_path))

    for root in result["foundation_roots"].values():
        assert list(iter_normalized_samples(root, "train"))
        assert list(iter_normalized_samples(root, "validation"))
        assert list(iter_normalized_samples(root, "test"))
    assert result["paired_logical_rows_equal"] is True
    assert result["max_estimated_token_count"] <= 256


def test_build_refuses_to_overwrite_immutable_output_root(tmp_path: Path) -> None:
    config = _pilot_config(tmp_path)
    config.output_root.mkdir(parents=True)
    (config.output_root / "existing.txt").write_text("immutable", encoding="utf-8")

    with pytest.raises(ValueError, match="non-empty output root"):
        build_dataset(config)


def test_pilot_snapshot_split_keeps_every_cold_user_out_of_train_and_in_test(tmp_path: Path) -> None:
    config = _pilot_config(tmp_path)
    result = build_dataset(config)
    cold_users = build_target_world(config).cold_start_user_ids
    root = next(iter(result["foundation_roots"].values()))

    train_users = {sample.user_id for sample in iter_normalized_samples(root, "train")}
    test_users = {sample.user_id for sample in iter_normalized_samples(root, "test")}

    assert not cold_users & train_users
    assert cold_users <= test_users


def test_pilot_cli_builds_and_writes_readiness_report(tmp_path: Path) -> None:
    config_payload = json.loads(Path("configs/green_sm_sim_pilot.json").read_text(encoding="utf-8"))
    config_payload["output_root"] = str(tmp_path / "out")
    config_path = tmp_path / "pilot.json"
    config_path.write_text(json.dumps(config_payload), encoding="utf-8")

    completed = subprocess.run(
        [sys.executable, "-m", "src.data.green_sm", "--config", str(config_path)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    report = tmp_path / "out" / "green_sm_sim_linked_v1" / "reports" / "data_quality_report.json"
    assert json.loads(report.read_text(encoding="utf-8"))["readiness_passed"] is True


def test_preprocessing_contract_publishes_geo_catalogue_and_green_sm_allowlists(tmp_path: Path) -> None:
    result = build_dataset(_pilot_config(tmp_path))
    preprocessing = json.loads(
        (Path(result["foundation_roots"]["linked"]) / "preprocessing_config.json").read_text(
            encoding="utf-8"
        )
    )

    contract = preprocessing["tokenizer_contract"]
    assert contract["profile_allowlist"] == [
        "home_location",
        "work_location",
        "budget_bucket",
        "preferred_service",
        "active_days_so_far",
    ]
    assert contract["context_allowlist"]["Ride"] == ["pickup_location", "dropoff_location"]
    assert contract["context_allowlist"]["Food"] == ["merchant_location", "food_category"]
    assert contract["coordinate_tolerance"] == pytest.approx(1e-6)
    assert len(contract["geo_catalogue"]) == 64
    assert set(contract["geo_catalogue"]["cell_00"]) == {"lat", "lon"}
