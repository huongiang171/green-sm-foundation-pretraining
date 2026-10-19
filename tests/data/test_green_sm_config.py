from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.data.green_sm.config import GreenSmConfig


def _write_pilot_config(root: Path, **overrides: object) -> Path:
    payload: dict[str, object] = {
        "data_version": "green_sm_sim_pilot_v1",
        "mode": "pilot",
        "seed": 7,
        "output_root": str(root / "out"),
        "pilot_fraction": 0.02,
        "user_count": 240,
        "timezone": "Asia/Ho_Chi_Minh",
        "total_days": 42,
        "split_days": {"train": 28, "validation": 7, "test": 7},
        "snapshot_counts": {"train": 1920, "validation": 480, "test": 480},
        "food_request_counts": {"train": 360, "validation": 60, "test": 60},
        "ads_request_counts": {"train": 480, "validation": 80, "test": 80},
        "max_history_events": 30,
        "max_tokens": 256,
        "thresholds": {
            "min_train_event_rows_per_event_type": 5000,
            "max_train_event_type_share": 0.45,
            "min_train_valid_next_action_rate": 0.75,
            "min_train_maskable_tokens": {
                "categorical": 100000,
                "numeric": 20000,
                "text": 10000,
                "geo": 50000,
            },
            "min_valid_geo_rate": 0.60,
            "max_non_user_oov_rate_validation_test": 0.01,
            "cold_start_test_user_rate": 0.15,
            "min_food_hard_negatives_per_request": 10,
            "food_positive_rate_range": [0.05, 0.20],
            "ads_click_rate_range": [0.02, 0.15],
            "min_ads_ranking_eligible_rate": 0.30,
            "max_control_auc_lift": 0.01,
        },
    }
    payload.update(overrides)
    path = root / "pilot.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_pilot_scales_counts_but_keeps_proportions(tmp_path: Path) -> None:
    config = GreenSmConfig.from_json(_write_pilot_config(tmp_path))

    assert config.user_count == 240
    assert config.snapshot_counts == {"train": 1920, "validation": 480, "test": 480}
    assert config.thresholds.min_train_event_rows_per_event_type == 100
    assert config.thresholds.min_ads_ranking_eligible_rate == 0.30
    assert config.thresholds.min_train_maskable_tokens["categorical"] == 2000


def test_config_rejects_invalid_pilot_fraction_and_split_total(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="pilot_fraction"):
        GreenSmConfig.from_json(_write_pilot_config(tmp_path, pilot_fraction=0.1))

    with pytest.raises(ValueError, match="split_days"):
        GreenSmConfig.from_json(
            _write_pilot_config(tmp_path, split_days={"train": 28, "validation": 6, "test": 7})
        )


def test_config_parses_an_existing_nonempty_output_root_without_authorizing_overwrite(tmp_path: Path) -> None:
    output = tmp_path / "occupied"
    output.mkdir()
    (output / "existing.txt").write_text("do not overwrite", encoding="utf-8")

    config = GreenSmConfig.from_json(_write_pilot_config(tmp_path, output_root=str(output)))

    assert config.output_root == output


def test_config_digest_changes_when_a_readiness_threshold_changes(tmp_path: Path) -> None:
    first_path = _write_pilot_config(tmp_path)
    payload = json.loads(first_path.read_text(encoding="utf-8"))
    payload["thresholds"]["min_valid_geo_rate"] = 0.70
    second_path = tmp_path / "changed-threshold.json"
    second_path.write_text(json.dumps(payload), encoding="utf-8")

    assert GreenSmConfig.from_json(first_path).config_digest != GreenSmConfig.from_json(second_path).config_digest


def test_shared_signal_poc_config_declares_immutable_policy_and_scale() -> None:
    config = GreenSmConfig.from_json("configs/green_sm_shared_signal_poc.json")

    assert config.data_version == "green_sm_shared_signal_poc_v1"
    assert config.behavior_policy == "shared_signal_poc_v1"
    assert dict(config.snapshot_counts) == {"train": 12000, "validation": 2000, "test": 2000}
    assert dict(config.food_request_counts) == {"train": 6000, "validation": 1000, "test": 1000}
    assert dict(config.ads_request_counts) == {"train": 6000, "validation": 1000, "test": 1000}
