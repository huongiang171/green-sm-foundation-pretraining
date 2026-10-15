from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from src.data.green_sm.build import build_dataset
from src.data.green_sm.config import GreenSmConfig
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
