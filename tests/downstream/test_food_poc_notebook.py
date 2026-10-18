from __future__ import annotations

import json
from pathlib import Path

import yaml


def _notebook_code(path: Path) -> str:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    return "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])


def test_food_poc_notebook_uses_shallow_input_discovery_and_poc_config() -> None:
    code = _notebook_code(Path("notebooks/food_recommendation_kaggle.ipynb"))

    assert ".rglob(" not in code
    assert "INPUT_ROOT.iterdir()" in code
    assert "green_sm_food_poc20_v1" in code
    assert "food_recommendation_poc20.yaml" in code
    assert "feat/food-poc-accelerated" in code
    assert "'epochs': 6" in code


def test_food_poc_config_limits_training_without_changing_full_data_identity() -> None:
    config = yaml.safe_load(Path("configs/food_recommendation_poc20.yaml").read_text(encoding="utf-8"))

    assert config["data"]["data_version"] == "green_sm_sim_full_v1"
    assert config["training"]["epochs"] == 6
    assert config["data"]["foundation_corpus"] == "linked"
