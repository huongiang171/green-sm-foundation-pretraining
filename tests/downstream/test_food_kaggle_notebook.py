from __future__ import annotations

import json
from pathlib import Path


def test_full_food_notebook_uses_bounded_input_discovery_and_prints_progress() -> None:
    notebook = json.loads(Path("notebooks/food_recommendation_kaggle.ipynb").read_text(encoding="utf-8"))
    code = "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )

    assert ".rglob(" not in code
    assert "kaggle_dataset_roots" in code
    assert "Food handoff validation passed" in code
    assert "Food training starts now" in code
