from __future__ import annotations

import json
from pathlib import Path

from src.evaluation.__main__ import main

from .conftest import write_baseline_prediction_dir


def test_kaggle_evaluation_notebook_only_invokes_committed_cli() -> None:
    notebook_path = Path("notebooks/downstream_evaluation_kaggle.ipynb")
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    code = "\n".join("".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code")

    assert "-m', 'src.evaluation'" in code
    assert "evaluate" in code
    assert "compare" in code
    assert "/kaggle/working" in code
    assert "sklearn" not in code
    assert "roc_auc_score" not in code
    assert "average_precision_score" not in code


def test_cli_evaluates_then_compares_reports(food_opportunities_root: Path, tmp_path: Path, capsys) -> None:
    prediction_dir = write_baseline_prediction_dir(tmp_path, food_opportunities_root)
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    comparison = tmp_path / "comparison.json"

    assert main(
        [
            "evaluate", "--task", "food_recommendation", "--split", "test",
            "--prediction-dir", str(prediction_dir), "--opportunities-root", str(food_opportunities_root),
            "--output", str(first),
        ]
    ) == 0
    assert main(
        [
            "evaluate", "--task", "food_recommendation", "--split", "test",
            "--prediction-dir", str(prediction_dir), "--opportunities-root", str(food_opportunities_root),
            "--output", str(second),
        ]
    ) == 0
    assert main(["compare", "--report", str(first), "--report", str(second), "--output", str(comparison)]) == 0

    assert "comparison" in capsys.readouterr().out
    assert len(json.loads(comparison.read_text(encoding="utf-8"))["runs"]) == 2
