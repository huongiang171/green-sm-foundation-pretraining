from __future__ import annotations

import subprocess
import sys


def test_pretraining_cli_exposes_reproducible_options() -> None:
    """Removing the module entry point must make the documented Kaggle command unusable."""
    result = subprocess.run(
        [sys.executable, "-m", "src.pretraining", "--help"], capture_output=True, text=True, check=False
    )

    assert result.returncode == 0, result.stderr
    assert "--config" in result.stdout
    assert "--output-dir" in result.stdout
