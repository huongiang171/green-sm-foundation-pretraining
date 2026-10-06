from src import __version__
from . import models

from . import baselines, data, downstream, evaluation


def test_package_skeleton_imports():
    assert __version__ == "0.1.0"
    assert all(module is not None for module in (baselines, data, downstream, evaluation, models))
