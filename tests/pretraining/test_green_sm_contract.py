from __future__ import annotations

import json
from pathlib import Path

from src.pretraining.green_sm_contract import load_green_sm_tokenizer_config


def test_load_green_sm_tokenizer_config_builds_the_published_contract(tmp_path: Path) -> None:
    root = tmp_path / "foundation"
    root.mkdir()
    (root / "preprocessing_config.json").write_text(
        json.dumps(
            {
                "schema_version": "omnibehavior_pretraining_v2",
                "foundation_batch_contract_version": 3,
                "tokenizer_contract": {
                    "profile_allowlist": ["home_location"],
                    "context_allowlist": {"Food": ["merchant_location", "food_category"]},
                    "geo_field_registry": ["profile.home_location", "context.merchant_location"],
                    "geo_catalogue": {
                        "cell_00": {"lat": 21.0, "lon": 105.0},
                        "cell_01": {"lat": 21.1, "lon": 105.1},
                    },
                    "geo_catalogue_fingerprint": "52d48821ee812e465fc6981feb68f1d7c66f04edd0c88f54ff0e7424a8adac83",
                    "coordinate_tolerance": 1e-6,
                },
            }
        ),
        encoding="utf-8",
    )

    config = load_green_sm_tokenizer_config(root)

    assert config.profile_allowlist == frozenset({"home_location"})
    assert config.context_allowlist["Food"] == ("merchant_location", "food_category")
    assert config.geo_catalogue["cell_01"] == (21.1, 105.1)
