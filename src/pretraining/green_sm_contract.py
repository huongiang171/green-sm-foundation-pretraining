"""Load the immutable Green SM tokenizer contract published with a corpus."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from src.foundation_input.tokenizer import TokenizerConfig


def _catalogue_fingerprint(catalogue: dict[str, tuple[float, float]]) -> str:
    payload = {
        cell: {"lat": coordinates[0], "lon": coordinates[1]}
        for cell, coordinates in sorted(catalogue.items())
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_green_sm_tokenizer_config(root: str | Path) -> TokenizerConfig:
    """Return the exact v3 tokenizer policy declared by one Foundation root."""

    path = Path(root) / "preprocessing_config.json"
    try:
        payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot read Green SM preprocessing contract: {path}") from error
    if payload.get("schema_version") != "omnibehavior_pretraining_v2":
        raise ValueError("Green SM preprocessing contract has an unsupported storage schema")
    if payload.get("foundation_batch_contract_version") != 3:
        raise ValueError("Green SM preprocessing contract must declare batch contract v3")
    contract = payload.get("tokenizer_contract")
    if not isinstance(contract, dict):
        raise ValueError("Green SM preprocessing contract lacks tokenizer_contract")
    try:
        profile = frozenset(str(value) for value in contract["profile_allowlist"])
        context = {
            str(scenario): tuple(str(value) for value in values)
            for scenario, values in dict(contract["context_allowlist"]).items()
        }
        fields = frozenset(str(value) for value in contract["geo_field_registry"])
        catalogue = {
            str(cell): (float(value["lat"]), float(value["lon"]))
            for cell, value in dict(contract["geo_catalogue"]).items()
        }
        tolerance = float(contract["coordinate_tolerance"])
        expected_fingerprint = str(contract["geo_catalogue_fingerprint"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Green SM tokenizer contract is malformed") from error
    if not profile or not context or not fields or not catalogue or tolerance < 0:
        raise ValueError("Green SM tokenizer contract has empty or invalid required fields")
    if _catalogue_fingerprint(catalogue) != expected_fingerprint:
        raise ValueError("Green SM tokenizer contract geo catalogue fingerprint mismatch")
    return TokenizerConfig(
        profile_allowlist=profile,
        context_allowlist=context,
        geo_field_registry=fields,
        geo_catalogue=catalogue,
        coordinate_tolerance=tolerance,
    )
