"""CLI entry point for immutable Green SM data construction."""

from __future__ import annotations

import argparse
import json

from .build import build_dataset
from .config import GreenSmConfig


def main() -> int:
    parser = argparse.ArgumentParser(description="Build and validate Green SM simulation artifacts.")
    parser.add_argument("--config", required=True, help="Path to a GreenSmConfig JSON file")
    args = parser.parse_args()
    config = GreenSmConfig.from_json(args.config)
    result = build_dataset(config)
    from .validate import validate_outputs

    result["validation"] = validate_outputs(config)
    print(json.dumps(result, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
