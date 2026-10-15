"""Command-line entry point for reproducible Green SM Foundation pretraining."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import torch

from .config import load_pretraining_config
from .gates import preflight_green_sm
from .runner import run_corpus


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Pretrain the Green SM user Foundation model")
    parser.add_argument("--config", required=True, type=Path, help="strict pretraining JSON recipe")
    parser.add_argument("--output-dir", type=Path, help="external run artifact directory")
    parser.add_argument("--device", default="cpu", help="PyTorch device, e.g. cuda")
    parser.add_argument("--max-steps", type=int, help="pilot override; recorded in artifacts")
    parser.add_argument("--corpus", choices=("linked", "control", "both"), default="both")
    parser.add_argument("--resume", type=Path, help="reserved for a checkpoint-resume run")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    config = load_pretraining_config(args.config)
    if args.output_dir is not None:
        config = replace(config, output_dir=args.output_dir)
    preflight = preflight_green_sm(config)
    corpora = ("linked", "control") if args.corpus == "both" else (args.corpus,)
    if args.resume is not None and len(corpora) != 1:
        raise ValueError("--resume requires exactly one --corpus")
    results = [
        run_corpus(
            config,
            preflight,
            corpus=corpus,
            output_dir=config.output_dir,
            device=args.device,
            max_steps=args.max_steps,
            resume=args.resume,
        )
        for corpus in corpora
    ]
    print(json.dumps({"config_digest": config.config_digest, "runs": results}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
