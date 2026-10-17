"""Command-line entry point for Green SM downstream evaluation."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence

from .artifacts import EvaluationArtifactError
from .compare import (
    EvaluationComparisonError,
    compare_reports,
    load_evaluation_report,
    write_comparison,
)
from .runner import evaluate_prediction_directory, write_evaluation_report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate and compare Green SM downstream predictions")
    commands = parser.add_subparsers(dest="command", required=True)
    evaluate = commands.add_parser("evaluate", help="evaluate one verified prediction directory")
    evaluate.add_argument("--task", required=True, choices=("food_recommendation", "ad_optimization"))
    evaluate.add_argument("--split", required=True, choices=("validation", "test"))
    evaluate.add_argument("--prediction-dir", required=True)
    evaluate.add_argument("--opportunities-root", required=True)
    evaluate.add_argument("--output", required=True)
    compare = commands.add_parser("compare", help="compare compatible evaluator reports")
    compare.add_argument("--report", required=True, action="append")
    compare.add_argument("--output", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run CLI command and return a process-style status code."""

    args = _parser().parse_args(argv)
    try:
        if args.command == "evaluate":
            report = evaluate_prediction_directory(
                task=args.task,
                split=args.split,
                prediction_dir=args.prediction_dir,
                opportunities_root=args.opportunities_root,
            )
            output = write_evaluation_report(report, args.output)
            print(json.dumps({"status": "evaluation_complete", "report": str(output)}))
            return 0
        reports = [load_evaluation_report(path) for path in args.report]
        output = write_comparison(compare_reports(reports), args.output)
        print(json.dumps({"status": "comparison_complete", "comparison": str(output)}))
        return 0
    except (EvaluationArtifactError, EvaluationComparisonError, FileExistsError) as error:
        print(f"Evaluation error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
