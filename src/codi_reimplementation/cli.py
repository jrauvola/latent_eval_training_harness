from __future__ import annotations

import argparse

from codi_reimplementation.eval.runner import run_evaluation_from_config
from codi_reimplementation.training.trainer import run_training_from_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CODI reimplementation CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    eval_parser = subparsers.add_parser("evaluate", help="Run benchmark evaluation")
    eval_parser.add_argument("--config", required=True, help="Path to evaluation YAML config")

    train_parser = subparsers.add_parser("train", help="Run CODI training")
    train_parser.add_argument("--config", required=True, help="Path to training YAML config")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "evaluate":
        run_evaluation_from_config(args.config)
        return
    if args.command == "train":
        run_training_from_config(args.config)
        return
    raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
