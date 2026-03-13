from __future__ import annotations

import argparse

from latent_harness.evaluation.runner import run_evaluation_from_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run latent reasoning evaluation")
    parser.add_argument("--config", required=True, help="Path to evaluation YAML config")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run_evaluation_from_config(args.config)


if __name__ == "__main__":
    main()
