from __future__ import annotations

import argparse

from latent_harness.training.trainer import run_training_from_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run latent reasoning training")
    parser.add_argument("--config", required=True, help="Path to training YAML config")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run_training_from_config(args.config)


if __name__ == "__main__":
    main()
