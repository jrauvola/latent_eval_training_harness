#!/usr/bin/env python3
"""
Run evaluation with the model list split into N parallel jobs (same GPU, different processes).

Usage (from latent_eval_training_harness repo root):

  uv run python scripts/run_parallel_eval.py --config configs/evaluation/broader_suite_gh200.yaml --splits 2
  uv run python scripts/run_parallel_eval.py --config configs/evaluation/broader_suite_gh200.yaml --splits 2 --dry-run

Generated configs are written next to the original with _part0, _part1, ... and corresponding output_dir.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="Split eval config by models and run N jobs in parallel")
    parser.add_argument("--config", type=str, required=True, help="Path to evaluation config YAML")
    parser.add_argument("--splits", type=int, default=2, help="Number of parallel jobs (default 2)")
    parser.add_argument(
        "--output-suffix",
        type=str,
        default="parallel",
        help="Suffix for output_dir and generated config filenames (default: parallel)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only write split configs and print commands, do not run",
    )
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    if not config_path.exists():
        print(f"Error: config not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    with config_path.open("r", encoding="utf-8") as f:
        payload = yaml.safe_load(f) or {}

    models = payload.get("models") or []
    if len(models) < args.splits:
        print(
            f"Warning: only {len(models)} models, splitting into {args.splits} parts will leave some parts empty.",
            file=sys.stderr,
        )

    # Split models into N chunks
    n = args.splits
    chunk_size = (len(models) + n - 1) // n
    chunks = [models[i : i + chunk_size] for i in range(0, len(models), chunk_size)]
    # Pad with empty lists so we have exactly n chunks
    while len(chunks) < n:
        chunks.append([])

    base_output_dir = str(payload.get("runtime", {}).get("output_dir", "artifacts/eval/out")).rstrip("/")
    if not base_output_dir.endswith(args.output_suffix):
        base_output_dir = f"{base_output_dir}_{args.output_suffix}"

    config_dir = config_path.parent
    config_stem = config_path.stem
    generated_configs: list[Path] = []

    for i, chunk in enumerate(chunks):
        if not chunk:
            print(f"Part {i}: no models, skipping.", file=sys.stderr)
            continue
        part_payload = dict(payload)
        part_payload["models"] = chunk
        if "runtime" not in part_payload:
            part_payload["runtime"] = {}
        part_payload["runtime"] = dict(part_payload["runtime"])
        part_payload["runtime"]["output_dir"] = f"{base_output_dir}_part{i}"
        part_config_path = config_dir / f"{config_stem}_{args.output_suffix}_part{i}.yaml"
        with part_config_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(part_payload, f, sort_keys=False, default_flow_style=False)
        generated_configs.append(part_config_path)
        print(f"Wrote {part_config_path} (models: {[m.get('name') for m in chunk]})")

    harness_root = Path(__file__).resolve().parent.parent

    if args.dry_run:
        print("\nDry run. To run in parallel, execute in separate terminals (from repo root):")
        for p in generated_configs:
            try:
                rel = p.relative_to(harness_root)
            except ValueError:
                rel = p
            print(f"  uv run python -c \"from latent_harness.evaluation.runner import run_evaluation_from_config; run_evaluation_from_config('{rel}')\"")
        return

    # Run each part in a subprocess (all in parallel)
    procs = []
    for p in generated_configs:
        try:
            rel = p.relative_to(harness_root)
        except ValueError:
            rel = p
        cmd = [
            sys.executable,
            "-c",
            f"from latent_harness.evaluation.runner import run_evaluation_from_config; run_evaluation_from_config('{rel}')",
        ]
        print(f"Starting: {rel}")
        proc = subprocess.Popen(
            cmd,
            cwd=str(harness_root),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        procs.append((proc, str(rel)))

    for proc, name in procs:
        proc.wait()
        if proc.returncode != 0:
            err = proc.stderr.read().decode() if proc.stderr else ""
            print(f"Process {name} exited with {proc.returncode}: {err}", file=sys.stderr)
        else:
            print(f"Finished: {name}")


if __name__ == "__main__":
    main()
