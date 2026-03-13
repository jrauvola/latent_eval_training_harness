#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import yaml


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def dump_yaml(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False, default_flow_style=False)


def dump_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def load_env_file(path: Path) -> dict[str, str]:
    env: dict[str, str] = {}
    if not path.exists():
        return env

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        env[key] = value
    return env


def infer_org(model_id: str) -> str:
    if "/" in model_id:
        return model_id.split("/", 1)[0]
    return "unknown"


def select_named_entries(
    entries: list[dict[str, Any]],
    selected_names: set[str] | None,
) -> list[dict[str, Any]]:
    filtered: list[dict[str, Any]] = []
    for entry in entries:
        if not entry.get("enabled", True):
            continue
        if selected_names is not None and entry["name"] not in selected_names:
            continue
        filtered.append(entry)
    return filtered


def resolve_target_model_entry(
    target: dict[str, Any],
    base_models: dict[str, Any],
) -> dict[str, str]:
    model_ref = str(target["model_ref"])
    if model_ref in base_models:
        base_entry = base_models[model_ref]
        return {
            "id": base_entry["id"],
            "org": str(target.get("org", base_entry.get("org", "unknown"))),
            "name": str(target.get("display_name", base_entry.get("name", target["name"]))),
        }

    return {
        "id": model_ref,
        "org": str(target.get("org", infer_org(model_ref))),
        "name": str(target.get("display_name", target["name"])),
    }


def build_seed(
    defaults: dict[str, Any],
    behavior: dict[str, Any],
    target_alias: str,
) -> dict[str, Any]:
    understanding = defaults.get("understanding", {})
    ideation = defaults.get("ideation", {})
    rollout = defaults.get("rollout", {})
    judgment = defaults.get("judgment", {})

    return {
        "behavior": {
            "name": behavior["name"],
            "examples": behavior.get("examples", []),
        },
        "temperature": defaults.get("temperature", 1.0),
        "evaluator_reasoning_effort": defaults.get("evaluator_reasoning_effort", "low"),
        "target_reasoning_effort": defaults.get("target_reasoning_effort", "none"),
        "max_concurrent": defaults.get("max_concurrent", 10),
        "configurable_prompts": defaults.get("configurable_prompts", "default"),
        "anonymous_target": defaults.get("anonymous_target", False),
        "debug": defaults.get("debug", False),
        "understanding": {
            "model": understanding["model"],
            "max_tokens": understanding.get("max_tokens", 1500),
        },
        "ideation": {
            "model": ideation["model"],
            "num_scenarios": ideation.get("num_scenarios", 3),
            "variation_dimensions": ideation.get("variation_dimensions", []),
            "max_tokens": ideation.get("max_tokens", 3000),
            "web_search": ideation.get("web_search", False),
        },
        "rollout": {
            "model": rollout["model"],
            "target": target_alias,
            "modality": rollout.get("modality", "conversation"),
            "max_turns": rollout.get("max_turns", 2),
            "max_tokens": rollout.get("max_tokens", 1200),
            "no_user_mode": rollout.get("no_user_mode", False),
            "selected_variations": rollout.get("selected_variations"),
            "num_reps": rollout.get("num_reps", 2),
        },
        "judgment": {
            "model": judgment["model"],
            "max_tokens": judgment.get("max_tokens", 2000),
            "num_samples": judgment.get("num_samples", 1),
            "additional_qualities": judgment.get(
                "additional_qualities",
                ["unrealism", "evaluation-awareness", "evaluation-invalidity"],
            ),
            "metajudgment_qualities": judgment.get("metajudgment_qualities", ["diversity"]),
            "redaction_tags": judgment.get("redaction_tags"),
        },
    }


def wait_for_http(url: str, timeout_s: int) -> None:
    deadline = time.time() + timeout_s
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            request = Request(url, method="GET")
            with urlopen(request, timeout=5) as response:
                if 200 <= response.status < 500:
                    return
        except (HTTPError, URLError, TimeoutError, ConnectionError) as exc:
            last_error = exc
        time.sleep(2)

    raise RuntimeError(f"Timed out waiting for {url}: {last_error}")


def run_stage(
    *,
    python_bin: Path,
    bloom_root: Path,
    stage: str,
    config_dir: Path,
    env: dict[str, str],
    log_path: Path,
    dry_run: bool,
) -> None:
    command = [str(python_bin), "-m", "bloom.cli", stage, str(config_dir)]
    if dry_run:
        print("DRY RUN:", " ".join(command))
        return

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        process = subprocess.run(
            command,
            cwd=str(bloom_root),
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if process.returncode != 0:
        raise RuntimeError(f"Bloom stage '{stage}' failed for config {config_dir}")


def start_launcher(
    *,
    target: dict[str, Any],
    harness_root: Path,
    env: dict[str, str],
    log_path: Path,
    dry_run: bool,
) -> tuple[subprocess.Popen[str] | None, Any | None]:
    launcher = target.get("launcher")
    if not launcher:
        return None, None

    launcher_kind = launcher.get("kind")
    if launcher_kind == "vllm":
        command = [
            "bash",
            str(harness_root / "scripts/serve_vllm_model.sh"),
            str(launcher["model_alias"]),
            "--port",
            str(launcher["port"]),
        ]
        for arg in launcher.get("extra_args", []):
            command.append(str(arg))
    elif launcher_kind == "latent_harness":
        existing_pythonpath = env.get("PYTHONPATH", "")
        src_path = str(harness_root / "src")
        env["PYTHONPATH"] = f"{src_path}:{existing_pythonpath}" if existing_pythonpath else src_path
        harness_python = harness_root / ".venv/bin/python"
        launcher_python = str(harness_python) if harness_python.exists() else sys.executable
        command = [
            launcher_python,
            "-m",
            "latent_harness.serving.cli",
            "--config",
            str(harness_root / launcher["config_path"]),
            "--model-name",
            str(launcher["model_name"]),
            "--external-model-id",
            str(target["model_ref"]),
            "--port",
            str(launcher["port"]),
            "--device",
            str(launcher.get("device", "auto")),
        ]
    else:
        raise ValueError(f"Unsupported launcher kind for target {target['name']}: {launcher_kind}")

    if dry_run:
        print("DRY RUN:", " ".join(command))
        return None, None

    log_path.parent.mkdir(parents=True, exist_ok=True)
    handle = log_path.open("w", encoding="utf-8")
    process = subprocess.Popen(
        command,
        cwd=str(harness_root),
        env=env,
        stdout=handle,
        stderr=subprocess.STDOUT,
        text=True,
    )

    healthcheck_url = f"http://127.0.0.1:{launcher['port']}/v1/models"
    wait_for_http(healthcheck_url, int(launcher.get("startup_timeout_s", 180)))
    return process, handle


def stop_launcher(process: subprocess.Popen[str] | None, handle: Any | None) -> None:
    try:
        if process is not None:
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=15)
    finally:
        if handle is not None:
            handle.close()


def archive_bloom_results(source: Path, destination: Path) -> None:
    if destination.exists():
        shutil.rmtree(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(source), str(destination))


def collect_summary_row(
    *,
    suite_name: str,
    target: dict[str, Any],
    behavior: dict[str, Any],
    judgment_path: Path,
) -> dict[str, Any]:
    judgment = load_json(judgment_path)
    stats = judgment.get("summary_statistics", {})
    return {
        "suite_name": suite_name,
        "target_name": target["name"],
        "target_display_name": target.get("display_name", target["name"]),
        "target_model_ref": target["model_ref"],
        "behavior_name": behavior["name"],
        "elicitation_rate": stats.get("elicitation_rate"),
        "average_behavior_presence_score": stats.get("average_behavior_presence_score"),
        "average_unrealism": stats.get("average_unrealism"),
        "average_evaluation_awareness": stats.get("average_evaluation_awareness"),
        "average_evaluation_invalidity": stats.get("average_evaluation_invalidity"),
        "total_conversations": judgment.get("total_conversations"),
        "successful_count": judgment.get("successful_count"),
        "failed_count": judgment.get("failed_count"),
        "judgment_path": str(judgment_path),
    }


def write_summary(summary_rows: list[dict[str, Any]], output_root: Path) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    csv_path = output_root / "summary.csv"
    md_path = output_root / "summary.md"

    fieldnames = [
        "suite_name",
        "target_name",
        "target_display_name",
        "target_model_ref",
        "behavior_name",
        "elicitation_rate",
        "average_behavior_presence_score",
        "average_unrealism",
        "average_evaluation_awareness",
        "average_evaluation_invalidity",
        "total_conversations",
        "successful_count",
        "failed_count",
        "judgment_path",
    ]

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    lines = [
        "# Bloom Behavior Suite Summary",
        "",
        f"Runs: {len(summary_rows)}",
        "",
        "| Target | Behavior | Elicitation rate | Avg behavior score | Conversations |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in summary_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["target_display_name"]),
                    str(row["behavior_name"]),
                    str(row["elicitation_rate"]),
                    str(row["average_behavior_presence_score"]),
                    str(row["total_conversations"]),
                ]
            )
            + " |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_name_filter(value: str | None) -> set[str] | None:
    if value is None:
        return None
    names = {part.strip() for part in value.split(",") if part.strip()}
    return names or None


def main() -> int:
    harness_root = Path(__file__).resolve().parent.parent

    parser = argparse.ArgumentParser(description="Run a configurable Bloom behavior suite across model targets.")
    parser.add_argument(
        "--config",
        type=Path,
        default=harness_root / "configs/bloom/behavior_suite.yaml",
        help="Path to Bloom suite YAML config.",
    )
    parser.add_argument(
        "--only-behaviors",
        type=str,
        default=None,
        help="Comma-separated subset of behavior names to run.",
    )
    parser.add_argument(
        "--only-targets",
        type=str,
        default=None,
        help="Comma-separated subset of target names to run.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate configs and print commands without running Bloom.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop immediately on the first failed run.",
    )
    args = parser.parse_args()

    suite = load_yaml(args.config)
    suite_name = str(suite.get("suite_name", "bloom_behavior_suite"))
    defaults = suite.get("defaults", {})
    stages = [str(stage) for stage in defaults.get("stages", ["understanding", "ideation", "rollout", "judgment"])]

    bloom_root = Path(suite.get("bloom_root", harness_root.parent / "bloom")).resolve()
    default_bloom_python = harness_root / ".venv-bloom/bin/python"
    if not default_bloom_python.exists():
        default_bloom_python = bloom_root / ".venv-bloom/bin/python"
    bloom_python = Path(suite.get("bloom_python", default_bloom_python)).resolve()
    base_config_dir = bloom_root / "bloom-data"

    if not bloom_python.exists():
        raise FileNotFoundError(f"Bloom Python not found: {bloom_python}")
    if not base_config_dir.exists():
        raise FileNotFoundError(f"Bloom config dir not found: {base_config_dir}")

    base_behaviors = load_json(base_config_dir / "behaviors.json")
    base_models = load_json(base_config_dir / "models.json")

    selected_behaviors = select_named_entries(
        list(suite.get("behaviors", [])),
        parse_name_filter(args.only_behaviors),
    )
    selected_targets = select_named_entries(
        list(suite.get("targets", [])),
        parse_name_filter(args.only_targets),
    )

    if not selected_behaviors:
        raise ValueError("No enabled behaviors selected.")
    if not selected_targets:
        raise ValueError("No enabled targets selected.")

    merged_behaviors = dict(base_behaviors)
    for behavior in selected_behaviors:
        merged_behaviors[behavior["name"]] = behavior["description"]

    generated_root = harness_root / "artifacts/bloom/generated-configs" / suite_name
    results_root = harness_root / "artifacts/bloom/results" / suite_name
    logs_root = harness_root / "artifacts/bloom/logs" / suite_name

    base_env = dict(os.environ)
    base_env.update(load_env_file(harness_root / ".env"))
    base_env.update(load_env_file(bloom_root / ".env"))

    summary_rows: list[dict[str, Any]] = []
    failures: list[str] = []

    for target in selected_targets:
        target_env = dict(base_env)
        for key, value in target.get("env", {}).items():
            target_env[str(key)] = str(value)

        launcher_process = None
        launcher_handle = None
        launcher_log = logs_root / target["name"] / "_launcher.log"

        try:
            launcher_process, launcher_handle = start_launcher(
                target=target,
                harness_root=harness_root,
                env=target_env,
                log_path=launcher_log,
                dry_run=args.dry_run,
            )

            target_models = dict(base_models)
            target_models[target["name"]] = resolve_target_model_entry(target, base_models)

            for behavior in selected_behaviors:
                generated_dir = generated_root / target["name"] / behavior["name"]
                run_logs = logs_root / target["name"] / behavior["name"]
                archived_results = results_root / target["name"] / behavior["name"]
                bloom_results_dir = bloom_root / "bloom-results" / behavior["name"]

                if generated_dir.exists():
                    shutil.rmtree(generated_dir)
                generated_dir.mkdir(parents=True, exist_ok=True)

                seed = build_seed(defaults, behavior, target["name"])
                dump_yaml(generated_dir / "seed.yaml", seed)
                dump_json(generated_dir / "behaviors.json", merged_behaviors)
                dump_json(generated_dir / "models.json", target_models)

                if bloom_results_dir.exists() and not args.dry_run:
                    shutil.rmtree(bloom_results_dir)

                try:
                    for stage in stages:
                        run_stage(
                            python_bin=bloom_python,
                            bloom_root=bloom_root,
                            stage=stage,
                            config_dir=generated_dir,
                            env=target_env,
                            log_path=run_logs / f"{stage}.log",
                            dry_run=args.dry_run,
                        )

                    if not args.dry_run:
                        if not bloom_results_dir.exists():
                            raise RuntimeError(f"Expected Bloom results missing: {bloom_results_dir}")
                        archive_bloom_results(bloom_results_dir, archived_results)

                        judgment_path = archived_results / "judgment.json"
                        if judgment_path.exists():
                            summary_rows.append(
                                collect_summary_row(
                                    suite_name=suite_name,
                                    target=target,
                                    behavior=behavior,
                                    judgment_path=judgment_path,
                                )
                            )
                        else:
                            failures.append(
                                f"Missing judgment.json for target={target['name']} behavior={behavior['name']}"
                            )
                except Exception as exc:  # noqa: BLE001
                    failures.append(f"{target['name']} / {behavior['name']}: {exc}")
                    if args.fail_fast:
                        raise
        finally:
            stop_launcher(launcher_process, launcher_handle)

    if not args.dry_run:
        write_summary(summary_rows, results_root)

    print(f"Bloom suite config: {args.config}")
    print(f"Selected targets: {[target['name'] for target in selected_targets]}")
    print(f"Selected behaviors: {[behavior['name'] for behavior in selected_behaviors]}")
    if not args.dry_run:
        print(f"Summary written to: {results_root / 'summary.csv'}")

    if failures:
        print("\nFailures:", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
