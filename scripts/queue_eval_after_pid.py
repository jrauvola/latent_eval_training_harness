from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Queue an evaluation run after a training PID exits")
    parser.add_argument("--wait-pid", type=int, required=True, help="Remote/local PID to wait for")
    parser.add_argument("--project-root", required=True, help="Harness project root on this machine")
    parser.add_argument("--eval-config", required=True, help="Config path relative to project root")
    parser.add_argument("--eval-log", required=True, help="Absolute path for queued eval stdout/stderr")
    parser.add_argument("--eval-pid-file", required=True, help="Absolute path for queued eval PID")
    parser.add_argument("--token-file", required=False, help="Optional file containing HF token")
    parser.add_argument("--poll-seconds", type=int, default=60, help="Wait interval while polling PID")
    return parser.parse_args()


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def load_eval_output_dir(project_root: Path, eval_config: Path) -> Path:
    payload = yaml.safe_load(eval_config.read_text(encoding="utf-8")) or {}
    runtime = payload.get("runtime", {})
    output_dir = runtime.get("output_dir", "artifacts/eval/default")
    return (project_root / output_dir).resolve()


def archive_if_exists(path: Path, suffix: str) -> None:
    if path.exists():
        archived = path.with_name(f"{path.name}_{suffix}")
        shutil.move(str(path), str(archived))


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root).expanduser().resolve()
    eval_config = (project_root / args.eval_config).resolve()
    eval_log = Path(args.eval_log).expanduser().resolve()
    eval_pid_file = Path(args.eval_pid_file).expanduser().resolve()
    token_file = Path(args.token_file).expanduser().resolve() if args.token_file else None

    print(
        f"[queue] waiting for pid={args.wait_pid} to exit before launching eval config={eval_config}",
        flush=True,
    )
    while pid_alive(args.wait_pid):
        time.sleep(max(args.poll_seconds, 5))

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = load_eval_output_dir(project_root, eval_config)
    archive_if_exists(output_dir, f"queued_prev_{stamp}")
    archive_if_exists(eval_log, f"queued_prev_{stamp}")
    archive_if_exists(eval_pid_file, f"queued_prev_{stamp}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    eval_log.parent.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    if token_file is not None and token_file.exists():
        token = token_file.read_text(encoding="utf-8").strip()
        if token:
            env["HF_TOKEN"] = token
            env["HUGGINGFACE_HUB_TOKEN"] = token
        token_file.unlink(missing_ok=True)
    env["PYTHONUNBUFFERED"] = "1"

    with eval_log.open("w", encoding="utf-8") as log_handle:
        proc = subprocess.Popen(
            [
                str(project_root / ".venv/bin/python"),
                "scripts/run_evaluation_entrypoint.py",
                "--config",
                str(eval_config.relative_to(project_root)),
            ],
            cwd=project_root,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
        )
    eval_pid_file.write_text(str(proc.pid), encoding="utf-8")
    print(
        f"[queue] launched eval pid={proc.pid} config={eval_config.name} log={eval_log}",
        flush=True,
    )


if __name__ == "__main__":
    main()
