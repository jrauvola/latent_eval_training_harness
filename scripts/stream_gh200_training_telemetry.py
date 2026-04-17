"""Sync GH200 training telemetry locally and refresh a lightweight dashboard."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent.parent
HARNESS_ROOT = ROOT / "latent_eval_training_harness"
DEFAULT_REMOTE_HOST = "192.222.51.198"
DEFAULT_REMOTE_USER = "ubuntu"
DEFAULT_REMOTE_PROJECT_ROOT = "/home/ubuntu/Latent_Reasoning_Project/latent_eval_training_harness"
DEFAULT_RUN_NAME = "gemma3_4b_codi_gh200"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--remote-host", default=DEFAULT_REMOTE_HOST)
    parser.add_argument("--remote-user", default=DEFAULT_REMOTE_USER)
    parser.add_argument("--remote-project-root", default=DEFAULT_REMOTE_PROJECT_ROOT)
    parser.add_argument("--run-name", default=DEFAULT_RUN_NAME)
    parser.add_argument("--remote-output-dir", default="")
    parser.add_argument("--remote-log-path", default="")
    parser.add_argument("--remote-pid-file", default="")
    parser.add_argument("--local-report-dir", default="")
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--keep-going", action="store_true")
    return parser.parse_args()


def env_file() -> Path:
    for candidate in (ROOT / ".env", HARNESS_ROOT / ".env"):
        if candidate.exists():
            return candidate
    raise FileNotFoundError("No .env file found for SSH key extraction")


def ssh_key_path() -> str:
    extractor = HARNESS_ROOT / "scripts" / "extract_spar_lambda_ssh_key.py"
    result = subprocess.run(
        ["python3", str(extractor), "--env-file", str(env_file())],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def ssh_base_command(*, key_path: str, remote_user: str, remote_host: str) -> list[str]:
    return [
        "ssh",
        "-i",
        key_path,
        "-o",
        "StrictHostKeyChecking=no",
        f"{remote_user}@{remote_host}",
    ]


def run_remote_status(
    *,
    key_path: str,
    remote_user: str,
    remote_host: str,
    remote_pid_file: str,
    remote_log_path: str,
    remote_live_dir: str,
    poll_index: int,
) -> dict[str, object]:
    remote_script = f"""
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

pid_file = Path({remote_pid_file!r})
log_path = Path({remote_log_path!r})
live_dir = Path({remote_live_dir!r})
pid_text = pid_file.read_text(encoding="utf-8").strip() if pid_file.exists() else ""
pid = int(pid_text) if pid_text.isdigit() else None
running = False
if pid is not None:
    try:
        os.kill(pid, 0)
    except OSError:
        running = False
    else:
        running = True

gpu = {{}}
try:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    row = result.stdout.strip().splitlines()[0]
    used, total, util = [item.strip() for item in row.split(",")]
    gpu = {{
        "gpu_memory_used_mib": int(float(used)),
        "gpu_memory_total_mib": int(float(total)),
        "gpu_utilization_pct": int(float(util)),
    }}
except Exception:
    gpu = {{}}

payload = {{
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "state": "running" if running else "stopped",
    "pid": pid,
    "poll_index": {poll_index},
    "remote_pid_file_exists": pid_file.exists(),
    "remote_log_exists": log_path.exists(),
    "remote_log_size_bytes": log_path.stat().st_size if log_path.exists() else 0,
    "remote_metrics_exists": (live_dir / "metrics.jsonl").exists(),
    "remote_events_exists": (live_dir / "events.jsonl").exists(),
    **gpu,
}}
print(json.dumps(payload, ensure_ascii=True))
"""
    remote_command = f"python3 -c {shlex.quote(remote_script)}"
    result = subprocess.run(
        ssh_base_command(key_path=key_path, remote_user=remote_user, remote_host=remote_host)
        + [remote_command],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout.strip())


def append_jsonl(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def rsync_if_exists(
    *,
    key_path: str,
    remote_user: str,
    remote_host: str,
    remote_path: str,
    local_path: Path,
) -> bool:
    local_path.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [
            "rsync",
            "-az",
            "-e",
            f"ssh -i {key_path} -o StrictHostKeyChecking=no",
            f"{remote_user}@{remote_host}:{remote_path}",
            str(local_path),
        ],
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


def build_dashboard(report_dir: Path) -> dict[str, object]:
    summary_path = report_dir / "dashboard_summary.json"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "visualizations" / "generate_training_dashboard.py"),
            "--metrics-jsonl",
            str(report_dir / "metrics.jsonl"),
            "--events-jsonl",
            str(report_dir / "events.jsonl"),
            "--log-path",
            str(report_dir / "remote_train.log"),
            "--status-history-jsonl",
            str(report_dir / "status_history.jsonl"),
            "--html-out",
            str(report_dir / "dashboard.html"),
            "--status-md-out",
            str(report_dir / "status.md"),
            "--summary-json-out",
            str(summary_path),
        ],
        check=True,
        cwd=ROOT,
    )
    if summary_path.exists():
        return json.loads(summary_path.read_text(encoding="utf-8"))
    return {}


def main() -> None:
    args = parse_args()
    key_path = ssh_key_path()
    remote_output_dir = args.remote_output_dir or f"artifacts/train/{args.run_name}"
    remote_log_path = args.remote_log_path or f"/tmp/gh200_runs/{args.run_name}.log"
    remote_pid_file = args.remote_pid_file or f"/tmp/gh200_runs/{args.run_name}.pid"
    remote_live_dir = f"{args.remote_project_root}/{remote_output_dir}/live"
    local_report_dir = (
        Path(args.local_report_dir).expanduser().resolve()
        if args.local_report_dir
        else (HARNESS_ROOT / "artifacts" / "reports" / "gh200_live" / args.run_name).resolve()
    )
    local_report_dir.mkdir(parents=True, exist_ok=True)

    poll_index = 0
    last_alert = None
    while True:
        poll_index += 1
        status = run_remote_status(
            key_path=key_path,
            remote_user=args.remote_user,
            remote_host=args.remote_host,
            remote_pid_file=remote_pid_file,
            remote_log_path=remote_log_path,
            remote_live_dir=remote_live_dir,
            poll_index=poll_index,
        )
        status["synced_at_utc"] = datetime.now(timezone.utc).isoformat()
        append_jsonl(local_report_dir / "status_history.jsonl", status)
        (local_report_dir / "latest_remote_status.json").write_text(
            json.dumps(status, ensure_ascii=True, indent=2) + "\n",
            encoding="utf-8",
        )

        if status.get("remote_log_exists"):
            rsync_if_exists(
                key_path=key_path,
                remote_user=args.remote_user,
                remote_host=args.remote_host,
                remote_path=remote_log_path,
                local_path=local_report_dir / "remote_train.log",
            )
        if status.get("remote_metrics_exists"):
            rsync_if_exists(
                key_path=key_path,
                remote_user=args.remote_user,
                remote_host=args.remote_host,
                remote_path=f"{remote_live_dir}/metrics.jsonl",
                local_path=local_report_dir / "metrics.jsonl",
            )
        if status.get("remote_events_exists"):
            rsync_if_exists(
                key_path=key_path,
                remote_user=args.remote_user,
                remote_host=args.remote_host,
                remote_path=f"{remote_live_dir}/events.jsonl",
                local_path=local_report_dir / "events.jsonl",
            )

        summary = build_dashboard(local_report_dir)
        alert_level = summary.get("alert_level", "unknown")
        if alert_level != last_alert:
            print(
                f"[stream] alert={alert_level} state={status.get('state')} step={summary.get('latest_global_step')} "
                f"gpu={(status.get('gpu_memory_used_mib', 0) or 0)/1024.0:.1f}GB/{status.get('gpu_utilization_pct', 0)}%",
                flush=True,
            )
            last_alert = alert_level
        else:
            print(
                f"[stream] state={status.get('state')} step={summary.get('latest_global_step')} "
                f"gpu={(status.get('gpu_memory_used_mib', 0) or 0)/1024.0:.1f}GB/{status.get('gpu_utilization_pct', 0)}%",
                flush=True,
            )

        if args.once:
            break
        if status.get("state") != "running" and not args.keep_going:
            break
        time.sleep(max(args.poll_seconds, 5))


if __name__ == "__main__":
    main()
