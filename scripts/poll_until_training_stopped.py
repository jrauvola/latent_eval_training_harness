#!/usr/bin/env python3
"""Poll stream_gh200_training_telemetry --once until remote training PID exits."""

from __future__ import annotations

import json
import random
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
HARNESS = ROOT / "latent_eval_training_harness"
SCRIPT = HARNESS / "scripts" / "stream_gh200_training_telemetry.py"


def main() -> None:
    run_name = sys.argv[1] if len(sys.argv) > 1 else "gemma_stabilized_full"
    remote_out = (
        sys.argv[2]
        if len(sys.argv) > 2
        else "artifacts/train/gemma3_4b_codi_gh200_stabilized_full"
    )
    status_path = HARNESS / "artifacts" / "reports" / "gh200_live" / run_name / "latest_remote_status.json"
    session_log = HARNESS / "artifacts" / "reports" / "gh200_live" / run_name / "poll_session.log"
    session_log.parent.mkdir(parents=True, exist_ok=True)

    i = 0
    while True:
        i += 1
        t = time.strftime("%Y-%m-%d %H:%M:%S")
        r = subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--run-name",
                run_name,
                "--remote-output-dir",
                remote_out,
                "--once",
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        out = (r.stdout or "").strip()
        err = (r.stderr or "").strip()
        block = f"{t} iter={i} rc={r.returncode}\n{out}\n{err}\n---\n"
        session_log.write_text(session_log.read_text(encoding="utf-8") + block if session_log.exists() else block, encoding="utf-8")
        print(block, flush=True)

        state = None
        if status_path.exists():
            try:
                state = json.loads(status_path.read_text(encoding="utf-8")).get("state")
            except (OSError, json.JSONDecodeError):
                pass
        print(f"PARSED_STATE={state!r}", flush=True)

        if state == "stopped":
            print("DONE: remote training process has exited (state=stopped).", flush=True)
            return

        delay = random.uniform(30.0, 90.0)
        print(f"Sleeping {delay:.1f}s ...", flush=True)
        time.sleep(delay)


if __name__ == "__main__":
    main()
