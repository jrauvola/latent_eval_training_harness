#!/usr/bin/env python3
"""Sync the harness to GH200, stage an HF token, and launch training in the background."""

from __future__ import annotations

import argparse
import os
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent.parent
HARNESS_ROOT = ROOT / "latent_eval_training_harness"
DEFAULT_REMOTE_HOST = "192.222.51.198"
DEFAULT_REMOTE_USER = "ubuntu"
DEFAULT_REMOTE_PROJECT_ROOT = "/home/ubuntu/Latent_Reasoning_Project/latent_eval_training_harness"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Training config path relative to harness root")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--remote-host", default=DEFAULT_REMOTE_HOST)
    parser.add_argument("--remote-user", default=DEFAULT_REMOTE_USER)
    parser.add_argument("--remote-project-root", default=DEFAULT_REMOTE_PROJECT_ROOT)
    parser.add_argument("--skip-sync", action="store_true")
    return parser.parse_args()


def env_files() -> list[Path]:
    return [ROOT / ".env", HARNESS_ROOT / ".env"]


def resolve_env_file() -> Path:
    for candidate in env_files():
        if candidate.exists():
            return candidate
    raise FileNotFoundError("No .env file found for SSH key extraction")


def ssh_key_path() -> str:
    extractor = HARNESS_ROOT / "scripts" / "extract_spar_lambda_ssh_key.py"
    result = subprocess.run(
        ["python3", str(extractor), "--env-file", str(resolve_env_file())],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def resolve_hf_token() -> str | None:
    for key in ("HF_TOKEN", "HUGGINGFACE_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        value = os.environ.get(key, "").strip()
        if value:
            return value
    for env_file in env_files():
        if not env_file.exists():
            continue
        for line in env_file.read_text(encoding="utf-8").splitlines():
            for key in ("HUGGINGFACE_TOKEN", "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
                if not line.startswith(f"{key}="):
                    continue
                value = line.split("=", 1)[1].strip().strip('"').strip("'")
                if value:
                    return value
    return None


def sync_project(*, key_path: str, remote_user: str, remote_host: str, remote_project_root: str) -> None:
    subprocess.run(
        [
            "rsync",
            "-az",
            "--exclude",
            ".venv",
            "--exclude",
            ".venv-*",
            "--exclude",
            ".cache",
            "--exclude",
            "artifacts",
            "--exclude",
            "__pycache__",
            "--exclude",
            "*.pyc",
            "-e",
            f"ssh -i {key_path} -o StrictHostKeyChecking=no",
            f"{HARNESS_ROOT}/",
            f"{remote_user}@{remote_host}:{remote_project_root}/",
        ],
        check=True,
        cwd=ROOT,
    )


def stage_hf_token(
    *,
    token: str | None,
    key_path: str,
    remote_user: str,
    remote_host: str,
    remote_token_path: str,
) -> None:
    if not token:
        return
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as handle:
        handle.write(token + "\n")
        local_token_path = Path(handle.name)
    try:
        os.chmod(local_token_path, 0o600)
        subprocess.run(
            [
                "rsync",
                "-az",
                "-e",
                f"ssh -i {key_path} -o StrictHostKeyChecking=no",
                str(local_token_path),
                f"{remote_user}@{remote_host}:{remote_token_path}",
            ],
            check=True,
            cwd=ROOT,
        )
    finally:
        local_token_path.unlink(missing_ok=True)


def launch_remote(
    *,
    key_path: str,
    remote_user: str,
    remote_host: str,
    remote_project_root: str,
    config_rel: str,
    run_name: str,
    output_dir_rel: str,
) -> str:
    remote_script = """#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$1"
CONFIG_REL="$2"
RUN_NAME="$3"
OUTPUT_DIR_REL="$4"

RUN_DIR="/tmp/gh200_runs"
LOG_FILE="${RUN_DIR}/${RUN_NAME}.log"
PID_FILE="${RUN_DIR}/${RUN_NAME}.pid"
TOKEN_FILE="${RUN_DIR}/${RUN_NAME}.hf_token"
ENV_FILE="${RUN_DIR}/${RUN_NAME}.env"
OUTPUT_DIR="${PROJECT_ROOT}/${OUTPUT_DIR_REL}"

mkdir -p "${RUN_DIR}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"

archive_if_exists() {
  local path="$1"
  if [[ -e "${path}" ]]; then
    mv "${path}" "${path}_${STAMP}"
  fi
}

archive_if_exists "${OUTPUT_DIR}"
archive_if_exists "${LOG_FILE}"
archive_if_exists "${PID_FILE}"
rm -f "${ENV_FILE}"

if [[ -f "${TOKEN_FILE}" ]]; then
  TOKEN="$(tr -d '\r\n' < "${TOKEN_FILE}")"
  rm -f "${TOKEN_FILE}"
  if [[ -n "${TOKEN}" ]]; then
    printf 'HF_TOKEN=%s\nHUGGING_FACE_HUB_TOKEN=%s\n' "${TOKEN}" "${TOKEN}" > "${ENV_FILE}"
    chmod 600 "${ENV_FILE}"
  fi
fi

if [[ ! -x "${PROJECT_ROOT}/.venv/bin/python" ]]; then
  echo "Missing remote virtualenv python at ${PROJECT_ROOT}/.venv/bin/python" >&2
  exit 1
fi

nohup bash -lc '
  cd "'"${PROJECT_ROOT}"'"
  if [[ -f "'"${ENV_FILE}"'" ]]; then
    set -a
    source "'"${ENV_FILE}"'"
    set +a
    rm -f "'"${ENV_FILE}"'"
  fi
  export PYTHONUNBUFFERED=1
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  export HF_DATASETS_VERBOSITY=info
  export TRANSFORMERS_VERBOSITY=info
  .venv/bin/python scripts/run_training_entrypoint.py --config "'"${CONFIG_REL}"'"
' > "${LOG_FILE}" 2>&1 < /dev/null &

echo "$!" > "${PID_FILE}"
printf 'launched pid=%s\nlog=%s\npid_file=%s\noutput_dir=%s\n' "$(cat "${PID_FILE}")" "${LOG_FILE}" "${PID_FILE}" "${OUTPUT_DIR}"
"""
    result = subprocess.run(
        [
            "ssh",
            "-i",
            key_path,
            "-o",
            "StrictHostKeyChecking=no",
            f"{remote_user}@{remote_host}",
            "bash",
            "-s",
            "--",
            remote_project_root,
            config_rel,
            run_name,
            output_dir_rel,
        ],
        input=remote_script,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def main() -> None:
    args = parse_args()
    key_path = ssh_key_path()
    output_dir_rel = args.output_dir or f"artifacts/train/{args.run_name}"
    remote_token_path = f"/tmp/gh200_runs/{args.run_name}.hf_token"

    if not args.skip_sync:
        sync_project(
            key_path=key_path,
            remote_user=args.remote_user,
            remote_host=args.remote_host,
            remote_project_root=args.remote_project_root,
        )

    stage_hf_token(
        token=resolve_hf_token(),
        key_path=key_path,
        remote_user=args.remote_user,
        remote_host=args.remote_host,
        remote_token_path=remote_token_path,
    )

    output = launch_remote(
        key_path=key_path,
        remote_user=args.remote_user,
        remote_host=args.remote_host,
        remote_project_root=args.remote_project_root,
        config_rel=args.config,
        run_name=args.run_name,
        output_dir_rel=output_dir_rel,
    )
    print(output)


if __name__ == "__main__":
    main()
