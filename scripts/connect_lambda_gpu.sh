#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
MONOREPO_ROOT="$(cd "${REPO_ROOT}/.." && pwd)"
RESOLVER="${SCRIPT_DIR}/resolve_lambda_gpu.py"
BOOTSTRAP_REMOTE_SCRIPT="${SCRIPT_DIR}/bootstrap_lambda_gpu_remote.sh"
SPAR_KEY_EXTRACTOR="${SCRIPT_DIR}/extract_spar_lambda_ssh_key.py"

# Monorepo .env often contains multiline PEM (SPAR_LAMBDA_PRIVATE_KEY); do not `source` it in bash.
# Optional harness-only .env: single-line vars only (sourcing multiline PEM will break).
LOCAL_ENV_FILE="${REPO_ROOT}/.env"
if [[ -f "${LOCAL_ENV_FILE}" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${LOCAL_ENV_FILE}" 2>/dev/null || true
  set +a
fi

_strip_env_value() {
  # shellcheck disable=SC2034
  local v="$1"
  v="${v#\"}"
  v="${v%\"}"
  v="${v#\'}"
  v="${v%\'}"
  printf '%s' "${v}"
}

# Load Cloud API bearer token from .env without sourcing the whole file (multiline PEM safe).
# Prefer SPAR_LAMBDA_API_KEY when it is not an SSH public key line; else LAMBDA_API_KEY.
_load_lambda_cloud_token_from_file() {
  local f="$1"
  [[ -f "$f" ]] || return 0
  if [[ -n "${LAMBDA_CLOUD_API_TOKEN:-}" || -n "${LAMBDA_CLOUD_TOKEN:-}" || -n "${LAMBDA_API_TOKEN:-}" ]]; then
    return 0
  fi
  local line val
  line="$(grep -E '^SPAR_LAMBDA_API_KEY=' "$f" 2>/dev/null | head -1)" || true
  if [[ -n "${line}" ]]; then
    val="$(_strip_env_value "${line#SPAR_LAMBDA_API_KEY=}")"
    if [[ "${val}" != ssh-rsa* && "${val}" != ssh-ed25519* && "${val}" != ecdsa-* && "${val}" != sk-* ]]; then
      export LAMBDA_CLOUD_API_TOKEN="${val}"
      return 0
    fi
  fi
  line="$(grep -E '^LAMBDA_API_KEY=' "$f" 2>/dev/null | head -1)" || true
  if [[ -n "${line}" ]]; then
    export LAMBDA_CLOUD_API_TOKEN="$(_strip_env_value "${line#LAMBDA_API_KEY=}")"
  fi
}
_load_lambda_cloud_token_from_file "${MONOREPO_ROOT}/.env"
if [[ -z "${LAMBDA_CLOUD_API_TOKEN:-}" ]]; then
  _load_lambda_cloud_token_from_file "${REPO_ROOT}/.env"
fi

HOST="${LAMBDA_GPU_HOST:-auto}"
USER_NAME="${LAMBDA_GPU_USER:-ubuntu}"
KEY_PATH="${LAMBDA_GPU_KEY_PATH:-}"
KEY_PATH="${KEY_PATH/#\~/$HOME}"
PROJECT_DIR="${LAMBDA_GPU_PROJECT_DIR:-/home/${USER_NAME}/Latent_Reasoning_Project/latent_eval_training_harness}"
STRICT_HOST_KEY_CHECKING="${LAMBDA_GPU_STRICT_HOST_KEY_CHECKING:-no}"
AUTO_BOOTSTRAP="${LAMBDA_GPU_AUTO_BOOTSTRAP:-1}"
BOOTSTRAP_INSTALL_VLLM="${LAMBDA_GPU_BOOTSTRAP_INSTALL_VLLM:-0}"
DEFAULT_REPO_URL="$(git -C "${REPO_ROOT}" remote get-url origin 2>/dev/null || true)"
REPO_URL="${LAMBDA_GPU_REPO_URL:-${DEFAULT_REPO_URL}}"

KEY_PATH="${KEY_PATH/#\~/$HOME}"

usage() {
  cat <<'EOF'
Connect to the Lambda GPU box we use for harness runs.

Usage:
  scripts/connect_lambda_gpu.sh
      On fresh boxes, clone the harness repo and create .venv, then open an interactive shell.

  scripts/connect_lambda_gpu.sh python3 -m pip list
      Ensure the remote repo/.venv exist, then run a command inside the harness repo.

  scripts/connect_lambda_gpu.sh --raw nvidia-smi
      Run a raw remote command without changing directories or activating .venv.

Optional env overrides:
  LAMBDA_GPU_HOST
  LAMBDA_GPU_USER
  LAMBDA_GPU_KEY_PATH
    If unset or file missing, uses SPAR_LAMBDA_PRIVATE_KEY from the monorepo ../.env
    (written to a temp PEM via scripts/extract_spar_lambda_ssh_key.py).
  LAMBDA_GPU_PROJECT_DIR
  LAMBDA_GPU_REPO_URL
  LAMBDA_GPU_STRICT_HOST_KEY_CHECKING
  LAMBDA_GPU_AUTO_BOOTSTRAP
  LAMBDA_GPU_BOOTSTRAP_INSTALL_VLLM
  LAMBDA_CLOUD_API_TOKEN (or LAMBDA_CLOUD_TOKEN / LAMBDA_API_TOKEN / LAMBDA_API_KEY in .env)
  LAMBDA_GPU_INSTANCE_ID
  LAMBDA_GPU_NAME
  LAMBDA_GPU_NAME_CONTAINS
  LAMBDA_GPU_INSTANCE_TYPE
  LAMBDA_GPU_REGION

Helper modes:
  scripts/connect_lambda_gpu.sh --list-instances
      List matching Lambda instances via the Cloud API instead of connecting.

  scripts/connect_lambda_gpu.sh --print-host
      Print the resolved Lambda host/IP and exit.

  scripts/connect_lambda_gpu.sh --bootstrap-only
      Ensure the remote repo and .venv exist, then exit without opening a shell.
EOF
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
  exit 0
fi

resolve_host() {
  if [[ -n "${HOST}" && "${HOST}" != "auto" ]]; then
    return 0
  fi
  if [[ ! -f "${RESOLVER}" ]]; then
    echo "Resolver script not found: ${RESOLVER}" >&2
    exit 1
  fi
  HOST="$(python3 "${RESOLVER}")"
}

bootstrap_remote_project() {
  if [[ "${AUTO_BOOTSTRAP}" != "1" ]]; then
    return 0
  fi
  if [[ ! -f "${BOOTSTRAP_REMOTE_SCRIPT}" ]]; then
    echo "Remote bootstrap script not found: ${BOOTSTRAP_REMOTE_SCRIPT}" >&2
    exit 1
  fi
  if [[ -z "${REPO_URL}" ]]; then
    echo "No repo URL found for remote bootstrap. Set LAMBDA_GPU_REPO_URL." >&2
    exit 1
  fi

  local bootstrap_cmd=(
    bash
    -s
    --
    --project-dir
    "${PROJECT_DIR}"
    --repo-url
    "${REPO_URL}"
  )
  if [[ "${BOOTSTRAP_INSTALL_VLLM}" == "1" ]]; then
    bootstrap_cmd+=(--install-vllm)
  fi

  "${SSH_BASE[@]}" "${bootstrap_cmd[@]}" < "${BOOTSTRAP_REMOTE_SCRIPT}"
}

if [[ "${1:-}" == "--list-instances" ]]; then
  shift
  exec python3 "${RESOLVER}" --list "$@"
fi

if [[ "${1:-}" == "--print-host" ]]; then
  shift
  if [[ -n "${HOST}" && "${HOST}" != "auto" && $# -eq 0 ]]; then
    printf '%s\n' "${HOST}"
    exit 0
  fi
  exec python3 "${RESOLVER}" "$@"
fi

if [[ -z "${KEY_PATH}" || ! -f "${KEY_PATH}" ]]; then
  if [[ -f "${MONOREPO_ROOT}/.env" && -f "${SPAR_KEY_EXTRACTOR}" ]]; then
    KEY_PATH="$(python3 "${SPAR_KEY_EXTRACTOR}" --env-file "${MONOREPO_ROOT}/.env")" || KEY_PATH=""
  fi
fi
if [[ -z "${KEY_PATH}" || ! -f "${KEY_PATH}" ]]; then
  KEY_PATH="${HOME}/.ssh/ubuntu-wsl"
fi
if [[ ! -f "${KEY_PATH}" ]]; then
  echo "SSH key not found: ${KEY_PATH}" >&2
  echo "Set LAMBDA_GPU_KEY_PATH or add SPAR_LAMBDA_PRIVATE_KEY to ${MONOREPO_ROOT}/.env" >&2
  exit 1
fi

resolve_host

SSH_BASE=(
  ssh
  -i "${KEY_PATH}"
  -o "StrictHostKeyChecking=${STRICT_HOST_KEY_CHECKING}"
  "${USER_NAME}@${HOST}"
)

if [[ "${1:-}" == "--bootstrap-only" ]]; then
  bootstrap_remote_project
  exit 0
fi

if [[ "${1:-}" == "--raw" ]]; then
  shift
  if [[ $# -eq 0 ]]; then
    exec "${SSH_BASE[@]}"
  fi

  REMOTE_CMD="$(printf '%q ' "$@")"
  exec "${SSH_BASE[@]}" "bash -lc '${REMOTE_CMD}'"
fi

bootstrap_remote_project

if [[ $# -eq 0 ]]; then
  exec "${SSH_BASE[@]}" -t "bash -lc 'cd ${PROJECT_DIR} && if [[ -f .venv/bin/activate ]]; then source .venv/bin/activate; fi; exec \${SHELL:-/bin/bash} -l'"
fi

REMOTE_CMD="$(printf '%q ' "$@")"
exec "${SSH_BASE[@]}" -t "bash -lc 'cd ${PROJECT_DIR} && if [[ -f .venv/bin/activate ]]; then source .venv/bin/activate; fi; ${REMOTE_CMD}'"
