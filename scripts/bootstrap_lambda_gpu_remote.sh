#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=""
REPO_URL=""
INSTALL_VLLM=0
SKIP_SETUP=0

usage() {
  cat <<'EOF'
Bootstrap a Lambda GPU box for the latent eval harness.

Usage:
  bash bootstrap_lambda_gpu_remote.sh --project-dir /home/ubuntu/Latent_Reasoning_Project/latent_eval_training_harness --repo-url https://github.com/example/repo.git

Options:
  --project-dir PATH   Absolute remote path where the harness repo should live.
  --repo-url URL       Git URL to clone when the box is fresh.
  --install-vllm       Install vLLM during environment setup.
  --skip-setup         Clone only; do not run scripts/setup_gpu_eval_env.sh.
  -h, --help           Show this help message.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --project-dir)
      PROJECT_DIR="${2:-}"
      shift 2
      ;;
    --repo-url)
      REPO_URL="${2:-}"
      shift 2
      ;;
    --install-vllm)
      INSTALL_VLLM=1
      shift
      ;;
    --skip-setup)
      SKIP_SETUP=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ -z "${PROJECT_DIR}" ]]; then
  echo "--project-dir is required." >&2
  exit 1
fi

if [[ "${PROJECT_DIR}" != /* ]]; then
  echo "--project-dir must be an absolute path." >&2
  exit 1
fi

if [[ -z "${REPO_URL}" ]]; then
  echo "--repo-url is required." >&2
  exit 1
fi

ensure_repo() {
  if [[ -d "${PROJECT_DIR}/.git" ]]; then
    echo "Remote harness repo already exists at ${PROJECT_DIR}."
    return 0
  fi

  if ! command -v git >/dev/null 2>&1; then
    echo "git is required on the remote machine to clone the harness repo." >&2
    exit 1
  fi

  if [[ ! -e "${PROJECT_DIR}" ]]; then
    mkdir -p "$(dirname "${PROJECT_DIR}")"
    echo "Fresh box detected. Cloning harness repo into ${PROJECT_DIR}."
    git clone "${REPO_URL}" "${PROJECT_DIR}"
    return 0
  fi

  if [[ -d "${PROJECT_DIR}" && -z "$(ls -A "${PROJECT_DIR}" 2>/dev/null)" ]]; then
    echo "Empty project directory detected. Cloning harness repo into ${PROJECT_DIR}."
    git clone "${REPO_URL}" "${PROJECT_DIR}"
    return 0
  fi

  echo "Remote project directory exists but is not a git repo: ${PROJECT_DIR}" >&2
  echo "Refusing to modify it automatically. Move it aside or set LAMBDA_GPU_PROJECT_DIR." >&2
  exit 1
}

ensure_setup() {
  if [[ "${SKIP_SETUP}" == "1" ]]; then
    echo "Skipping environment setup by request."
    return 0
  fi

  if [[ -x "${PROJECT_DIR}/.venv/bin/python" ]]; then
    echo "Remote Python environment already exists at ${PROJECT_DIR}/.venv."
    return 0
  fi

  if ! command -v python3 >/dev/null 2>&1; then
    echo "python3 is required on the remote machine to set up the harness environment." >&2
    exit 1
  fi

  if [[ ! -f "${PROJECT_DIR}/scripts/setup_gpu_eval_env.sh" ]]; then
    echo "Missing setup script at ${PROJECT_DIR}/scripts/setup_gpu_eval_env.sh" >&2
    exit 1
  fi

  echo "Creating remote harness environment in ${PROJECT_DIR}/.venv."
  cd "${PROJECT_DIR}"
  INSTALL_VLLM="${INSTALL_VLLM}" bash scripts/setup_gpu_eval_env.sh
}

ensure_repo
ensure_setup
