#!/usr/bin/env bash
set -euo pipefail

# Run a CODI training job on the team's Lambda GPU box.
#
# Usage:
#   bash scripts/run_lambda_codi_train.sh
#   TRAIN_CONFIG=configs/training/llama32_1b_codi.yaml \
#     EVAL_CONFIG=configs/evaluation/llama32_1b_codi_paper_core.yaml \
#     bash scripts/run_lambda_codi_train.sh
#
# Notes:
# - Defaults to the smoke config so bring-up is cheap and safe.
# - Reuses connect_lambda_gpu.sh for host resolution, SSH key extraction, and
#   remote bootstrap.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MONOREPO_ROOT="$(cd "${ROOT_DIR}/.." && pwd)"
CONNECT_SCRIPT="${ROOT_DIR}/scripts/connect_lambda_gpu.sh"

TRAIN_CONFIG="${TRAIN_CONFIG:-configs/training/llama32_1b_codi_smoke.yaml}"
EVAL_CONFIG="${EVAL_CONFIG:-}"
ALLOW_DIRTY="${LAMBDA_ALLOW_DIRTY:-0}"

load_single_line_env_var() {
  local key="$1"
  local file="$2"
  python3 - "$key" "$file" <<'PY'
from pathlib import Path
import sys

key = sys.argv[1]
path = Path(sys.argv[2])
if not path.is_file():
    raise SystemExit(0)

for line in path.read_text(encoding="utf-8").splitlines():
    if not line.startswith(f"{key}="):
        continue
    value = line.split("=", 1)[1].strip().strip('"').strip("'")
    if value:
        print(value)
    break
PY
}

resolve_hf_token() {
  if [[ -n "${HF_TOKEN:-}" ]]; then
    printf '%s\n' "${HF_TOKEN}"
    return 0
  fi
  if [[ -n "${HUGGINGFACE_TOKEN:-}" ]]; then
    printf '%s\n' "${HUGGINGFACE_TOKEN}"
    return 0
  fi
  if [[ -n "${HUGGING_FACE_HUB_TOKEN:-}" ]]; then
    printf '%s\n' "${HUGGING_FACE_HUB_TOKEN}"
    return 0
  fi

  local file
  local key
  for file in "${MONOREPO_ROOT}/.env" "${ROOT_DIR}/.env"; do
    for key in HUGGINGFACE_TOKEN HF_TOKEN HUGGING_FACE_HUB_TOKEN; do
      local value
      value="$(load_single_line_env_var "${key}" "${file}")"
      if [[ -n "${value}" ]]; then
        printf '%s\n' "${value}"
        return 0
      fi
    done
  done
}

resolve_local_branch() {
  git -C "${ROOT_DIR}" rev-parse --abbrev-ref HEAD
}

resolve_local_head() {
  git -C "${ROOT_DIR}" rev-parse HEAD
}

resolve_remote_branch_head() {
  local branch="$1"
  git -C "${ROOT_DIR}" ls-remote --exit-code --heads origin "${branch}" | awk '{print $1}'
}

ensure_local_repo_is_syncable() {
  local branch
  local local_head
  local remote_head
  local dirty

  branch="$(resolve_local_branch)"
  if [[ "${branch}" == "HEAD" ]]; then
    echo "Detached HEAD detected in ${ROOT_DIR}. Check out a branch before launching a remote run." >&2
    exit 1
  fi

  local_head="$(resolve_local_head)"
  remote_head="$(resolve_remote_branch_head "${branch}" || true)"
  if [[ -z "${remote_head}" ]]; then
    echo "Could not resolve origin/${branch}. Push the branch before launching a remote run." >&2
    exit 1
  fi
  if [[ "${local_head}" != "${remote_head}" ]]; then
    echo "Local HEAD (${local_head}) does not match origin/${branch} (${remote_head})." >&2
    echo "Push or pull first so the GH200 runs the exact code you intend." >&2
    exit 1
  fi

  dirty="$(git -C "${ROOT_DIR}" status --porcelain)"
  if [[ -n "${dirty}" && "${ALLOW_DIRTY}" != "1" ]]; then
    echo "Local working tree has uncommitted changes." >&2
    echo "Commit/stash them, or set LAMBDA_ALLOW_DIRTY=1 to run the pushed commit anyway." >&2
    exit 1
  fi

  printf '%s\n' "${branch}" "${local_head}"
}

if [[ ! -f "${CONNECT_SCRIPT}" ]]; then
  echo "Missing Lambda helper: ${CONNECT_SCRIPT}" >&2
  exit 1
fi

mapfile -t GIT_SYNC_STATE < <(ensure_local_repo_is_syncable)
LOCAL_BRANCH="${GIT_SYNC_STATE[0]}"
LOCAL_HEAD="${GIT_SYNC_STATE[1]}"
HF_TOKEN_VALUE="$(resolve_hf_token || true)"
ENV_ARGS=(
  "LAMBDA_GPU_GIT_BRANCH=${LOCAL_BRANCH}"
  "LAMBDA_GPU_GIT_REF=${LOCAL_HEAD}"
  "PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
  "PYTHONPATH=src"
)
if [[ -n "${HF_TOKEN_VALUE}" ]]; then
  ENV_ARGS+=("HF_TOKEN=${HF_TOKEN_VALUE}" "HUGGING_FACE_HUB_TOKEN=${HF_TOKEN_VALUE}")
else
  echo "WARN: no HF token found locally; gated model downloads may fail on the remote box." >&2
fi

REMOTE_SCRIPT="echo 'Remote git commit: '\"\$(git rev-parse HEAD)\" && python -m latent_harness.training.cli --config '${TRAIN_CONFIG}'"
if [[ -n "${EVAL_CONFIG}" ]]; then
  REMOTE_SCRIPT="${REMOTE_SCRIPT} && python -m latent_harness.evaluation.cli --config '${EVAL_CONFIG}'"
fi

echo "Pinned branch:      ${LOCAL_BRANCH}"
echo "Pinned git commit:  ${LOCAL_HEAD}"
echo "Remote train config: ${TRAIN_CONFIG}"
if [[ -n "${EVAL_CONFIG}" ]]; then
  echo "Remote eval config:  ${EVAL_CONFIG}"
fi

bash "${CONNECT_SCRIPT}" env "${ENV_ARGS[@]}" bash -lc "${REMOTE_SCRIPT}"
