#!/usr/bin/env bash
# Launch the V2/V3/V4 detach-variant training chain on a Lambda GPU box.
#
# What it does:
#   1. Ensures the local feature/detach-variants branch is pushed to origin.
#   2. Extracts the SSH key from .env (multi-line PEM safe).
#   3. Copies run_detach_phase_chain_remote.sh to the remote box.
#   4. Starts a new detached tmux session running the chain.
#
# After this script returns, the remote orchestrator runs autonomously. Use
# ``pull_detach_artifacts.sh`` in a separate local shell to periodically
# rsync artifacts back.
#
# Env overrides:
#   LAMBDA_GPU_HOST   - IP of the GPU box. Default: 192.222.51.198 (Test Key).
#   LAMBDA_GPU_USER   - SSH user. Default: ubuntu.
#   BARRIER_PID       - Remote PID the chain should wait for before starting.
#                       Default: empty (run immediately).
#   TMUX_SESSION      - tmux session name on the remote box. Default: detach_chain.
#   CONFIG_LIST_FILE  - File with one config path per line. If unset, uses the
#                       default queue hardcoded below.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
MONOREPO_ROOT="$(cd "${ROOT_DIR}/.." && pwd)"

LAMBDA_GPU_HOST="${LAMBDA_GPU_HOST:-192.222.51.198}"
LAMBDA_GPU_USER="${LAMBDA_GPU_USER:-ubuntu}"
TMUX_SESSION="${TMUX_SESSION:-detach_chain}"
BARRIER_PID="${BARRIER_PID:-}"
FEATURE_BRANCH="${FEATURE_BRANCH:-feature/detach-variants}"

# Default queue: Phase 0 first, then V2/V3/V4 full runs in that order.
DEFAULT_QUEUE=(
  "configs/training/gemma3_4b_codi_gh200_phase0_fp32_no_detach.yaml"
  "configs/training/gemma3_4b_codi_gh200_v2_keep_last_2.yaml"
  "configs/training/gemma3_4b_codi_gh200_v3_reasoning_only.yaml"
  "configs/training/gemma3_4b_codi_gh200_v4_strict_no_latent_detach.yaml"
)

if [[ -n "${CONFIG_LIST_FILE:-}" ]]; then
  if [[ ! -f "${CONFIG_LIST_FILE}" ]]; then
    echo "ERROR: CONFIG_LIST_FILE=${CONFIG_LIST_FILE} not found" >&2
    exit 2
  fi
  mapfile -t QUEUE < "${CONFIG_LIST_FILE}"
else
  QUEUE=("${DEFAULT_QUEUE[@]}")
fi

CONFIGS_PAYLOAD="$(printf '%s\n' "${QUEUE[@]}")"

echo "Queue (${#QUEUE[@]} configs):"
printf '  - %s\n' "${QUEUE[@]}"
echo ""

# --- Push local branch so remote can fetch it -------------------------------
echo "Pushing ${FEATURE_BRANCH} to origin..."
pushd "${ROOT_DIR}" > /dev/null
LOCAL_BRANCH="$(git rev-parse --abbrev-ref HEAD)"
LOCAL_HEAD="$(git rev-parse HEAD)"
if [[ "${LOCAL_BRANCH}" != "${FEATURE_BRANCH}" ]]; then
  echo "WARNING: local HEAD is on '${LOCAL_BRANCH}', not '${FEATURE_BRANCH}'."
  echo "         The chain will pin to ${FEATURE_BRANCH}'s tip on origin, not ${LOCAL_HEAD}."
fi
git push origin "${FEATURE_BRANCH}"
popd > /dev/null
echo ""

# --- Extract SSH key (multi-line PEM safe) ----------------------------------
ENV_FILE="${MONOREPO_ROOT}/.env"
if [[ ! -f "${ENV_FILE}" ]]; then
  ENV_FILE="${ROOT_DIR}/.env"
fi
if [[ ! -f "${ENV_FILE}" ]]; then
  echo "ERROR: .env not found in monorepo or harness root" >&2
  exit 3
fi

TMPKEY="$(mktemp)"
trap 'rm -f "${TMPKEY}"' EXIT
awk '
  /^SPAR_LAMBDA_PRIVATE_KEY=/{found=1; sub(/^SPAR_LAMBDA_PRIVATE_KEY=/,""); print; next}
  found && /-----END RSA PRIVATE KEY-----/{print; exit}
  found{print}
' "${ENV_FILE}" > "${TMPKEY}"
chmod 600 "${TMPKEY}"

SSH_OPTS=(-i "${TMPKEY}" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=15)
SCP_OPTS=("${SSH_OPTS[@]}")

# --- Copy the remote orchestrator script ------------------------------------
REMOTE_SCRIPT_PATH="/tmp/run_detach_phase_chain_remote.sh"
echo "Uploading orchestrator to ${LAMBDA_GPU_HOST}:${REMOTE_SCRIPT_PATH}"
scp "${SCP_OPTS[@]}" \
  "${SCRIPT_DIR}/run_detach_phase_chain_remote.sh" \
  "${LAMBDA_GPU_USER}@${LAMBDA_GPU_HOST}:${REMOTE_SCRIPT_PATH}"

# --- Build the remote command to start the tmux session ---------------------
# We pass CONFIGS via a file to avoid quoting hell across ssh+tmux.
REMOTE_CONFIG_FILE="/tmp/detach_chain_configs_$$.txt"
printf '%s' "${CONFIGS_PAYLOAD}" | ssh "${SSH_OPTS[@]}" \
  "${LAMBDA_GPU_USER}@${LAMBDA_GPU_HOST}" "cat > ${REMOTE_CONFIG_FILE}"

# The tmux new-session command. Use send-keys-free form: bash -c with env pre-set.
# If the session already exists, refuse to overwrite (ask user to pick a new name).
REMOTE_START_CMD="$(cat <<EOF
if tmux has-session -t ${TMUX_SESSION} 2>/dev/null; then
  echo "ERROR: tmux session '${TMUX_SESSION}' already exists. Attach with 'tmux attach -t ${TMUX_SESSION}' or set TMUX_SESSION=<newname>."
  exit 1
fi
chmod +x ${REMOTE_SCRIPT_PATH}
tmux new-session -d -s ${TMUX_SESSION} \
  "BARRIER_PID='${BARRIER_PID}' \
   FEATURE_BRANCH='${FEATURE_BRANCH}' \
   CONFIGS=\"\$(cat ${REMOTE_CONFIG_FILE})\" \
   ABORT_ON_FAIL=1 \
   bash ${REMOTE_SCRIPT_PATH} 2>&1 | tee /tmp/detach_chain_tee.log; \
   echo; echo '--- chain ended. press enter to close tmux window ---'; read"
echo "Chain started in tmux session '${TMUX_SESSION}'."
echo "Attach with: tmux attach -t ${TMUX_SESSION}"
echo "Progress log: /tmp/detach_chain_tee.log (remote) or artifacts/chain/ (remote)"
EOF
)"

echo ""
echo "Starting remote chain via tmux..."
ssh "${SSH_OPTS[@]}" "${LAMBDA_GPU_USER}@${LAMBDA_GPU_HOST}" "${REMOTE_START_CMD}"

echo ""
echo "Launch complete. In another shell, run:"
echo "  bash scripts/pull_detach_artifacts.sh"
echo "to periodically rsync artifacts back to local."
