#!/usr/bin/env bash
# Periodically rsync training artifacts from the Lambda GPU box back to local.
#
# Runs forever until Ctrl-C. Every POLL_INTERVAL_SEC seconds:
#   1. SSHes in and lists the chain status + recent train dirs.
#   2. rsyncs each ``artifacts/train/<run_name>/`` directory back to local.
#      Uses ``--partial --append-verify`` so interrupted transfers resume.
#   3. Prints which runs have ``.done`` / ``.failed`` markers.
#
# Env overrides:
#   LAMBDA_GPU_HOST     - IP of the GPU box. Default: 192.222.51.198.
#   LAMBDA_GPU_USER     - SSH user. Default: ubuntu.
#   POLL_INTERVAL_SEC   - Seconds between syncs. Default: 900 (15 min).
#   REMOTE_ROOT         - Remote harness dir. Default: ~/Latent_Reasoning_Project/latent_eval_training_harness.
#   SYNC_FILTER         - Extra --include/--exclude patterns (advanced).
#
# Artifacts land at: <harness_root>/artifacts/train/<run_name>/ locally.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
MONOREPO_ROOT="$(cd "${ROOT_DIR}/.." && pwd)"

LAMBDA_GPU_HOST="${LAMBDA_GPU_HOST:-192.222.51.198}"
LAMBDA_GPU_USER="${LAMBDA_GPU_USER:-ubuntu}"
POLL_INTERVAL_SEC="${POLL_INTERVAL_SEC:-900}"
REMOTE_ROOT="${REMOTE_ROOT:-/home/${LAMBDA_GPU_USER}/Latent_Reasoning_Project/latent_eval_training_harness}"
LOCAL_ARTIFACTS_ROOT="${ROOT_DIR}/artifacts/train"
LOCAL_CHAIN_LOG_DIR="${ROOT_DIR}/artifacts/chain"

mkdir -p "${LOCAL_ARTIFACTS_ROOT}" "${LOCAL_CHAIN_LOG_DIR}"

# --- Extract SSH key (multi-line PEM safe) ----------------------------------
ENV_FILE="${MONOREPO_ROOT}/.env"
if [[ ! -f "${ENV_FILE}" ]]; then
  ENV_FILE="${ROOT_DIR}/.env"
fi
if [[ ! -f "${ENV_FILE}" ]]; then
  echo "ERROR: .env not found" >&2
  exit 1
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

REMOTE_HOST="${LAMBDA_GPU_USER}@${LAMBDA_GPU_HOST}"

echo "Polling ${REMOTE_HOST}:${REMOTE_ROOT}/artifacts/train/"
echo "Local destination: ${LOCAL_ARTIFACTS_ROOT}/"
echo "Poll interval: ${POLL_INTERVAL_SEC}s"
echo "Chain status (updated in real time):"
echo ""

# Main loop. Press Ctrl-C to stop.
while :; do
  ts="[$(date -u +%Y-%m-%dT%H:%M:%SZ)]"

  # 1) Pull chain status file for at-a-glance view.
  if rsync -az -e "ssh ${SSH_OPTS[*]}" --quiet \
       "${REMOTE_HOST}:${REMOTE_ROOT}/artifacts/chain/" \
       "${LOCAL_CHAIN_LOG_DIR}/" 2>/dev/null; then
    if [[ -f "${LOCAL_CHAIN_LOG_DIR}/chain_status.txt" ]]; then
      last_line="$(tail -n 1 "${LOCAL_CHAIN_LOG_DIR}/chain_status.txt" 2>/dev/null || true)"
      echo "${ts} chain status: ${last_line}"
    fi
  fi

  # 1b) Pull probe research artifacts (dgrad CSVs, heatmaps) when any exist.
  LOCAL_PROBE_DIR="${ROOT_DIR}/../research_findings/dgrad_probe"
  mkdir -p "${LOCAL_PROBE_DIR}"
  rsync -az --partial -e "ssh ${SSH_OPTS[*]}" \
       "${REMOTE_HOST}:${REMOTE_ROOT}/research_findings/dgrad_probe/" \
       "${LOCAL_PROBE_DIR}/" 2>/dev/null \
    && echo "${ts} probe artifacts synced -> ${LOCAL_PROBE_DIR}/" \
    || true

  # 2) List train dirs on remote.
  train_dirs="$(ssh "${SSH_OPTS[@]}" "${REMOTE_HOST}" "ls -1 ${REMOTE_ROOT}/artifacts/train/ 2>/dev/null" || true)"

  if [[ -z "${train_dirs}" ]]; then
    echo "${ts} no train dirs yet; sleeping ${POLL_INTERVAL_SEC}s"
  else
    while IFS= read -r run_name; do
      [[ -z "${run_name}" ]] && continue
      # Sync the run directory. --partial lets us resume interrupted transfers.
      # Exclude raw shards / optimizer states we don't need locally (tune as needed).
      rsync -avz --partial --append-verify \
        --exclude='optimizer.pt' \
        --exclude='scheduler.pt' \
        --exclude='trainer_state.json.tmp*' \
        -e "ssh ${SSH_OPTS[*]}" \
        "${REMOTE_HOST}:${REMOTE_ROOT}/artifacts/train/${run_name}/" \
        "${LOCAL_ARTIFACTS_ROOT}/${run_name}/" \
        > "${LOCAL_CHAIN_LOG_DIR}/rsync_${run_name}.log" 2>&1 \
        && status="synced" || status="rsync_err(see rsync_${run_name}.log)"

      # Marker summary.
      marker=""
      if [[ -f "${LOCAL_ARTIFACTS_ROOT}/${run_name}/.done" ]]; then marker="DONE"; fi
      if [[ -f "${LOCAL_ARTIFACTS_ROOT}/${run_name}/.failed" ]]; then
        rc="?"
        [[ -f "${LOCAL_ARTIFACTS_ROOT}/${run_name}/.exit_code" ]] && rc="$(cat "${LOCAL_ARTIFACTS_ROOT}/${run_name}/.exit_code")"
        marker="FAILED(exit=${rc})"
      fi
      echo "${ts}   ${run_name}: ${status}${marker:+ [${marker}]}"
    done <<< "${train_dirs}"
  fi

  echo ""
  sleep "${POLL_INTERVAL_SEC}"
done
