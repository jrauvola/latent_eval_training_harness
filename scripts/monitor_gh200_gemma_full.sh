#!/usr/bin/env bash
# Poll remote GH200 training until the latent_harness.training.cli process exits.
# Logs to artifacts/gh200_gemma3_4b_codi_full_monitor.log (random 25–95s between polls).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HARNESS_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
MONO_ROOT="$(cd "${HARNESS_ROOT}/.." && pwd)"
LOG_FILE="${HARNESS_ROOT}/artifacts/gh200_gemma3_4b_codi_full_monitor.log"
RESOLVER="${SCRIPT_DIR}/extract_spar_lambda_ssh_key.py"

REMOTE_HOST="${GH200_MONITOR_HOST:-192.222.51.198}"
REMOTE_USER="${GH200_MONITOR_USER:-ubuntu}"
REMOTE_CONFIG="${GH200_REMOTE_CONFIG:-configs/training/gemma3_4b_codi_gh200.yaml}"
REMOTE_PID_FILE="${GH200_REMOTE_PID_FILE:-/tmp/gh200_runs/gemma3_4b_codi_gh200.pid}"
REMOTE_TRAIN_LOG="${GH200_REMOTE_TRAIN_LOG:-/tmp/gh200_runs/gemma3_4b_codi_gh200.log}"
REMOTE_ARTIFACTS="${GH200_REMOTE_ARTIFACTS:-/home/ubuntu/Latent_Reasoning_Project/latent_eval_training_harness/artifacts/train/gemma3_4b_codi_gh200}"

ENV_FILE="${MONO_ROOT}/.env"
if [[ ! -f "${ENV_FILE}" ]]; then
  ENV_FILE="${HARNESS_ROOT}/.env"
fi

KEY_PATH="$(python3 "${RESOLVER}" --env-file "${ENV_FILE}")"
SSH_BASE=(ssh -i "${KEY_PATH}" -o StrictHostKeyChecking=no "${REMOTE_USER}@${REMOTE_HOST}")

mkdir -p "$(dirname "${LOG_FILE}")"

log() {
  # shellcheck disable=SC2329
  printf '%s\n' "$*" | tee -a "${LOG_FILE}"
}

remote_state() {
  # RUNNING | STOPPED | SSH_ERROR
  pid="$(remote_pid)"
  if [[ "${pid}" == "SSH_ERROR" ]]; then
    echo SSH_ERROR
    return
  fi
  if [[ -n "${pid}" ]]; then
    echo RUNNING
  else
    echo STOPPED
  fi
}

remote_pid() {
  output="$("${SSH_BASE[@]}" "bash -lc 'if [[ -f \"${REMOTE_PID_FILE}\" ]]; then pid=\$(<\"${REMOTE_PID_FILE}\"); if [[ -n \"\${pid}\" ]] && kill -0 \"\${pid}\" 2>/dev/null; then printf \"%s\\n\" \"\${pid}\"; fi; fi'" 2>/dev/null)" || {
    echo SSH_ERROR
    return
  }
  printf '%s\n' "${output}"
}

poll_round=0
log "=== monitor start $(date -u +%Y-%m-%dT%H:%M:%SZ) host=${REMOTE_HOST} ==="
log "remote_config=${REMOTE_CONFIG} remote_pid_file=${REMOTE_PID_FILE} remote_train_log=${REMOTE_TRAIN_LOG} remote_output_dir=${REMOTE_ARTIFACTS}"

while true; do
  poll_round=$((poll_round + 1))
  sleep $((RANDOM % 71 + 25))

  state="$(remote_state)"

  log ""
  log "=== poll #${poll_round} $(date -u +%Y-%m-%dT%H:%M:%SZ) state=${state} ==="
  if [[ "${state}" == "SSH_ERROR" ]]; then
    log "(ssh unreachable; will retry next poll)"
    continue
  fi

  pid="$(remote_pid)"
  log "remote_pid=${pid:-missing}"
  "${SSH_BASE[@]}" "bash -lc 'if [[ -n \"${pid}\" ]]; then echo \"--- process\"; ps -p ${pid} -o pid,etime,%cpu,%mem,rss,vsz,state,comm,args= 2>&1; echo \"--- network\"; lsof -a -p ${pid} -i -n -P 2>&1 | sed -n \"1,12p\"; fi; echo \"--- train-log-meta\"; ls -l \"${REMOTE_TRAIN_LOG}\" 2>&1; echo \"--- tail ${REMOTE_TRAIN_LOG}\"; tail -n 25 \"${REMOTE_TRAIN_LOG}\" 2>&1; echo \"--- gpu\"; nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu,utilization.memory --format=csv,noheader 2>&1'" >> "${LOG_FILE}" 2>&1 || log "(ssh tail failed)"

  if [[ "${state}" == "STOPPED" ]]; then
    log "--- training process not found; final snapshots ---"
    "${SSH_BASE[@]}" "ls -la '${REMOTE_ARTIFACTS}' 2>&1; echo '---'; tail -n 120 '${REMOTE_TRAIN_LOG}' 2>&1" >> "${LOG_FILE}" 2>&1 || true
    log "=== monitor end $(date -u +%Y-%m-%dT%H:%M:%SZ) (STOPPED) ==="
    exit 0
  fi
done
