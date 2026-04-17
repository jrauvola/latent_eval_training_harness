#!/usr/bin/env bash
# Remote orchestrator for the V2/V3/V4 detach-variant training chain.
#
# This script is invoked ON THE LAMBDA GPU BOX (not locally). It:
#   1. Waits for a barrier PID to exit (if set via BARRIER_PID) so the current
#      Ouro eval finishes before we touch the GPU.
#   2. Fetches and checks out the pinned git ref (FEATURE_BRANCH + FEATURE_REF).
#   3. Iterates over a list of config YAML paths and runs each sequentially via
#      ``python -m latent_harness.training.cli --config <cfg>``.
#   4. After each run, writes either a ``.done`` or ``.failed`` marker file in
#      the run's ``output_dir``, so the local rsync poller can detect completion.
#   5. Logs each run's stdout+stderr to ``<output_dir>/train.log``.
#
# Env vars (all optional unless noted):
#   BARRIER_PID      - Wait for this PID to exit before starting. Empty = skip.
#   FEATURE_BRANCH   - Git branch to check out. Default: feature/detach-variants
#   FEATURE_REF      - Git ref to pin (SHA). If set, overrides the branch tip.
#   CONFIGS          - Newline-separated list of config paths (REQUIRED).
#   ABORT_ON_FAIL    - "1" to stop the chain if any run fails. Default: "1".
#   CHAIN_LOG_DIR    - Dir for chain-level progress log. Default: artifacts/chain.

set -euo pipefail

HARNESS_DIR="${HARNESS_DIR:-$HOME/Latent_Reasoning_Project/latent_eval_training_harness}"
FEATURE_BRANCH="${FEATURE_BRANCH:-feature/detach-variants}"
FEATURE_REF="${FEATURE_REF:-}"
CHAIN_LOG_DIR="${CHAIN_LOG_DIR:-${HARNESS_DIR}/artifacts/chain}"
SMOKE_LOG_DIR="${CHAIN_LOG_DIR}/smoke"

if [[ -z "${FEATURE_REF}" ]]; then
  echo "ERROR: FEATURE_REF (expected git SHA) is required" >&2
  exit 2
fi

if [[ -z "${CONFIGS:-}" ]]; then
  echo "ERROR: CONFIGS env var is required (newline-separated list of YAML paths)"
  exit 2
fi

mkdir -p "${CHAIN_LOG_DIR}"
CHAIN_LOG="${CHAIN_LOG_DIR}/chain_$(date -u +%Y%m%dT%H%M%SZ).log"
STATUS_FILE="${CHAIN_LOG_DIR}/chain_status.txt"

# Truncate status file so the local poller knows this is a fresh chain.
: > "${STATUS_FILE}"

log() {
  local msg="[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"
  echo "${msg}" | tee -a "${CHAIN_LOG}"
}

update_status() {
  # Append a single status line. Local poller tails this file.
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" >> "${STATUS_FILE}"
}

# Runs all 6 pre-flight checks. Returns 0 if all pass, non-zero otherwise.
# Prints a human-readable report to stdout and the chain log.
run_preflight_checks() {
  log "Running pre-flight checks..."
  local failures=0

  # Check 1: Disk free >= 100 GB on /home/ubuntu
  local kb_free
  kb_free="$(df --output=avail /home/ubuntu | tail -1 | tr -d ' ')"
  local gb_free=$((kb_free / 1024 / 1024))
  if (( gb_free < 100 )); then
    log "FAIL preflight[1/6] disk: only ${gb_free}GB free, need >=100GB"
    failures=$((failures + 1))
  else
    log "ok   preflight[1/6] disk: ${gb_free}GB free"
  fi

  # Check 2: HF auth + tokenizer sanity for google/gemma-3-4b-it
  if "${HARNESS_DIR}/.venv/bin/python" -c "
from transformers import AutoTokenizer
AutoTokenizer.from_pretrained('google/gemma-3-4b-it')
" > /tmp/preflight_hf.log 2>&1; then
    log "ok   preflight[2/6] hf auth + tokenizer: gemma-3-4b-it reachable"
  else
    log "FAIL preflight[2/6] hf auth + tokenizer: see /tmp/preflight_hf.log"
    failures=$((failures + 1))
  fi

  # Check 3: Harness imports from venv
  if "${HARNESS_DIR}/.venv/bin/python" -c "
import sys; sys.path.insert(0, '${HARNESS_DIR}/src')
from latent_harness.core.runtime import _apply_boundary_detach, _resolve_should_detach, _detach_cache
" > /tmp/preflight_harness.log 2>&1; then
    log "ok   preflight[3/6] harness imports: runtime helpers resolvable"
  else
    log "FAIL preflight[3/6] harness imports: see /tmp/preflight_harness.log"
    failures=$((failures + 1))
  fi

  # Check 4: Dry-run loader for each queued config
  local cfg bad_cfgs=""
  for cfg in "${CONFIG_LIST[@]}"; do
    if ! "${HARNESS_DIR}/.venv/bin/python" -c "
import sys; sys.path.insert(0, '${HARNESS_DIR}/src')
import yaml
from latent_harness.training.config import TrainingConfig
TrainingConfig.from_dict(yaml.safe_load(open('${HARNESS_DIR}/${cfg}')))
" > /tmp/preflight_loader.log 2>&1; then
      bad_cfgs="${bad_cfgs}${cfg} "
    fi
  done
  if [[ -z "${bad_cfgs}" ]]; then
    log "ok   preflight[4/6] dry-run loader: ${#CONFIG_LIST[@]} configs parse"
  else
    log "FAIL preflight[4/6] dry-run loader fail on: ${bad_cfgs}(see /tmp/preflight_loader.log)"
    failures=$((failures + 1))
  fi

  # Check 5: Git working tree clean
  cd "${HARNESS_DIR}"
  if [[ -z "$(git status --porcelain)" ]]; then
    log "ok   preflight[5/6] git clean: no uncommitted changes"
  else
    log "FAIL preflight[5/6] git dirty:"
    git status --short | sed 's/^/       /' | tee -a "${CHAIN_LOG}"
    failures=$((failures + 1))
  fi

  # Check 6: git HEAD == FEATURE_REF (exact SHA pin)
  local head_sha
  head_sha="$(git rev-parse HEAD)"
  if [[ "${head_sha}" == "${FEATURE_REF}" ]]; then
    log "ok   preflight[6/6] git HEAD == FEATURE_REF (${FEATURE_REF})"
  else
    log "FAIL preflight[6/6] git HEAD (${head_sha}) != FEATURE_REF (${FEATURE_REF})"
    failures=$((failures + 1))
  fi

  if (( failures > 0 )); then
    log "pre-flight: ${failures} check(s) failed; aborting chain"
    update_status "preflight_failed: ${failures}"
    return 1
  fi
  log "pre-flight: all 6 checks passed"
  update_status "preflight_passed"
  return 0
}

# -------- Phase 0: wait for barrier PID (the current Ouro eval) -------------
if [[ -n "${BARRIER_PID:-}" ]]; then
  log "Waiting for BARRIER_PID=${BARRIER_PID} to exit..."
  update_status "waiting: BARRIER_PID=${BARRIER_PID}"
  while kill -0 "${BARRIER_PID}" 2>/dev/null; do
    sleep 30
  done
  log "Barrier PID ${BARRIER_PID} has exited. Proceeding."
fi

# -------- Phase 1: sync git to the pinned ref -------------------------------
cd "${HARNESS_DIR}"
log "Fetching origin and checking out ${FEATURE_BRANCH}${FEATURE_REF:+ @ ${FEATURE_REF}}"
git fetch origin --prune
git checkout "${FEATURE_BRANCH}"
git pull --ff-only origin "${FEATURE_BRANCH}"
if [[ -n "${FEATURE_REF}" ]]; then
  git checkout "${FEATURE_REF}"
fi
REMOTE_HEAD="$(git rev-parse HEAD)"
log "Remote HEAD pinned at ${REMOTE_HEAD}"
update_status "git_checked_out: ${REMOTE_HEAD}"

# -------- Phase 2: iterate over configs --------------------------------------
# Parse CONFIGS as a newline-separated list (preserves ordering).
mapfile -t CONFIG_LIST < <(printf '%s\n' "${CONFIGS}" | sed '/^$/d')

log "Chain begin: ${#CONFIG_LIST[@]} config(s) queued"
for cfg_path in "${CONFIG_LIST[@]}"; do
  log "========================================================"
  log "Starting run: ${cfg_path}"

  if [[ ! -f "${HARNESS_DIR}/${cfg_path}" ]]; then
    log "FAIL (missing config): ${cfg_path}"
    update_status "missing_config: ${cfg_path}"
    [[ "${ABORT_ON_FAIL}" == "1" ]] && exit 3
    continue
  fi

  # Extract output_dir from the config for marker placement.
  out_dir=$(python3 -c "
import yaml, os, sys
with open(sys.argv[1]) as f:
    cfg = yaml.safe_load(f)
print(cfg.get('trainer', {}).get('output_dir', ''))
" "${HARNESS_DIR}/${cfg_path}")
  if [[ -z "${out_dir}" ]]; then
    log "FAIL (no output_dir in config): ${cfg_path}"
    update_status "no_output_dir: ${cfg_path}"
    [[ "${ABORT_ON_FAIL}" == "1" ]] && exit 4
    continue
  fi

  mkdir -p "${HARNESS_DIR}/${out_dir}"
  run_log="${HARNESS_DIR}/${out_dir}/train.log"

  update_status "running: ${cfg_path} -> ${out_dir}"
  log "output_dir=${out_dir}"
  log "log file: ${run_log}"

  # Run training. Use the harness's own CLI. PYTHONPATH=src is required because
  # the package uses a src/ layout (matches run_lambda_codi_train.sh convention).
  set +e
  PYTHONPATH=src \
  PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
    "${HARNESS_DIR}/.venv/bin/python" \
    -m latent_harness.training.cli --config "${cfg_path}" \
    > "${run_log}" 2>&1
  rc=$?
  set -e

  if [[ ${rc} -eq 0 ]]; then
    touch "${HARNESS_DIR}/${out_dir}/.done"
    log "DONE: ${cfg_path} (exit=0)"
    update_status "done: ${cfg_path}"
  else
    touch "${HARNESS_DIR}/${out_dir}/.failed"
    echo "${rc}" > "${HARNESS_DIR}/${out_dir}/.exit_code"
    log "FAILED: ${cfg_path} (exit=${rc})"
    update_status "failed: ${cfg_path} exit=${rc}"
    if [[ "${ABORT_ON_FAIL}" == "1" ]]; then
      log "ABORT_ON_FAIL=1; stopping chain."
      exit "${rc}"
    fi
  fi
done

log "Chain complete."
update_status "chain_complete"
