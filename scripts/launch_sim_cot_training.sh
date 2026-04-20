#!/usr/bin/env bash
# Launch SIM-CoT Phase 2a training on GH200.
#
# This is the canonical launch wrapper for the
# ``qwen3_4b_codi_gh200_sim_cot_v2_detach`` config. Before invoking, make
# sure:
#   1. The remote worktree at ``/tmp/harness_phase2_sim_cot/`` (or the
#      equivalent dedicated path — see spec: avoid the main tree where
#      Phase 1 eval / 2b LT-Tuning may be running) is checked out on
#      ``feature/phase2-sim-cot``.
#   2. GPU utilization is <70% AND free memory is sufficient for a full-LM
#      aux decoder (~40 GB for Qwen3-4B bf16 x2 + optimizer states).
#      If memory is tight, flip ``aux_decoder_full_lm -> false`` in the
#      YAML before launching and flag the deviation in the writeup.
#   3. The Phase 1 eval tmux session (``phase1_eval_20260420T051926Z``) is
#      either finished or has a compatible GPU-sharing profile.
#
# Usage:
#   bash scripts/launch_sim_cot_training.sh
#
# To dry-run (print the command without executing):
#   DRY_RUN=1 bash scripts/launch_sim_cot_training.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HARNESS_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

CONFIG_REL="${SIM_COT_CONFIG:-configs/training/qwen3_4b_codi_gh200_sim_cot_v2_detach.yaml}"
RUN_NAME="${SIM_COT_RUN_NAME:-qwen3_4b_sim_cot_v2_detach_$(date -u +%Y%m%dT%H%M%SZ)}"
OUTPUT_DIR="${SIM_COT_OUTPUT_DIR:-artifacts/train/${RUN_NAME}}"

REMOTE_HOST="${REMOTE_HOST:-192.222.51.198}"
REMOTE_USER="${REMOTE_USER:-ubuntu}"
REMOTE_PROJECT_ROOT="${REMOTE_PROJECT_ROOT:-/home/ubuntu/Latent_Reasoning_Project/harness_phase2_sim_cot}"

CMD=(
  python3 "${HARNESS_ROOT}/scripts/launch_gh200_training_run.py"
  --config "${CONFIG_REL}"
  --run-name "${RUN_NAME}"
  --output-dir "${OUTPUT_DIR}"
  --remote-host "${REMOTE_HOST}"
  --remote-user "${REMOTE_USER}"
  --remote-project-root "${REMOTE_PROJECT_ROOT}"
)

printf '>>> Launching SIM-CoT training with:\n'
printf '    CONFIG:  %s\n' "${CONFIG_REL}"
printf '    RUN:     %s\n' "${RUN_NAME}"
printf '    OUTPUT:  %s\n' "${OUTPUT_DIR}"
printf '    REMOTE:  %s@%s:%s\n' "${REMOTE_USER}" "${REMOTE_HOST}" "${REMOTE_PROJECT_ROOT}"
printf '>>> Command: %s\n' "${CMD[*]}"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  printf '>>> DRY_RUN=1: not executing.\n'
  exit 0
fi

exec "${CMD[@]}"
