#!/usr/bin/env bash
# Launch Qwen3-4B LT-Tuning training on the GH200.
#
# Do NOT run this while Phase 1 eval or the cartography probe are using >70%
# of the GPU. This script is a no-op when $DRY_RUN=1, which is the default.
#
# Usage (from remote worktree on the GH200):
#   DRY_RUN=0 bash scripts/run_lt_tuning_qwen3_4b.sh
#
# HF upload target (set after training):
#   jrauvola/qwen3-4b-codi-lt-tuning-bf16-v2detach (private)

set -euo pipefail

CONFIG="${CONFIG:-configs/training/qwen3_4b_codi_gh200_lt_tuning_v2_detach.yaml}"
LOG_DIR="${LOG_DIR:-artifacts/train/qwen3_4b_codi_gh200_lt_tuning_v2_detach/logs}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOG_FILE="${LOG_DIR}/run_${STAMP}.log"
DRY_RUN="${DRY_RUN:-1}"

mkdir -p "${LOG_DIR}"

echo "[lt-tuning] config=${CONFIG}"
echo "[lt-tuning] log=${LOG_FILE}"
echo "[lt-tuning] DRY_RUN=${DRY_RUN}"

if command -v nvidia-smi >/dev/null 2>&1; then
    echo "[lt-tuning] GPU snapshot:"
    nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv
fi

if [[ "${DRY_RUN}" == "1" ]]; then
    echo "[lt-tuning] DRY_RUN=1 — NOT launching. Re-run with DRY_RUN=0 once GPU is free."
    echo "[lt-tuning] Command that would run:"
    echo "  python -m latent_harness.training.cli --config${CONFIG} 2>&1 | tee ${LOG_FILE}"
    exit 0
fi

# Real launch path.
cd "$(dirname "${BASH_SOURCE[0]}")/.."

python -m latent_harness.training.cli --config"${CONFIG}" 2>&1 | tee "${LOG_FILE}"

# Once training completes, push final checkpoint to HF.
#   python -m latent_harness.serving.upload \
#     --source artifacts/train/qwen3_4b_codi_gh200_lt_tuning_v2_detach \
#     --repo jrauvola/qwen3-4b-codi-lt-tuning-bf16-v2detach \
#     --private
