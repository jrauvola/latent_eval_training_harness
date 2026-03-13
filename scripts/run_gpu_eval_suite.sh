#!/usr/bin/env bash
set -euo pipefail

# Run the combined canonical + P3 + Gemma evaluation suite on a GPU box.
#
# Usage:
#   bash scripts/run_gpu_eval_suite.sh
#   CONFIG=configs/evaluation/broader_suite_plus_p3_gemma3_gh200.yaml SPLITS=2 bash scripts/run_gpu_eval_suite.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

CONFIG="${CONFIG:-configs/evaluation/broader_suite_plus_p3_gemma3_gh200.yaml}"
SPLITS="${SPLITS:-1}"
OUTPUT_SUFFIX="${OUTPUT_SUFFIX:-gpu}"

if [[ -f ".env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source ".env"
  set +a
fi

if [[ -d ".venv" ]]; then
  # shellcheck disable=SC1091
  source ".venv/bin/activate"
fi

export HF_HOME="${HF_HOME:-$ROOT_DIR/.cache/huggingface}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "Running eval config: $CONFIG"
echo "Using HF_HOME: $HF_HOME"
echo "Using splits: $SPLITS"

if [[ "$SPLITS" -gt 1 ]]; then
  python scripts/run_parallel_eval.py \
    --config "$CONFIG" \
    --splits "$SPLITS" \
    --output-suffix "$OUTPUT_SUFFIX"
else
  PYTHONPATH=src python -m latent_harness.evaluation.cli --config "$CONFIG"
fi
