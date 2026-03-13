#!/usr/bin/env bash
set -euo pipefail

# Prepare a Linux GPU machine for the latent eval harness.
#
# Usage:
#   bash scripts/setup_gpu_eval_env.sh
#   INSTALL_VLLM=1 bash scripts/setup_gpu_eval_env.sh
#
# Notes:
# - Run from the `latent_eval_training_harness/` repo root.
# - Expects Python 3.10+ to already be available on the machine.
# - If you plan to use gated models, export `HF_TOKEN` before running evals.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${VENV_DIR:-$ROOT_DIR/.venv}"
INSTALL_VLLM="${INSTALL_VLLM:-0}"

cd "$ROOT_DIR"

python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip
python -m pip install -e ".[dev]"

# bitsandbytes is optional in the harness, but useful on GPU boxes when using
# 4-bit loading paths.
python -m pip install bitsandbytes

if [[ "$INSTALL_VLLM" == "1" ]]; then
  python -m pip install vllm
fi

cat <<'EOF'

GPU eval environment is ready.

Recommended next steps:
  1. source .venv/bin/activate
  2. export HF_TOKEN=...           # needed for gated or license-protected models
  3. export ANTHROPIC_API_KEY=...  # needed for Bloom API-target evals
  4. bash scripts/run_gpu_eval_suite.sh

EOF
