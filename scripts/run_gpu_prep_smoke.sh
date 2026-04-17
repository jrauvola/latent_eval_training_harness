#!/usr/bin/env bash
set -euo pipefail
# Run gpu_prep_smoke eval before a full GPU sweep. Requires network + HF cache writes.
# Optional: export HF_TOKEN for gated Llama/Gemma.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

CONFIG="${CONFIG:-configs/evaluation/gpu_prep_smoke.yaml}"

if [[ -f ".env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source ".env"
  set +a
fi

if [[ -n "${HUGGINGFACE_TOKEN:-}" ]]; then
  export HF_TOKEN="${HF_TOKEN:-$HUGGINGFACE_TOKEN}"
  export HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN:-$HUGGINGFACE_TOKEN}"
fi

if [[ -d ".venv" ]]; then
  # shellcheck disable=SC1091
  source ".venv/bin/activate"
fi

export HF_HOME="${HF_HOME:-$ROOT_DIR/.cache/huggingface}"
export PYTHONPATH="${PYTHONPATH:-}:$ROOT_DIR/src"

echo "GPU prep smoke: $CONFIG"
echo "HF_HOME=$HF_HOME"
if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "WARN: HF_TOKEN unset — meta-llama / Gemma may fail gate checks."
fi

python -m latent_harness.evaluation.cli --config "$CONFIG"
echo "OK — artifacts are under runtime.output_dir in $CONFIG"
