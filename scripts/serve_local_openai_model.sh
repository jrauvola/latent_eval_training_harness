#!/usr/bin/env bash
set -euo pipefail

# Serve a named harness evaluation model behind a local OpenAI-compatible API.
#
# Usage:
#   bash scripts/serve_local_openai_model.sh codi_gpt2_official --port 8101
#   bash scripts/serve_local_openai_model.sh codi_llama32_1b_official --port 8102

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <model-name> [--config PATH] [--port PORT] [--device DEVICE] [--external-model-id ID]"
  exit 1
fi

MODEL_NAME="$1"
shift

CONFIG="configs/evaluation/broader_suite_plus_p3_gemma3_gh200.yaml"
PORT="8101"
DEVICE="auto"
EXTERNAL_MODEL_ID=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      CONFIG="$2"
      shift 2
      ;;
    --port)
      PORT="$2"
      shift 2
      ;;
    --device)
      DEVICE="$2"
      shift 2
      ;;
    --external-model-id)
      EXTERNAL_MODEL_ID="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1"
      exit 1
      ;;
  esac
done

if [[ -d ".venv" ]]; then
  # shellcheck disable=SC1091
  source ".venv/bin/activate"
fi

export PYTHONPATH="${PYTHONPATH:-$ROOT_DIR/src}"

CMD=(
  python -m latent_harness.serving.cli
  --config "$CONFIG"
  --model-name "$MODEL_NAME"
  --port "$PORT"
  --device "$DEVICE"
)

if [[ -n "$EXTERNAL_MODEL_ID" ]]; then
  CMD+=(--external-model-id "$EXTERNAL_MODEL_ID")
fi

exec "${CMD[@]}"
