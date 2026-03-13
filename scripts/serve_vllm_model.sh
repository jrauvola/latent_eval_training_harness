#!/usr/bin/env bash
set -euo pipefail

# Serve a vLLM-compatible baseline model behind the OpenAI-compatible API.
#
# This is intended for standard-generation baseline models only.
# It is NOT suitable for CODI / latent-cot checkpoints, which require the
# custom latent inference path in the harness.
#
# Usage:
#   bash scripts/serve_vllm_model.sh gemma3-4b
#   bash scripts/serve_vllm_model.sh qwen3-4b --port 8004
#   bash scripts/serve_vllm_model.sh llama32-1b --port 8002
#   MODEL_ID=google/gemma-3-4b-it bash scripts/serve_vllm_model.sh custom

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ -d ".venv" ]]; then
  # shellcheck disable=SC1091
  source ".venv/bin/activate"
fi

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <gemma3-4b|qwen3-4b|llama32-1b|gpt2|custom> [--port PORT] [extra vllm args...]"
  exit 1
fi

MODEL_ALIAS="$1"
shift

PORT=""
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --port)
      PORT="$2"
      shift 2
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

case "$MODEL_ALIAS" in
  gemma3-4b|gemma)
    MODEL_ID="${MODEL_ID:-google/gemma-3-4b-it}"
    DEFAULT_PORT=8001
    ;;
  qwen3-4b|qwen3|qwen)
    MODEL_ID="${MODEL_ID:-Qwen/Qwen3-4B-Instruct-2507}"
    DEFAULT_PORT=8004
    ;;
  llama32-1b|llama)
    MODEL_ID="${MODEL_ID:-meta-llama/Llama-3.2-1B-Instruct}"
    DEFAULT_PORT=8002
    ;;
  gpt2)
    MODEL_ID="${MODEL_ID:-gpt2}"
    DEFAULT_PORT=8003
    ;;
  custom)
    MODEL_ID="${MODEL_ID:-}"
    DEFAULT_PORT=8004
    if [[ -z "$MODEL_ID" ]]; then
      echo "For 'custom', set MODEL_ID=..."
      exit 1
    fi
    ;;
  *)
    echo "Unknown model alias: $MODEL_ALIAS"
    exit 1
    ;;
esac

AUTO_INSTALL_VLLM_IF_MISSING="${AUTO_INSTALL_VLLM_IF_MISSING:-1}"

if ! command -v vllm >/dev/null 2>&1; then
  if [[ "${AUTO_INSTALL_VLLM_IF_MISSING}" != "1" ]]; then
    echo "vllm is not installed in the current environment."
    echo "Run: INSTALL_VLLM=1 bash scripts/setup_gpu_eval_env.sh"
    exit 1
  fi

  if [[ ! -f "scripts/setup_gpu_eval_env.sh" ]]; then
    echo "Cannot auto-install vllm: scripts/setup_gpu_eval_env.sh is missing."
    exit 1
  fi

  echo "vllm is missing; installing it into the harness environment."
  INSTALL_VLLM=1 bash scripts/setup_gpu_eval_env.sh

  if [[ -d ".venv" ]]; then
    # shellcheck disable=SC1091
    source ".venv/bin/activate"
  fi

  if ! command -v vllm >/dev/null 2>&1; then
    echo "vllm is still unavailable after installation attempt."
    exit 1
  fi
fi

PORT="${PORT:-$DEFAULT_PORT}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
DTYPE="${DTYPE:-bfloat16}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"

cat <<EOF
Starting vLLM server:
  model: $MODEL_ID
  port: $PORT
  dtype: $DTYPE
  max_model_len: $MAX_MODEL_LEN
  tensor_parallel_size: $TENSOR_PARALLEL_SIZE

OpenAI-compatible endpoint:
  http://0.0.0.0:$PORT/v1
EOF

exec vllm serve "$MODEL_ID" \
  --host 0.0.0.0 \
  --port "$PORT" \
  --dtype "$DTYPE" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --max-model-len "$MAX_MODEL_LEN" \
  --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
  "${EXTRA_ARGS[@]}"
