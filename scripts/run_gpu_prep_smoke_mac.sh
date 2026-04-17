#!/usr/bin/env bash
set -euo pipefail
# Laptop-friendly prep smoke (no Laura 8B). See configs/evaluation/gpu_prep_smoke_mac.yaml

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export CONFIG="${CONFIG:-configs/evaluation/gpu_prep_smoke_mac.yaml}"
exec "$ROOT_DIR/scripts/run_gpu_prep_smoke.sh"
