#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UPROP_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${UPROP_DIR}/../.." && pwd)"

RAW_DIR="${RAW_DIR:-${REPO_ROOT}/outputs/MSP}"
OUTPUT_DIR="${OUTPUT_DIR:-${UPROP_DIR}/results}"
GENERATION_RUNTIME_PATH="${GENERATION_RUNTIME_PATH:-${OUTPUT_DIR}/medqa/generation_runtime}"
PYTHON_BIN="${PYTHON_BIN:-python}"

if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate "${CONDA_ENV:-mas}"
fi

cd "${UPROP_DIR}"

"${PYTHON_BIN}" code/uprop_reproduction.py \
  --raw-dir "${RAW_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --generation-runtime-path "${GENERATION_RUNTIME_PATH}"
