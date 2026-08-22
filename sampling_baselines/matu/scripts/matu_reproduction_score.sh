#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MATU_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${MATU_DIR}/../.." && pwd)"

cd "${MATU_DIR}"
export PYTHONUNBUFFERED=1

if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate "${CONDA_ENV:-mas}"
fi

TASK="${TASK:-medqa}"
TASKS="${TASKS:-${TASK}}"
RAW_DIR="${RAW_DIR:-${REPO_ROOT}/outputs/ASK4CONF}"
OUTPUT_DIR="${OUTPUT_DIR:-${MATU_DIR}/results}"
EMBEDDING_DEVICE="${EMBEDDING_DEVICE:-cuda}"
EMBEDDING_BATCH_SIZE="${EMBEDDING_BATCH_SIZE:-32}"
PYTHON_BIN="${PYTHON_BIN:-python}"

for task in ${TASKS}; do
  if [[ "${task}" == "mbbpplus" ]]; then
    task="mbppplus"
  fi

  extra_args=()
  generation_time_json="${OUTPUT_DIR}/${task}/matu_${task}_generation_time_per_uncertainty_example.json"
  if [ -f "${generation_time_json}" ]; then
    extra_args+=(--generation-time-per-uncertainty-example-json "${generation_time_json}")
  fi

  "${PYTHON_BIN}" -u code/matu_reproduction.py \
    --task "${task}" \
    --raw-dir "${RAW_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --local-files-only \
    --embedding-device "${EMBEDDING_DEVICE}" \
    --embedding-batch-size "${EMBEDDING_BATCH_SIZE}" \
    "${extra_args[@]}"
done
