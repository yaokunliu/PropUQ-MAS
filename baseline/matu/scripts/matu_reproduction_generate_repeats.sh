#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MATU_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${MATU_DIR}/../.." && pwd)"

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-8B}"
RUN_ROOT="${RUN_ROOT:-${REPO_ROOT}}"
TASK="${TASK:-medqa}"
TASKS="${TASKS:-${TASK}}"
SEEDS="${SEEDS:-42 43 44 45 46 47 48 49 50 51}"
MAX_SAMPLES="${MAX_SAMPLES:--1}"
TEMPERATURE="${TEMPERATURE:-0.9}"
GENERATE_BS="${GENERATE_BS:-16}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-4096}"
USE_VLLM="${USE_VLLM:-1}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"
PYTHON_BIN="${PYTHON_BIN:-python}"

cd "${RUN_ROOT}"
export PYTHONUNBUFFERED=1

if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate "${CONDA_ENV:-mas}"
fi

read -r -a seed_array <<< "${SEEDS}"
if [ "${#seed_array[@]}" -ne 10 ]; then
  echo "Strict MATU reproduction requires exactly 10 seeds; got ${#seed_array[@]}: ${SEEDS}" >&2
  exit 1
fi

runner_topology() {
  local topology="$1"
  if [[ "${LEGACY_TOPOLOGY_NAMES:-0}" == "1" ]]; then
    case "${topology}" in
      sequential) echo "chain" ;;
      hierarchical) echo "star_convergent" ;;
      *) echo "${topology}" ;;
    esac
  else
    echo "${topology}"
  fi
}

run_one() {
  local task="$1"
  local topology="$2"
  local prompt="$3"
  local seed="$4"
  local runner_topology_name
  runner_topology_name="$(runner_topology "${topology}")"

  local extra_args=()
  if [[ "${USE_VLLM}" == "1" ]]; then
    extra_args+=(--use_vllm --tensor_parallel_size "${TENSOR_PARALLEL_SIZE}" --gpu_memory_utilization "${GPU_MEMORY_UTILIZATION}")
  fi

  "${PYTHON_BIN}" -u "${RUN_ROOT}/run.py" \
    --method mas \
    --model_name "${MODEL_NAME}" \
    --task "${task}" \
    --mas_topology "${runner_topology_name}" \
    --mas_prompt "${prompt}" \
    --mas_node_num 4 \
    --uncertainty_mode ASK4CONF \
    --seed "${seed}" \
    --max_samples "${MAX_SAMPLES}" \
    --temperature "${TEMPERATURE}" \
    --generate_bs "${GENERATE_BS}" \
    --max_new_tokens "${MAX_NEW_TOKENS}" \
    --generate_only \
    "${extra_args[@]}"
}

for task in ${TASKS}; do
  if [[ "${task}" == "mbbpplus" ]]; then
    task="mbppplus"
  fi
  for seed in "${seed_array[@]}"; do
    run_one "${task}" sequential role "${seed}"
    run_one "${task}" hierarchical role "${seed}"
  done
done
