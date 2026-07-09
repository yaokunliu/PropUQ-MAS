#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UPROP_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${UPROP_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONUNBUFFERED=1

if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate "${CONDA_ENV:-mas}"
fi

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-8B}"
SEEDS="${SEEDS:-42 43 44 45 46 47 48 49 50 51}"
MAX_SAMPLES="${MAX_SAMPLES:--1}"
TEMPERATURE="${TEMPERATURE:-0.8}"
TOP_P="${TOP_P:-0.95}"
GENERATE_BS="${GENERATE_BS:-16}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
USE_VLLM="${USE_VLLM:-1}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"
RUNTIME_DIR="${RUNTIME_DIR:-${REPO_ROOT}/baseline/uprop/results/medqa/generation_runtime}"
PYTHON_BIN="${PYTHON_BIN:-python}"
mkdir -p "${RUNTIME_DIR}"

read -r -a seed_array <<< "${SEEDS}"
if [ "${#seed_array[@]}" -ne 10 ]; then
  echo "Strict UProp reproduction requires exactly 10 seeds; got ${#seed_array[@]}: ${SEEDS}" >&2
  exit 1
fi

for seed in "${seed_array[@]}"; do
  extra_args=()
  if [[ "${USE_VLLM}" == "1" ]]; then
    extra_args+=(--use_vllm --tensor_parallel_size "${TENSOR_PARALLEL_SIZE}" --gpu_memory_utilization "${GPU_MEMORY_UTILIZATION}")
  fi
  run_log="$(mktemp)"
  runtime_json="${RUNTIME_DIR}/uprop_medqa_seed${seed}.json"
  "${PYTHON_BIN}" -u run.py \
    --method mas \
    --model_name "${MODEL_NAME}" \
    --task medqa \
    --mas_topology sequential \
    --mas_prompt role \
    --mas_node_num 4 \
    --local_uncertainty_mode MSP \
    --seed "${seed}" \
    --max_samples "${MAX_SAMPLES}" \
    --temperature "${TEMPERATURE}" \
    --top_p "${TOP_P}" \
    --generate_bs "${GENERATE_BS}" \
    --max_new_tokens "${MAX_NEW_TOKENS}" \
    --generate_only \
    --force_generation \
    "${extra_args[@]}" \
    2>&1 | tee "${run_log}"
  "${PYTHON_BIN}" - "${run_log}" "${runtime_json}" "${seed}" <<'PY'
import json
import sys
from pathlib import Path

log_path = Path(sys.argv[1])
runtime_path = Path(sys.argv[2])
seed = int(sys.argv[3])
summary = None
for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
    line = line.strip()
    if not line.startswith("{"):
        continue
    try:
        obj = json.loads(line)
    except json.JSONDecodeError:
        continue
    if "total_time_sec" in obj:
        summary = obj
if summary is None:
    raise SystemExit(f"No runtime JSON with total_time_sec found in {log_path}")
summary["seed"] = seed
summary.pop("runtime_log", None)
runtime_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
PY
  rm -f "${run_log}"
done
