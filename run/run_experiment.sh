#!/bin/bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export PYTHONUNBUFFERED=1

if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate "${CONDA_ENV:-mas}"
fi

if [ -f "${HF_ENV_FILE:-$HOME/.hf_token_env}" ]; then
  # shellcheck disable=SC1090
  source "${HF_ENV_FILE:-$HOME/.hf_token_env}"
fi

model="Qwen/Qwen3-8B"
task="medqa"
topology="hierarchical"
mas_prompt="role"
node_num="4"
local_uncertainty_mode="Verb"

max_samples="${MAX_SAMPLES:--1}"
max_new_tokens="${MAX_NEW_TOKENS:-8192}"
generate_bs="${GENERATE_BS:-16}"
seed="${SEED:-42}"
split="${SPLIT:-test}"
use_vllm="${USE_VLLM:-1}"
gpu_memory_utilization="${GPU_MEMORY_UTILIZATION:-0.9}"
force_generation="${FORCE_GENERATION:-0}"
force_uq="${FORCE_UQ:-1}"

current_args=(
  --method mas
  --model_name "$model"
  --task "$task"
  --split "$split"
  --mas_topology "$topology"
  --mas_node_num "$node_num"
  --mas_prompt "$mas_prompt"
  --local_uncertainty_mode "$local_uncertainty_mode"
  --max_samples "$max_samples"
  --max_new_tokens "$max_new_tokens"
  --generate_bs "$generate_bs"
  --seed "$seed"
  --gpu_memory_utilization "$gpu_memory_utilization"
)

if [ "$use_vllm" = "1" ]; then
  current_args+=(--use_vllm)
fi
if [ "$force_generation" = "1" ]; then
  current_args+=(--force_generation)
fi
if [ "$force_uq" = "1" ]; then
  current_args+=(--force_uq)
fi

echo "MODEL: $model"
echo "TASK: $task"
echo "MAS_TOPOLOGY: $topology"
echo "MAS_NODE_NUM: $node_num"
echo "MAS_PROMPT: $mas_prompt"
echo "LOCAL_UNCERTAINTY_MODE: $local_uncertainty_mode"
printf 'ARGS:'
printf ' %q' "${current_args[@]}"
printf '\n'

python -u run.py "${current_args[@]}"
