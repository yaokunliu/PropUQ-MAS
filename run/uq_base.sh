#!/bin/bash

set -euo pipefail

ROOT_DIR="/u/yliu105/MAS_UQ"
cd "$ROOT_DIR"

export PYTHONUNBUFFERED=1

if [ -f ~/.bashrc ]; then
  # shellcheck disable=SC1090
  source ~/.bashrc
fi

eval "$(conda shell.bash hook)"
conda activate mas

mkdir -p "$ROOT_DIR/log" "$ROOT_DIR/preds/uq"

declare -a models
models[0]="google/gemma-3-12b-it"
models[1]="Qwen/Qwen3-8B"

declare -a prompts
prompts[0]="sequential"
prompts[1]="hierarchical"

declare -a tasks
# tasks[0]="mbppplus"
# tasks[0]="aime2024"
# tasks[0]="aime2025"
tasks[0]="humanevalplus"

max_samples="${MAX_SAMPLES:--1}"
max_new_tokens_default="${MAX_NEW_TOKENS_DEFAULT:-8192}"
max_new_tokens_code="${MAX_NEW_TOKENS_CODE:-32768}"
generate_bs_default="${GENERATE_BS_DEFAULT:-16}"
uncertainty_mode="${UNCERTAINTY_MODE:-anchor}" # continuous | anchor
split="${SPLIT:-test}"
seed="${SEED:-42}"
force_uq="${FORCE_UQ:-0}"

num_models=${#models[@]}
num_prompts=${#prompts[@]}
num_tasks=${#tasks[@]}
total_runs=$(( num_models * num_prompts * num_tasks ))

usage() {
  echo "Usage: bash run/posthoc_uq_local.sh [all|TASK_ID]"
  echo "  all     Run all combinations locally (default)"
  echo "  TASK_ID Run one combination using the same indexing as base.sbatch"
}

run_one() {
  local task_id="$1"
  local model_idx remainder prompt_idx dataset_idx
  local model prompt task max_new_tokens gpu_memory_utilization generate_bs
  local model_sanitized stem raw_preds_path current_args

  if [ "$task_id" -lt 0 ] || [ "$task_id" -ge "$total_runs" ]; then
    echo "Invalid TASK_ID=$task_id; expected 0 <= TASK_ID < $total_runs"
    exit 1
  fi

  model_idx=$(( task_id / (num_prompts * num_tasks) ))
  remainder=$(( task_id % (num_prompts * num_tasks) ))
  prompt_idx=$(( remainder / num_tasks ))
  dataset_idx=$(( remainder % num_tasks ))

  model="${models[$model_idx]}"
  prompt="${prompts[$prompt_idx]}"
  task="${tasks[$dataset_idx]}"

  if [ "$task" = "mbppplus" ] || [ "$task" = "humanevalplus" ]; then
    max_new_tokens="$max_new_tokens_code"
  else
    max_new_tokens="$max_new_tokens_default"
  fi

  if [ "$task" = "aime2024" ] || [ "$task" = "aime2025" ]; then
    gpu_memory_utilization="0.65"
    generate_bs="1"
  else
    gpu_memory_utilization="0.9"
    generate_bs="$generate_bs_default"
  fi

  model_sanitized="${model//\//_}"
  stem="mas_${model_sanitized}_${task}_${prompt}_${uncertainty_mode}_${split}_seed${seed}_n${max_samples}"
  raw_preds_path="$ROOT_DIR/preds/raw/raw_preds_${stem}.jsonl"

  if [ ! -f "$raw_preds_path" ]; then
    echo "Missing raw preds cache for TASK_ID=$task_id"
    echo "Expected: $raw_preds_path"
    echo "This script only replays posthoc UQ and will not regenerate raw preds."
    exit 1
  fi

  current_args="--method mas --use_vllm --model_name $model --prompt $prompt --max_samples $max_samples --max_new_tokens $max_new_tokens --generate_bs $generate_bs --task $task --gpu_memory_utilization $gpu_memory_utilization --uncertainty_mode $uncertainty_mode --split $split --seed $seed --raw_preds_path $raw_preds_path"
  if [ "$force_uq" = "1" ]; then
    current_args="$current_args --force_uq"
  fi

  echo "STARTING: local_task=$task_id host=$(hostname)"
  echo "MODEL: $model"
  echo "PROMPT: $prompt"
  echo "TASK: $task"
  echo "RAW_PREDS: $raw_preds_path"
  echo "ARGS: $current_args"
  echo "TIME: $(date)"

  python -u run.py $current_args

  echo "COMPLETED: local_task=$task_id host=$(hostname) time=$(date)"
}

selected="${1:-all}"

if [ "$selected" = "-h" ] || [ "$selected" = "--help" ]; then
  usage
  exit 0
fi

if [ "$selected" = "all" ]; then
  for ((task_id=0; task_id<total_runs; task_id++)); do
    run_one "$task_id"
  done
else
  run_one "$selected"
fi
