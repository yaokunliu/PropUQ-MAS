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

mkdir -p "$ROOT_DIR/outputs" "$ROOT_DIR/outputs/mas_graphs"

# Configure these lists directly.
models=(
  "google/gemma-3-12b-it"
  # "Qwen/Qwen3-8B"
)

topologies=(
  # "chain"
  # "star_convergent"
  # "star_divergent"
  # "net"
  "random"
)

tasks=(
  "mbppplus"
  # "gsm8k"
  # "aime2024"
  # "aime2025"
  # "gpqa"
  # "arc_easy"
  # "arc_challenge"
  # "humanevalplus"
  # "medqa"
)

mas_prompts=(
  "norole"
  # "role"
)

node_counts=(
  "5"
  # "2"
  # "4"
  # "6"
  # "8"
  # "10"
)

# Used only when topology == random.
random_edge_counts=(
  "4"
  "5"
  "6"
  "7"
  "8"
  "9"
  "10"
)

uncertainty_mode_sets=(
  "ASK4CONF"
  # "MSP"
  # "NLL"
  # "LOGIT_UQ_BUNDLE"
)

# Posthoc MAS adoption used during UQ replay.
# "original": use recorded adoption scores
# "all_one": force every incoming adoption score to 1
uq_adoption_modes=(
  # "original"
  "all_one"
)

max_samples="${MAX_SAMPLES:--1}"
max_new_tokens_default="${MAX_NEW_TOKENS_DEFAULT:-8192}"
max_new_tokens_code="${MAX_NEW_TOKENS_CODE:-32768}"
generate_bs_default="${GENERATE_BS_DEFAULT:-16}"
split="${SPLIT:-test}"
seed="${SEED:-42}"
force_uq="${FORCE_UQ:-1}"

declare -a spec_models
declare -a spec_topologies
declare -a spec_tasks
declare -a spec_mas_prompts
declare -a spec_node_counts
declare -a spec_random_edge_counts
declare -a spec_uncertainty_mode_sets
declare -a spec_uq_adoption_modes

for model in "${models[@]}"; do
  for topology in "${topologies[@]}"; do
    for task in "${tasks[@]}"; do
      for mas_prompt in "${mas_prompts[@]}"; do
        for node_num in "${node_counts[@]}"; do
          for uncertainty_mode_set in "${uncertainty_mode_sets[@]}"; do
            for uq_adoption_mode in "${uq_adoption_modes[@]}"; do
              if [ "$topology" = "random" ]; then
                for edge_count in "${random_edge_counts[@]}"; do
                  spec_models+=("$model")
                  spec_topologies+=("$topology")
                  spec_tasks+=("$task")
                  spec_mas_prompts+=("$mas_prompt")
                  spec_node_counts+=("$node_num")
                  spec_random_edge_counts+=("$edge_count")
                  spec_uncertainty_mode_sets+=("$uncertainty_mode_set")
                  spec_uq_adoption_modes+=("$uq_adoption_mode")
                done
              else
                spec_models+=("$model")
                spec_topologies+=("$topology")
                spec_tasks+=("$task")
                spec_mas_prompts+=("$mas_prompt")
                spec_node_counts+=("$node_num")
                spec_random_edge_counts+=("")
                spec_uncertainty_mode_sets+=("$uncertainty_mode_set")
                spec_uq_adoption_modes+=("$uq_adoption_mode")
              fi
            done
          done
        done
      done
    done
  done
done

total_runs=${#spec_models[@]}
if [ "$total_runs" -eq 0 ]; then
  echo "No runs configured. Please set at least one model, topology, task, node count, uncertainty_mode_set, and uq_adoption_mode."
  exit 1
fi

echo "Configured total runs: $total_runs"

normalize_uncertainty_mode_dir() {
  local mode="${1:-ASK4CONF}"
  case "$mode" in
    ""|"continuous")
      printf '%s\n' "ASK4CONF"
      ;;
    "LOGIT_UQ_BUNDLE")
      printf '%s\n' "MSP_NLL"
      ;;
    *)
      printf '%s\n' "${mode// /_}"
      ;;
  esac
}

run_one() {
  local task_id="$1"
  local model topology task mas_node_num random_edge_count
  local max_new_tokens gpu_memory_utilization generate_bs
  local stem raw_preds_path model_root uncertainty_mode_dir
  local -a current_args

  if [ "$task_id" -lt 0 ] || [ "$task_id" -ge "$total_runs" ]; then
    echo "Invalid TASK_ID=$task_id; expected 0 <= TASK_ID < $total_runs"
    exit 1
  fi

  model="${spec_models[$task_id]}"
  topology="${spec_topologies[$task_id]}"
  task="${spec_tasks[$task_id]}"
  mas_prompt="${spec_mas_prompts[$task_id]}"
  mas_node_num="${spec_node_counts[$task_id]}"
  random_edge_count="${spec_random_edge_counts[$task_id]}"
  uncertainty_mode_set="${spec_uncertainty_mode_sets[$task_id]}"
  uq_adoption_mode="${spec_uq_adoption_modes[$task_id]}"

  uncertainty_mode_dir="$(normalize_uncertainty_mode_dir "$uncertainty_mode_set")"
  model_root="$ROOT_DIR/outputs/$uncertainty_mode_dir/${model//\//\/}"

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

  stem="mas_${task}_${topology}_nodes${mas_node_num}"
  if [ -n "$random_edge_count" ]; then
    stem="${stem}_edges${random_edge_count}"
  fi
  stem="${stem}_seed${seed}_n${max_samples}"
  raw_preds_path="$model_root/raw/raw_preds_${stem}.jsonl"

  if [ ! -f "$raw_preds_path" ]; then
    echo "SKIP: missing raw preds cache for TASK_ID=$task_id"
    echo "Expected: $raw_preds_path"
    echo "This script only replays posthoc UQ and will not regenerate raw preds."
    return 0
  fi

  current_args=(
    --method mas
    --use_vllm
    --model_name "$model"
    --max_samples "$max_samples"
    --max_new_tokens "$max_new_tokens"
    --generate_bs "$generate_bs"
    --task "$task"
    --gpu_memory_utilization "$gpu_memory_utilization"
    --mas_topology "$topology"
    --mas_node_num "$mas_node_num"
    --mas_prompt "$mas_prompt"
    --split "$split"
    --seed "$seed"
    --uq_adoption_mode "$uq_adoption_mode"
  )
  IFS=',' read -r -a uncertainty_modes <<< "$uncertainty_mode_set"
  for uq_mode in "${uncertainty_modes[@]}"; do
    uq_mode="${uq_mode#"${uq_mode%%[![:space:]]*}"}"
    uq_mode="${uq_mode%"${uq_mode##*[![:space:]]}"}"
    [ -z "$uq_mode" ] && continue
    current_args+=(--uncertainty_mode "$uq_mode")
  done
  if [ -n "$random_edge_count" ]; then
    current_args+=(--random_edge_count "$random_edge_count")
  fi
  if [ "$force_uq" = "1" ]; then
    current_args+=(--force_uq)
  fi

  echo "STARTING: local_task=$task_id host=$(hostname)"
  echo "RUN_INDEX: $((task_id + 1))/$total_runs"
  echo "MODEL: $model"
  echo "MAS_TOPOLOGY: $topology"
  echo "MAS_NODE_NUM: $mas_node_num"
  echo "MAS_PROMPT: $mas_prompt"
  if [ -n "$random_edge_count" ]; then
    echo "RANDOM_EDGE_COUNT: $random_edge_count"
  fi
  echo "TASK: $task"
  echo "UNCERTAINTY_MODES: $uncertainty_mode_set"
  echo "UQ_ADOPTION_MODE: $uq_adoption_mode"
  echo "RAW_PREDS: $raw_preds_path"
  printf 'ARGS:'
  printf ' %q' "${current_args[@]}"
  printf '\n'
  echo "TIME: $(date)"

  python -u run.py "${current_args[@]}"

  echo "COMPLETED: local_task=$task_id host=$(hostname) time=$(date)"
}

for ((task_id=0; task_id<total_runs; task_id++)); do
  run_one "$task_id"
done
