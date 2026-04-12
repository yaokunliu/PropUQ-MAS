#!/bin/bash
#SBATCH --job-name=test
#SBATCH --account=bgfy-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --time=0:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --array=0
#SBATCH --output=/u/yliu105/MAS_UQ/test/test_%x_%A_%a.out
#SBATCH --error=/u/yliu105/MAS_UQ/test/test_%x_%A_%a.err

set -euo pipefail

cd /u/yliu105/MAS_UQ
mkdir -p /u/yliu105/MAS_UQ/test
export PYTHONUNBUFFERED=1

# Optional: load shell env/conda env if your cluster setup needs it.
if [ -f ~/.bashrc ]; then
  source ~/.bashrc
fi
if ! command -v module >/dev/null 2>&1; then
  echo "module command not found; cannot load cuda-compat/12.9"
  exit 1
fi
module load cuda-compat/12.9
if command -v nvidia-smi >/dev/null 2>&1; then
  cuda_version="$(nvidia-smi | sed -n 's/.*CUDA Version: \([0-9.]*\).*/\1/p' | head -n1)"
  if [[ "${cuda_version:-}" != 12.9* ]]; then
    echo "Expected CUDA Version: 12.9, got: ${cuda_version:-unknown}"
    exit 1
  fi
fi
eval "$(conda shell.bash hook)"
conda activate mas
HF_ENV_FILE="${HF_ENV_FILE:-$HOME/.hf_token_env}"
if [ -f "$HF_ENV_FILE" ]; then
  # shellcheck disable=SC1090
  source "$HF_ENV_FILE"
else
  echo "HF token env file not found: $HF_ENV_FILE"
  exit 1
fi

if [ -z "${HUGGINGFACE_HUB_TOKEN:-${HF_TOKEN:-}}" ]; then
  echo "HF token missing. Set HUGGINGFACE_HUB_TOKEN or HF_TOKEN in $HF_ENV_FILE"
  exit 1
fi

export HF_HOME="${HF_HOME:-/u/yliu105/.cache/huggingface}"
export HUGGINGFACE_HUB_TOKEN="${HUGGINGFACE_HUB_TOKEN:-$HF_TOKEN}"
export HF_TOKEN="${HF_TOKEN:-$HUGGINGFACE_HUB_TOKEN}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

max_samples=16
max_new_tokens_long=8192
model="Qwen/Qwen3-8B" # Qwen/Qwen3-8B | google/gemma-3-12b-it
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
  # "norole"
  "role"
)
topologies=(
  # "chain"
  "star_convergent"
  # "star_divergent"
  # "tree"
  # "net"
  # "random"
)
node_counts=(
  "4"
)
uncertainty_mode_sets=(
  "ASK4CONF"
  # "LOGIT_UQ_BUNDLE"
  # "MSP"
  # "NLL"
)

declare -a spec_topologies
declare -a spec_tasks
declare -a spec_node_counts
declare -a spec_mas_prompts
declare -a spec_uncertainty_mode_sets

for topology in "${topologies[@]}"; do
  for task in "${tasks[@]}"; do
    for mas_prompt in "${mas_prompts[@]}"; do
      for node_num in "${node_counts[@]}"; do
        for uncertainty_mode_set in "${uncertainty_mode_sets[@]}"; do
          spec_topologies+=("$topology")
          spec_tasks+=("$task")
          spec_mas_prompts+=("$mas_prompt")
          spec_node_counts+=("$node_num")
          spec_uncertainty_mode_sets+=("$uncertainty_mode_set")
        done
      done
    done
  done
done

total_runs=${#spec_topologies[@]}
if [ "$total_runs" -eq 0 ]; then
  echo "No runs configured. Please set at least one task, topology, node count, and uncertainty_mode_set."
  exit 1
fi
task_id="${SLURM_ARRAY_TASK_ID}"
if [ "$task_id" -lt 0 ] || [ "$task_id" -ge "$total_runs" ]; then
  echo "Invalid SLURM_ARRAY_TASK_ID=$task_id; valid range is 0..$((total_runs - 1))"
  echo "Update #SBATCH --array to match the configured lists."
  exit 1
fi

mas_topology="${spec_topologies[$task_id]}"
task="${spec_tasks[$task_id]}"
mas_prompt="${spec_mas_prompts[$task_id]}"
mas_node_num="${spec_node_counts[$task_id]}"
uncertainty_mode_set="${spec_uncertainty_mode_sets[$task_id]}"

current_args=(
  --method mas
  --use_vllm
  --model_name "$model"
  --max_samples "$max_samples"
  --max_new_tokens "$max_new_tokens_long"
  --task "$task"
  --mas_topology "$mas_topology"
  --mas_node_num "$mas_node_num"
  --mas_prompt "$mas_prompt"
  --force_generation
)
if [ "$mas_topology" = "random" ]; then
  current_args+=(--random_edge_count 7)
fi
IFS=',' read -r -a uncertainty_modes <<< "$uncertainty_mode_set"
for uq_mode in "${uncertainty_modes[@]}"; do
  uq_mode="${uq_mode#"${uq_mode%%[![:space:]]*}"}"
  uq_mode="${uq_mode%"${uq_mode##*[![:space:]]}"}"
  [ -z "$uq_mode" ] && continue
  current_args+=(--uncertainty_mode "$uq_mode")
done

echo "STARTING: job=$SLURM_JOB_ID array_task=$SLURM_ARRAY_TASK_ID node=$(hostname)"
echo "MODEL: $model"
echo "TASK: $task"
echo "MAS_TOPOLOGY: $mas_topology"
echo "MAS_NODE_NUM: $mas_node_num"
echo "MAS_PROMPT: $mas_prompt"
echo "UNCERTAINTY_MODES: $uncertainty_mode_set"
printf 'ARGS:'
printf ' %q' "${current_args[@]}"
printf '\n'
echo "TIME: $(date)"

python -u run.py "${current_args[@]}"

echo "COMPLETED: job=$SLURM_JOB_ID array_task=$SLURM_ARRAY_TASK_ID node=$(hostname) time=$(date)"
