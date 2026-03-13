#!/bin/bash
#SBATCH --job-name=base_aime2024_anchor
#SBATCH --account=bgfy-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --time=5:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --array=0-3
#SBATCH --output=/u/yliu105/MAS_UQ/log/%x_%A_%a.out
#SBATCH --error=/u/yliu105/MAS_UQ/log/%x_%A_%a.err

set -euo pipefail

cd /u/yliu105/MAS_UQ
mkdir -p /u/yliu105/MAS_UQ/log
export PYTHONUNBUFFERED=1

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

# Active runs depend on how many tasks below are uncommented:
# total_runs = 2 models x 2 prompts x num_enabled_datasets.
declare -a models
models[0]="google/gemma-3-12b-it"
models[1]="Qwen/Qwen3-8B"

declare -a prompts
prompts[0]="sequential"
prompts[1]="hierarchical"

declare -a tasks
# tasks[0]="mbppplus"
tasks[0]="aime2024"
# tasks[0]="aime2025"
# tasks[0]="humanevalplus"

max_samples="${MAX_SAMPLES:--1}"
max_new_tokens_default="${MAX_NEW_TOKENS_DEFAULT:-8192}"
max_new_tokens_code="${MAX_NEW_TOKENS_CODE:-65536}"
generate_bs_default="${GENERATE_BS_DEFAULT:-16}"

task_id="${SLURM_ARRAY_TASK_ID}"
num_tasks=${#tasks[@]}
num_prompts=${#prompts[@]}

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

current_args="--method mas --use_vllm --model_name $model --prompt $prompt --max_samples $max_samples --max_new_tokens $max_new_tokens --generate_bs $generate_bs --task $task --gpu_memory_utilization $gpu_memory_utilization --uncertainty_mode anchor --generate_only" # continuous | anchor

echo "STARTING: job=$SLURM_JOB_ID array_task=$SLURM_ARRAY_TASK_ID node=$(hostname)"
echo "MODEL: $model"
echo "PROMPT: $prompt"
echo "TASK: $task"
echo "GENERATE_BS: $generate_bs"
echo "GPU_MEMORY_UTILIZATION: $gpu_memory_utilization"
echo "ARGS: $current_args"
echo "TIME: $(date)"

python -u run.py $current_args

echo "COMPLETED: job=$SLURM_JOB_ID array_task=$SLURM_ARRAY_TASK_ID node=$(hostname) time=$(date)"
