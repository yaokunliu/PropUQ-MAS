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
#SBATCH --array=1,3
#SBATCH --output=/u/yliu105/MAS_UQ/test/%x_%A_%a.out
#SBATCH --error=/u/yliu105/MAS_UQ/test/%x_%A_%a.err

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

# posthoc UQ test:
# 1) sequential + self_report(anchor)
# 2) sequential + self_report(continuous)
# 3) hierarchical + self_report(anchor)
# 4) hierarchical + self_report(continuous)
max_samples=16
max_new_tokens_long=8192
task="medqa"
# model="google/gemma-3-12b-it"
model="Qwen/Qwen3-8B"
# model="mistralai/Ministral-3-8B-Instruct-2512"

# 4 runs with vLLM + self_report uncertainty mode, followed by posthoc UQ.
declare -a args
args[1]="--method mas --use_vllm --model_name $model --prompt sequential --max_samples $max_samples --max_new_tokens $max_new_tokens_long --task $task --uncertainty_mode anchor"
# args[2]="--method mas --use_vllm --model_name $model --prompt sequential --max_samples $max_samples --max_new_tokens $max_new_tokens_long --task $task --uncertainty_mode continuous"
args[3]="--method mas --use_vllm --model_name $model --prompt hierarchical --max_samples $max_samples --max_new_tokens $max_new_tokens_long --task $task --uncertainty_mode anchor"
# args[4]="--method mas --use_vllm --model_name $model --prompt hierarchical --max_samples $max_samples --max_new_tokens $max_new_tokens_long --task $task --uncertainty_mode continuous"

current_args="${args[$SLURM_ARRAY_TASK_ID]}"
if [ -z "${current_args:-}" ]; then
  echo "No args configured for SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID"
  exit 1
fi

echo "STARTING: job=$SLURM_JOB_ID array_task=$SLURM_ARRAY_TASK_ID node=$(hostname)"
echo "ARGS: $current_args"
echo "TIME: $(date)"

python -u run.py $current_args

echo "COMPLETED: job=$SLURM_JOB_ID array_task=$SLURM_ARRAY_TASK_ID node=$(hostname) time=$(date)"
