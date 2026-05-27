#!/bin/bash
#SBATCH --job-name=propuq_test
#SBATCH --account=bgfy-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --output=log/%x_%A.out
#SBATCH --error=log/%x_%A.err

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
mkdir -p "$ROOT_DIR/log"

export PYTHONUNBUFFERED=1
export CONDA_ENV="${CONDA_ENV:-mas}"
export HF_ENV_FILE="${HF_ENV_FILE:-$HOME/.hf_token_env}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if [ -f ~/.bashrc ]; then
  # shellcheck disable=SC1090
  source ~/.bashrc
fi

if [ -f "$HF_ENV_FILE" ]; then
  # shellcheck disable=SC1090
  source "$HF_ENV_FILE"
fi

if [ -n "${HF_TOKEN:-}" ] && [ -z "${HUGGINGFACE_HUB_TOKEN:-}" ]; then
  export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"
fi
if [ -n "${HUGGINGFACE_HUB_TOKEN:-}" ] && [ -z "${HF_TOKEN:-}" ]; then
  export HF_TOKEN="$HUGGINGFACE_HUB_TOKEN"
fi

export MAX_SAMPLES="${MAX_SAMPLES:-10}"
export USE_VLLM="${USE_VLLM:-1}"
export FORCE_GENERATION="${FORCE_GENERATION:-0}"
export FORCE_UQ="${FORCE_UQ:-1}"

echo "STARTING: job=$SLURM_JOB_ID node=$(hostname)"
echo "ROOT_DIR: $ROOT_DIR"
echo "MAX_SAMPLES: $MAX_SAMPLES"
echo "USE_VLLM: $USE_VLLM"
echo "FORCE_GENERATION: $FORCE_GENERATION"
echo "FORCE_UQ: $FORCE_UQ"
echo "TIME: $(date)"

bash run/run_experiment.sh

echo "COMPLETED: job=$SLURM_JOB_ID node=$(hostname) time=$(date)"
