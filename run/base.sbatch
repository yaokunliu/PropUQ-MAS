#!/bin/bash
#SBATCH --job-name=nll_4_gsm8k_role
#SBATCH --account=bgfy-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --time=5:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --array=5
#SBATCH --output=/u/yliu105/MAS_UQ/log/%x_%A_%a.out
#SBATCH --error=/u/yliu105/MAS_UQ/log/%x_%A_%a.err

set -euo pipefail

cd /u/yliu105/MAS_UQ
mkdir -p /u/yliu105/MAS_UQ/log
export PYTHONUNBUFFERED=1

if [ -f ~/.bashrc ]; then
  source ~/.bashrc
fi
cuda_version="unknown"
driver_version="unknown"
cuda_compat_action="skipped"
cuda_version_ge_12_9=0
if command -v nvidia-smi >/dev/null 2>&1; then
  cuda_version="$(nvidia-smi | sed -n 's/.*CUDA Version: \([0-9.]*\).*/\1/p' | head -n1)"
  driver_version="$(nvidia-smi | sed -n 's/.*Driver Version: \([0-9.]*\).*/\1/p' | head -n1)"
fi
if [[ "${cuda_version:-}" =~ ^[0-9]+(\.[0-9]+)*$ ]] && [ "$(printf '%s\n' "12.9" "${cuda_version}" | sort -V | head -n1)" = "12.9" ]; then
  cuda_version_ge_12_9=1
fi
if [ "${cuda_version_ge_12_9}" = "1" ]; then
  echo "Detected CUDA Version ${cuda_version} on driver ${driver_version}; skipping cuda-compat/12.9 because current support is already >= 12.9"
else
  if ! command -v module >/dev/null 2>&1; then
    echo "module command not found; cannot load cuda-compat/12.9"
    exit 1
  fi
  echo "Detected CUDA Version ${cuda_version:-unknown}; loading cuda-compat/12.9"
  module load cuda-compat/12.9
  cuda_compat_action="loaded"
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
export CUDA_COMPAT_ACTION="${cuda_compat_action}"
export CUDA_DETECTED_VERSION="${cuda_version:-unknown}"
export CUDA_PREFLIGHT_DRIVER_VERSION="${driver_version:-unknown}"
export CUDA_PREFLIGHT_NODE_NAME="$(hostname)"
CUDA_PREFLIGHT="${CUDA_PREFLIGHT:-1}"
CUDA_PREFLIGHT_REQUEUE_ON_FAIL="${CUDA_PREFLIGHT_REQUEUE_ON_FAIL:-1}"
CUDA_PREFLIGHT_MAX_REQUEUES="${CUDA_PREFLIGHT_MAX_REQUEUES:-5}"

slurm_requeue_target() {
  if [ -n "${SLURM_ARRAY_JOB_ID:-}" ] && [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
    printf '%s_%s\n' "$SLURM_ARRAY_JOB_ID" "$SLURM_ARRAY_TASK_ID"
    return
  fi
  printf '%s\n' "${SLURM_JOB_ID:-}"
}

handle_cuda_preflight_failure() {
  local restart_count max_requeues target
  restart_count="${SLURM_RESTART_COUNT:-0}"
  max_requeues="${CUDA_PREFLIGHT_MAX_REQUEUES:-5}"
  target="$(slurm_requeue_target)"

  echo "CUDA preflight: failure detected"
  echo "CUDA preflight: restart_count=${restart_count}"
  echo "CUDA preflight: max_requeues=${max_requeues}"

  if [ "${CUDA_PREFLIGHT_REQUEUE_ON_FAIL}" != "1" ]; then
    echo "CUDA preflight: auto-requeue disabled"
    return 1
  fi

  if ! command -v scontrol >/dev/null 2>&1; then
    echo "CUDA preflight: scontrol not found; cannot requeue"
    return 1
  fi

  if [ -z "${target}" ]; then
    echo "CUDA preflight: no SLURM job target found; cannot requeue"
    return 1
  fi

  if [ "${restart_count}" -ge "${max_requeues}" ]; then
    echo "CUDA preflight: requeue limit reached; not requeueing"
    return 1
  fi

  echo "CUDA preflight: requeueing ${target} (attempt $((restart_count + 1))/${max_requeues})"
  if ! scontrol requeue "${target}"; then
    echo "CUDA preflight: requeue request failed; scheduler likely disables requeue for this job/account"
    return 1
  fi
  echo "CUDA preflight: requeue request submitted"
  exit 0
}

run_cuda_preflight() {
  echo "CUDA preflight: start"
  echo "CUDA preflight: node=$(hostname)"
  echo "CUDA preflight: which python=$(command -v python)"
  python -V
  if command -v module >/dev/null 2>&1; then
    echo "CUDA preflight: module list"
    module list 2>&1 || true
  fi
  if command -v nvidia-smi >/dev/null 2>&1; then
    echo "CUDA preflight: nvidia-smi -L"
    nvidia-smi -L || true
    echo "CUDA preflight: nvidia-smi"
    nvidia-smi || true
  fi
  echo "CUDA preflight: CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
  echo "CUDA preflight: LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-<unset>}"
  echo "CUDA preflight: PATH=${PATH}"
  echo "CUDA preflight: detected_cuda_version=${CUDA_DETECTED_VERSION:-unknown}"
  echo "CUDA preflight: cuda_compat_action=${CUDA_COMPAT_ACTION:-unknown}"

  python - <<'PY'
import os
import sys

try:
    import torch
except Exception as exc:
    print(f"CUDA preflight failed: could not import torch: {exc}", file=sys.stderr)
    raise SystemExit(1)

print(f"CUDA preflight: torch={torch.__version__}")
print(f"CUDA preflight: torch.version.cuda={torch.version.cuda}")
print(f"CUDA preflight: cuda.is_available={torch.cuda.is_available()}")
print(f"CUDA preflight: detected_cuda_version={os.environ.get('CUDA_DETECTED_VERSION', 'unknown')}")
print(f"CUDA preflight: cuda_compat_action={os.environ.get('CUDA_COMPAT_ACTION', 'unknown')}")

try:
    device_count = torch.cuda.device_count()
    print(f"CUDA preflight: cuda.device_count={device_count}")
except Exception as exc:
    print(f"CUDA preflight failed during device_count(): {exc}", file=sys.stderr)
    raise SystemExit(1)

if device_count < 1:
    print("CUDA preflight failed: no CUDA devices visible", file=sys.stderr)
    raise SystemExit(1)

try:
    torch.cuda.init()
    print("CUDA preflight: torch.cuda.init() ok")
except Exception as exc:
    print(f"CUDA preflight failed during torch.cuda.init(): {exc}", file=sys.stderr)
    if "Error 803" in str(exc):
        driver_version = os.environ.get("CUDA_PREFLIGHT_DRIVER_VERSION", "unknown")
        node_name = os.environ.get("CUDA_PREFLIGHT_NODE_NAME", "unknown")
        print(
            "CUDA preflight hint: this is a node-level CUDA/driver mismatch, not a model bug.",
            file=sys.stderr,
        )
        print(
            f"CUDA preflight hint: node={node_name} driver={driver_version}. "
            "Try a different node or keep this host excluded.",
            file=sys.stderr,
        )
    raise SystemExit(1)

try:
    torch.cuda.set_device(0)
    props = torch.cuda.get_device_properties(0)
    print(f"CUDA preflight: device_0={props.name}")
    tensor = torch.zeros(1, device="cuda")
    print(f"CUDA preflight: tensor_device={tensor.device}")
except Exception as exc:
    print(f"CUDA preflight failed during CUDA tensor test: {exc}", file=sys.stderr)
    raise SystemExit(1)

print("CUDA preflight: success")
PY
}

# Configure these lists directly.
models=(
  "google/gemma-3-12b-it"
  "Qwen/Qwen3-8B"
)

topologies=(
  "chain"
  "star_convergent"
  # "star_divergent"
  "net"
  # "random"
)

tasks=(
  # "mbppplus"
  "gsm8k"
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

node_counts=(
  # "5"
  # "2"
  "4"
  # "6"
  # "8"
  # "10"
)

# Used only when topology == random.
random_edge_counts=(
  # "4"
  # "5"
  # "6"
  "7"
  # "8"
  # "9"
  # "10"
)

max_samples="${MAX_SAMPLES:--1}"
max_new_tokens_default="${MAX_NEW_TOKENS_DEFAULT:-8192}"
max_new_tokens_code="${MAX_NEW_TOKENS_CODE:-65536}"
generate_bs_default="${GENERATE_BS_DEFAULT:-16}"
generate_only="${GENERATE_ONLY:-1}"
SEED="${SEED:-42}"
uncertainty_mode_sets=(
  # "ASK4CONF"
  # "MSP"
  "NLL"
  # "LOGIT_UQ_BUNDLE"
)

declare -a spec_models
declare -a spec_topologies
declare -a spec_tasks
declare -a spec_mas_prompts
declare -a spec_node_counts
declare -a spec_random_edge_counts
declare -a spec_uncertainty_mode_sets

for model in "${models[@]}"; do
  for topology in "${topologies[@]}"; do
    for task in "${tasks[@]}"; do
      for mas_prompt in "${mas_prompts[@]}"; do
        for node_num in "${node_counts[@]}"; do
          for uncertainty_mode_set in "${uncertainty_mode_sets[@]}"; do
            if [ "$topology" = "random" ]; then
              for edge_count in "${random_edge_counts[@]}"; do
                spec_models+=("$model")
                spec_topologies+=("$topology")
                spec_tasks+=("$task")
                spec_mas_prompts+=("$mas_prompt")
                spec_node_counts+=("$node_num")
                spec_random_edge_counts+=("$edge_count")
                spec_uncertainty_mode_sets+=("$uncertainty_mode_set")
              done
            else
              spec_models+=("$model")
              spec_topologies+=("$topology")
              spec_tasks+=("$task")
              spec_mas_prompts+=("$mas_prompt")
              spec_node_counts+=("$node_num")
              spec_random_edge_counts+=("")
              spec_uncertainty_mode_sets+=("$uncertainty_mode_set")
            fi
          done
        done
      done
    done
  done
done

total_runs=${#spec_models[@]}
if [ "$total_runs" -eq 0 ]; then
  echo "No runs configured. Please set at least one model, topology, task, node count, and uncertainty_mode_set."
  exit 1
fi
task_id="${SLURM_ARRAY_TASK_ID}"
if [ "$task_id" -lt 0 ] || [ "$task_id" -ge "$total_runs" ]; then
  echo "Invalid SLURM_ARRAY_TASK_ID=$task_id; valid range is 0..$((total_runs - 1))"
  echo "Update #SBATCH --array to match the configured lists."
  exit 1
fi

model="${spec_models[$task_id]}"
topology="${spec_topologies[$task_id]}"
task="${spec_tasks[$task_id]}"
mas_prompt="${spec_mas_prompts[$task_id]}"
mas_node_num="${spec_node_counts[$task_id]}"
random_edge_count="${spec_random_edge_counts[$task_id]}"
uncertainty_mode_set="${spec_uncertainty_mode_sets[$task_id]}"

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
  --seed "$SEED"
  --force_generation
)
if [ -n "$random_edge_count" ]; then
  current_args+=(--random_edge_count "$random_edge_count")
fi
if [ "$generate_only" = "1" ]; then
  current_args+=(--generate_only)
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
echo "MAS_TOPOLOGY: $topology"
echo "MAS_NODE_NUM: $mas_node_num"
echo "MAS_PROMPT: $mas_prompt"
if [ -n "$random_edge_count" ]; then
  echo "RANDOM_EDGE_COUNT: $random_edge_count"
fi
echo "TASK: $task"
echo "UNCERTAINTY_MODES: $uncertainty_mode_set"
echo "GENERATE_BS: $generate_bs"
echo "GPU_MEMORY_UTILIZATION: $gpu_memory_utilization"
printf 'ARGS:'
printf ' %q' "${current_args[@]}"
printf '\n'
echo "TIME: $(date)"

if [ "$CUDA_PREFLIGHT" = "1" ]; then
  if ! run_cuda_preflight; then
    handle_cuda_preflight_failure
    exit 1
  fi
fi

python -u run.py "${current_args[@]}"

echo "COMPLETED: job=$SLURM_JOB_ID array_task=$SLURM_ARRAY_TASK_ID node=$(hostname) time=$(date)"
