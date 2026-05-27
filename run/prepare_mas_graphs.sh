#!/bin/bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export PYTHONUNBUFFERED=1

if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate "${CONDA_ENV:-mas}"
fi

seed="${SEED:-42}"
render_svg="${RENDER_SVG:-1}"

specs=(
  "sequential role 4"
  "hierarchical role 4"
  "sequential norole 2"
  "sequential norole 4"
  "sequential norole 6"
  "sequential norole 8"
  "sequential norole 10"
  "hierarchical norole 2"
  "hierarchical norole 4"
  "hierarchical norole 6"
  "hierarchical norole 8"
  "hierarchical norole 10"
  "decentralized norole 4"
)

for spec in "${specs[@]}"; do
  read -r topology mas_prompt node_num <<< "$spec"
  current_args=(
    --method mas
    --mas_topology "$topology"
    --mas_node_num "$node_num"
    --mas_prompt "$mas_prompt"
    --seed "$seed"
    --prepare_mas_graph_only
  )
  if [ "$render_svg" = "1" ]; then
    current_args+=(--render_mas_graph_svg)
  fi

  echo "STARTING: topology=$topology node_num=$node_num mas_prompt=$mas_prompt seed=$seed"
  printf 'ARGS:'
  printf ' %q' "${current_args[@]}"
  printf '\n'

  python -u run.py "${current_args[@]}"

  echo "COMPLETED: topology=$topology node_num=$node_num seed=$seed time=$(date)"
  echo
done
