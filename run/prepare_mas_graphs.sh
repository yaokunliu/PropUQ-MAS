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

seed="${SEED:-42}"
render_svg="${RENDER_SVG:-1}"

# Edit these lists directly to choose which topologies and node counts to prepare.
topologies=(${TOPOLOGIES:-net}) # chain star_convergent star_divergent net random
node_counts=(${NODE_COUNTS:-4}) # 2 4 6 8 10
# node_counts=(${NODE_COUNTS:-5})
random_edge_counts=(${RANDOM_EDGE_COUNTS:-4 5 6 7 8 9 10})
mas_prompts=(${MAS_PROMPTS:-norole}) # role

# Graph artifacts depend only on topology/node count, not prompt style.
# We still pass one configurable prompt value for CLI consistency.
mas_prompt="${mas_prompts[0]:-norole}"

for topology in "${topologies[@]}"; do
  for node_num in "${node_counts[@]}"; do
    if [ "$topology" = "random" ]; then
      for edge_count in "${random_edge_counts[@]}"; do
        current_args="--method mas --mas_topology $topology --mas_node_num $node_num --mas_prompt $mas_prompt --random_edge_count $edge_count --seed $seed --prepare_mas_graph_only"
        if [ "$render_svg" = "1" ]; then
          current_args="$current_args --render_mas_graph_svg"
        fi

        echo "STARTING: topology=$topology node_num=$node_num mas_prompt=$mas_prompt random_edge_count=$edge_count seed=$seed host=$(hostname)"
        echo "ARGS: $current_args"
        echo "TIME: $(date)"

        python -u run.py $current_args

        echo "COMPLETED: topology=$topology node_num=$node_num random_edge_count=$edge_count seed=$seed time=$(date)"
        echo
      done
    else
      current_args="--method mas --mas_topology $topology --mas_node_num $node_num --mas_prompt $mas_prompt --seed $seed --prepare_mas_graph_only"
      if [ "$render_svg" = "1" ]; then
        current_args="$current_args --render_mas_graph_svg"
      fi

      echo "STARTING: topology=$topology node_num=$node_num mas_prompt=$mas_prompt seed=$seed host=$(hostname)"
      echo "ARGS: $current_args"
      echo "TIME: $(date)"

      python -u run.py $current_args

      echo "COMPLETED: topology=$topology node_num=$node_num seed=$seed time=$(date)"
      echo
    fi
  done
done
