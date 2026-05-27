# PropUQ-MAS

Propagation-Aware Uncertainty Quantification for LLM Multi-Agent Systems.

PropUQ-MAS represents a multi-agent execution as a directed communication graph and computes propagation-aware uncertainty by combining each agent's local uncertainty with uncertainty inherited from upstream agents through normalized acceptance weights.

## Supported Scope

- Datasets: `gsm8k`, `medqa`, `mbppplus`
- Models: `Qwen/Qwen3-4B`, `Qwen/Qwen3-8B`, `Qwen/Qwen3-14B`, `google/gemma-3-12b-it`
- MAS topologies: `sequential`, `hierarchical`, `decentralized`
- Local uncertainty modes: `Verb`, `MSP`
- Metrics: AUROC and PRR

## Installation

Create an environment and install dependencies:

```bash
conda create -n mas python=3.10 -y
conda activate mas
pip install -r requirements.txt
```

`graphviz` in `requirements.txt` installs the Python package. To export SVG graph files, the system `dot` executable is also required:

```bash
conda install -c conda-forge graphviz -y
```

If you use gated Hugging Face models, authenticate before running:

```bash
huggingface-cli login
```

## Data

GSM8K and MBPP-Plus are loaded from Hugging Face datasets:

- `gsm8k`
- `evalplus/mbppplus`

MedQA is loaded from:

```text
data/medqa.json
```

The MedQA file should be a JSON dataset compatible with `data.py`, containing `query`, `options`, and `answer` fields.

## Run

The repository keeps two runnable scripts under `run/`.

Prepare MAS graph artifacts only:

```bash
bash run/prepare_mas_graphs.sh
```

Run the default experiment and compute metrics:

```bash
bash run/run_experiment.sh
```

The default experiment is:

```text
Qwen/Qwen3-8B + medqa + hierarchical + role + Verb + 4 agents
```

You can also call `run.py` directly:

```bash
python run.py \
  --method mas \
  --model_name Qwen/Qwen3-8B \
  --task medqa \
  --mas_topology hierarchical \
  --mas_node_num 4 \
  --mas_prompt role \
  --local_uncertainty_mode Verb \
  --use_vllm
```

For a quick smoke test, limit the number of examples:

```bash
MAX_SAMPLES=10 bash run/run_experiment.sh
```

Useful environment variables for `run/run_experiment.sh`:

- `MAX_SAMPLES`: number of examples; default `-1` means all examples.
- `USE_VLLM`: set to `1` to use vLLM, `0` to use Hugging Face generation.
- `FORCE_GENERATION`: set to `1` to ignore cached raw predictions.
- `FORCE_UQ`: set to `1` to overwrite UQ outputs.
- `CONDA_ENV`: conda environment name; default `mas`.
- `HF_ENV_FILE`: optional file containing Hugging Face token exports.

## Outputs

All generated files are written under `outputs/`.

Raw predictions:

```text
outputs/<local_uncertainty_mode>/<model>/raw/raw_preds_*.jsonl
```

Posthoc UQ predictions:

```text
outputs/<local_uncertainty_mode>/<model>/uq/uq_preds_*.jsonl
```

Metrics:

```text
outputs/<local_uncertainty_mode>/<model>/uq/uq_metrics_*.json
```

MAS graph artifacts:

```text
outputs/mas_graphs/<graph_config>/mas_graph.json
outputs/mas_graphs/<graph_config>/mas_graph.txt
outputs/mas_graphs/<graph_config>/mas_graph.svg
```

The metrics JSON reports AUROC and PRR for local uncertainty and propagation-aware uncertainty.

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{propuqmas2026,
  title = {PropUQ-MAS: Propagation-Aware Uncertainty Quantification for LLM Multi-Agent Systems},
  author = {Anonymous},
  booktitle = {Proceedings of EMNLP},
  year = {2026}
}
```

## License

This project is released under the MIT License.
