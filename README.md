# PropUQ-MAS

[![EMNLP 2026](https://img.shields.io/badge/EMNLP_2026-Main_Conference-7b2cbf)](https://2026.emnlp.org/)
[![Python 3.10](https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **Accepted to the EMNLP 2026 Main Conference.**

Official implementation of **PropUQ-MAS: Propagation-Aware Uncertainty Quantification for LLM Multi-Agent Systems**.

PropUQ-MAS models an LLM multi-agent system (MAS) as a directed communication graph. It estimates each agent's local uncertainty and propagates uncertainty along communication edges according to how strongly a receiving agent accepts the upstream message. This produces an interaction-aware uncertainty estimate without requiring repeated MAS trajectories.

## Highlights

- **Propagation-aware:** captures uncertainty inherited through multi-agent communication rather than considering only the final agent.
- **Interaction-aware:** weights incoming uncertainty using edge-level acceptance scores.
- **Post-hoc and lightweight:** replays uncertainty propagation from a single recorded MAS execution.
- **Flexible:** supports multiple MAS topologies, LLM backbones, tasks, and local uncertainty estimators.
- **Reproducible:** includes experiment launchers, graph exports, structured outputs, metrics, and baseline reproductions.

## Method

For agent \(i\), PropUQ-MAS combines its local uncertainty \(u_i\) with propagated uncertainty from its parent agents \(\mathcal{P}(i)\):

$$
\tilde{u}_i
=
1-(1-u_i)
\prod_{j\in\mathcal{P}(i)}
\left(1-\bar{\alpha}_{ji}\tilde{u}_j\right),
$$

where \(\bar{\alpha}_{ji}\) is the normalized acceptance weight assigned by agent \(i\) to the message from agent \(j\). The implementation can use either verbalized uncertainty (`Verb`) or token-probability-based uncertainty (`MSP`) as the local estimator.

## Results Snapshot

The table below reports the MedQA results included with this release for `Qwen/Qwen3-8B`, four role-specialized agents, and 300 examples. Higher is better.

| Topology | Metric | MSP | MSP + PropUQ | Verb. | Verb. + PropUQ |
|---|---|---:|---:|---:|---:|
| Sequential | AUROC | 0.754 | **0.842** | 0.600 | **0.791** |
| Sequential | PRR | 0.496 | **0.624** | 0.217 | **0.482** |
| Hierarchical | AUROC | 0.658 | **0.711** | 0.604 | **0.722** |
| Hierarchical | PRR | 0.277 | **0.349** | 0.223 | **0.362** |

See [`baseline/comparison.md`](baseline/comparison.md) for the full comparison with MATU and UProp, including runtime measurements and result provenance.

## Repository Structure

```text
PropUQ-MAS/
├── run.py                              # Main experiment and evaluation entry point
├── data.py                             # GSM8K, MedQA, and MBPP-Plus loaders
├── models.py                           # Hugging Face and vLLM model backends
├── mas_graph.py                        # MAS graph construction and export
├── prompts.py                          # Generic and role-specialized prompts
├── methods/
│   ├── mas.py                          # Multi-agent execution
│   └── uncertainty_quantification.py   # PropUQ replay and UQ metrics
├── run/
│   ├── run_experiment.sh               # Default MedQA reproduction
│   └── prepare_mas_graphs.sh           # Graph artifact generation
├── baseline/
│   ├── comparison.md                   # Consolidated baseline comparison
│   ├── matu/                           # MATU reproduction
│   └── uprop/                          # UProp reproduction
├── data/medqa.json                     # MedQA evaluation subset used by this release
└── requirements.txt
```

## Installation

The recommended setup uses Python 3.10 on Linux with an NVIDIA GPU and a CUDA-compatible PyTorch installation.

```bash
git clone https://github.com/yaokunliu/PropUQ-MAS.git
cd PropUQ-MAS

conda create -n mas python=3.10 -y
conda activate mas
pip install -r requirements.txt
```

The default launcher uses [vLLM](https://docs.vllm.ai/). To use the Hugging Face Transformers backend instead, set `USE_VLLM=0` when invoking the launcher or omit `--use_vllm` when calling `run.py` directly.

Graphviz is optional. JSON and text graph descriptions are always generated; SVG export additionally requires the system `dot` executable:

```bash
conda install -c conda-forge graphviz -y
```

If a selected Hugging Face model requires authentication, log in before running:

```bash
huggingface-cli login
```

## Data

| Task | Source | Loading behavior |
|---|---|---|
| GSM8K | [`gsm8k`](https://huggingface.co/datasets/openai/gsm8k) | Downloaded automatically through `datasets` |
| MBPP-Plus | [`evalplus/mbppplus`](https://huggingface.co/datasets/evalplus/mbppplus) | Downloaded automatically through `datasets` |
| MedQA | `data/medqa.json` | Loaded from the repository-local JSON file |

The MedQA loader expects each record to contain `query`, `options`, and `answer`. Please follow the licenses and terms of the original datasets.

## Quick Start

Run a 10-example smoke test with the default configuration:

```bash
MAX_SAMPLES=10 bash run/run_experiment.sh
```

The default configuration is:

```text
Qwen/Qwen3-8B + MedQA + hierarchical topology
+ role-specialized prompts + Verb uncertainty + 4 agents
```

Run the full 300-example MedQA experiment:

```bash
bash run/run_experiment.sh
```

To run without vLLM:

```bash
USE_VLLM=0 MAX_SAMPLES=10 bash run/run_experiment.sh
```

The first run downloads the requested model and dataset resources and can take substantially longer than later cached runs.

## Custom Experiments

Call `run.py` directly to change the task, topology, model, or uncertainty estimator:

```bash
python run.py \
  --method mas \
  --model_name Qwen/Qwen3-8B \
  --task medqa \
  --mas_topology hierarchical \
  --mas_node_num 4 \
  --mas_prompt role \
  --local_uncertainty_mode Verb \
  --max_samples 10 \
  --seed 42 \
  --use_vllm
```

### Supported configurations

| Option | Supported values |
|---|---|
| `--task` | `gsm8k`, `medqa`, `mbppplus` |
| `--model_name` | `Qwen/Qwen3-4B`, `Qwen/Qwen3-8B`, `Qwen/Qwen3-14B`, `google/gemma-3-12b-it` |
| `--mas_topology` | `sequential`, `hierarchical`, `decentralized` |
| `--local_uncertainty_mode` | `Verb`, `MSP` |
| `--mas_prompt` | `norole`, `role` |

Role-specialized prompts are defined for four-agent sequential and hierarchical systems. For other topology/node-count combinations, `--mas_prompt role` automatically falls back to the generic graph prompt.

Useful execution options include:

| Option | Purpose |
|---|---|
| `--max_samples N` | Evaluate only the first `N` examples; `-1` uses the full split |
| `--generate_only` | Save raw MAS predictions without post-hoc UQ evaluation |
| `--force_generation` | Ignore a matching raw-prediction cache and regenerate outputs |
| `--force_uq` | Overwrite matching UQ output files |
| `--tensor_parallel_size N` | Shard vLLM inference across `N` GPUs |
| `--gpu_memory_utilization F` | Set the vLLM GPU-memory target |
| `--render_mas_graph_svg` | Export a Graphviz SVG in addition to JSON and text |

Run `python run.py --help` for the complete CLI reference.

### Launcher environment variables

`run/run_experiment.sh` accepts the following environment variables without modifying the script:

| Variable | Default | Description |
|---|---:|---|
| `MAX_SAMPLES` | `-1` | Number of examples; `-1` uses all examples |
| `USE_VLLM` | `1` | Use vLLM (`1`) or Transformers (`0`) |
| `FORCE_GENERATION` | `0` | Regenerate raw predictions when set to `1` |
| `FORCE_UQ` | `1` | Overwrite matching UQ outputs when set to `1` |
| `MAX_NEW_TOKENS` | `8192` | Maximum new tokens per agent |
| `GENERATE_BS` | `16` | Generation batch size |
| `GPU_MEMORY_UTILIZATION` | `0.9` | vLLM memory-utilization target |
| `SEED` | `42` | Random seed |
| `CONDA_ENV` | `mas` | Conda environment activated by the launcher |
| `HF_ENV_FILE` | `~/.hf_token_env` | Optional shell file containing Hugging Face token exports |

## MAS Graph Artifacts

Generate all graph configurations used by the release without loading a model:

```bash
bash run/prepare_mas_graphs.sh
```

Disable SVG rendering when Graphviz is unavailable:

```bash
RENDER_SVG=0 bash run/prepare_mas_graphs.sh
```

Artifacts are written to:

```text
outputs/mas_graphs/<graph_config>/mas_graph.json
outputs/mas_graphs/<graph_config>/mas_graph.txt
outputs/mas_graphs/<graph_config>/mas_graph.dot   # when SVG rendering is enabled
outputs/mas_graphs/<graph_config>/mas_graph.svg   # when SVG rendering is enabled
```

## Outputs and Evaluation

Experiments write structured artifacts under `outputs/`:

```text
outputs/<uq_mode>/<model>/raw/raw_preds_*.jsonl
outputs/<uq_mode>/<model>/uq/uq_preds_*.jsonl
outputs/<uq_mode>/<model>/uq/uq_metrics_*.json
```

The metrics JSON contains experiment metadata, accuracy, runtime, and final-output UQ metrics grouped by local estimator:

- **AUROC:** area under the receiver operating characteristic curve for error detection.
- **PRR:** prediction rejection ratio, evaluated up to a rejection rate of 0.5.
- **Local-Uncertainty-Final:** uncertainty reported by the final agent alone.
- **Prop-Uncertainty-Final:** uncertainty after PropUQ-MAS propagation.

Raw-prediction caches are keyed by the task, topology, number of agents, prompt mode, seed, and sample count. Use `--force_generation` whenever generation settings change but those cache-key fields remain the same.

## Baseline Reproduction

This release includes self-contained MedQA reproductions for two uncertainty baselines:

- [`baseline/matu/`](baseline/matu/README.md): MATU with 10 repeated trajectories and low-rank reconstruction uncertainty.
- [`baseline/uprop/`](baseline/uprop/README.md): UProp with 10 repeated trajectories and token-log-probability-based predictive entropy.

Each baseline directory documents its commands, configuration, outputs, and runtime. The consolidated numbers are reported in [`baseline/comparison.md`](baseline/comparison.md).

## Reproducibility Notes

- The default random seed is `42`; stochastic generation may still vary across hardware, CUDA, PyTorch, Transformers, and vLLM versions.
- Full experiments require GPU memory appropriate for the selected model, context length, batch size, and inference backend. Reduce `GENERATE_BS` if memory is limited.
- `Verb` asks agents to emit structured local uncertainty and acceptance scores. `MSP` derives local uncertainty from generation probabilities.
- Generated predictions and metrics are intentionally ignored by Git. Preserve the relevant `outputs/` directory when archiving a run.

## Citation

If you use this repository, please cite the paper. The record below is provisional and should be replaced by the official ACL Anthology BibTeX after the proceedings metadata becomes available.

```bibtex
@inproceedings{propuqmas2026,
  title     = {{PropUQ-MAS}: Propagation-Aware Uncertainty Quantification for {LLM} Multi-Agent Systems},
  author    = {{PropUQ-MAS Authors}},
  booktitle = {Proceedings of the 2026 Conference on Empirical Methods in Natural Language Processing},
  year      = {2026},
  url       = {https://github.com/yaokunliu/PropUQ-MAS}
}
```

## License

This project is released under the [MIT License](LICENSE). Models and datasets are subject to their respective licenses and terms of use.
