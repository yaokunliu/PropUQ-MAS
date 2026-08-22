# MATU Sampling-Based Baseline

This folder contains the MATU sampling-based baseline reproduction used for comparison with PropUQ-MAS.

MATU is implemented at the sample level: for each fixed `(topology, mas_prompt, sample_idx)`, the scorer collects 10 repeated Qwen3-8B trajectories, embeds all intermediate agent responses with `Qwen/Qwen3-Embedding-0.6B`, computes low-rank reconstruction loss, and writes one uncertainty value.

## Layout

```text
sampling_baselines/matu/
  code/matu_reproduction.py
  scripts/matu_reproduction_generate_repeats.sh
  scripts/matu_reproduction_score.sh
  results/medqa/
  results/runtime/
```

The main MedQA result files are:

```text
sampling_baselines/matu/results/medqa/matu_medqa_scores.csv
sampling_baselines/matu/results/medqa/matu_medqa_main_results.csv
sampling_baselines/matu/results/medqa/matu_medqa_results.md
sampling_baselines/matu/results/medqa/matu_medqa_runtime.md
sampling_baselines/matu/results/medqa/matu_medqa_runtime.json
```

## MedQA Results

The copied MedQA results use:

- Model: `Qwen/Qwen3-8B`
- Embedding model: `Qwen/Qwen3-Embedding-0.6B`
- Repeated trajectories: exactly 10 seeds, `42-51`
- MAS configurations: `sequential, role`; `hierarchical, role`
- Label mode: `first`

| MAS config | n | Accuracy | AUROC | PRR |
|---|---:|---:|---:|---:|
| sequential,role | 300 | 0.7933 | 0.7177 | 0.3731 |
| hierarchical,role | 300 | 0.7800 | 0.6264 | 0.2633 |

## Run

Generate repeated trajectories:

```bash
bash sampling_baselines/matu/scripts/matu_reproduction_generate_repeats.sh
```

Select tasks with `TASKS`:

```bash
TASKS="gsm8k mbppplus" bash sampling_baselines/matu/scripts/matu_reproduction_generate_repeats.sh
```

After generation, run scoring:

```bash
bash sampling_baselines/matu/scripts/matu_reproduction_score.sh
```

The scripts default to the current repository. To score raw files from another checkout, pass `RAW_DIR=/path/to/outputs/ASK4CONF`. For legacy raw filenames, pass `LEGACY_TOPOLOGY_NAMES=1` during generation. The scorer enforces exactly 10 repeated seeds and only writes strict sample-level MATU outputs.

The scorer writes task-separated results under:

```text
sampling_baselines/matu/results/<task>/
```

Each scoring run also writes runtime files inside the same task folder:

```text
sampling_baselines/matu/results/<task>/matu_<task>_runtime.md
sampling_baselines/matu/results/<task>/matu_<task>_runtime.json
```

## Runtime

See `results/runtime/medqa_runtime_summary.md` for the manually summarized MedQA generation runtime. New scoring runs additionally write task-local runtime files automatically. In short, Qwen3-8B trajectory generation plus MATU scoring takes about 48.73 seconds per MedQA sample, where one sample is one `(topology, mas_prompt, sample_idx)`.
