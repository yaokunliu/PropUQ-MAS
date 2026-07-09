# MATU Baseline

This folder contains the MATU baseline reproduction used for comparison with PropUQ-MAS.

MATU is implemented at the sample level: for each fixed `(topology, mas_prompt, sample_idx)`, the scorer collects 10 repeated Qwen3-8B trajectories, embeds all intermediate agent responses with `Qwen/Qwen3-Embedding-0.6B`, computes low-rank reconstruction loss, and writes one uncertainty value.

## Layout

```text
baseline/matu/
  code/matu_reproduction.py
  scripts/matu_reproduction_generate_repeats.sh
  scripts/matu_reproduction_score.sh
  results/medqa/
  results/runtime/
```

The main MedQA result files are:

```text
baseline/matu/results/medqa/matu_medqa_scores.csv
baseline/matu/results/medqa/matu_medqa_main_results.csv
baseline/matu/results/medqa/matu_medqa_results.md
baseline/matu/results/medqa/matu_medqa_runtime.md
baseline/matu/results/medqa/matu_medqa_runtime.json
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
bash baseline/matu/scripts/matu_reproduction_generate_repeats.sh
```

Select tasks with `TASKS`:

```bash
TASKS="gsm8k mbppplus" bash baseline/matu/scripts/matu_reproduction_generate_repeats.sh
```

After generation, run scoring:

```bash
bash baseline/matu/scripts/matu_reproduction_score.sh
```

The scripts default to the current repository. To score raw files from another checkout, pass `RAW_DIR=/path/to/outputs/ASK4CONF`. For legacy raw filenames, pass `LEGACY_TOPOLOGY_NAMES=1` during generation. The scorer enforces exactly 10 repeated seeds and only writes strict sample-level MATU outputs.

The scorer writes task-separated results under:

```text
baseline/matu/results/<task>/
```

Each scoring run also writes runtime files inside the same task folder:

```text
baseline/matu/results/<task>/matu_<task>_runtime.md
baseline/matu/results/<task>/matu_<task>_runtime.json
```

## Runtime

See `results/runtime/medqa_runtime_summary.md` for the manually summarized MedQA generation runtime. New scoring runs additionally write task-local runtime files automatically. In short, Qwen3-8B trajectory generation plus MATU scoring takes about 48.73 seconds per MedQA sample, where one sample is one `(topology, mas_prompt, sample_idx)`.
