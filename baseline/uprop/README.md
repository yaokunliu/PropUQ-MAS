# UProp Baseline

This folder contains a MedQA-only UProp reproduction for comparison with PropUQ-MAS.

The implementation follows the paper's trajectory-dependent uncertainty propagation idea using newly generated MAS outputs with token log-probabilities:

- Task: `medqa`
- Model: `Qwen/Qwen3-8B`
- MAS configuration: `sequential, role`
- Repeated trajectories: exactly 10 seeds, `42-51`
- Sampling: multinomial sampling with temperature `0.8`
- Max new tokens: `512`
- Intrinsic uncertainty: predictive entropy from generated token log-probabilities
- Decision distance: token-level fuzzy matching (`token_f1`; Python `difflib` is also available)
- Output unit: one uncertainty value per `(topology, mas_prompt, sample_idx)`

## Layout

```text
baseline/uprop/
  code/uprop_reproduction.py
  scripts/uprop_generate_medqa.sh
  scripts/uprop_score_medqa.sh
  results/medqa/
```

## Run

Generate UProp trajectories first. This writes MSP raw files with per-token log-probabilities under `outputs/MSP`.

```bash
bash baseline/uprop/scripts/uprop_generate_medqa.sh
```

Then score UProp:

```bash
bash baseline/uprop/scripts/uprop_score_medqa.sh
```

## Outputs

```text
baseline/uprop/results/medqa/uprop_medqa_scores.csv
baseline/uprop/results/medqa/uprop_medqa_results.csv
baseline/uprop/results/medqa/uprop_medqa_results.md
baseline/uprop/results/medqa/uprop_medqa_runtime.json
baseline/uprop/results/medqa/generation_runtime/
```

`uprop_medqa_scores.csv` contains one sample-level UProp uncertainty per MedQA sample.
`uprop_medqa_runtime.json` reports total time per sample, including all 10 repeated trajectory generations plus UProp scoring.

## MedQA Results

| MAS config | n | Accuracy | AUROC | PRR |
|---|---:|---:|---:|---:|
| sequential,role | 300 | 0.2867 | 0.6501 | 0.3303 |

Runtime including 10 repeated trajectory generations plus UProp scoring:

| MAS config | Time/sample |
|---|---:|
| sequential,role | 11.3106s |
