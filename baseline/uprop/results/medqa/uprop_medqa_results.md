# UProp Reproduction on medqa

This reproduces a UProp-style uncertainty estimator from `12720_UProp_Investigating_the_ (1).pdf` using token log-probabilities from 10 repeated Qwen3-8B sequential-role MAS trajectories.

Configuration:
- Task: `medqa`
- MAS structure: `sequential, role`
- Model: `Qwen/Qwen3-8B`
- Repeated trajectories: exactly 10 seeds (`42-51`)
- Target prediction seed: `42`
- Decision distance: `token_f1`
- Intrinsic uncertainty: PE from token log-probabilities (`sequence` normalization)
- Kernel sharpness: `10.0`
- Length normalization: `paper`

## Main Results

| MAS config | n | Accuracy | AUROC | PRR | Mean UProp | Mean IU | Mean EU | Mean Step PE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| sequential,role | 300 | 0.2867 | 0.6501 | 0.3303 | 136.9547 | 548.7206 | 5.4527 | 137.1802 |
