# MedQA Baseline Comparison

Results are reported on MedQA for `sequential, role` and `hierarchical, role` MAS settings using `Qwen/Qwen3-8B`. Missing results are marked as `-`. The best AUROC/PRR value in each topology-metric row is bolded; time is reported without bolding.

| MAS | Metric | MATU | UProp | MSP | MSP+PropUQ | Verb. | Verb.+PropUQ |
|---|---|---:|---:|---:|---:|---:|---:|
| Sequential | AUROC | 0.718 | 0.650 | 0.754 | **0.842** | 0.600 | 0.791 |
| Sequential | PRR | 0.373 | 0.330 | 0.496 | **0.624** | 0.217 | 0.482 |
| Sequential | Time/sample (s) | 41.410 | 11.310 | 1.129 | 1.328 | 4.059 | 4.258 |
| Hierarchical | AUROC | 0.626 | - | 0.658 | 0.711 | 0.604 | **0.722** |
| Hierarchical | PRR | 0.263 | - | 0.277 | 0.349 | 0.223 | **0.362** |
| Hierarchical | Time/sample (s) | 56.040 | - | 1.536 | 1.735 | 5.522 | 5.721 |

Notes:
- MATU and UProp values are from `baseline/matu/results/medqa/` and `baseline/uprop/results/medqa/`.
- Time/sample is end-to-end time per sample when available, including repeated trajectory generation for MATU/UProp.
- MSP sequential time is from the existing MSP MedQA generation logs. Hierarchical MSP time is scaled by MATU's measured hierarchical/sequential one-trajectory generation ratio. Verb. times use MATU's measured one-trajectory MedQA generation time. PropUQ times add the observed posthoc propagation/UQ overhead from the available hierarchical Verb.+PropUQ log.
- PropUQ values are transcribed from the provided MedQA table.
