# UQ Metric Summary

| task | prompt | model | mode | accuracy | best_auroc_metric | best_auroc | best_ece_metric | best_ece | best_brier_metric | best_brier |
| --- | --- | --- | --- | ---: | --- | ---: | --- | ---: | --- | ---: |
| aime2024 | hierarchical | Qwen/Qwen3-8B | anchor | 0.5667 | Uncertainty-Final | 0.5000 | Uncertainty-Max | 0.1848 | Hazard-h-Final | 0.2518 |
| aime2024 | hierarchical | google/gemma-3-12b-it | anchor | 0.2333 | Uncertainty-Final | 0.7292 | Formula-Uncertainty-Max | 0.1396 | Formula-Uncertainty-Max | 0.2040 |
| aime2024 | sequential | Qwen/Qwen3-8B | anchor | 0.6333 | Uncertainty-Final | 0.7895 | Uncertainty-Mean | 0.1210 | Uncertainty-Final | 0.1755 |
| aime2024 | sequential | google/gemma-3-12b-it | anchor | 0.1667 | Hazard-h-Final | 0.5875 | Hazard-h-Final | 0.1396 | Hazard-h-Final | 0.1673 |
| humanevalplus | hierarchical | Qwen/Qwen3-8B | anchor | 0.9024 | Uncertainty-Mean | 0.7981 | Uncertainty-Mean | 0.0476 | Uncertainty-Mean | 0.0743 |
| humanevalplus | hierarchical | google/gemma-3-12b-it | anchor | 0.7683 | Uncertainty-Final | 0.6400 | Uncertainty-Final | 0.1479 | Uncertainty-Final | 0.1917 |
| humanevalplus | sequential | Qwen/Qwen3-8B | anchor | 0.8902 | Uncertainty-Mean | 0.8202 | Uncertainty-Mean | 0.1139 | Uncertainty-Mean | 0.0919 |
| humanevalplus | sequential | google/gemma-3-12b-it | anchor | 0.7378 | Uncertainty-Mean | 0.6462 | Uncertainty-Mean | 0.0974 | Uncertainty-Mean | 0.1934 |
| mbppplus | hierarchical | Qwen/Qwen3-8B | anchor | 0.7963 | Uncertainty-Mean | 0.7022 | Uncertainty-Max | 0.0463 | Hazard-h-Mean | 0.1527 |
| mbppplus | hierarchical | Qwen/Qwen3-8B | continuous | 0.7963 | Uncertainty-Mean | 0.6850 | System-Uncertainty-Final | 0.0258 | Hazard-h-Mean | 0.1542 |
| mbppplus | hierarchical | google/gemma-3-12b-it | anchor | 0.7540 | Uncertainty-Final | 0.6030 | Uncertainty-Final | 0.1850 | Uncertainty-Final | 0.2127 |
| mbppplus | hierarchical | google/gemma-3-12b-it | continuous | 0.7698 | Uncertainty-Mean | 0.6425 | Formula-Uncertainty-Max | 0.0379 | Formula-Uncertainty-Max | 0.1675 |
| mbppplus | sequential | Qwen/Qwen3-8B | anchor | 0.7910 | Formula-Uncertainty-Final | 0.7187 | Uncertainty-Final | 0.0562 | Uncertainty-Mean | 0.1458 |
| mbppplus | sequential | Qwen/Qwen3-8B | continuous | 0.8095 | Hazard-h-Final | 0.6240 | Uncertainty-Max | 0.0396 | Uncertainty-Max | 0.1461 |
| mbppplus | sequential | google/gemma-3-12b-it | anchor | 0.6958 | Uncertainty-Mean | 0.6953 | Uncertainty-Mean | 0.0673 | Uncertainty-Mean | 0.1977 |
| mbppplus | sequential | google/gemma-3-12b-it | continuous | 0.6958 | Hazard-h-Final | 0.7598 | Formula-Uncertainty-Max | 0.0833 | Hazard-h-Final | 0.1801 |
