# UQ Method Accuracy Summary

| task | prompt | model | no_uq_acc | anchor_acc | continuous_acc | anchor_vs_no_uq | continuous_vs_no_uq | best_method | best_acc | best_vs_no_uq |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| aime2024 | hierarchical | Qwen/Qwen3-8B | 0.5000 | 0.5667 | NA | +0.0667 | NA | anchor | 0.5667 | +0.0667 |
| aime2024 | hierarchical | google/gemma-3-12b-it | 0.2333 | 0.2333 | NA | +0.0000 | NA | anchor/no_uq | 0.2333 | +0.0000 |
| aime2024 | sequential | Qwen/Qwen3-8B | 0.5667 | 0.6333 | NA | +0.0667 | NA | anchor | 0.6333 | +0.0667 |
| aime2024 | sequential | google/gemma-3-12b-it | 0.2667 | 0.1667 | NA | -0.1000 | NA | no_uq | 0.2667 | +0.0000 |
| humanevalplus | hierarchical | Qwen/Qwen3-8B | 0.8780 | 0.9024 | NA | +0.0244 | NA | anchor | 0.9024 | +0.0244 |
| humanevalplus | hierarchical | google/gemma-3-12b-it | 0.7500 | 0.7683 | NA | +0.0183 | NA | anchor | 0.7683 | +0.0183 |
| humanevalplus | sequential | Qwen/Qwen3-8B | 0.9085 | 0.8902 | NA | -0.0183 | NA | no_uq | 0.9085 | +0.0000 |
| humanevalplus | sequential | google/gemma-3-12b-it | 0.7561 | 0.7378 | NA | -0.0183 | NA | no_uq | 0.7561 | +0.0000 |
| mbppplus | hierarchical | Qwen/Qwen3-8B | 0.8042 | 0.7963 | 0.7963 | -0.0079 | -0.0079 | no_uq | 0.8042 | +0.0000 |
| mbppplus | hierarchical | google/gemma-3-12b-it | 0.7646 | 0.7540 | 0.7698 | -0.0106 | +0.0053 | continuous | 0.7698 | +0.0053 |
| mbppplus | sequential | Qwen/Qwen3-8B | 0.8095 | 0.7910 | 0.8095 | -0.0185 | +0.0000 | continuous/no_uq | 0.8095 | +0.0000 |
| mbppplus | sequential | google/gemma-3-12b-it | 0.6799 | 0.6958 | 0.6958 | +0.0159 | +0.0159 | anchor/continuous | 0.6958 | +0.0159 |
