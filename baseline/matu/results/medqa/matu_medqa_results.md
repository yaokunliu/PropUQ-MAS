# MATU Reproduction on medqa

This reproduces MATU-style low-rank reconstruction uncertainty from `2604.08708v1.pdf` on repeated Qwen3-8B MAS trajectories.

Configuration:
- UQ source runs: ASK4CONF raw MAS outputs
- Task: `medqa`
- MAS structures: `sequential, role`; `hierarchical, role`
- Model: `Qwen/Qwen3-8B`
- Repeated trajectories: 10 seeds (`42,43,44,45,46,47,48,49,50,51`)
- Correctness label mode: `first`
- Embedding backend: `qwen3`
- Embedding model: `Qwen/Qwen3-Embedding-0.6B`
- Low-rank reconstruction ranks: 1..3, capped by available runs

## Main Results

| MAS config | n | Accuracy | AUROC | PRR |
|---|---:|---:|---:|---:|
| sequential,role | 300 | 0.7933 | 0.7177 | 0.3731 |
| hierarchical,role | 300 | 0.7800 | 0.6264 | 0.2633 |
