# MATU Runtime on medqa

- Started UTC: `2026-07-09T09:54:52`
- Finished UTC: `2026-07-09T10:07:10`
- Uncertainty example unit: `(topology, mas_prompt, sample_idx)`
- Uncertainty examples: `600`
- Unique samples: `300`
- MAS configs: `2`
- Repeated trajectories per sample: `10`
- Time per uncertainty example: `48.7250` seconds
- End-to-end elapsed: `29235.00` seconds (`487.25` minutes)
- Trajectory generation elapsed: `28743.00` seconds (`479.05` minutes)
- Trajectory generation time per uncertainty example: `47.9050` seconds
- Uncertainty scoring elapsed: `492.00` seconds (`8.20` minutes)
- Uncertainty scoring time per uncertainty example: `0.8200` seconds
- Scoring script total elapsed: `492.00` seconds (`8.20` minutes)
- Note: trajectory generation times are estimated from the measured MedQA generation runtime summary; uncertainty scoring time is estimated from the retained two-config scoring runtime because the original run did not separately instrument the uncertainty block.

## Time By MAS Config

| MAS config | Uncertainty examples | Trajectory generation / example | Uncertainty scoring / example | Time / uncertainty example |
|---|---:|---:|---:|---:|
| sequential,role | 300 | 40.5900 | 0.8200 | 41.4100 |
| hierarchical,role | 300 | 55.2200 | 0.8200 | 56.0400 |

## Configuration

- Model: `Qwen/Qwen3-8B`
- Embedding backend: `qwen3`
- Embedding model: `Qwen/Qwen3-Embedding-0.6B`
- Embedding device: `cuda`
- Embedding batch size: `32`
- Max rank: `3`
- Label mode: `first`
- Raw dir: `external_outputs/ASK4CONF`
