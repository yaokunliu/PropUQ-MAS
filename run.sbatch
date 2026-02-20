#!/bin/bash
#SBATCH --job-name=TaxedFixedCircuit
#SBATCH --account=bgfy-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/u/yliu105/DiFAIR/logs/Analysis_%x_%j.log
#SBATCH --error=/u/yliu105/DiFAIR/logs/Analysis_%x_%j.err

set -euo pipefail

cd /u/yliu105/DiFAIR
export HF_HUB_DISABLE_PROGRESS_BARS=1
export TRANSFORMERS_VERBOSITY=error

PYTHON_BIN="/u/yliu105/miniforge3/envs/mi/bin/python"
SAMPLES_PER_CLASS="${SAMPLES_PER_CLASS:-80}"  # Must be in [50, 100]
# GH200 120GB has enough headroom for a larger batch than 2 in this workload.
# Override at submit time if needed, e.g. --export=ALL,BATCH_SIZE=10
BATCH_SIZE="${BATCH_SIZE:-8}"
OBJECTIVE="${OBJECTIVE:-logit_diff}"
FEATURE_OUTPUT_DIR="${FEATURE_OUTPUT_DIR:-/work/nvme/bfnx/yliu105/analysis_features}"

# visualize.py knobs
VIS_TOP_K="${VIS_TOP_K:-15}"
MIN_ABS_SIGNIFICANCE_DIFF="${MIN_ABS_SIGNIFICANCE_DIFF:-2.0}"
MAX_POOLED_STD="${MAX_POOLED_STD:-0.12}"
MIN_VALID_FRACTION="${MIN_VALID_FRACTION:-0.97}"
MAX_NONFINITE="${MAX_NONFINITE:-2}"

# build_report.py knobs
REPORT_TOP_HEAD="${REPORT_TOP_HEAD:-30}"
REPORT_TOP_MLP="${REPORT_TOP_MLP:-10}"

OUTPUT_DIR="mi_analysis/results/taxed_fixed_${SAMPLES_PER_CLASS}_job_${SLURM_JOB_ID:-manual}"

if (( SAMPLES_PER_CLASS < 50 || SAMPLES_PER_CLASS > 100 )); then
    echo "[ERROR] SAMPLES_PER_CLASS must be between 50 and 100, got: ${SAMPLES_PER_CLASS}" >&2
    exit 1
fi

mkdir -p "${OUTPUT_DIR}" "${FEATURE_OUTPUT_DIR}"

echo "=== Taxed/Fixed circuit pipeline started ==="
date
echo "SAMPLES_PER_CLASS=${SAMPLES_PER_CLASS} (for taxed and fixed each)"
echo "BATCH_SIZE=${BATCH_SIZE}"
echo "OBJECTIVE=${OBJECTIVE}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "FEATURE_OUTPUT_DIR=${FEATURE_OUTPUT_DIR}"

# Step 1: Analysis
"${PYTHON_BIN}" -u mi_analysis/code/run_analysis.py \
    --samples_per_class "${SAMPLES_PER_CLASS}" \
    --batch_size "${BATCH_SIZE}" \
    --objective "${OBJECTIVE}" \
    --output_dir "${OUTPUT_DIR}" \
    --feature_output_dir "${FEATURE_OUTPUT_DIR}"

# Step 2: Build comparison + visualizations
"${PYTHON_BIN}" -u mi_analysis/code/visualize.py \
    --run_dir "${OUTPUT_DIR}" \
    --top_k "${VIS_TOP_K}" \
    --min_abs_significance_diff "${MIN_ABS_SIGNIFICANCE_DIFF}" \
    --max_pooled_std "${MAX_POOLED_STD}" \
    --min_valid_fraction "${MIN_VALID_FRACTION}" \
    --max_nonfinite "${MAX_NONFINITE}"

# Step 3: Build report
"${PYTHON_BIN}" -u mi_analysis/code/build_report.py \
    --run_dir "${OUTPUT_DIR}" \
    --top_head "${REPORT_TOP_HEAD}" \
    --top_mlp "${REPORT_TOP_MLP}" \
    --feature_cache_dir "${FEATURE_OUTPUT_DIR}"

echo "=== Pipeline complete ==="
date
echo "Run directory: ${OUTPUT_DIR}"
echo "Analysis visualization: ${OUTPUT_DIR}/taxed_fixed_visualization.png"
echo "Comparison dataframe: ${OUTPUT_DIR}/taxed_fixed_comparison_dataframe.csv"
echo "Report: ${OUTPUT_DIR}/report.md"
echo "Sample-level selected stats: ${OUTPUT_DIR}/selected_component_sample_level_stats.csv"