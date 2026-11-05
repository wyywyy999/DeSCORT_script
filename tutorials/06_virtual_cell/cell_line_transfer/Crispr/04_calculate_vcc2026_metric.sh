#!/usr/bin/env bash
set -euo pipefail

# Metric options match the notebook; paths follow 03_run_inference.sh.
MODEL_NAME="${MODEL_NAME:-${EXPERIMENT_NAME:-A549_all_genes_cell_line_transfer_50epochs}}"
# H5AD input paths match the inference filenames configured below.
DATA_DIR="/fse/home/wangyongyan/Project/CellPainting/DeSCORT/other_model/DeSCORT/tutorials/06_virtual_cell/cell_line_transfer/Crispr/test_anndata/A549_all_genes_cell_line_transfer_50epochs"
GROUND_TRUTH_PATH="/fse/home/wangyongyan/Project/CellPainting/DeSCORT/other_model/DeSCORT/tutorials/06_virtual_cell/cell_line_transfer/Crispr/test_anndata/A549_all_genes_cell_line_transfer_50epochs/ground_truth.h5ad"
PREDICTION_PATH="/fse/home/wangyongyan/Project/CellPainting/DeSCORT/other_model/DeSCORT/tutorials/06_virtual_cell/cell_line_transfer/Crispr/test_anndata/A549_all_genes_cell_line_transfer_50epochs/A549_all_genes_cell_line_transfer_50epochs_pred_data.h5ad"
BASELINE_PATH="/fse/home/wangyongyan/Project/CellPainting/DeSCORT/other_model/DeSCORT/tutorials/06_virtual_cell/cell_line_transfer/Crispr/test_anndata/A549_all_genes_cell_line_transfer_50epochs/perturb_mean_data.h5ad"
OUTDIR="/fse/home/wangyongyan/Project/CellPainting/DeSCORT/other_model/DeSCORT/tutorials/06_virtual_cell/cell_line_transfer/Crispr/result/A549_all_genes_cell_line_transfer_50epochs"
CACHE_DIR="/fse/home/wangyongyan/Project/CellPainting/DeSCORT/other_model/DeSCORT/tutorials/06_virtual_cell/cell_line_transfer/Crispr/result/de_cache"
METRIC_SCRIPT=04_calculate_vcc2026_metric.py

PERT_COL="${PERT_COL:-Perturb_Gene}"
CONTROL_NAME="${CONTROL_NAME:-nontargeting}"
GROUND_TRUTH_LAYER="${GROUND_TRUTH_LAYER:-}"
PREDICTION_LAYER="${PREDICTION_LAYER:-}"
BASELINE_LAYER="${BASELINE_LAYER:-}"
DEDUP_CONTROLS="${DEDUP_CONTROLS:-true}"
STRICT_ALIGNMENT="${STRICT_ALIGNMENT:-true}"
COPY_TRUTH_CONTROLS="${COPY_TRUTH_CONTROLS:-false}"
FDR_SCOPE="${FDR_SCOPE:-per_pert}"
N_JOBS="${N_JOBS:-8}"
BLAS_THREADS="${BLAS_THREADS:-4}"
SCIPY_CHECKS="${SCIPY_CHECKS:-100}"
REUSE_CACHE="${REUSE_CACHE:-true}"
SHOW_PROGRESS="${SHOW_PROGRESS:-true}"
# Set RUN_BASELINE=false, or edit BASELINE_PATH="", to skip baseline evaluation.
RUN_BASELINE="${RUN_BASELINE:-true}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$BLAS_THREADS}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-$BLAS_THREADS}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-$BLAS_THREADS}"

PARAMS=(
    --ground_truth_path "$GROUND_TRUTH_PATH"
    --prediction_path "$PREDICTION_PATH"
    --outdir "$OUTDIR"
    --cache_dir "$CACHE_DIR"
    --pert_col "$PERT_COL"
    --control_name "$CONTROL_NAME"
    --dedup_controls "$DEDUP_CONTROLS"
    --strict_alignment "$STRICT_ALIGNMENT"
    --copy_truth_controls "$COPY_TRUTH_CONTROLS"
    --fdr_scope "$FDR_SCOPE"
    --n_jobs "$N_JOBS"
    --blas_threads "$BLAS_THREADS"
    --scipy_checks "$SCIPY_CHECKS"
    --reuse_cache "$REUSE_CACHE"
    --show_progress "$SHOW_PROGRESS"
)
if [[ "$RUN_BASELINE" == true || "$RUN_BASELINE" == 1 || "$RUN_BASELINE" == yes ]]; then
    if [[ -n "$BASELINE_PATH" ]]; then PARAMS+=(--baseline_path "$BASELINE_PATH"); fi
elif [[ "$RUN_BASELINE" != false && "$RUN_BASELINE" != 0 && "$RUN_BASELINE" != no ]]; then
    echo "RUN_BASELINE must be true or false" >&2
    exit 2
fi
if [[ -n "$GROUND_TRUTH_LAYER" ]]; then PARAMS+=(--ground_truth_layer "$GROUND_TRUTH_LAYER"); fi
if [[ -n "$PREDICTION_LAYER" ]]; then PARAMS+=(--prediction_layer "$PREDICTION_LAYER"); fi
if [[ -n "$BASELINE_LAYER" ]]; then PARAMS+=(--baseline_layer "$BASELINE_LAYER"); fi

exec python -u "$METRIC_SCRIPT" "${PARAMS[@]}" "$@"
