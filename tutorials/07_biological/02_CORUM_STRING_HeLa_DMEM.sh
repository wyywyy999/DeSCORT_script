#!/usr/bin/env bash
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
RESOURCE_ROOT="${DESCORT_RESOURCE_ROOT:-$PROJECT_ROOT/../../common_files}"
cd "$SCRIPT_DIR"
# Common hit genes -> PCA -> L2 normalization -> subtract nonedge similarity background.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
WORKSPACE_ROOT="$(cd -- "$PROJECT_ROOT/.." && pwd)"
# ============================================================================
# Data input and output paths: edit the explicit paths below.
# ============================================================================
# Directory containing model embedding parquet files.
DEFAULT_MODEL_INPUT="/fse/home/wangyongyan/Project/CellPainting/Paper_LLm/tutorials/01_mAP/well_level_new_site_pretrain_emb_attention_mask/cpg0021_HeLa_DeSCORT_1615_nosampling_layernorm_finetune/20210422_6W_CP257/normalization_clip_data"

# Directory containing original morphology parquet files.
DEFAULT_ORIGINAL_INPUT="/fse/DC/CP_data/feature_data/20210422_6W_CP257/raw_data/well_level_new/normalization_clip_data"

# Save all analysis tables, figures and logs under this directory.
DEFAULT_OUTPUT_DIR="$PROJECT_ROOT/tutorials/07_biological/result/02_CORUM_STRING/cpg0021_HeLa_DMEM"

# Set to an empty string to analyze all common/available genes.
DEFAULT_HIT_FILE="/fse/home/wangyongyan/Project/CellPainting/LLLM/common_files/HeLa_DMEM_hits.xlsx"

# CORUM complex membership table.
DEFAULT_CORUM_FILE="/fse/home/wangyongyan/Project/CellPainting/LLLM/common_files/CORUM_humanComplexes.txt"

# STRING gene-pair scores.
DEFAULT_STRING_FILE="/fse/home/wangyongyan/Project/CellPainting/LLLM/common_files/STRING_data.csv"

# Candidate original morphology feature names.
DEFAULT_ORIGINAL_FEATURES_PKL="$RESOURCE_ROOT/feature_intersection_union/cpg0021_HeLa/cpg0021_HeLa_union_feature_1615.pkl"

# Runtime settings. Environment variables and trailing CLI arguments can override defaults.
PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_INPUT="${MODEL_INPUT:-$DEFAULT_MODEL_INPUT}"
ORIGINAL_INPUT="${ORIGINAL_INPUT:-$DEFAULT_ORIGINAL_INPUT}"
OUTPUT_DIR="${OUTPUT_DIR:-$DEFAULT_OUTPUT_DIR}"
HIT_FILE="${HIT_FILE-$DEFAULT_HIT_FILE}"
CORUM_FILE="${CORUM_FILE:-$DEFAULT_CORUM_FILE}"
STRING_FILE="${STRING_FILE:-$DEFAULT_STRING_FILE}"
ORIGINAL_FEATURES_PKL="${ORIGINAL_FEATURES_PKL:-$DEFAULT_ORIGINAL_FEATURES_PKL}"

PARAMS=(
    --dataset-name HeLa_DMEM
    --pca-variance 0.95
    --corum-min-coverage 0.333
    --string-low-score 300
    --string-high-score 700
    --string-chunk-size 500000
    --pair-chunk-size 50000
    --model-feature-prefix emb_
    --model-feature-dim 256
    --gene-column Metadata_Foci_Barcode_MatchedTo_GeneCode
    --guide-column Metadata_Foci_Barcode_MatchedTo_Barcode
    --control-key-word nontargeting
    --plates CP257A CP257B CP257D CP257F CP257H
    --figure-formats png pdf
    --png-dpi 300
    --seed 20260918
    --threads 8
    --memory-limit 24GB
)
export MPLBACKEND=Agg USE_TF=0
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/descort_corum_matplotlib_${UID}}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
exec "$PYTHON_BIN" -u "$SCRIPT_DIR/02_CORUM_STRING.py" \
    --model-input "$MODEL_INPUT" --original-input "$ORIGINAL_INPUT" \
    --hit-file "$HIT_FILE" --corum-file "$CORUM_FILE" --string-file "$STRING_FILE" \
    --original-features-pkl "$ORIGINAL_FEATURES_PKL" --output-dir "$OUTPUT_DIR" \
    "${PARAMS[@]}" "$@"
