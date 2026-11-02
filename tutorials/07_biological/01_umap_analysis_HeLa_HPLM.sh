#!/usr/bin/env bash
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
RESOURCE_ROOT="${DESCORT_RESOURCE_ROOT:-$PROJECT_ROOT/../../common_files}"
cd "$SCRIPT_DIR"
# Run from any working directory. Extra CLI arguments override the defaults below.
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
DEFAULT_OUTPUT_DIR="$PROJECT_ROOT/tutorials/07_biological/result/01_umap_analysis/cpg0021_HeLa_HPLM"

# Set to an empty string to analyze all common/available genes.
DEFAULT_HIT_FILE="/fse/home/wangyongyan/Project/CellPainting/LLLM/common_files/HeLa_HPLM_hits.xlsx"

# Annotated pathway gene sets.
DEFAULT_ANNOTATED_GENE_SETS="/fse/home/wangyongyan/Project/CellPainting/LLLM/annotated_gene_sets.json"

# Candidate original morphology feature names.
DEFAULT_ORIGINAL_FEATURES_PKL="/fse/home/wangyongyan/Project/CellPainting/Final_well_level_new_LLM/common_files/00_pretrain/cpg0021_HeLa_union_feature_1615.pkl"

# Selected manual pathways from the existing HeLa_HPLM comparison notebook.
DEFAULT_SELECTED_PATHWAYS_JSON="$SCRIPT_DIR/01_umap_analysis_HeLa_HPLM_selected_pathways.json"

# Runtime settings. Environment variables and trailing CLI arguments can override defaults.
PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_INPUT="${MODEL_INPUT:-$DEFAULT_MODEL_INPUT}"
ORIGINAL_INPUT="${ORIGINAL_INPUT:-$DEFAULT_ORIGINAL_INPUT}"
OUTPUT_DIR="${OUTPUT_DIR:-$DEFAULT_OUTPUT_DIR}"
HIT_FILE="${HIT_FILE-$DEFAULT_HIT_FILE}"
ANNOTATED_GENE_SETS="${ANNOTATED_GENE_SETS:-$DEFAULT_ANNOTATED_GENE_SETS}"
ORIGINAL_FEATURES_PKL="${ORIGINAL_FEATURES_PKL:-$DEFAULT_ORIGINAL_FEATURES_PKL}"
DATASET_NAME="${DATASET_NAME:-HeLa_HPLM}"
SELECTED_PATHWAYS_JSON="${SELECTED_PATHWAYS_JSON:-$DEFAULT_SELECTED_PATHWAYS_JSON}"

# Defaults match the current notebook values, including percentile 100 (all valid pathways).
PARAMS=(
    --pca-variance 0.95
    --umap-n-neighbors 10
    --umap-min-dist 0.05
    --umap-metric cosine
    --distance-metric cosine
    --min-go-kegg-genes 5
    --max-go-kegg-genes 200
    # ANNOTATED renders all valid pathways; these two limits apply only to other databases.
    --max-pathway-plots 0
    --pathway-plot-fraction 0.3
    --min-annotated-genes 2
    --compact-percentile 100
    --top-fraction 0.30
    --top-rank-by model_distance
    --exclude-databases ANNOTATED
    --model-feature-prefix emb_
    --model-feature-dim 256
    --gene-column Metadata_Foci_Barcode_MatchedTo_GeneCode
    --guide-column Metadata_Foci_Barcode_MatchedTo_Barcode
    --control-key-word nontargeting
    --plates CP257J CP257K CP257L CP257N
    --show-gene-labels
    --max-gene-labels 40
    --png-dpi 300
    --comparison-dpi 600
    --figure-formats png pdf
    --seed 42
    --threads 8
    --memory-limit 24GB
)
export MPLBACKEND=Agg
export USE_TF=0
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/descort_umap_matplotlib_${UID}}"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/tmp/descort_umap_numba_${UID}}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
exec "$PYTHON_BIN" -u "$SCRIPT_DIR/01_umap_analysis.py" \
    --model-input "$MODEL_INPUT" \
    --original-input "$ORIGINAL_INPUT" \
    --output-dir "$OUTPUT_DIR" \
    --hit-file "$HIT_FILE" \
    --annotated-gene-sets "$ANNOTATED_GENE_SETS" \
    --original-features-pkl "$ORIGINAL_FEATURES_PKL" \
    --dataset-name "$DATASET_NAME" \
    --selected-pathways-json "$SELECTED_PATHWAYS_JSON" \
    "${PARAMS[@]}" "$@"
