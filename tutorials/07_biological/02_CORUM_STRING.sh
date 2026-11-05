#!/usr/bin/env bash
# Common hit genes -> PCA -> L2 normalization -> subtract nonedge similarity background.
# To redraw only a saved CORUM distribution, pass --stage corum-plot and set
# OUTPUT_DIR to the existing dataset result directory (containing results/).
# Use --stage string-plot to redraw the STRING line in fixed [-1, 1] intervals.
set -euo pipefail
# ============================================================================
# Data input and output paths: edit the explicit paths below.
# ============================================================================
# Directory containing model embedding parquet files.
DEFAULT_MODEL_INPUT="/fse/home/wangyongyan/Project/CellPainting/Paper_LLm/tutorials/01_mAP/well_level_new_site_pretrain_emb_attention_mask/cpg0021_A549_DeSCORT_1382_nosampling_layernorm_finetune_A549/20200805_A549_WG_Screen/normalization_clip_data"

# Directory containing original morphology parquet files.
DEFAULT_ORIGINAL_INPUT="/fse/DC/CP_data/feature_data/20200805_A549_WG_Screen/raw_data/well_level_new/normalization_clip_data"

# Save all analysis tables, figures and logs under this directory.
DEFAULT_OUTPUT_DIR="/fse/home/wangyongyan/Project/CellPainting/DeSCORT/other_model/DeSCORT/tutorials/07_biological/result/02_CORUM_STRING"

# Set to an empty string to analyze all common/available genes.
DEFAULT_HIT_FILE="/fse/home/wangyongyan/Project/CellPainting/LLLM/common_files/A549_DMEM_hits.xlsx"

# CORUM complex membership table.
DEFAULT_CORUM_FILE="/fse/home/wangyongyan/Project/CellPainting/LLLM/common_files/CORUM_humanComplexes.txt"

# STRING gene-pair scores.
DEFAULT_STRING_FILE="/fse/home/wangyongyan/Project/CellPainting/LLLM/common_files/STRING_data.csv"

# Candidate original morphology feature names.
DEFAULT_ORIGINAL_FEATURES_PKL="/fse/home/wangyongyan/Project/CellPainting/DeSCORT/common_files/feature_intersection_union/cpg0021_A549/cpg0021_A549_union_feature_1382.pkl"

# Runtime settings. Environment variables and trailing CLI arguments can override defaults.
MODEL_INPUT="/fse/home/wangyongyan/Project/CellPainting/Paper_LLm/tutorials/01_mAP/well_level_new_site_pretrain_emb_attention_mask/cpg0021_A549_DeSCORT_1382_nosampling_layernorm_finetune_A549/20200805_A549_WG_Screen/normalization_clip_data"
ORIGINAL_INPUT="/fse/DC/CP_data/feature_data/20200805_A549_WG_Screen/raw_data/well_level_new/normalization_clip_data"
OUTPUT_DIR="/fse/home/wangyongyan/Project/CellPainting/DeSCORT/other_model/DeSCORT/tutorials/07_biological/result/02_CORUM_STRING"
HIT_FILE="/fse/home/wangyongyan/Project/CellPainting/LLLM/common_files/A549_DMEM_hits.xlsx"
CORUM_FILE="/fse/home/wangyongyan/Project/CellPainting/LLLM/common_files/CORUM_humanComplexes.txt"
STRING_FILE="/fse/home/wangyongyan/Project/CellPainting/LLLM/common_files/STRING_data.csv"
ORIGINAL_FEATURES_PKL="/fse/home/wangyongyan/Project/CellPainting/DeSCORT/common_files/feature_intersection_union/cpg0021_A549/cpg0021_A549_union_feature_1382.pkl"

PARAMS=(
    --dataset-name A549
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
    --plates CP186A CP186B CP186C CP186D CP186E CP186F CP186G CP186H CP186N
    --figure-formats png pdf
    --png-dpi 300
    --seed 20260918
    --threads 8
    --memory-limit 24GB
)
export MPLBACKEND=Agg USE_TF=0
export MPLCONFIGDIR="/tmp/descort_corum_matplotlib_1021"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
exec python -u "02_CORUM_STRING.py" \
    --model-input "$MODEL_INPUT" --original-input "$ORIGINAL_INPUT" \
    --hit-file "$HIT_FILE" --corum-file "$CORUM_FILE" --string-file "$STRING_FILE" \
    --original-features-pkl "$ORIGINAL_FEATURES_PKL" --output-dir "$OUTPUT_DIR" \
    "${PARAMS[@]}" "$@"
