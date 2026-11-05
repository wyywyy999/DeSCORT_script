#!/usr/bin/env bash
# Run from this script's directory. Extra CLI arguments override the defaults below.
set -euo pipefail

# ============================================================================
# Data input and output paths: edit the explicit paths below.
# ============================================================================
# Directory containing model embedding parquet files.
DEFAULT_MODEL_INPUT="/fse/home/wangyongyan/Project/CellPainting/Paper_LLm/tutorials/01_mAP/well_level_new_site_pretrain_emb_attention_mask/cpg0021_A549_DeSCORT_1382_nosampling_layernorm_finetune_A549/20200805_A549_WG_Screen/normalization_clip_data"

# Directory containing original morphology parquet files.
DEFAULT_ORIGINAL_INPUT="/fse/DC/CP_data/feature_data/20200805_A549_WG_Screen/raw_data/well_level_new/normalization_clip_data"

# Save all analysis tables, figures and logs under this directory.
DEFAULT_OUTPUT_DIR="/fse/home/wangyongyan/Project/CellPainting/DeSCORT/other_model/DeSCORT/tutorials/07_biological/result/01_umap_analysis/cpg0021_A549_DMEM"

# Set to an empty string to analyze all common/available genes.
DEFAULT_HIT_FILE="/fse/home/wangyongyan/Project/CellPainting/LLLM/common_files/A549_DMEM_hits.xlsx"

# Annotated pathway gene sets.
DEFAULT_ANNOTATED_GENE_SETS="/fse/home/wangyongyan/Project/CellPainting/LLLM/annotated_gene_sets.json"

# Candidate original morphology feature names.
DEFAULT_ORIGINAL_FEATURES_PKL="/fse/home/wangyongyan/Project/CellPainting/Final_well_level_new_LLM/common_files/00_pretrain/cpg0021_A549_union_feature_1382.pkl"

# Figure 7 computes PCA/nearest neighbors directly from this script's data inputs.
DEFAULT_CORUM_FILE="/fse/home/wangyongyan/Project/CellPainting/LLLM/common_files/CORUM_humanComplexes.txt"
DEFAULT_STRING_FILE="/fse/home/wangyongyan/Project/CellPainting/LLLM/common_files/STRING_data.csv"

# Runtime settings. Environment variables and trailing CLI arguments can override defaults.
MODEL_INPUT="/fse/home/wangyongyan/Project/CellPainting/Paper_LLm/tutorials/01_mAP/well_level_new_site_pretrain_emb_attention_mask/cpg0021_A549_DeSCORT_1382_nosampling_layernorm_finetune_A549/20200805_A549_WG_Screen/normalization_clip_data"
ORIGINAL_INPUT="/fse/DC/CP_data/feature_data/20200805_A549_WG_Screen/raw_data/well_level_new/normalization_clip_data"
OUTPUT_DIR="/fse/home/wangyongyan/Project/CellPainting/DeSCORT/other_model/DeSCORT/tutorials/07_biological/result/01_umap_analysis/cpg0021_A549_DMEM"
HIT_FILE="/fse/home/wangyongyan/Project/CellPainting/LLLM/common_files/A549_DMEM_hits.xlsx"
ANNOTATED_GENE_SETS="/fse/home/wangyongyan/Project/CellPainting/LLLM/annotated_gene_sets.json"
ORIGINAL_FEATURES_PKL="/fse/home/wangyongyan/Project/CellPainting/Final_well_level_new_LLM/common_files/00_pretrain/cpg0021_A549_union_feature_1382.pkl"
DATASET_NAME="${DATASET_NAME:-A549}"
CORUM_FILE="/fse/home/wangyongyan/Project/CellPainting/LLLM/common_files/CORUM_humanComplexes.txt"
STRING_FILE="/fse/home/wangyongyan/Project/CellPainting/LLLM/common_files/STRING_data.csv"

# Combined direct-label UMAPs are also generated for model and original.
# Figure 7 only: bash 01_umap_analysis.sh --stage query --query-gene PSMA1 ATP6V0C
# Replot cached results only: bash 01_umap_analysis.sh --stage combined-plot
# Override the notebook pathway list with --selected-pathways-json /absolute/path/list.json
# Defaults match the current notebook values, including percentile 100 (all valid pathways).
PARAMS=(
    # Optional: uncomment to enable figure 7. Omitted: no neighbor figures.
    --query-gene PSMA1 ATP6V0C
    --query-top-n 20
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
    --plates CP186A CP186B CP186C CP186D CP186E CP186F CP186G CP186H CP186N
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
export MPLCONFIGDIR="/tmp/descort_umap_matplotlib_1021"
export NUMBA_CACHE_DIR="/tmp/descort_umap_numba_1021"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
exec python -u "01_umap_analysis.py" \
    --model-input "$MODEL_INPUT" \
    --original-input "$ORIGINAL_INPUT" \
    --output-dir "$OUTPUT_DIR" \
    --hit-file "$HIT_FILE" \
    --annotated-gene-sets "$ANNOTATED_GENE_SETS" \
    --original-features-pkl "$ORIGINAL_FEATURES_PKL" \
    --dataset-name "$DATASET_NAME" \
    --corum-file "$CORUM_FILE" \
    --string-file "$STRING_FILE" \
    "${PARAMS[@]}" "$@"
