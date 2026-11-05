#!/usr/bin/env bash
# Embedding-only clustering, differential morphology, biological validation and neighbour queries.
# Enrichment statistics are saved here; use 04_cluster_enrichment.sh for cluster bubble plots.
set -euo pipefail

# Export cached neighbor tables only (figures are in 01_umap_analysis.sh): bash 03_de_cluster.sh --stage query --query-gene PSMA1 ATP6V0C
# Redraw revised panels from cached results: bash 03_de_cluster.sh --stage plot
# Data input and output paths: edit these explicit defaults.
DEFAULT_MODEL_INPUT="/fse/home/wangyongyan/Project/CellPainting/Paper_LLm/tutorials/01_mAP/well_level_new_site_pretrain_emb_attention_mask/cpg0021_A549_DeSCORT_1382_nosampling_layernorm_finetune_A549/20200805_A549_WG_Screen/normalization_clip_data"
DEFAULT_ORIGINAL_INPUT="/fse/DC/CP_data/feature_data/20200805_A549_WG_Screen/raw_data/well_level_new/normalization_clip_data"
DEFAULT_OUTPUT_DIR="/fse/home/wangyongyan/Project/CellPainting/DeSCORT/other_model/DeSCORT/tutorials/07_biological/result/03_de_cluster/hit_10_cluster"
# Empty or nonexistent hit file: analyze all common valid non-control genes.
DEFAULT_HIT_FILE="/fse/home/wangyongyan/Project/CellPainting/DeSCORT/common_files/biological/A549_DMEM_hits.xlsx"
DEFAULT_ORIGINAL_FEATURES_PKL="/fse/home/wangyongyan/Project/CellPainting/DeSCORT/common_files/feature_intersection_union/cpg0021_A549/cpg0021_A549_union_feature_1382.pkl"
DEFAULT_CORUM_FILE="/fse/home/wangyongyan/Project/CellPainting/LLLM/common_files/CORUM_humanComplexes.txt"
DEFAULT_STRING_FILE="/fse/home/wangyongyan/Project/CellPainting/LLLM/common_files/STRING_data.csv"
# Local annotation snapshot: needs database, pathway, genes columns. Distances/ranks are ignored.
DEFAULT_PATHWAY_SNAPSHOT="/fse/home/wangyongyan/Project/CellPainting/DeSCORT/tutorials/07_biological/GO_KEGG_ANNOTATED_pathway_UMAP/results/all_pathway_distances.csv"
DEFAULT_SELECTED_PATHWAYS_JSON="/fse/home/wangyongyan/Project/CellPainting/DeSCORT/other_model/DeSCORT/tutorials/07_biological/03_de_cluster_selected_pathways.json"

MODEL_INPUT="/fse/home/wangyongyan/Project/CellPainting/Paper_LLm/tutorials/01_mAP/well_level_new_site_pretrain_emb_attention_mask/cpg0021_A549_DeSCORT_1382_nosampling_layernorm_finetune_A549/20200805_A549_WG_Screen/normalization_clip_data"
ORIGINAL_INPUT="/fse/DC/CP_data/feature_data/20200805_A549_WG_Screen/raw_data/well_level_new/normalization_clip_data"
OUTPUT_DIR="/fse/home/wangyongyan/Project/CellPainting/DeSCORT/other_model/DeSCORT/tutorials/07_biological/result/03_de_cluster/hit_10_cluster"
HIT_FILE="/fse/home/wangyongyan/Project/CellPainting/DeSCORT/common_files/biological/A549_DMEM_hits.xlsx"
ORIGINAL_FEATURES_PKL="/fse/home/wangyongyan/Project/CellPainting/DeSCORT/common_files/feature_intersection_union/cpg0021_A549/cpg0021_A549_union_feature_1382.pkl"
CORUM_FILE="/fse/home/wangyongyan/Project/CellPainting/LLLM/common_files/CORUM_humanComplexes.txt"
STRING_FILE="/fse/home/wangyongyan/Project/CellPainting/LLLM/common_files/STRING_data.csv"
PATHWAY_SNAPSHOT="/fse/home/wangyongyan/Project/CellPainting/DeSCORT/tutorials/07_biological/GO_KEGG_ANNOTATED_pathway_UMAP/results/all_pathway_distances.csv"
SELECTED_PATHWAYS_JSON="/fse/home/wangyongyan/Project/CellPainting/DeSCORT/other_model/DeSCORT/tutorials/07_biological/03_de_cluster_selected_pathways.json"
PARAMS=(
    --dataset-name "A549 DMEM"
    --model-feature-prefix emb_
    --model-feature-dim 256
    --gene-column Metadata_Foci_Barcode_MatchedTo_GeneCode
    --guide-column Metadata_Foci_Barcode_MatchedTo_Barcode
    --control-key-word nontargeting
    --plates CP186A CP186B CP186C CP186D CP186E CP186F CP186G CP186H CP186N
    --hit-column Gene
    --pca-variance 0.95
    --n-clusters 10
    --stability-seeds 5
    --kmeans-n-init 20
    --top-k-export 50
    --top-k-eval 20
    --pair-permutations 200
    --cluster-permutations 200
    --case-top-n 10
    --case-min-overlap 6
    --case-one-per-cluster
    --case-random-repeats 10000
    --pathway-random-repeats 10000
    --de-fdr 0.05
    --min-cohens-d 0.5
    --min-mean-difference 0.2
    --dotplot-top-n 5
    --umap-n-neighbors 10
    --umap-min-dist 0.05
    --umap-seed 42
    --seed 20260926
    --figure-formats pdf svg png
    --png-dpi 400
    --threads 8
    --memory-limit 24GB
)
export USE_TF=0 MPLBACKEND=Agg
export CUDA_VISIBLE_DEVICES=""
export TF_CPP_MIN_LOG_LEVEL=2
export MPLCONFIGDIR="/tmp/descort_de_cluster_matplotlib_1021"
export NUMBA_CACHE_DIR="/tmp/descort_de_cluster_numba_1021"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
exec python -u "03_de_cluster.py" \
    --model-input "$MODEL_INPUT" --original-input "$ORIGINAL_INPUT" \
    --original-features-pkl "$ORIGINAL_FEATURES_PKL" --hit-file "$HIT_FILE" \
    --corum-file "$CORUM_FILE" --string-file "$STRING_FILE" \
    --pathway-snapshot "$PATHWAY_SNAPSHOT" --selected-pathways-json "$SELECTED_PATHWAYS_JSON" \
    --output-dir "$OUTPUT_DIR" "${PARAMS[@]}" "$@"
