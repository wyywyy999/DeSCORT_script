#!/usr/bin/env bash
# Run: bash /fse/home/wangyongyan/Project/CellPainting/DeSCORT/other_model/DeSCORT/tutorials/08_three_type/06_umap.sh
# Run one input only: UMAP_INPUT_MODE=model (or original) bash /absolute/path/to/this_script.sh
set -euo pipefail
PYTHON_BIN="${PYTHON_BIN:-/data/home/wangyongyan/miniconda3/envs/DeSCORT/bin/python}"
script_path="/fse/home/wangyongyan/Project/CellPainting/DeSCORT/other_model/DeSCORT/tutorials/08_three_type/05_umap.py"

# Input mode: model / original / both. Analyze all Parquet files for each modality in the input directories.
input_mode="${UMAP_INPUT_MODE:-both}"
model_data_dir="/fse/home/wangyongyan/Project/CellPainting/DeSCORT/tutorials/08_three_type/data/A549_150epochs"
# The three original modalities may use independent directories at different nesting depths.
original_compound_dir="/fse/DC/CP_data/feature_data/cpg000_jump_pilot/raw_data_bulk_download/A549/Compound/raw_data/normalization_clip_data"
original_crispr_dir="/fse/DC/CP_data/feature_data/cpg000_jump_pilot/raw_data_bulk_download/A549/Crispr/raw_data/normalization_clip_data"
original_orf_dir="/fse/DC/CP_data/feature_data/cpg000_jump_pilot/raw_data_bulk_download/A549/Orf/raw_data/normalization_clip_data"
original_feature_list="/fse/home/wangyongyan/Project/CellPainting/DeSCORT/common_files/feature_intersection_union/cpg0000_three_type/vocab_union_feature_3344.pkl"
output_dir="/fse/home/wangyongyan/Project/CellPainting/DeSCORT/other_model/DeSCORT/tutorials/08_three_type/06_umap/outputs/A549_150epochs"

# native uses all selected input dimensions; pca95 fits PCA independently on all aggregated profiles for each input.
representation="pca95"
pca_variance=0.95
top_n=20
n_neighbors=15
min_dist=0.5
umap_metric="cosine"
seed=0
threads=4
dpi=450
figure_formats=(png pdf)  # Choose one or more formats: png, pdf, svg, tiff.
font_family="Arial"
point_area=18
dry_run=false
qa_tools_dir="/data/home/wangyongyan/.codex/skills/nature-figure/scripts"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl_descort_cross_modality}"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/tmp/numba_descort_cross_modality}"
export OPENBLAS_NUM_THREADS="$threads"
export OMP_NUM_THREADS="$threads"

args=(
    --output-dir "$output_dir"
    --representation "$representation"
    --pca-variance "$pca_variance"
    --top-n "$top_n"
    --n-neighbors "$n_neighbors"
    --min-dist "$min_dist"
    --umap-metric "$umap_metric"
    --seed "$seed"
    --threads "$threads"
    --dpi "$dpi"
    --figure-formats "${figure_formats[@]}"
    --font-family "$font_family"
    --point-area "$point_area"
    --qa-tools-dir "$qa_tools_dir"
)
case "$input_mode" in
    model|original|both) ;;
    *) echo "input_mode must be model, original, or both" >&2; exit 2 ;;
esac
if [[ "$input_mode" == model || "$input_mode" == both ]]; then
    args+=(--model-data-dir "$model_data_dir")
fi
if [[ "$input_mode" == original || "$input_mode" == both ]]; then
    args+=(--original-compound-dir "$original_compound_dir"
           --original-crispr-dir "$original_crispr_dir"
           --original-orf-dir "$original_orf_dir"
           --original-feature-list "$original_feature_list")
fi
if [[ "$dry_run" == true ]]; then args+=(--dry-run); fi
"$PYTHON_BIN" -u "$script_path" "${args[@]}" "$@"
