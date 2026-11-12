#!/usr/bin/env bash
# Run profile comparison and built-in temporal retrieval together; configure all settings here.
set -euo pipefail
PYTHON_BIN="${PYTHON_BIN:-/data/home/wangyongyan/miniconda3/envs/DeSCORT/bin/python}"
script_path="/fse/home/wangyongyan/Project/CellPainting/DeSCORT/other_model/DeSCORT/tutorials/08_three_type/06_compare.py"
# comparison: distances/background/retrieval/rankings/up-down; temporal: built-in time-stratified MRR/Hit.
stages=(comparison temporal)
cell_line="A549"
model_epochs=150
model_data_dir="/fse/home/wangyongyan/Project/CellPainting/DeSCORT/tutorials/08_three_type/data/A549_150epochs"
# Leave empty to aggregate model_data_dir on each run, or specify an existing UMAP model output directory.
model_profiles_dir=""
# Use the UMAP original output directory or a compatible legacy aggregation directory.
original_profiles_dir="/fse/home/wangyongyan/Project/CellPainting/DeSCORT/tutorials/08_three_type/biological/A549/old_version_orignal"
output_dir="/fse/home/wangyongyan/Project/CellPainting/DeSCORT/other_model/DeSCORT/tutorials/08_three_type/07_compare_results/A549_150epochs"

# Profile comparison settings: pool all timepoints and fit PCA independently for each representation.
pca_variance=0.95
top_k=(20 50)
# Model–Original overlap curves for all three modality pairs (per-gene mean similarity).
overlap_curve_step=5
overlap_curve_max=100
bootstraps=5000
seed=20261012
threads=4
dpi=600
figure_formats=(png pdf)  # Profile figures support png, pdf, svg, and tiff.
font_family="DejaVu Sans"
font_dir=""
qa_tools_dir="/data/home/wangyongyan/.codex/skills/nature-figure/scripts"

# Time-stratified retrieval settings (uses descort.multimodality.temporal_retrieval).
original_results="/fse/home/wangyongyan/Project/CellPainting/DeSCORT/tutorials/08_three_type/mAP/A549_original/mAP_result_original"
descort_results="/fse/home/wangyongyan/Project/CellPainting/DeSCORT/tutorials/08_three_type/mAP/A549_150epochs/mAP_result"
descort_data_root="/fse/home/wangyongyan/Project/CellPainting/DeSCORT/tutorials/08_three_type/data/A549_150epochs"
raw_feature_list="/fse/home/wangyongyan/Project/CellPainting/DeSCORT/common_files/feature_intersection_union/cpg0000_three_type/vocab_union_feature_3344.pkl"
feature_vocab="/fse/home/wangyongyan/Project/CellPainting/DeSCORT/common_files/feature_vocab/feature2index_vocabulary_7366.pkl"
target_modalities=(crispr orf)
batch_scope="primary"
temporal_representation="pca95"
reference_seed=42
descort_top_k=5
original_worse_than=20
minimum_rank_gain=20
dry_run=false

export MPLBACKEND=Agg
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl_descort_compare07}"
export OPENBLAS_NUM_THREADS="$threads"
export OMP_NUM_THREADS="$threads"
export MKL_NUM_THREADS="$threads"
args=(
    --stages "${stages[@]}"
    --cell-line "$cell_line"
    --model-epochs "$model_epochs"
    --model-data-dir "$model_data_dir"
    --original-profiles-dir "$original_profiles_dir"
    --output-dir "$output_dir"
    --pca-variance "$pca_variance"
    --top-k "${top_k[@]}"
    --overlap-curve-step "$overlap_curve_step"
    --overlap-curve-max "$overlap_curve_max"
    --bootstraps "$bootstraps"
    --seed "$seed"
    --threads "$threads"
    --dpi "$dpi"
    --figure-formats "${figure_formats[@]}"
    --font-family "$font_family"
    --qa-tools-dir "$qa_tools_dir"
    --original-results "$original_results"
    --descort-results "$descort_results"
    --raw-feature-list "$raw_feature_list"
    --feature-vocab "$feature_vocab"
    --target-modalities "${target_modalities[@]}"
    --batch-scope "$batch_scope"
    --temporal-representation "$temporal_representation"
    --reference-seed "$reference_seed"
    --descort-top-k "$descort_top_k"
    --original-worse-than "$original_worse_than"
    --minimum-rank-gain "$minimum_rank_gain"
)
if [[ -n "$model_profiles_dir" ]]; then args+=(--model-profiles-dir "$model_profiles_dir"); fi
if [[ -n "$descort_data_root" ]]; then args+=(--descort-data-root "$descort_data_root"); fi
if [[ -n "$font_dir" ]]; then args+=(--font-dir "$font_dir"); fi
if [[ "$dry_run" == true ]]; then args+=(--dry-run); fi
"$PYTHON_BIN" -u "$script_path" "${args[@]}" "$@"
