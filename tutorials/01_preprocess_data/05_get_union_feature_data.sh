#!/bin/bash
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
RESOURCE_ROOT="${DESCORT_RESOURCE_ROOT:-$PROJECT_ROOT/../../common_files}"
cd "$SCRIPT_DIR"

# ============================================================================
# 05_get_union_feature_data.sh
#
# Launch 05_get_union_feature_data.py with argparse parameters.
# Edit the variables below to match your data / environment.
# ============================================================================

# -------------------------------
# Source data paths
# -------------------------------
# U2OS ORF
source_data_paths="/fse/DC/CP_data/feature_data/cpg000_jump_pilot/raw_data_bulk_download/U2OS/Orf/raw_data/normalization_clip_data"
# U2OS CRISPR
# source_data_paths="/fse/DC/CP_data/feature_data/cpg000_jump_pilot/raw_data_bulk_download/U2OS/Crispr/raw_data/normalization_clip_data"
# U2OS Compound
# source_data_paths="/fse/DC/CP_data/feature_data/cpg000_jump_pilot/raw_data_bulk_download/U2OS/Compound/raw_data/normalization_clip_data"
# A549 ORF
# source_data_paths="/fse/DC/CP_data/feature_data/cpg000_jump_pilot/raw_data_bulk_download/A549/Orf/raw_data/normalization_clip_data"
# A549 CRISPR
# source_data_paths="/fse/DC/CP_data/feature_data/cpg000_jump_pilot/raw_data_bulk_download/A549/Crispr/raw_data/normalization_clip_data"
# A549 Compound
# source_data_paths="/fse/DC/CP_data/feature_data/cpg000_jump_pilot/raw_data_bulk_download/A549/Compound/raw_data/normalization_clip_data"
# cpg0016 source_4
# source_data_paths="/fse/DC/CP_data/feature_data/cpg0016_jump/source_4/normalization_clip_data"

# -------------------------------
# Feature list (union / intersection pickle)
# -------------------------------
feature_list_pkl="/fse/home/wangyongyan/Project/CellPainting/Final_well_level_new_LLM/common_files/00_pretrain/cpg0000_U2OS/cpg0000_U2OS_all_union_feature_2705.pkl"
# feature_list_pkl="/fse/home/wangyongyan/Project/CellPainting/Final_well_level_new_LLM/common_files/00_pretrain/cpg0000_A549/cpg0000_A549_all_union_feature_2967.pkl"
# feature_list_pkl="/fse/home/wangyongyan/Project/CellPainting/Final_well_level_new_LLM/common_files/00_pretrain/cpg0016/cpg0016_select_union_feature_3183.pkl"

# -------------------------------
# Output
# -------------------------------
output_path="/fse/DC/CP_data/feature_data/cpg000_jump_pilot/raw_data_bulk_download/U2OS/Orf/raw_data/normalization_clip_data_select_feature"

# -------------------------------
# Logging
# -------------------------------
log_file="$PROJECT_ROOT/tutorials/01_preprocess_data/get_union_feature_data.log"
log_level="INFO"

# ============================================================================
# Build argument list
# ============================================================================
ARGS=(
    --source_data_paths "$source_data_paths"
    --feature_list_pkl "$feature_list_pkl"
    --output_path "$output_path"
    --log_file "$log_file"
    --log_level "$log_level"
)

# ============================================================================
# Run
# ============================================================================
python "05_get_union_feature_data.py" "${ARGS[@]}" "$@"
