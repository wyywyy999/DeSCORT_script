#!/bin/bash

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
source_data_paths="/fse/DC/CP_data/feature_data/github_code_test/raw_data/well_level_new/normalization_clip_data"

# -------------------------------
# Feature list (union / intersection pickle)
# -------------------------------
feature_list_pkl="/fse/home/wangyongyan/Project/CellPainting/DeSCORT_Script/common_files/intersection_union_feature/github_code_test_union_feature_894.pkl"
# feature_list_pkl="/fse/home/wangyongyan/Project/CellPainting/Final_well_level_new_LLM/common_files/00_pretrain/cpg0000_A549/cpg0000_A549_all_union_feature_2967.pkl"
# feature_list_pkl="/fse/home/wangyongyan/Project/CellPainting/Final_well_level_new_LLM/common_files/00_pretrain/cpg0016/cpg0016_select_union_feature_3183.pkl"

# -------------------------------
# Output
# -------------------------------
output_path="/fse/DC/CP_data/feature_data/cpg000_jump_pilot/raw_data_bulk_download/U2OS/Orf/raw_data/normalization_clip_data_select_feature"

# -------------------------------
# Logging
# -------------------------------
log_file="/fse/home/wangyongyan/Project/CellPainting/DeSCORT/other_model/DeSCORT/tutorials/01_preprocess_data/get_union_feature_data.log"
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
