#!/bin/bash

# ============================================================================
# 04_select_intersection_union_feature.sh
#
# Launch 04_select_intersection_union_feature.py with argparse parameters.
# Edit the variables below to match your data / environment.
# ============================================================================

# -------------------------------
# Source data paths
# -------------------------------
# cpg0016 source_4
# source_data_paths="/fse/DC/CP_data/feature_data/cpg0016_jump/source_4/MAD_feature_select_clip_data"
# A549 + U2OS JUMP pilot (all types)
source_data_paths=(
    /fse/DC/CP_data/feature_data/github_code_test/raw_data/well_level_new/normalization_select_feature_data
)

# -------------------------------
# Vocabulary (optional)
# -------------------------------
vocab_file="/fse/home/wangyongyan/Project/CellPainting/DeSCORT/common_files/feature_vocab/feature2index_vocabulary_7366.pkl"
# vocab_file=""  # set to empty to skip vocab filtering

# -------------------------------
# Output
# -------------------------------
output_dir="/fse/home/wangyongyan/Project/CellPainting/DeSCORT_Script/common_files/intersection_union_feature"
output_prefix="github_code_test"

# -------------------------------
# Logging
# -------------------------------
log_file="/fse/home/wangyongyan/Project/CellPainting/DeSCORT_Script/common_files/intersection_union_feature/select_features.log"
log_level="INFO"

# ============================================================================
# Build argument list
# ============================================================================
ARGS=(
    --source_data_paths "${source_data_paths[@]}"
    --output_dir "$output_dir"
    --output_prefix "$output_prefix"
    --log_file "$log_file"
    --log_level "$log_level"
)

if [ -n "$vocab_file" ]; then
    ARGS+=(--vocab_file "$vocab_file")
fi

# ============================================================================
# Run
# ============================================================================
python "04_select_intersection_union_feature.py" "${ARGS[@]}" "$@"
