#!/bin/bash
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
RESOURCE_ROOT="${DESCORT_RESOURCE_ROOT:-$PROJECT_ROOT/../../common_files}"
cd "$SCRIPT_DIR"

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
source_data_paths='/fse/DC/CP_data/feature_data/cpg000_jump_pilot/raw_data_bulk_download/A549/Compound/raw_data/normalization_clip_data_select_feature /fse/DC/CP_data/feature_data/cpg000_jump_pilot/raw_data_bulk_download/U2OS/Compound/raw_data/normalization_clip_data_select_feature'

# -------------------------------
# Vocabulary (optional)
# -------------------------------
vocab_file="/fse/home/wangyongyan/Project/CellPainting/Final_well_level_new_LLM/common_files/00_pretrain/feature2index_vocabulary_7366.pkl"
# vocab_file=""  # set to empty to skip vocab filtering

# -------------------------------
# Output
# -------------------------------
output_dir="${OUTPUT_DIR:-${DESCORT_OUTPUT_ROOT:-$PROJECT_ROOT/outputs}/01_preprocess_data/output_dir}"
output_prefix="vocab"

# -------------------------------
# Logging
# -------------------------------
log_file="$PROJECT_ROOT/tutorials/01_preprocess_data/select_features.log"
log_level="INFO"

# ============================================================================
# Build argument list
# ============================================================================
ARGS=(
    --source_data_paths $source_data_paths
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
