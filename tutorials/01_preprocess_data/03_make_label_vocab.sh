#!/bin/bash
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
RESOURCE_ROOT="${DESCORT_RESOURCE_ROOT:-$PROJECT_ROOT/../../common_files}"
cd "$SCRIPT_DIR"

# ============================================================================
# 03_make_label_vocab.sh
#
# Launch 03_make_label_vocab.py with argparse parameters.
# Edit the variables below to match your data / environment.
# ============================================================================

# -------------------------------
# Source data paths
# -------------------------------
# JUMP pilot A549 + U2OS (Compound / CRISPR / ORF)
# source_data_paths="/fse/DC/CP_data/feature_data/MAD_merged_data_NNLLM/intersection_data/paper_well_level_new_1_99_clip/cpg0016/batch_effect/raw_data"
source_data_paths='/fse/DC/CP_data/feature_data/cpg000_jump_pilot/raw_data_bulk_download/A549/Crispr/raw_data/normalization_clip_data /fse/DC/CP_data/feature_data/cpg000_jump_pilot/raw_data_bulk_download/A549/Compound/raw_data/normalization_clip_data /fse/DC/CP_data/feature_data/cpg000_jump_pilot/raw_data_bulk_download/A549/Orf/raw_data/normalization_clip_data /fse/DC/CP_data/feature_data/cpg000_jump_pilot/raw_data_bulk_download/U2OS/Crispr/raw_data/normalization_clip_data /fse/DC/CP_data/feature_data/cpg000_jump_pilot/raw_data_bulk_download/U2OS/Compound/raw_data/normalization_clip_data /fse/DC/CP_data/feature_data/cpg000_jump_pilot/raw_data_bulk_download/U2OS/Orf/raw_data/normalization_clip_data'
# -------------------------------
# Label column
# -------------------------------
label_column="Metadata_Broad_Sample"
# label_column="Metadata_JCP2022"
# label_column="Metadata_Treatment"
# label_column="Metadata_Foci_Barcode_MatchedTo_GeneCode"

# -------------------------------
# Output
# -------------------------------
output_dir="${OUTPUT_DIR:-${DESCORT_OUTPUT_ROOT:-$PROJECT_ROOT/outputs}/01_preprocess_data/output_dir}"
output_prefix="cpg0000_three_type"

# -------------------------------
# Logging
# -------------------------------
log_file="$PROJECT_ROOT/tutorials/01_preprocess_data/make_label_vocab.log"
log_level="INFO"

# ============================================================================
# Build argument list
# ============================================================================
ARGS=(
    --source_data_paths $source_data_paths
    --label_column "$label_column"
    --output_dir "$output_dir"
    --output_prefix "$output_prefix"
    --log_file "$log_file"
    --log_level "$log_level"
)

# ============================================================================
# Run
# ============================================================================
python "03_make_label_vocab.py" "${ARGS[@]}" "$@"
