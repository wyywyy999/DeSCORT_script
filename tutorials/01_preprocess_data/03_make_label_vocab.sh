#!/bin/bash

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
# source_data_paths=("/path/to/another/dataset")
source_data_paths=(
    /fse/DC/CP_data/feature_data/github_code_test/raw_data/well_level_new/normalization_clip_data
)
# -------------------------------
# Label column
# -------------------------------
# label_column="Metadata_Broad_Sample"
# label_column="Metadata_JCP2022"
# label_column="Metadata_Treatment"
label_column="Metadata_Foci_Barcode_MatchedTo_GeneCode"

# -------------------------------
# Output
# -------------------------------
output_dir="/fse/home/wangyongyan/Project/CellPainting/DeSCORT_Script/common_files/label_vocab"
output_prefix="github_code_test"

# -------------------------------
# Logging
# -------------------------------
log_file="/fse/home/wangyongyan/Project/CellPainting/DeSCORT_Script/common_files/label_vocab/make_label_vocab.log"
log_level="INFO"

# ============================================================================
# Build argument list
# ============================================================================
ARGS=(
    --source_data_paths "${source_data_paths[@]}"
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
