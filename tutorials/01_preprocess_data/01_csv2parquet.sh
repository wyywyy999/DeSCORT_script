#!/bin/bash

# ============================================================================
# 01_csv2parquet.sh
#
# Launch 01_csv2parquet.py with argparse parameters.
# Edit the variables below to match your data / environment.
# ============================================================================

# -------------------------------
# Input / output paths
# -------------------------------
csv_dir="/fse/DC/CP_data/feature_data/github_code_test/raw_data/csv_raw_data"
out_dir="/fse/DC/CP_data/feature_data/github_code_test/raw_data/raw_data"

# -------------------------------
# Search settings
# -------------------------------
pattern="*.csv"

# ============================================================================
# Build argument list
# ============================================================================
ARGS=(
    --csv_dir "$csv_dir"
    --out_dir "$out_dir"
    --pattern "$pattern"
)

# ============================================================================
# Run
# ============================================================================


python "01_csv2parquet.py" "${ARGS[@]}" "$@"
