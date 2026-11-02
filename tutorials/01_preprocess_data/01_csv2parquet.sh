#!/bin/bash
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
RESOURCE_ROOT="${DESCORT_RESOURCE_ROOT:-$PROJECT_ROOT/../../common_files}"
cd "$SCRIPT_DIR"

# ============================================================================
# 01_csv2parquet.sh
#
# Launch 01_csv2parquet.py with argparse parameters.
# Edit the variables below to match your data / environment.
# ============================================================================

# -------------------------------
# Input / output paths
# -------------------------------
csv_dir="/fse/DC/CP_data/feature_data/20200805_A549_WG_Screen/raw_data/paper_code_test/csv_raw_data"
out_dir="/fse/DC/CP_data/feature_data/20200805_A549_WG_Screen/raw_data/paper_code_test/raw_data"

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
