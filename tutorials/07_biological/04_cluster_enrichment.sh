#!/usr/bin/env bash
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
RESOURCE_ROOT="${DESCORT_RESOURCE_ROOT:-$PROJECT_ROOT/../../common_files}"
cd "$SCRIPT_DIR"
# Plot saved cluster enrichment bubbles; trailing CLI arguments override these defaults.
set -euo pipefail
PYTHON_SCRIPT="$PROJECT_ROOT/tutorials/07_biological/04_cluster_enrichment.py"

# Existing hit_10_cluster enrichment table. Valid cluster IDs are 0 through 9.
DEFAULT_ENRICHMENT_FILE="$PROJECT_ROOT/tutorials/07_biological/result/03_de_cluster/hit_10_cluster/results/cluster_pathway_complex_enrichment.csv"
DEFAULT_OUTPUT_DIR="$PROJECT_ROOT/tutorials/07_biological/result/04_cluster_enrichment"
PYTHON_BIN="${PYTHON_BIN:-python}"
ENRICHMENT_FILE="${ENRICHMENT_FILE:-$DEFAULT_ENRICHMENT_FILE}"
OUTPUT_DIR="${OUTPUT_DIR:-$DEFAULT_OUTPUT_DIR}"
# Examples: CLUSTER="3" or CLUSTER="2 3 4".
CLUSTER="${CLUSTER:-2 3 4}"
read -r -a CLUSTER_IDS <<< "$CLUSTER"
PARAMS=(
    # Maximum number of pathways per cluster after raw P < 0.05 filtering.
    --top-n 10
    --fdr 0.05
    --figure-formats pdf svg png
    --png-dpi 400
)
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/descort_cluster_enrichment_matplotlib_${UID}}"
exec "$PYTHON_BIN" -u "$PYTHON_SCRIPT" \
    --enrichment-file "$ENRICHMENT_FILE" --output-dir "$OUTPUT_DIR" \
    --cluster "${CLUSTER_IDS[@]}" "${PARAMS[@]}" "$@"
