#!/usr/bin/env bash
# Plot saved cluster enrichment bubbles; trailing CLI arguments override these defaults.
set -euo pipefail
PYTHON_SCRIPT=04_cluster_enrichment.py

# Existing hit_10_cluster enrichment table. Valid cluster IDs are 0 through 9.
DEFAULT_ENRICHMENT_FILE="/fse/home/wangyongyan/Project/CellPainting/DeSCORT/other_model/DeSCORT/tutorials/07_biological/result/03_de_cluster/hit_10_cluster/results/cluster_pathway_complex_enrichment.csv"
DEFAULT_OUTPUT_DIR="/fse/home/wangyongyan/Project/CellPainting/DeSCORT/other_model/DeSCORT/tutorials/07_biological/result/04_cluster_enrichment"
ENRICHMENT_FILE="/fse/home/wangyongyan/Project/CellPainting/DeSCORT/other_model/DeSCORT/tutorials/07_biological/result/03_de_cluster/hit_10_cluster/results/cluster_pathway_complex_enrichment.csv"
OUTPUT_DIR="/fse/home/wangyongyan/Project/CellPainting/DeSCORT/other_model/DeSCORT/tutorials/07_biological/result/04_cluster_enrichment"
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
export MPLCONFIGDIR="/tmp/descort_cluster_enrichment_matplotlib_1021"
exec python -u "$PYTHON_SCRIPT" \
    --enrichment-file "$ENRICHMENT_FILE" --output-dir "$OUTPUT_DIR" \
    --cluster "${CLUSTER_IDS[@]}" "${PARAMS[@]}" "$@"
