#!/usr/bin/env bash
# JUMP pilot: download five A549 DMSO control fields, with five channels per field.
# Preview the selection first: bash /absolute/path/to/this_script.sh --dry-run
set -euo pipefail
PYTHON_BIN="${PYTHON_BIN:-/data/home/wangyongyan/miniconda3/envs/DeSCORT/bin/python}"
script_path="/fse/home/wangyongyan/Project/CellPainting/DeSCORT/other_model/DeSCORT/tutorials/08_three_type/07_get_image.py"

# Edit the query arrays; set unused perturbation types to ().
compound_queries=(DMSO)
crispr_queries=()
orf_queries=()

# Use absolute resource/cache paths. Reuse annotations, indexes, and image-location metadata without copying the 2 GB annotation table.
annotation="/fse/home/wangyongyan/Project/CellPainting/DeSCORT/tutorials/000_get_image/metadata/cpg0000-jump-pilot_draft.csv"
load_data_dir="/fse/home/wangyongyan/Project/CellPainting/DeSCORT/tutorials/000_get_image/metadata/load_data_csv"
index_dir="/fse/home/wangyongyan/Project/CellPainting/DeSCORT/tutorials/000_get_image/metadata"
output_dir="/fse/home/wangyongyan/Project/CellPainting/DeSCORT/other_model/DeSCORT/tutorials/08_three_type/07_get_image_results"

# Optional filters: leave empty for no restriction. Query DMSO with compound_queries=(DMSO).
cell_line="A549"
batch="2020_11_04_CPJUMP1"
plate=""
well=""
timepoint=""
site=1
# One site per well; select up to five distinct DMSO control wells.
max_fields=5
all_sites=false
all_fields=false

# Display settings: retain raw TIFFs; export five pseudocolor PNGs, an RGB composite PNG, and a channel montage PNG.
low_percentile=1
high_percentile=99.8
dpi=600
sync_metadata=false
rebuild_index=false
dry_run=false

# Use the local HTTP proxy by default; set GET_IMAGE_USE_PROXY=false for direct access.
# This only changes the script process and its children, not the parent shell.
use_proxy="${GET_IMAGE_USE_PROXY:-true}"
proxy_url="${GET_IMAGE_PROXY_URL:-http://127.0.0.1:17890}"
case "$use_proxy" in
    false) unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy ;;
    true)
        export HTTP_PROXY="$proxy_url" HTTPS_PROXY="$proxy_url" ALL_PROXY="$proxy_url"
        export http_proxy="$proxy_url" https_proxy="$proxy_url" all_proxy="$proxy_url"
        ;;
    *) echo 'GET_IMAGE_USE_PROXY must be true or false.' >&2; exit 2 ;;
esac
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/jump_pilot_image_matplotlib}"

args=(
    --annotation "$annotation"
    --load-data-dir "$load_data_dir"
    --index-dir "$index_dir"
    --output-dir "$output_dir"
    --site "$site"
    --max-fields "$max_fields"
    --low-percentile "$low_percentile"
    --high-percentile "$high_percentile"
    --dpi "$dpi"
)
if [[ -n "$cell_line" ]]; then args+=(--cell-line "$cell_line"); fi
if [[ -n "$batch" ]]; then args+=(--batch "$batch"); fi
if [[ -n "$plate" ]]; then args+=(--plate "$plate"); fi
if [[ -n "$well" ]]; then args+=(--well "$well"); fi
if [[ -n "$timepoint" ]]; then args+=(--timepoint "$timepoint"); fi
if [[ "$all_sites" == true ]]; then args+=(--all-sites); fi
if [[ "$all_fields" == true ]]; then args+=(--all-fields); fi
if [[ "$sync_metadata" == true ]]; then args+=(--sync-metadata); fi
if [[ "$rebuild_index" == true ]]; then args+=(--rebuild-index); fi
if [[ "$dry_run" == true ]]; then args+=(--dry-run); fi
if (( ${#compound_queries[@]} + ${#crispr_queries[@]} + ${#orf_queries[@]} == 0 )); then
    echo 'Configure at least one compound, crispr, or orf query.' >&2
    exit 2
fi
for compound in "${compound_queries[@]}"; do
    "$PYTHON_BIN" -u "$script_path" --compound "$compound" --perturbation-type compound "${args[@]}" "$@"
done
for gene in "${crispr_queries[@]}"; do
    "$PYTHON_BIN" -u "$script_path" --gene "$gene" --perturbation-type crispr "${args[@]}" "$@"
done
for gene in "${orf_queries[@]}"; do
    "$PYTHON_BIN" -u "$script_path" --gene "$gene" --perturbation-type orf "${args[@]}" "$@"
done
