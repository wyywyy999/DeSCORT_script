#!/bin/bash
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/../../../.." && pwd)"
RESOURCE_ROOT="${DESCORT_RESOURCE_ROOT:-$PROJECT_ROOT/../../common_files}"
cd "$SCRIPT_DIR"
set -euo pipefail

# Source parquet files and output directories for U2OS compound training.
SOURCE_DATA_PATHS=(
    '/fse/DC/CP_data/feature_data/cpg000_jump_pilot/raw_data_bulk_download/U2OS/Compound/raw_data/normalization_clip_data'
)

train_output_dir="${TRAIN_OUTPUT_DIR:-${DESCORT_OUTPUT_ROOT:-$PROJECT_ROOT/outputs}/06_virtual_cell_cell_line_transfer_Compound/train_output_dir}"
train_save_dir="${TRAIN_SAVE_DIR:-${DESCORT_OUTPUT_ROOT:-$PROJECT_ROOT/outputs}/06_virtual_cell_cell_line_transfer_Compound/train_save_dir}"
gene2index_pkl="$RESOURCE_ROOT/06_vc_compound/Morgan/smiles2index_577.pkl"
feature2index_pkl="/fse/home/wangyongyan/Project/CellPainting/Final_well_level_new_LLM/common_files/00_pretrain/feature2index_vocabulary_7366.pkl"
need_feature_list_pkl="$RESOURCE_ROOT/feature_intersection_union/cpg0000_compound/vocab_intersect_feature_2319.pkl"

key_gene_column="Metadata_Smiles"
control_key_word="CS(=O)C"
# Empty means all parquet files, including subdirectories. Otherwise use
# whitespace-separated path substrings, e.g. "BR00117010 BR00117011".
train_data_names=""
# Concurrent parquet workers; override with NPROC=1 to process serially.
nproc="${NPROC:-4}"
merge_nproc="${MERGE_NPROC:-2}"
verbosity=1

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DATAPARAMS=(
    --source_data_paths "${SOURCE_DATA_PATHS[@]}"
    --train_output_dir "$train_output_dir"
    --train_save_dir "$train_save_dir"
    --gene2index_pkl "$gene2index_pkl"
    --feature2index_pkl "$feature2index_pkl"
    --need_feature_list_pkl "$need_feature_list_pkl"
    --key_gene_column "$key_gene_column"
    --control_key_word "$control_key_word"
    --nproc "$nproc"
    --merge_nproc "$merge_nproc"
    --verbosity "$verbosity"
)
if [[ -n "$train_data_names" ]]; then
    read -r -a selected_data_names <<< "$train_data_names"
    DATAPARAMS+=(--train_data_names "${selected_data_names[@]}")
fi

"${PYTHON_BIN:-python}" "$SCRIPT_DIR/01_process_data.py" "${DATAPARAMS[@]}" "$@"
