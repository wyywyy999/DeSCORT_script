#!/usr/bin/env bash
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/../../../.." && pwd)"
RESOURCE_ROOT="${DESCORT_RESOURCE_ROOT:-$PROJECT_ROOT/../../common_files}"
cd "$SCRIPT_DIR"
set -euo pipefail

# Process one cell line at a time. Defaults follow the HeLa DMEM CRISPR script.
# Change source, output and plate names together when processing another cell line.
SOURCE_DATA_PATHS=(
    /fse/DC/CP_data/feature_data/20210422_6W_CP257/raw_data/well_level_new/normalization_clip_data
)
# A549 source:
# /fse/DC/CP_data/feature_data/20200805_A549_WG_Screen/raw_data/well_level_new/normalization_clip_data

train_output_dir="${TRAIN_OUTPUT_DIR:-${DESCORT_OUTPUT_ROOT:-$PROJECT_ROOT/outputs}/06_virtual_cell_cell_line_transfer_Crispr/train_output_dir}"
train_save_dir="${TRAIN_SAVE_DIR:-${DESCORT_OUTPUT_ROOT:-$PROJECT_ROOT/outputs}/06_virtual_cell_cell_line_transfer_Crispr/train_save_dir}"

gene2index_pkl=$RESOURCE_ROOT/ESM2_files/gene2index_21611.pkl
esm2_dict_pkl=$RESOURCE_ROOT/ESM2_files/ESM2_pert_features_with_nontargeting.pkl
feature2index_pkl=$RESOURCE_ROOT/feature_vocab/feature2index_vocabulary_7366.pkl
need_feature_list_pkl=/fse/home/wangyongyan/Project/CellPainting/Final_well_level_new_LLM/common_files/04_new_data_generation/intersection_columns_1107.pkl

# Empty: process all mapped genes with ESM2, without a train/test gene split.
# Set a pickle path only when an explicit gene subset is needed.
train_gene_list_pkl=""
key_gene_column=Metadata_Foci_Barcode_MatchedTo_GeneCode
control_key_word=nontargeting
train_data_names=(CP257A CP257B CP257D CP257F CP257H)
# Use train_data_names=() to select every parquet file under SOURCE_DATA_PATHS.

nproc="${NPROC:-1}"
merge_nproc="${MERGE_NPROC:-2}"
verbosity=1
SCRIPT=$PROJECT_ROOT/tutorials/06_virtual_cell/cell_line_transfer/Crispr/01_process_data.py
DATAPARAMS=(
    --source_data_paths "${SOURCE_DATA_PATHS[@]}"
    --train_output_dir "$train_output_dir"
    --train_save_dir "$train_save_dir"
    --gene2index_pkl "$gene2index_pkl"
    --esm2_dict_pkl "$esm2_dict_pkl"
    --feature2index_pkl "$feature2index_pkl"
    --need_feature_list_pkl "$need_feature_list_pkl"
    --key_gene_column "$key_gene_column"
    --control_key_word "$control_key_word"
    --nproc "$nproc"
    --merge_nproc "$merge_nproc"
    --verbosity "$verbosity"
)
if [[ -n "$train_gene_list_pkl" ]]; then
    DATAPARAMS+=(--train_gene_list_pkl "$train_gene_list_pkl")
fi
if ((${#train_data_names[@]})); then
    DATAPARAMS+=(--train_data_names "${train_data_names[@]}")
fi
python "$SCRIPT" "${DATAPARAMS[@]}" "$@"
