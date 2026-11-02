#!/bin/bash
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/../../../.." && pwd)"
RESOURCE_ROOT="${DESCORT_RESOURCE_ROOT:-$PROJECT_ROOT/../../common_files}"
cd "$SCRIPT_DIR"
set -euo pipefail

# Compound settings follow 06_vc_compound/01_process_data.sh.
# Train/test SMILES filtering follows 06_vc/01_process_data.sh.
SOURCE_DATA_PATHS=(
    '/fse/DC/CP_data/feature_data/cpg000_jump_pilot/raw_data_bulk_download/A549/Compound/raw_data/normalization_clip_data'
)

# Keep this split separate from the existing all-compound dataset.
output_root="${OUTPUT_ROOT:-/fse/DC/CP_data/feature_data/Paper_MAD_unseen_data_generation_clip/06_vc_compound/unseen_gene/Morgan/A549}"
train_output_dir="${TRAIN_OUTPUT_DIR:-${output_root}/train_data/tmp}"
train_save_dir="${TRAIN_SAVE_DIR:-${output_root}/train_data/hit}"
test_output_dir="${TEST_OUTPUT_DIR:-${output_root}/test_data/tmp}"
test_save_dir="${TEST_SAVE_DIR:-${output_root}/test_data/hit}"

# Preprocessing needs SMILES -> integer ID, not fingerprint -> list of IDs.
gene2index_pkl="$RESOURCE_ROOT/06_vc_compound/Morgan/smiles2index_577.pkl"
feature2index_pkl="/fse/home/wangyongyan/Project/CellPainting/Final_well_level_new_LLM/common_files/00_pretrain/feature2index_vocabulary_7366.pkl"
need_feature_list_pkl="$RESOURCE_ROOT/feature_intersection_union/cpg0000_compound/vocab_intersect_feature_2319.pkl"
train_gene_list_pkl="$RESOURCE_ROOT/06_vc/random_gene_list_compound_unseen_gene/smiles_214_train.pkl"
test_gene_list_pkl="$RESOURCE_ROOT/06_vc/random_gene_list_compound_unseen_gene/smiles_94_test.pkl"

key_gene_column="Metadata_Smiles"
control_key_word="CS(=O)C"
# Empty arrays process all parquet files recursively. Otherwise use path substrings.
train_data_names=()
test_data_names=()

nproc="${NPROC:-4}"
merge_nproc="${MERGE_NPROC:-2}"
verbosity=1

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DATAPARAMS=(
    --source_data_paths "${SOURCE_DATA_PATHS[@]}"
    --train_output_dir "$train_output_dir"
    --train_save_dir "$train_save_dir"
    --test_output_dir "$test_output_dir"
    --test_save_dir "$test_save_dir"
    --gene2index_pkl "$gene2index_pkl"
    --feature2index_pkl "$feature2index_pkl"
    --train_gene_list_pkl "$train_gene_list_pkl"
    --test_gene_list_pkl "$test_gene_list_pkl"
    --need_feature_list_pkl "$need_feature_list_pkl"
    --key_gene_column "$key_gene_column"
    --control_key_word "$control_key_word"
    --nproc "$nproc"
    --merge_nproc "$merge_nproc"
    --verbosity "$verbosity"
)
if ((${#train_data_names[@]})); then
    DATAPARAMS+=(--train_data_names "${train_data_names[@]}")
fi
if ((${#test_data_names[@]})); then
    DATAPARAMS+=(--test_data_names "${test_data_names[@]}")
fi

"${PYTHON_BIN:-python}" "$SCRIPT_DIR/01_process_data.py" "${DATAPARAMS[@]}" "$@"
