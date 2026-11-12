#!/bin/bash
set -euo pipefail

# Compound settings follow 06_vc_compound/01_process_data.sh.
# Train/test SMILES filtering follows 06_vc/01_process_data.sh.
SOURCE_DATA_PATHS=(
    '/fse/DC/CP_data/feature_data/cpg000_jump_pilot/raw_data_bulk_download/U2OS/Compound/raw_data/normalization_clip_data'
)

# Keep this split separate from the existing all-compound dataset.
output_root="/fse/DC/CP_data/feature_data/Paper_MAD_unseen_data_generation_clip/06_vc_compound/cell_line_transfer/Morgan_few_shot/50_gene/U2OS"
train_output_dir="/fse/DC/CP_data/feature_data/Paper_MAD_unseen_data_generation_clip/06_vc_compound/cell_line_transfer/Morgan_few_shot/50_gene/U2OS/train_data/tmp"
train_save_dir="/fse/DC/CP_data/feature_data/Paper_MAD_unseen_data_generation_clip/06_vc_compound/cell_line_transfer/Morgan_few_shot/50_gene/U2OS/train_data/hit"
test_output_dir="/fse/DC/CP_data/feature_data/Paper_MAD_unseen_data_generation_clip/06_vc_compound/cell_line_transfer/Morgan_few_shot/50_gene/U2OS/test_data/tmp"
test_save_dir="/fse/DC/CP_data/feature_data/Paper_MAD_unseen_data_generation_clip/06_vc_compound/cell_line_transfer/Morgan_few_shot/50_gene/U2OS/test_data/hit"

# Preprocessing needs SMILES -> integer ID, not fingerprint -> list of IDs.
gene2index_pkl="/fse/home/wangyongyan/Project/CellPainting/DeSCORT/common_files/06_vc_compound/Morgan/smiles2index_577.pkl"
feature2index_pkl="/fse/home/wangyongyan/Project/CellPainting/Final_well_level_new_LLM/common_files/00_pretrain/feature2index_vocabulary_7366.pkl"
need_feature_list_pkl="/fse/home/wangyongyan/Project/CellPainting/DeSCORT/common_files/feature_intersection_union/cpg0000_compound/vocab_intersect_feature_2319.pkl"
train_gene_list_pkl="/fse/home/wangyongyan/Project/CellPainting/DeSCORT/common_files/06_vc/random_gene_list_compound_few_shot/50_gene/smiles_51_train.pkl"
test_gene_list_pkl="/fse/home/wangyongyan/Project/CellPainting/DeSCORT/common_files/06_vc/random_gene_list_compound_few_shot/50_gene/smiles_257_test.pkl"

key_gene_column="Metadata_Smiles"
control_key_word="CS(=O)C"
# Empty arrays process all parquet files recursively. Otherwise use path substrings.
train_data_names=()
test_data_names=()

nproc="${NPROC:-4}"
merge_nproc="${MERGE_NPROC:-2}"
verbosity=1

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

python "01_process_data.py" "${DATAPARAMS[@]}" "$@"
