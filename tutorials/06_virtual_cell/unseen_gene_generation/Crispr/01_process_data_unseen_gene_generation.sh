#!/bin/bash

set -euo pipefail

# ============================================================================
# 01_orocess_data_unseen_gene_generation.sh
#
# Launch 01_orocess_data_unseen_gene_generation.py with argparse parameters.
# Edit the variables below to match your data / environment.
# ============================================================================

# -------------------------------
# Source data paths
# -------------------------------
SOURCE_DATA_PATHS=(
    '/fse/DC/CP_data/feature_data/20210422_6W_CP257/raw_data/well_level_new/normalization_clip_data'
)

# SOURCE_DATA_PATHS=(
#     "/fse/DC/CP_data/feature_data/20200805_A549_WG_Screen/raw_data/well_level_new/normalization_clip_data"
# )


# -------------------------------
# Output directories
# -------------------------------
train_output_dir="/fse/home/wangyongyan/Project/CellPainting/DeSCORT/other_model/DeSCORT/outputs/06_virtual_cell_unseen_gene_generation_Crispr/train_output_dir"
train_save_dir="/fse/home/wangyongyan/Project/CellPainting/DeSCORT/other_model/DeSCORT/outputs/06_virtual_cell_unseen_gene_generation_Crispr/train_save_dir"
test_output_dir="/fse/home/wangyongyan/Project/CellPainting/DeSCORT/other_model/DeSCORT/outputs/06_virtual_cell_unseen_gene_generation_Crispr/test_output_dir"
test_save_dir="/fse/home/wangyongyan/Project/CellPainting/DeSCORT/other_model/DeSCORT/outputs/06_virtual_cell_unseen_gene_generation_Crispr/test_save_dir"

# -------------------------------
# Pickle / vocabulary files
# -------------------------------
gene2index_pkl="/fse/home/wangyongyan/Project/CellPainting/DeSCORT/common_files/ESM2_files/gene2index_21611.pkl"
esm2_dict_pkl="/fse/home/wangyongyan/Project/CellPainting/DeSCORT/common_files/ESM2_files/ESM2_pert_features_with_nontargeting.pkl"
feature2index_pkl="/fse/home/wangyongyan/Project/CellPainting/DeSCORT/common_files/feature_vocab/feature2index_vocabulary_7366.pkl"
# train_gene_list_pkl="/fse/home/wangyongyan/Project/CellPainting/DeSCORT/common_files/06_vc/random_gene_list/gene_501_train.pkl"
# test_gene_list_pkl="/fse/home/wangyongyan/Project/CellPainting/DeSCORT/common_files/06_vc/random_gene_list/gene_501_test.pkl"
train_gene_list_pkl="/fse/home/wangyongyan/Project/CellPainting/DeSCORT/common_files/06_vc/random_gene_list/few_shot/gene_50_train.pkl"
test_gene_list_pkl="/fse/home/wangyongyan/Project/CellPainting/DeSCORT/common_files/06_vc/random_gene_list/few_shot/gene_50_test.pkl"
need_feature_list_pkl="/fse/home/wangyongyan/Project/CellPainting/Final_well_level_new_LLM/common_files/04_new_data_generation/intersection_columns_1107.pkl"

# -------------------------------
# Column / data-name settings
# -------------------------------
key_gene_column="Metadata_Foci_Barcode_MatchedTo_GeneCode"
control_key_word="nontargeting"
train_data_names=("CP257A" "CP257B" "CP257D" "CP257F" "CP257H" )

test_data_names=("CP257A" "CP257B" "CP257D" "CP257F" "CP257H" )

# train_data_names=()

# test_data_names=()

# train_data_names=("CP257J" "CP257K" "CP257L" "CP257N"  )

# test_data_names=("CP257J" "CP257K" "CP257L" "CP257N" )
# -------------------------------
# Processing options
# -------------------------------
nproc=1
merge_nproc=2
verbosity=1

# ============================================================================
# Build and run command
# ============================================================================

DATAPARAMS=(
    --source_data_paths "${SOURCE_DATA_PATHS[@]}"
    --train_output_dir "$train_output_dir"
    --train_save_dir "$train_save_dir"
    --test_output_dir "$test_output_dir"
    --test_save_dir "$test_save_dir"
    --gene2index_pkl "$gene2index_pkl"
    --esm2_dict_pkl "$esm2_dict_pkl"
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
