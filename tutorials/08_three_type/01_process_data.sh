#!/bin/bash

# ============================================================================
# 01_process_data.sh
#
# Launch 01_process_data.py with argparse parameters.
# Edit the variables below to match your data / environment.
# ============================================================================

# -------------------------------
# Source data paths
# -------------------------------

SOURCE_DATA_PATHS='/fse/DC/CP_data/feature_data/cpg000_jump_pilot/raw_data_bulk_download/U2OS/Orf/raw_data/normalization_clip_data'

# -------------------------------
# Output directories
# -------------------------------
train_output_dir="/fse/DC/CP_data/feature_data/MAD_merged_data_NNLLM/intersection_data/paper_well_level_new_1_99_clip/cpg0000_three_type/U2OS/Orf/train_tmp_data"
train_save_dir="/fse/DC/CP_data/feature_data/MAD_merged_data_NNLLM/intersection_data/paper_well_level_new_1_99_clip/cpg0000_three_type/U2OS/Orf/train_merged_data"

# -------------------------------
# Pickle / vocabulary files
# -------------------------------
gene2index_pkl="/fse/home/wangyongyan/Project/CellPainting/DeSCORT/common_files/label_vocab/cpg0000/cpg0000_three_type/cpg0000_three_type_label2index_161.pkl"
feature2index_pkl="/fse/home/wangyongyan/Project/CellPainting/DeSCORT/common_files/feature_vocab/feature2index_vocabulary_7366.pkl"
need_feature_list_pkl='/fse/home/wangyongyan/Project/CellPainting/DeSCORT/common_files/feature_intersection_union/cpg0000_three_type/vocab_union_feature_3344.pkl'

# -------------------------------
# Column / gene / data names
# -------------------------------
key_gene_column="Metadata_Treatment"
# Optional pickle file containing the gene list. Empty: use all gene2index_pkl keys.
# need_gene_list="/fse/home/wangyongyan/Project/CellPainting/DeSCORT/common_files/label_vocab/cpg0000_A549_Crispr_gene_list_161.pkl"
need_gene_list=''

# Empty array/string: recursively process every parquet file in SOURCE_DATA_PATHS.
# Nonempty array/string: filter paths by the supplied name substrings.
train_data_names=()
# train_data_names="CP257A CP257B CP257D CP257F"
# train_data_names="CP186N"
# train_data_names="BR00117000 BR00117001 BR00117002 BR00117003 BR00117004 BR00117005 BR00118041 BR00118042 BR00118043 BR00118044"
# train_data_names="BR00116996 BR00116997 BR00116998 BR00116999 BR00118045 BR00118046 BR00118047 BR00118048"
# test_data_names="CP186A CP186H"

# -------------------------------
# Processing options
# -------------------------------
nproc=1
merge_nproc=2
verbosity=1

# ============================================================================
# Build argument arrays
# ============================================================================
read -r -a source_paths_array <<< "$SOURCE_DATA_PATHS"
read -r -a train_names_array <<< "${train_data_names[*]}"
DATAPARAMS=(
    --source_data_paths "${source_paths_array[@]}"
    --train_output_dir "$train_output_dir"
    --train_save_dir "$train_save_dir"
    --gene2index_pkl "$gene2index_pkl"
    --feature2index_pkl "$feature2index_pkl"
    --need_feature_list_pkl "$need_feature_list_pkl"
    --key_gene_column "$key_gene_column"
    --nproc "$nproc"
    --merge_nproc "$merge_nproc"
    --verbosity "$verbosity"
)
if (( ${#train_names_array[@]} )); then
    DATAPARAMS+=(--train_data_names "${train_names_array[@]}")
fi
if [[ -n "$need_gene_list" ]]; then
    DATAPARAMS+=(--need_gene_list "$need_gene_list")
fi

python "/fse/home/wangyongyan/Project/CellPainting/DeSCORT/other_model/DeSCORT/tutorials/08_three_type/01_process_data.py" "${DATAPARAMS[@]}" "$@"
