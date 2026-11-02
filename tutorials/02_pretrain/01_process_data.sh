#!/bin/bash

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# ============================================================================
# 01_process_data.sh
#
# Launch 01_process_data.py with argparse parameters.
# Edit the variables below to match your data / environment.
# ============================================================================

# -------------------------------
# Source data paths
# -------------------------------
# SOURCE_DATA_PATHS="/fse/DC/CP_data/feature_data/20210422_6W_CP257/raw_data/well_level_new/normalization_clip_data"
# SOURCE_DATA_PATHS="/fse/DC/CP_data/feature_data/20200805_A549_WG_Screen/raw_data/well_level_new/normalization_clip_data"
SOURCE_DATA_PATHS="/fse/DC/CP_data/feature_data/cpg000_jump_pilot/raw_data_bulk_download/A549/Crispr/raw_data/normalization_clip_data"
#   /fse/home/wangyongyan/Project/CellPainting/Final_well_level_new_LLM/tutorials/02_batch/Target_1_Compound/raw_data
#   /fse/DC/CP_data/feature_data/cpg000_jump_pilot/raw_data_bulk_download/U2OS/Compoud/raw_data/normalization_clip_data_select_feature
#   /fse/DC/CP_data/feature_data/20210422_6W_CP257/raw_data/well_level_new/MAD_Zscore_data
#   /fse/DC/CP_data/feature_data/cpg000_jump_pilot/well_data_new/A549/normalization_data_clip_data
#   /fse/DC/CP_data/feature_data/cpg0016_jump/source_1/MAD_data_clip_data
#   /fse/DC/CP_data/feature_data/cpg0016_jump/source_2/MAD_data_clip_data
#   /fse/DC/CP_data/feature_data/cpg0016_jump/source_3/MAD_data_clip_data
#   /fse/DC/CP_data/feature_data/cpg0016_jump/source_4/MAD_data_clip_data
#   /fse/DC/CP_data/feature_data/cpg0016_jump/source_5/MAD_data_clip_data
#   /fse/DC/CP_data/feature_data/cpg0016_jump/source_6/MAD_data_clip_data
#   /fse/DC/CP_data/feature_data/cpg0016_jump/source_7/MAD_data_clip_data
#   /fse/DC/CP_data/feature_data/cpg0016_jump/source_8/MAD_data_clip_data
#   /fse/DC/CP_data/feature_data/cpg0016_jump/source_9/MAD_data_clip_data
#   /fse/DC/CP_data/feature_data/cpg0016_jump/source_10/MAD_data_clip_data
#   /fse/DC/CP_data/feature_data/cpg0016_jump/source_11/MAD_data_clip_data
#   /fse/DC/CP_data/feature_data/cpg0016_jump/source_13/MAD_data_clip_data
#   /fse/DC/CP_data/feature_data/cpg0016_jump/source_15/MAD_data_clip_data
#   /fse/DC/CP_data/feature_data/cpg000_jump_pilot/well_data_new/U2OS/MAD_clip_data
#   /fse/DC/CP_data/feature_data/cpg000_jump_pilot/well_data_new/U2OS/normalization_data_clip_data

# -------------------------------
# Output directories
# -------------------------------
train_output_dir="${TRAIN_OUTPUT_DIR:-${DESCORT_OUTPUT_ROOT:-$PROJECT_ROOT/outputs}/02_pretrain/train_output_dir}"
train_save_dir="${TRAIN_SAVE_DIR:-${DESCORT_OUTPUT_ROOT:-$PROJECT_ROOT/outputs}/02_pretrain/train_save_dir}"

# test_output_dir="/fse/DC/CP_data/feature_data/MAD_merged_data_NNLLM/intersection_data/paper_well_level_new_1_99_clip/cpg0021_HeLa_HPLM/test_tmp_data"
# test_save_dir="/fse/DC/CP_data/feature_data/MAD_merged_data_NNLLM/intersection_data/paper_well_level_new_1_99_clip/cpg0021_HeLa_HPLM/test_merged_data"

# -------------------------------
# Pickle / vocabulary files
# -------------------------------
gene2index_pkl="/fse/home/wangyongyan/Project/CellPainting/Final_well_level_new_LLM/common_files/00_pretrain/gene2index_21611.pkl"
feature2index_pkl="/fse/home/wangyongyan/Project/CellPainting/Final_well_level_new_LLM/common_files/00_pretrain/feature2index_vocabulary_7366.pkl"
# need_feature_list_pkl="/fse/home/wangyongyan/Project/CellPainting/Final_well_level_new_LLM/common_files/00_pretrain/cpg0021_HeLa_union_feature_1615.pkl"
# need_feature_list_pkl="/fse/home/wangyongyan/Project/CellPainting/Final_well_level_new_LLM/common_files/00_pretrain/cpg0021_A549_union_feature_1382.pkl"
need_feature_list_pkl="/fse/home/wangyongyan/Project/CellPainting/DeSCORT/common_files/feature_intersection_union/cpg0000_A549/cpg0000_A549_all_union_feature_2967.pkl"

# -------------------------------
# Column / gene / data names
# -------------------------------
# key_gene_column="Metadata_Foci_Barcode_MatchedTo_GeneCode"
# Optional pickle file containing the gene list. Empty: use all gene2index_pkl keys.
need_gene_list="/fse/home/wangyongyan/Project/CellPainting/DeSCORT/common_files/label_vocab/cpg0000_A549_Crispr_gene_list_161.pkl"
# need_gene_list="/absolute/path/to/need_gene_list.pkl"
key_gene_column="Metadata_Treatment"

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

python "$SCRIPT_DIR/01_process_data.py" "${DATAPARAMS[@]}" "$@"
