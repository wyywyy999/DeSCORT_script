#!/bin/bash
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
RESOURCE_ROOT="${DESCORT_RESOURCE_ROOT:-$PROJECT_ROOT/../../common_files}"
cd "$SCRIPT_DIR"

# ============================================================================
# 01_calculate_emb.sh
#
# Launch 01_calculate_emb.py with argparse parameters.
# Edit the variables below to match your data / environment.
# ============================================================================

# -------------------------------
# Source data paths
# -------------------------------
# source_data_paths="/fse/DC/CP_data/feature_data/20200805_A549_WG_Screen/raw_data/well_level_new/normalization_clip_data"
# source_data_paths="/fse/DC/CP_data/feature_data/20210422_6W_CP257/raw_data/well_level_new/normalization_clip_data"
#   /fse/DC/CP_data/feature_data/cpg000_jump_pilot/raw_data_bulk_download/A549/Crispr/raw_data/normalization_clip_data_select_feature
#   /fse/DC/CP_data/feature_data/cpg000_jump_pilot/raw_data_bulk_download/A549/Orf/raw_data/normalization_clip_data_select_feature
#   /fse/DC/CP_data/feature_data/cpg000_jump_pilot/raw_data_bulk_download/A549/Compound/raw_data/normalization_clip_data_select_feature
source_data_paths="/fse/DC/CP_data/feature_data/cpg000_jump_pilot/raw_data_bulk_download/U2OS/Crispr/raw_data/normalization_clip_data/"
#   /fse/DC/CP_data/feature_data/cpg0016_jump/source_4/normalization_clip_data_select_feature_only_source_4

# -------------------------------
# Pickle / vocabulary files
# -------------------------------
# gene2index_pkl="/fse/home/wangyongyan/Project/CellPainting/Final_well_level_new_LLM/common_files/00_pretrain/gene2index_21611.pkl"
# gene2index_pkl="/fse/home/wangyongyan/Project/CellPainting/Final_well_level_new_LLM/common_files/00_pretrain/cpg0016/cpg0016_source_4_and_10_Compound2Index_vocab_73202.pkl"
gene2index_pkl="/fse/home/wangyongyan/Project/CellPainting/DeSCORT/common_files/label_vocab/cpg0000_U2OS_Crispr/cpg0000_U2OS_Crispr_label2index_161.pkl"
# gene2index_pkl="/fse/home/wangyongyan/Project/CellPainting/Final_well_level_new_LLM/common_files/00_pretrain/cpg0016/cpg0016_source_all_Compound2Index_vocab_138907.pkl"
# gene2index_pkl="/fse/home/wangyongyan/Project/CellPainting/Final_well_level_new_LLM/common_files/00_pretrain/cpg0000_three_type/cpg0000_all_Compound2Index_vocab_817.pkl"
# gene2index_pkl="/fse/home/wangyongyan/Project/CellPainting/Final_well_level_new_LLM/common_files/00_pretrain/cpg0000_Orf/cpg0000_Orf_all_Gene2Index_vocab_161.pkl"

feature2index_pkl="/fse/home/wangyongyan/Project/CellPainting/Final_well_level_new_LLM/common_files/00_pretrain/feature2index_vocabulary_7366.pkl"

# need_feature_list_pkl="/fse/home/wangyongyan/Project/CellPainting/Final_well_level_new_LLM/common_files/00_pretrain/cpg0021_A549_union_feature_1382.pkl"
# need_feature_list_pkl="/fse/home/wangyongyan/Project/CellPainting/Final_well_level_new_LLM/common_files/00_pretrain/cpg0000_A549/cpg0000_A549_all_union_feature_2967.pkl"
# need_feature_list_pkl="/fse/home/wangyongyan/Project/CellPainting/Final_well_level_new_LLM/common_files/00_pretrain/cpg0016/cpg0016_source_4_select_union_feature_1963.pkl"
need_feature_list_pkl="/fse/home/wangyongyan/Project/CellPainting/DeSCORT/common_files/feature_intersection_union/cpg0000_U2OS/cpg0000_U2OS_all_union_feature_2705.pkl"
# need_feature_list_pkl="/fse/home/wangyongyan/Project/CellPainting/Final_well_level_new_LLM/common_files/00_pretrain/cpg0021_HeLa_union_feature_1615.pkl"

# -------------------------------
# Model settings
# -------------------------------
# pretrained_model_path="/fse/DC/CP_data/feature_data/benchmark_MAD_output/All_MAD_model/paper_model/foundation_model/DeSCORT/models"
pretrained_model_path="/fse/DC/CP_data/feature_data/benchmark_MAD_output/All_MAD_model/paper_model/01_mAP/cpg0000_U2OS_Crispr_1024_50epochs_161/models"

save_path="/fse/home/wangyongyan/Project/CellPainting/Paper_LLm/tutorials/01_mAP/well_level_new_site_pretrain_emb_attention_mask/cpg0000_U2OS_Crispr_2705_50epochs_161"

# -------------------------------
# Column / gene / data filtering
# -------------------------------
# key_gene_column="Metadata_Foci_Barcode_MatchedTo_GeneCode"
key_gene_column="Metadata_Treatment"
# key_gene_column="Metadata_JCP2022"
# key_gene_column="Metadata_Broad_Sample"

# need_plate_list="CP186A CP186B CP186C CP186D CP186E CP186F CP186G CP186N"
# need_plate_list="CP186F CP186G CP186N"
# need_plate_list="CP186H"
# need_plate_list="CP257N CP257H"
# need_plate_list="CP257A CP257B CP257D CP257F CP257J CP257K CP257L CP257N CP257H"
# need_plate_list="CP257D CP257F CP257J CP257K CP257L CP257N CP257H"
# need_plate_list="BR00117000 BR00117001 BR00117002 BR00117003 BR00117004 BR00117005 BR00118041 BR00118042 BR00118043 BR00118044"
# need_plate_list="BR00116996 BR00116997 BR00116998 BR00116999 BR00118045 BR00118046 BR00118047 BR00118048"
# need_plate_list=""

# -------------------------------
# Device / batch settings
# -------------------------------
device="cuda:7"
# device="cuda:0"
# device="cpu"

batch_size=64
max_length=2705
# max_length=2048

# -------------------------------
# Collator / processing options
# -------------------------------
do_mlm=false
mlm_probability=0.0
# mlm_probability=0.15
sampling=false
nproc=1
verbosity=1

# -------------------------------
# Logging
# -------------------------------
log_file="$PROJECT_ROOT/tutorials/04_mAP/logs/calculate_emb_cpg0000_U2OS_Crispr_1024_50epochs_161_1024.log"
log_level="INFO"

# ============================================================================
# Build argument strings
# ============================================================================
args=(
    --source_data_paths "$source_data_paths"
    --gene2index_pkl "$gene2index_pkl"
    --feature2index_pkl "$feature2index_pkl"
    --need_feature_list_pkl "$need_feature_list_pkl"
    --pretrained_model_path "$pretrained_model_path"
    --save_path "$save_path"
    --key_gene_column "$key_gene_column"
    --device "$device"
    --batch_size "$batch_size"
    --max_length "$max_length"
    --mlm_probability "$mlm_probability"
    --nproc "$nproc"
    --verbosity "$verbosity"
    --log_file "$log_file"
    --log_level "$log_level"
)

if [ -n "$need_plate_list" ]; then
    args+=(--need_plate_list $need_plate_list)
fi

# 添加布尔标志
if [ "$do_mlm" = true ]; then
    args+=(--do_mlm)
fi

if [ "$sampling" = true ]; then
    args+=(--sampling)
fi

# 执行
python 01_calculate_emb.py "${args[@]}" "$@"
