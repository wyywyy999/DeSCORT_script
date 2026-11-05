#!/bin/bash

# ============================================================================
# 04_run_emb.sh
#
# Launch 04_run_emb.py with argparse parameters.
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
# source_data_paths="/fse/DC/CP_data/feature_data/cpg000_jump_pilot/raw_data_bulk_download/U2OS/Crispr/raw_data/normalization_clip_data/"
source_data_paths="/fse/DC/CP_data/feature_data/github_code_test/raw_data/well_level_new/normalization_clip_data_select_feature/normalization_clip_data"
#   /fse/DC/CP_data/feature_data/cpg0016_jump/source_4/normalization_clip_data_select_feature_only_source_4

# -------------------------------
# Pickle / vocabulary files
# -------------------------------
gene2index_pkl="/fse/home/wangyongyan/Project/CellPainting/DeSCORT/common_files/label_vocab/cpg0021/gene2index_21611.pkl"

feature2index_pkl="/fse/home/wangyongyan/Project/CellPainting/DeSCORT/common_files/feature_vocab/feature2index_vocabulary_7366.pkl"

need_feature_list_pkl='/fse/home/wangyongyan/Project/CellPainting/DeSCORT_Script/common_files/intersection_union_feature/github_code_test_union_feature_894.pkl'
# -------------------------------
# Model settings
# -------------------------------
# pretrained_model_path="/fse/DC/CP_data/feature_data/benchmark_MAD_output/All_MAD_model/paper_model/foundation_model/DeSCORT_layernorm/models"
# pretrained_model_path="/fse/DC/CP_data/feature_data/benchmark_MAD_output/All_MAD_model/paper_model/01_mAP/cpg0000_U2OS_Crispr_1024_50epochs_161/models"
pretrained_model_path="/fse/home/wangyongyan/Project/CellPainting/DeSCORT_Script/output_model/pretrain/models"
# pretrained_model_path="/fse/DC/CP_data/feature_data/benchmark_MAD_output/All_MAD_model/paper_model/03_biological/cpg0021_A549_DMEM_1epochs_layernorm/models"

# save_path="/fse/home/wangyongyan/Project/CellPainting/Paper_LLm/tutorials/01_mAP/well_level_new_site_pretrain_emb_attention_mask/cpg0021_A549_DeSCORT_1382_nosampling_layernorm_finetune_A549"
save_path="/fse/home/wangyongyan/Project/CellPainting/DeSCORT_Script/outputs/02_pretrain/"

# -------------------------------
# Column / gene / data filtering
# -------------------------------
key_gene_column="Metadata_Foci_Barcode_MatchedTo_GeneCode"
# key_gene_column="Metadata_Treatment"
# key_gene_column="Metadata_JCP2022"
# key_gene_column="Metadata_Broad_Sample"

# need_plate_list="CP186A CP186B CP186C CP186D"
# CP186A CP186B CP186C  CP186E CP186F CP186G CP186N
# need_plate_list="CP186F CP186G CP186N"
# need_plate_list="CP186E CP186F CP186G CP186N CP186H"
# need_plate_list="CP257N CP257H"
# need_plate_list="CP257A CP257B CP257D CP257F CP257H"
# need_plate_list="CP257J CP257K CP257L CP257N"
# need_plate_list="BR00117000 BR00117001 BR00117002 BR00117003 BR00117004 BR00117005 BR00118041 BR00118042 BR00118043 BR00118044"
# need_plate_list="BR00116996 BR00116997 BR00116998 BR00116999 BR00118045 BR00118046 BR00118047 BR00118048"
need_plate_list=""

# -------------------------------
# Device / batch settings
# -------------------------------
device="cuda:2"
# device="cuda:0"
# device="cpu"

batch_size=64
max_length=3505
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
log_file="/fse/home/wangyongyan/Project/CellPainting/DeSCORT/other_model/DeSCORT/tutorials/02_pretrain/logs/pretrain_embeddings.log"
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

# Add boolean flags.
if [ "$do_mlm" = true ]; then
    args+=(--do_mlm)
fi

if [ "$sampling" = true ]; then
    args+=(--sampling)
fi

# Run the embedding extraction entry.
python 03_run_emb.py "${args[@]}" "$@"
