#!/bin/bash
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
RESOURCE_ROOT="${DESCORT_RESOURCE_ROOT:-$PROJECT_ROOT/../../common_files}"
cd "$SCRIPT_DIR"

# ============================================================================
# 02_calculate_mAP.sh
#
# Launch 02_calculate_mAP.py with argparse parameters.
# Edit the variables below to match your data / environment.
# ============================================================================

# -------------------------------
# mAP type
# -------------------------------
# replicates_activity | sister_activity | sister_consistency
# mAP_type="replicates_activity"
# mAP_type="sister_activity"
mAP_type="sister_consistency"

# -------------------------------
# Source data paths
# -------------------------------
source_data_paths="/fse/home/wangyongyan/Project/CellPainting/DeSCORT_Script/outputs/02_pretrain/raw_data/normalization_clip_data"

# -------------------------------
# Pickle / vocabulary files
# -------------------------------
need_feature_list_pkl="/fse/home/wangyongyan/Project/CellPainting/DeSCORT_Script/common_files/intersection_union_feature/github_code_test_union_feature_894.pkl"


feature2index_pkl="/fse/home/wangyongyan/Project/CellPainting/DeSCORT/common_files/feature_vocab/feature2index_vocabulary_7366.pkl"

# -------------------------------
# Output
# -------------------------------
# save_path="/fse/home/wangyongyan/Project/CellPainting/Paper_LLm/tutorials/01_mAP/well_level_attention_mask_result/sister_level/perturbation_activity/well-level/condition_pretrain_finetune/cpg0021_HeLa_DeSCORT_1024_sampling"
# save_path='/fse/home/wangyongyan/Project/CellPainting/Paper_LLm/tutorials/01_mAP/well_level_attention_mask_result/replicates_level/perturbation_activity/well-level/condition_pretrain_finetune/cpg0021_HeLa_DeSCORT_1615_nosampling_CP257H'
save_path='/fse/home/wangyongyan/Project/CellPainting/DeSCORT_Script/outputs/01_mAP/model/model_cpg0021_HeLa_1615_CP257N'
# save_path="/fse/home/wangyongyan/Project/CellPainting/Paper_LLm/tutorials/01_mAP/well_level_attention_mask_result/sister_level/perturbation_consistency/well-level/condition_pretrain_finetune_layernrom/cpg0021_HeLa_DeSCORT_1615_nosampling_layernorm_finetune_CP257N"
# -------------------------------
# Column / gene settings
# -------------------------------
perturbation_gene_key="Metadata_Foci_Barcode_MatchedTo_GeneCode"
# perturbation_gene_key="Metadata_Treatment"
# perturbation_gene_key="Metadata_Broad_Sample"

sgRNA_barcode_key="Metadata_Foci_Barcode_MatchedTo_Barcode"
# sgRNA_barcode_key="Metadata_Target_Sequence"
# sgRNA_barcode_key="Metadata_Broad_Sample"

negative_control_list="nontargeting"

# -------------------------------
# Aggregation / processing
# -------------------------------
choose_type="well-level"

# Whether to intersect features across plates(original_data)
do_intersect_features=false
# Whether to group by need_columns and take mean
do_groupby_mean=true

# need_columns
need_columns="Metadata_Foci_Barcode_MatchedTo_GeneCode Metadata_Foci_Barcode_MatchedTo_Barcode Metadata_Foci_well"
# need_columns="Metadata_Treatment Metadata_Target_Sequence Metadata_Well"

# group_by_columns
group_by_columns="Metadata_Foci_Barcode_MatchedTo_GeneCode Metadata_Foci_Barcode_MatchedTo_Barcode Metadata_Foci_well"
# group_by_columns="Metadata_Treatment Metadata_Target_Sequence Metadata_Well"

# -------------------------------
# Data filtering
# -------------------------------
# need_plate_list="CP257N"
# need_plate_list="CP186H"
# need_plate_list="CP186A CP186B CP186C CP186D CP186E CP186F CP186G CP186H CP186N"
# need_plate_list="CP186A CP186B CP186C CP186D CP186E CP186F CP186G CP186H CP186N"
# need_plate_list="CP257A CP257B CP257D CP257F CP257H CP257J CP257K CP257L CP257N"
# need_plate_list="BR00117000 BR00117001 BR00117002 BR00117003 BR00117004 BR00117005 BR00118041 BR00118042 BR00118043 BR00118044"
need_plate_list=""

# -------------------------------
# mAP calculation settings
# -------------------------------
# random_seeds="42 2026 2025 2024 2023"
random_seeds="42"
n_jobs=20
sample_method="random"
# sample_method="all"
n_number=2

# -------------------------------
# Logging
# -------------------------------
log_file="/fse/home/wangyongyan/Project/CellPainting/DeSCORT_Script/outputs/01_mAP/logs/original_cpg0021_A549_1382_CP186H.log"
log_level="INFO"

# ============================================================================
# Build argument strings
# ============================================================================
args=(
    --mAP_type "$mAP_type"
    --source_data_paths "$source_data_paths"
    --need_feature_list_pkl "$need_feature_list_pkl"
    --feature2index_pkl "$feature2index_pkl"
    --save_path "$save_path"
    --perturbation_gene_key "$perturbation_gene_key"
    --sgRNA_barcode_key "$sgRNA_barcode_key"
    --choose_type "$choose_type"
    --need_columns $need_columns
    --group_by_columns $group_by_columns
    --random_seeds $random_seeds
    --n_jobs "$n_jobs"
    --sample_method "$sample_method"
    --n_number "$n_number"
    --log_file "$log_file"
    --log_level "$log_level"
)


if [ -n "$need_plate_list" ]; then
    args+=(--need_plate_list "$need_plate_list")
fi

if [ -n "$negative_control_list" ]; then
    args+=(--negative_control_list "$negative_control_list")
fi


if [ "$do_intersect_features" = true ]; then
    args+=(--do_intersect_features)
fi

if [ "$do_groupby_mean" = true ]; then
    args+=(--do_groupby_mean)
fi

# 执行
python 01_run_mAP.py "${args[@]}" "$@"
