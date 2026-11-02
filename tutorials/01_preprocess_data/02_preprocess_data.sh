#!/bin/bash
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
RESOURCE_ROOT="${DESCORT_RESOURCE_ROOT:-$PROJECT_ROOT/../../common_files}"
cd "$SCRIPT_DIR"

# ============================================================================
# 02_preprocess_data.sh
#
# Launch 02_preprocess_data.py with argparse parameters.
# Edit the variables below to match your data / environment.
# ============================================================================

# -------------------------------
# Source data paths
# -------------------------------
source_data_paths="/fse/DC/CP_data/feature_data/20200805_A549_WG_Screen/raw_data/paper_code_test/raw_data"
# source_data_paths="/fse/DC/CP_data/feature_data/20210422_6W_CP257/raw_data/raw_data"

# -------------------------------
# Output paths
# -------------------------------
base_output="/fse/DC/CP_data/feature_data/20200805_A549_WG_Screen/raw_data/paper_code_test/well_level_new"
# base_output="/fse/DC/CP_data/feature_data/20210422_6W_CP257/raw_data/well_level_new"

# Intermediate outputs are NOT saved by default. Set the corresponding
# do_save_* flag to true and provide the path to keep them.
# median_output_path="${base_output}/median_data"
# normalization_output_path="${base_output}/normalization_data"
normalization_select_feature_output_path="${base_output}/normalization_select_feature_data"
clip_output_path="${base_output}/normalization_clip_data"

# -------------------------------
# Save intermediate outputs
# -------------------------------
do_save_median=false
do_save_normalization=false

# -------------------------------
# Data filtering
# -------------------------------
# need_plate_list="CP186A CP186B CP186C CP186D CP186E CP186F CP186G CP186H CP186N"
# need_plate_list="CP257A CP257B CP257D CP257F CP257H CP257J CP257K CP257L CP257N"
need_plate_list=""

# -------------------------------
# Aggregation / processing
# -------------------------------
data_level="sc"

need_columns="Metadata_Foci_Barcode_MatchedTo_GeneCode Metadata_Foci_Barcode_MatchedTo_Barcode Metadata_Foci_well Metadata_Foci_plate"
# need_columns="Metadata_Treatment Metadata_Target_Sequence Metadata_Well Metadata_Plate"

do_map_channel_names=true

# -------------------------------
# Normalization
# -------------------------------
normalize_method="mad_robustize"
normalize_samples_query="Metadata_Foci_Barcode_MatchedTo_GeneCode == 'nontargeting'"
# normalize_samples_query="Metadata_Treatment == 'nontargeting'"

# -------------------------------
# Clip
# -------------------------------
clip_lower_percentile=1
clip_upper_percentile=99

# -------------------------------
# Feature selection
# -------------------------------
feature_select_operations="variance_threshold correlation_threshold drop_na_columns blocklist"
na_cutoff=0.05
corr_threshold=0.9

# -------------------------------
# Logging
# -------------------------------
log_file="$PROJECT_ROOT/tutorials/01_preprocess_data/preprocess_data.log"
log_level="INFO"

# ============================================================================
# Build argument list
# ============================================================================
ARGS=(
    --source_data_paths "$source_data_paths"
    --clip_output_path "$clip_output_path"
    --normalization_select_feature_output_path "$normalization_select_feature_output_path"
    --data_level "$data_level"
    --need_columns $need_columns
    --normalize_method "$normalize_method"
    --normalize_samples_query "$normalize_samples_query"
    --clip_lower_percentile "$clip_lower_percentile"
    --clip_upper_percentile "$clip_upper_percentile"
    --feature_select_operations $feature_select_operations
    --na_cutoff "$na_cutoff"
    --corr_threshold "$corr_threshold"
    --log_file "$log_file"
    --log_level "$log_level"
)

if [ -n "$need_plate_list" ]; then
    ARGS+=(--need_plate_list $need_plate_list)
fi

if [ "$do_map_channel_names" = true ]; then
    ARGS+=(--do_map_channel_names)
fi

if [ "$do_save_median" = true ]; then
    ARGS+=(--do_save_median --median_output_path "$median_output_path")
fi

if [ "$do_save_normalization" = true ]; then
    ARGS+=(--do_save_normalization --normalization_output_path "$normalization_output_path")
fi

# ============================================================================
# Run
# ============================================================================
python "02_preprocess_data.py" "${ARGS[@]}" "$@"
