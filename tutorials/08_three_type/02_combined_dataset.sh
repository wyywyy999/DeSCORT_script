#!/bin/bash

# ============================================================================
# combined_dataset.sh
#
# Launch combined_dataset.py with argparse parameters.
# Edit the variables below to match your data / environment.
# ============================================================================

# -------------------------------
# Input / output paths
# -------------------------------
base_dir="/fse/DC/CP_data/feature_data/MAD_merged_data_NNLLM/intersection_data/paper_well_level_new_1_99_clip/cpg0000_three_type/U2OS"
save_path="/fse/DC/CP_data/feature_data/MAD_merged_data_NNLLM/intersection_data/paper_well_level_new_1_99_clip/cpg0000_three_type/U2OS/all_data"

# -------------------------------
# Search settings
# -------------------------------
subdir_name="train_merged_data"
exclude_dirs="all_data"

# ============================================================================
# Build argument list
# ============================================================================
ARGS=(
    --base_dir "$base_dir"
    --save_path "$save_path"
    --subdir_name "$subdir_name"
    --exclude $exclude_dirs
)

# ============================================================================
# Run
# ============================================================================

python "/fse/home/wangyongyan/Project/CellPainting/DeSCORT/other_model/DeSCORT/tutorials/08_three_type/02_combined_dataset.py" "${ARGS[@]}" "$@"
