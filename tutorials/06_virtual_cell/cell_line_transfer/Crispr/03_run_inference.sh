#!/bin/bash
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/../../../.." && pwd)"
RESOURCE_ROOT="${DESCORT_RESOURCE_ROOT:-$PROJECT_ROOT/../../common_files}"
cd "$SCRIPT_DIR"

set -euo pipefail

# ============================================================================
# 03_run_inference.sh
#
# Launch virtual-cell inference and generate three AnnData outputs:
# ground truth, model prediction, and train-mean baseline.
# ============================================================================

# -------------------------------
# Experiment names
# -------------------------------
model_name="${MODEL_NAME:-${EXPERIMENT_NAME:-A549_all_genes_cell_line_transfer_50epochs}}"

# -------------------------------
# Model and vocabulary paths
# -------------------------------
model_root="/fse/DC/CP_data/feature_data/benchmark_MAD_output/All_MAD_model/paper_model/06_vc_new/cell_line_transfer/Crispr"
# Keep the output filename and loaded checkpoint tied to the same experiment.
# MODEL_PATH remains available for explicitly evaluating an individual
# checkpoint directory (for example checkpoint-97).
default_model_path="${model_root}/${model_name}/models"

pretrained_model_path="${MODEL_PATH:-$default_model_path}"
index2gene_pkl="$RESOURCE_ROOT/label_vocab/cpg0021/index2gene_21611.pkl"
index2esm2_pkl="$RESOURCE_ROOT/ESM2_files/Index2ESM2_pert_features_with_nontargeting_19791.pkl"
index2feature_pkl="$RESOURCE_ROOT/feature_vocab/index2feature_vocabulary_7366.pkl"

# -------------------------------
# Dataset and output paths
# -------------------------------
# These datasets match this task's preprocessing outputs.
train_dataset_path="${TRAIN_DATASET_PATH:-/fse/DC/CP_data/feature_data/Paper_MAD_unseen_data_generation_clip/06_vc/cell_line_transfer/A549/hit}"
test_dataset_path="${TEST_DATASET_PATH:-/fse/DC/CP_data/feature_data/Paper_MAD_unseen_data_generation_clip/06_vc/cell_line_transfer/HeLa_DMEM/hit}"
output_dir="${OUTPUT_DIR:-$PROJECT_ROOT/tutorials/06_virtual_cell/cell_line_transfer/Crispr/test_anndata/${model_name}}"

prediction_filename="${PREDICTION_FILENAME:-${model_name}_pred_data.h5ad}"
ground_truth_filename="${GROUND_TRUTH_FILENAME:-ground_truth.h5ad}"
mean_baseline_filename="${MEAN_BASELINE_FILENAME:-perturb_mean_data.h5ad}"

# -------------------------------
# Device / DataLoader settings
# -------------------------------
# CUDA_VISIBLE_DEVICES selects physical GPUs; DEVICE is the logical visible index.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-7}"
device="${DEVICE:-cuda:0}"
python_bin="${PYTHON_BIN:-python}"
batch_size="${BATCH_SIZE:-64}"
max_length="${MAX_LENGTH:-1107}"
pad_value=100
keep_first_n_tokens=0
sampling=false

dataloader_num_workers=4
dataloader_prefetch_factor=2
dataloader_pin_memory=true
dataloader_persistent_workers=true

# -------------------------------
# Output metadata
# -------------------------------
gene_column="Perturb_Gene"
control_label="nontargeting"
# -------------------------------
# Logging
# -------------------------------
log_file="${LOG_FILE:-$PROJECT_ROOT/tutorials/06_virtual_cell/cell_line_transfer/Crispr/logs/inference_${model_name}.log}"
log_level="INFO"

# ============================================================================
# Build and run command
# ============================================================================
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

args=(
    --pretrained_model_path "$pretrained_model_path"
    --index2gene_pkl "$index2gene_pkl"
    --index2esm2_pkl "$index2esm2_pkl"
    --index2feature_pkl "$index2feature_pkl"
    --train_dataset_path "$train_dataset_path"
    --test_dataset_path "$test_dataset_path"
    --output_dir "$output_dir"
    --prediction_filename "$prediction_filename"
    --ground_truth_filename "$ground_truth_filename"
    --mean_baseline_filename "$mean_baseline_filename"
    --device "$device"
    --batch_size "$batch_size"
    --max_length "$max_length"
    --pad_value "$pad_value"
    --keep_first_n_tokens "$keep_first_n_tokens"
    --dataloader_num_workers "$dataloader_num_workers"
    --dataloader_prefetch_factor "$dataloader_prefetch_factor"
    --gene_column "$gene_column"
    --control_label "$control_label"
    --log_file "$log_file"
    --log_level "$log_level"
)

if [ "$sampling" = true ]; then
    args+=(--sampling)
else
    args+=(--no_sampling)
fi

if [ "$dataloader_pin_memory" = true ]; then
    args+=(--dataloader_pin_memory)
else
    args+=(--no_dataloader_pin_memory)
fi

if [ "$dataloader_persistent_workers" = true ]; then
    args+=(--dataloader_persistent_workers)
else
    args+=(--no_dataloader_persistent_workers)
fi

"$python_bin" "$SCRIPT_DIR/03_inference.py" "${args[@]}" "$@"
