#!/bin/bash
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/../../../.." && pwd)"
RESOURCE_ROOT="${DESCORT_RESOURCE_ROOT:-$PROJECT_ROOT/../../common_files}"
cd "$SCRIPT_DIR"
set -euo pipefail

# Match the U2OS Morgan experiment configured in 02_finetune.sh.
model_name="${MODEL_NAME:-${EXPERIMENT_NAME:-A549_214_compound_20epochs_Morgan_cell_line_transfer}}"
# model_path="${MODEL_PATH:-/fse/DC/CP_data/feature_data/benchmark_MAD_output/All_MAD_model/paper_model/06_vc_compound/unseen_gene/Morgan/${model_name}/models}"
model_path="${MODEL_PATH:-/fse/DC/CP_data/feature_data/benchmark_MAD_output/All_MAD_model/paper_model/06_vc_compound/cell_line_transfer/Morgan/${model_name}/models}"
train_dataset_path="${TRAIN_DATASET_PATH:-/fse/DC/CP_data/feature_data/Paper_MAD_unseen_data_generation_clip/06_vc_compound/cell_line_transfer/Morgan/A549/hit}"
test_dataset_path="${TEST_DATASET_PATH:-/fse/DC/CP_data/feature_data/Paper_MAD_unseen_data_generation_clip/06_vc_compound/cell_line_transfer/Morgan/U2OS/hit}"
output_dir="${OUTPUT_DIR:-$PROJECT_ROOT/tutorials/06_vc_compound/test_anndata/cell_line_transfer/Morgan/A549_2_U2OS/DeSCORT/${model_name}}"

smiles2index_pkl="${SMILES2INDEX_PKL:-$RESOURCE_ROOT/06_vc_compound/Morgan/smiles2index_577.pkl}"
index2esm2_pkl="${INDEX2ESM2_PKL:-${INDEX2UNIMOL_PKL:-${INDEX2MORGAN_PKL:-$RESOURCE_ROOT/06_vc_compound/Morgan/index2morgan_577.pkl}}}"
# Use exactly the vocabulary used by 01_process_data.sh; invert it for feature names.
feature2index_pkl="${FEATURE2INDEX_PKL:-/fse/home/wangyongyan/Project/CellPainting/Final_well_level_new_LLM/common_files/00_pretrain/feature2index_vocabulary_7366.pkl}"
python_bin="${PYTHON_BIN:-python}"
inference_script="$PROJECT_ROOT/tutorials/06_virtual_cell/cell_line_transfer/Compound/03_inference.py"
# Physical GPU selection; DEVICE is the logical index within the visible set.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4}"
export USE_TF=0

# Default to the sequence length saved in the checkpoint, including padding.
EXTRA_PARAMS=()
if [[ -n "${MAX_LENGTH:-}" ]]; then EXTRA_PARAMS+=(--max_length "$MAX_LENGTH"); fi

"$python_bin" "$inference_script" \
    --pretrained_model_path "$model_path" \
    --smiles2index_pkl "$smiles2index_pkl" \
    --index2esm2_pkl "$index2esm2_pkl" \
    --feature2index_pkl "$feature2index_pkl" \
    --train_dataset_path "$train_dataset_path" \
    --test_dataset_path "$test_dataset_path" \
    --output_dir "$output_dir" \
    --prediction_filename "${PREDICTION_FILENAME:-${model_name}_pred_data.h5ad}" \
    --ground_truth_filename "${GROUND_TRUTH_FILENAME:-ground_truth.h5ad}" \
    --mean_baseline_filename "${MEAN_BASELINE_FILENAME:-perturb_mean_data.h5ad}" \
    --device "${DEVICE:-cuda:0}" \
    --batch_size "${BATCH_SIZE:-32}" \
    --use_sdpa "${USE_SDPA:-false}" \
    --pad_value 100 \
    --dataloader_num_workers "${DATALOADER_NUM_WORKERS:-4}" \
    --dataloader_prefetch_factor "${DATALOADER_PREFETCH_FACTOR:-2}" \
    --gene_column Perturb_Gene \
    --control_key_word "${CONTROL_KEY_WORD:-CS(=O)C}" \
    --log_file "${LOG_FILE:-$PROJECT_ROOT/tutorials/06_vc_compound/logs/inference_${model_name}.log}" \
    "${EXTRA_PARAMS[@]}" \
    "$@"
