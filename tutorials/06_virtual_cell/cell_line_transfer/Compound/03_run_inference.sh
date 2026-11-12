#!/bin/bash
set -euo pipefail

# Match the U2OS Morgan experiment configured in 02_finetune.sh.
model_name="A549_2_U2OS_100_gene"
# model_path="/fse/DC/CP_data/feature_data/benchmark_MAD_output/All_MAD_model/paper_model/06_vc_compound/unseen_gene/Morgan/A549_214_compound_20epochs_Morgan_cell_line_transfer/models"
model_path="/fse/DC/CP_data/feature_data/benchmark_MAD_output/All_MAD_model/paper_model/06_vc_compound/cell_line_transfer/Morgan_few_shot/A549_2_U2OS_100_gene/models"
train_dataset_path="/fse/DC/CP_data/feature_data/Paper_MAD_unseen_data_generation_clip/06_vc_compound/cell_line_transfer/Morgan_few_shot/100_gene/U2OS/train_data/hit"
test_dataset_path="/fse/DC/CP_data/feature_data/Paper_MAD_unseen_data_generation_clip/06_vc_compound/cell_line_transfer/Morgan_few_shot/100_gene/U2OS/test_data/hit"
output_dir="/fse/home/wangyongyan/Project/CellPainting/DeSCORT/tutorials/06_vc_compound/test_anndata/cell_line_transfer/Morgan_few_shot/A549_to_U2OS/epoch_20/100_gene/DeSCORT"

smiles2index_pkl="/fse/home/wangyongyan/Project/CellPainting/DeSCORT/common_files/06_vc_compound/Morgan/smiles2index_577.pkl"
index2esm2_pkl="/fse/home/wangyongyan/Project/CellPainting/DeSCORT/common_files/06_vc_compound/Morgan/index2morgan_577.pkl"
# Use exactly the vocabulary used by 01_process_data.sh; invert it for feature names.
feature2index_pkl="/fse/home/wangyongyan/Project/CellPainting/Final_well_level_new_LLM/common_files/00_pretrain/feature2index_vocabulary_7366.pkl"
inference_script=03_inference.py
# Physical GPU selection; DEVICE is the logical index within the visible set.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-7}"
export USE_TF=0

# Default to the sequence length saved in the checkpoint, including padding.
EXTRA_PARAMS=()
if [[ -n "${MAX_LENGTH:-}" ]]; then EXTRA_PARAMS+=(--max_length "$MAX_LENGTH"); fi

python "$inference_script" \
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
    --log_file "/fse/home/wangyongyan/Project/CellPainting/DeSCORT/other_model/DeSCORT/tutorials/06_vc_compound/logs/inference_A549_214_compound_20epochs_Morgan_cell_line_transfer.log" \
    "${EXTRA_PARAMS[@]}" \
    "$@"
