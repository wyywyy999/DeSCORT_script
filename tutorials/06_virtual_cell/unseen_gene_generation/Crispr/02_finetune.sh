#!/bin/bash
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/../../../.." && pwd)"
RESOURCE_ROOT="${DESCORT_RESOURCE_ROOT:-$PROJECT_ROOT/../../common_files}"
cd "$SCRIPT_DIR"

set -euo pipefail

# ============================================================================
# 02_finetune.sh
#
# Fine-tune DeSCORT for virtual-cell perturbation prediction.
# Edit the variables and parameter arrays below to match your environment.
# ============================================================================

# -------------------------------
# Paths
# -------------------------------
experiment_name="${EXPERIMENT_NAME:-HeLa_DMEM_50genes_unseen_gene_generation_50epochs}"
default_train_dataset_path="/fse/DC/CP_data/feature_data/Paper_MAD_unseen_data_generation_clip/few_shot/HeLa_DMEM_13k_5k_50_genes/train_data/hit"
train_dataset_path="${TRAIN_DATASET_PATH:-$default_train_dataset_path}"
output_dir="${OUTPUT_DIR:-${DESCORT_OUTPUT_ROOT:-$PROJECT_ROOT/outputs}/06_virtual_cell_unseen_gene_generation_Crispr/output_dir}"
deepspeed_config="$PROJECT_ROOT/configs/deepspeed/zero2.json"
default_pretrained_model_path="/fse/DC/CP_data/feature_data/benchmark_MAD_output/All_MAD_model/paper_model/06_vc/unseen_gene/DeSCORT/A549_13k_5k_direct_mlm_mean1_dir1_marg01_3ep/models"
pretrianed_model_path="${PRETRAINED_MODEL_PATH:-$default_pretrained_model_path}"
index2esm2_pkl="$RESOURCE_ROOT/ESM2_files/Index2ESM2_pert_features_with_nontargeting_19791.pkl"

# Dataset column used to construct same-gene batches.
gene_column="Perturb_Gene"

# -------------------------------
# Distributed launch settings
# -------------------------------
# Exactly two GPUs are used by default. Override with GPU_IDS only when needed.
gpu_ids="${GPU_IDS:-0,7}"
deepspeed_bin="${DEEPSPEED_BIN:-deepspeed}"

# -------------------------------
# Model parameters
# -------------------------------
# Must match the embedding width in index2esm2_pkl.
esm2_input_dim="${ESM2_INPUT_DIM:-5120}"
MODELPARAMS=(
    --hidden_size 256
    --bias True
    --pretraining_tp 1
    --hidden_act silu
    --max_seq_len 1107
    --bin_num 10
    --bin_alpha 1
    --temperature 1.0
    --mask_token_value -100
    --pad_token_value 100
    --feature_name_vocab_size 7366
    --pad_token_id 0
    --num_attention_heads 8
    --attention_dropout 0.0
    --intermediate_size 1024
    --n_routed_experts 8
    --num_experts_per_tok 4
    --first_k_dense_replace 0
    --moe_layer_freq 1
    --moe_intermediate_size 1024
    --n_shared_experts 1
    --scoring_func softmax
    --seq_aux True
    --aux_loss_alpha 0.001
    --norm_topk_prob True
    --rms_norm_eps 1e-6
    --n_layers 8
    --initializer_range 0.02
    --hidden_dropout 0.1
    --esm2_input_dim "$esm2_input_dim"
    --esm2_projection_dim 0
    --normalize_esm2_embedding False
    # Direct-absolute MLM head: directly output post-perturbation morphology
    # features. Control_Expr is not added back to the prediction.
    --predict_residual False
    --prediction_head_type mlm
    # L = 1.0 * Mean + 1.0 * Direction + 0.1 * Marginal Quantile.
    --mean_weight "${MEAN_WEIGHT:-1.0}"
    --direction_weight "${DIRECTION_WEIGHT:-1.0}"
    --marginal_quantile_weight "${MARGINAL_QUANTILE_WEIGHT:-0.1}"
    --use_global_delta_prior False
    --freeze_global_delta_prior False
    --freeze_morphology_backbone False
)

# -------------------------------
# Dataset and virtual-cell parameters
# -------------------------------
DATAPARAMS=(
    --train_dataset_path "$train_dataset_path"
    --max_length 1107
    --sampling False
    --pad_value 100
    --keep_first_n_tokens 0
)

FINETUNEPARAMS=(
    --pretrianed_model_path "$pretrianed_model_path"
    --index2esm2_pkl "$index2esm2_pkl"
    --gene_column "$gene_column"
)

# -------------------------------
# Trainer parameters
# -------------------------------
TRAINPARAMS=(
    --do_train True
    --do_eval False
    --seed 42
    --output_dir "$output_dir"
    --overwrite_output_dir True
    --report_to tensorboard
    --logging_steps 10
    --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS:-1}"
    --num_train_epochs "${EPOCHS:-50}"
    --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE:-64}"
    --learning_rate "${LEARNING_RATE:-2e-4}"
    --lr_scheduler_type cosine
    --warmup_ratio 0.1
    --weight_decay 0
    --max_grad_norm 1.0
    --optim adamw_torch
    --save_strategy epoch
    --save_total_limit "${SAVE_TOTAL_LIMIT:-5}"
    --bf16 True
    --dataloader_pin_memory True
    --dataloader_prefetch_factor 2
    --dataloader_num_workers 4
    --dataloader_persistent_workers True
    --ddp_find_unused_parameters True
    --gradient_checkpointing False
    --remove_unused_columns False
)

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RANDOM_PORT="$(shuf -i 1024-65535 -n 1)"
echo "Master port: $RANDOM_PORT"

"$deepspeed_bin" \
    --include="localhost:$gpu_ids" \
    --master_port="$RANDOM_PORT" \
    "$SCRIPT_DIR/02_finetune.py" \
    --deepspeed "$deepspeed_config" \
    "${MODELPARAMS[@]}" \
    "${DATAPARAMS[@]}" \
    "${FINETUNEPARAMS[@]}" \
    "${TRAINPARAMS[@]}" \
    "$@"
