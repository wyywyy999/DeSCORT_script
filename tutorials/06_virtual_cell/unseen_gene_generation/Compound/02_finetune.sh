#!/bin/bash

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
experiment_name="${EXPERIMENT_NAME:-A549_214_compound_40epochs_Morgan_unseen_gene}"
default_train_dataset_path="/fse/DC/CP_data/feature_data/Paper_MAD_unseen_data_generation_clip/06_vc_compound/unseen_gene/Morgan/A549/train_data/hit"
train_dataset_path="/fse/DC/CP_data/feature_data/Paper_MAD_unseen_data_generation_clip/06_vc_compound/unseen_gene/Morgan/A549/train_data/hit"
output_dir="/fse/home/wangyongyan/Project/CellPainting/DeSCORT/other_model/DeSCORT/outputs/06_virtual_cell_unseen_gene_generation_Compound/output_dir"
deepspeed_config="/fse/home/wangyongyan/Project/CellPainting/DeSCORT/other_model/DeSCORT/configs/deepspeed/zero2.json"
default_pretrained_model_path="/fse/DC/CP_data/feature_data/benchmark_MAD_output/All_MAD_model/paper_model/foundation_model/DeSCORT_layernorm/models"
pretrianed_model_path="/fse/DC/CP_data/feature_data/benchmark_MAD_output/All_MAD_model/paper_model/foundation_model/DeSCORT_layernorm/models"
index2esm2_pkl="/fse/home/wangyongyan/Project/CellPainting/DeSCORT/common_files/06_vc_compound/Morgan/index2morgan_577.pkl"
# index2esm2_pkl="/fse/home/wangyongyan/Project/CellPainting/DeSCORT/common_files/06_vc_compound/index2Unimol_577.pkl"

# Dataset column used to construct same-gene batches.
gene_column="Perturb_Gene"

# -------------------------------
# Distributed launch settings
# -------------------------------
# GPU device IDs; use a comma-separated list to launch on multiple GPUs.
gpu_ids="${GPU_IDS:-0,2,3,7}"
deepspeed_bin="${DEEPSPEED_BIN:-deepspeed}"

# -------------------------------
# Model parameters
# -------------------------------
# Must match the embedding width in index2esm2_pkl.
esm2_input_dim="${ESM2_INPUT_DIM:-2048}"
MODELPARAMS=(
    --hidden_size 256
    --bias True
    --pretraining_tp 1
    --hidden_act silu
    --max_seq_len 2297
    --bin_num 10
    --bin_alpha 1
    --temperature 1.0
    --mask_token_value -100
    --pad_token_value 100
    --feature_name_vocab_size 7366
    --pad_token_id 0
    --num_attention_heads 8
    # Use the original explicit QK/softmax/V attention calculation.
    --use_sdpa False
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
    --max_length 2297
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
    --num_train_epochs 20
    --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE:-64}"
    --learning_rate "${LEARNING_RATE:-2e-4}"
    --lr_scheduler_type cosine
    --warmup_ratio 0.1
    --weight_decay 0
    --max_grad_norm 1.0
    --optim adamw_torch
    --save_steps 0.25
    --bf16 True
    --dataloader_pin_memory True
    --dataloader_prefetch_factor 2
    --dataloader_num_workers 4
    --dataloader_persistent_workers True
    --ddp_find_unused_parameters True
    # Recompute layer activations to reduce memory use at length 3318.
    --gradient_checkpointing True
    --remove_unused_columns False
)

RANDOM_PORT="$(shuf -i 1024-65535 -n 1)"
echo "使用的端口号: $RANDOM_PORT"

"$deepspeed_bin" \
    --include="localhost:$gpu_ids" \
    --master_port="$RANDOM_PORT" \
    "02_finetune.py" \
    --deepspeed "$deepspeed_config" \
    "${MODELPARAMS[@]}" \
    "${DATAPARAMS[@]}" \
    "${FINETUNEPARAMS[@]}" \
    "${TRAINPARAMS[@]}" \
    "$@"
