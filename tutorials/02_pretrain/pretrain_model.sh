#!/bin/bash
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
RESOURCE_ROOT="${DESCORT_RESOURCE_ROOT:-$PROJECT_ROOT/../../common_files}"
cd "$SCRIPT_DIR"
# train_dataset_path='/fse/DC/CP_data/feature_data/MAD_merged_data_NNLLM/intersection_data/well_level_new_clip_data/all_data_clip'
# train_dataset_path='/fse/DC/CP_data/feature_data/MAD_merged_data_NNLLM/intersection_data/well_level_new_clip_data/cpg0021_A549/train_merged_data'
# train_dataset_path='/fse/DC/CP_data/feature_data/MAD_merged_data_NNLLM/intersection_data/well_level_new_clip_data/cpg0021_HeLa/train_merged_data'
# train_dataset_path='/fse/DC/CP_data/feature_data/MAD_merged_data_NNLLM/intersection_data/well_level_new_clip_data/cpg0000_A549/train_merged_data'
# train_dataset_path='/fse/DC/CP_data/feature_data/MAD_merged_data_NNLLM/intersection_data/well_level_new_clip_data_union/all_data_clip_cpg0000'
train_dataset_path='/fse/DC/CP_data/feature_data/MAD_merged_data_NNLLM/intersection_data/paper_well_level_new_1_99_clip/cpg0021/all_data'
# train_dataset_path='/fse/DC/CP_data/feature_data/MAD_merged_data_NNLLM/intersection_data/well_level_new_clip_data/all_data_clip_cpg0000'
# train_dataset_path='/fse/DC/CP_data/feature_data/MAD_merged_data_NNLLM/intersection_data/well_level_new_clip_Zscore/cpg0021_A549/train_merged_data'
# train_dataset_path='/fse/DC/CP_data/feature_data/all_merged_data_select_feature_intersection/normaliztion_data_chunk/A549/A549_merged_data'
output_dir="${OUTPUT_DIR:-${DESCORT_OUTPUT_ROOT:-$PROJECT_ROOT/outputs}/02_pretrain/output_dir}"
deepspeed_config="$PROJECT_ROOT/configs/deepspeed/zero2.json"



# Persist the conditioning embedding width in the pretrained model config.
esm2_input_dim="${ESM2_INPUT_DIM:-5120}"
MODELPARAMS="
    --hidden_size=256 \
    --bias=True \
    --pretraining_tp=1 \
    --hidden_act="silu" \
    --max_seq_len=1024 \
    --bin_num=10 \
    --bin_alpha=1 \
    --mask_token_value=-100 \
    --pad_token_value=100 \
    --temperature=1.0 \
    --feature_name_vocab_size=7366\
    --pad_token_id=0\
    --num_attention_heads=8 \
    --attention_dropout=0.0 \
    --n_routed_experts=8 \
    --first_k_dense_replace=0 \
    --moe_layer_freq=1 \
    --num_experts_per_tok=4 \
    --moe_intermediate_size=1024 \
    --n_shared_experts=1 \
    --intermediate_size=1024 \
    --scoring_func="softmax" \
    --seq_aux=True \
    --aux_loss_alpha=0.001 \
    --norm_topk_prob=True \
    --rms_norm_eps=1e-6 \
    --n_layers=8 \
    --hidden_dropout=0.1 \
    --esm2_input_dim=$esm2_input_dim \
    --initializer_range=0.02 
"

DATAPARAMS="
    --train_dataset_path=$train_dataset_path \
    --mlm_probability=0.3 \
    --do_mlm=True \
    --mask_value=-100 \
    --pad_value=100 \
    --max_length=1024 \
    --sampling=True\
    --keep_first_n_tokens=0
"

TRAINPARAMS="
    --do_train=True \
    --do_eval=False \
    --seed=42 \
    --output_dir=$output_dir \
    --overwrite_output_dir=True \
    --report_to="tensorboard" \
    --logging_steps=20 \
    --gradient_accumulation_steps=3 \
    --num_train_epochs=1 \
    --per_device_train_batch_size=64 \
    --learning_rate=1e-3 \
    --lr_scheduler_type="cosine" \
    --warmup_ratio=0.1 \
    --weight_decay=0 \
    --max_grad_norm=1.0 \
    --optim="adamw_torch" \
    --save_steps=0.5 \
    --bf16=True \
    --dataloader_pin_memory=True \
    --dataloader_prefetch_factor=2 \
    --dataloader_num_workers=4 \
    --dataloader_persistent_workers=True \
    --ddp_find_unused_parameters=True \
    --gradient_checkpointing=False \
    --remove_unused_columns=False "

RADOM_PORT=$(shuf -i 1024-65535 -n 1)
echo "使用的端口号: $RADOM_PORT"


# python 02_pretrain_model.py $MODELPARAMS $DATAPARAMS $TRAINPARAMS

# # deepspeed
deepspeed --include=localhost:1,2,3 --master_port=$RADOM_PORT 02_pretrain_model.py \
    --deepspeed=$deepspeed_config \
    $MODELPARAMS \
    $DATAPARAMS \
    $TRAINPARAMS "$@"
