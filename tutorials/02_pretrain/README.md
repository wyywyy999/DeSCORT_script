# 02 — Representation Pretraining



## Pipeline Overview

```text
              Normalized Cell Painting Parquet files
                               │
                               ▼
          ┌─────────────────────────────────────────┐
          │             01_process_data             │
          │   Filter profiles and encode features   │
          │   Save Arrow shards and merge datasets  │
          └────────────────────┬────────────────────┘
                               │
                               ▼
          ┌─────────────────────────────────────────┐
          │           02_combined_dataset           │
          │   Optional: combine prepared datasets   │
          │   Skip when Step 1 already provides     │
          │          the desired training set      │
          └────────────────────┬────────────────────┘
                               │
                               ▼
          ┌─────────────────────────────────────────┐
          │            03_pretrain_model            │
          │     Dual-view masked reconstruction    │
          │          + contrastive learning        │
          └────────────────────┬────────────────────┘
                               │
                               ▼
                      output_dir/models/
                               │
                               ▼
          ┌─────────────────────────────────────────┐
          │               04_run_emb                │
          │   Extract pretrained representations   │
          │     Save metadata + embedding columns  │
          └────────────────────┬────────────────────┘
                               │
                               ▼
                     Embedding Parquet files
```

## Before You Start

Complete [data preprocessing](../01_preprocess_data/README.md) first. Prepare normalized numerical features, consistent label and feature mappings, and the feature list needed by the experiment. These files and the experimental datasets are not included in the Python package.

Activate the DeSCORT environment and enter this tutorial directory:

```bash
conda activate DeSCORT
cd /fse/home/wangyongyan/Project/CellPainting/DeSCORT/other_model/DeSCORT/tutorials/02_pretrain
```

Run the numbered Bash scripts from this directory because they launch Python scripts by filename and do not change the working directory. Replace every `/absolute/path/...` placeholder below with your actual absolute path.


For example, inspect an entry without processing data:

```bash
bash 01_process_data.sh --help
bash 02_combined_dataset.sh --help
bash 04_run_emb.sh --help
```

Pretraining uses PyTorch, Transformers, datasets, and DeepSpeed. Follow the [installation instructions](../../docs/installation.md) to prepare the GPU environment. Steps 1 and 2 run on CPU; Step 3 uses the GPUs configured in its Bash script. Step 4 extracts embeddings using one selected GPU or CPU.

## Step 1 — Prepare Arrow Datasets

### Key arguments

| Argument | Description |
|---|---|
| `--source_data_paths` | One or more directories containing normalized Parquet files; searched recursively |
| `--train_output_dir` | Directory for per-file Arrow dataset shards |
| `--train_save_dir` | Directory for the merged training dataset |
| `--gene2index_pkl` | Label-to-integer mapping; labels must match `--key_gene_column` |
| `--feature2index_pkl` | Feature-name-to-integer vocabulary; index 0 is reserved for padding |
| `--need_feature_list_pkl` | Selected feature list, such as the union/intersection list from preprocessing |
| `--key_gene_column` | Metadata column containing perturbation labels |
| `--need_gene_list` | Optional pickle containing a label subset; omitted by the supplied Bash preset |
| `--train_data_names` | Optional filename/path substrings; omitted or empty means no filename filtering |
| `--nproc` | Processor configuration, including dataset serialization workers; default 1 |
| `--merge_nproc` | Number of concurrent shard-loading workers; default 2 |
| `--verbosity` | Logging verbosity; default 1 |


### Example

```bash
bash 01_process_data.sh \
  --source_data_paths /absolute/path/normalization_clip_data_select_feature \
  --train_output_dir /absolute/path/pretrain_data/Crispr/train_tmp_data \
  --train_save_dir /absolute/path/pretrain_data/Crispr/train_merged_data \
  --gene2index_pkl /absolute/path/label_vocab/my_dataset_label2index_161.pkl \
  --feature2index_pkl /absolute/path/feature_vocab/feature2index_vocabulary_7366.pkl \
  --need_feature_list_pkl /absolute/path/feature_intersection_union/my_dataset_union_feature_894.pkl \
  --key_gene_column Metadata_Foci_Barcode_MatchedTo_GeneCode \
  --nproc 1 \
  --merge_nproc 2 \
  --verbosity 1
```

### Output

Each merged dataset row contains:

| Field | Content |
|---|---|
| `Perturb_Gene` | Perturbation label |
| `Feature_expression` | Float32 morphological feature values |
| `Feature_name_index` | Integer feature IDs aligned with the values |
| `Perturb_Gene_Index` | Integer perturbation label ID |

Example output layout:

```text
pretrain_data/
└── Crispr/
    ├── train_tmp_data/
    │   └── <parent>_<input_file>_0_index/
    └── train_merged_data/
        ├── dataset_info.json
        ├── state.json
        └── data-*.arrow
```


## Step 2 — Combine Prepared Datasets (Optional)

For example:

```text
pretrain_data/
├── Crispr/train_merged_data/
├── Compound/train_merged_data/
├── Orf/train_merged_data/
└── all_data/                         # Combined output; excluded from input search
```

### Key arguments

| Argument | Description |
|---|---|
| `--base_dir` | Parent directory containing perturbation-type subdirectories |
| `--save_path` | Output directory for the combined dataset |
| `--subdir_name` | Dataset subdirectory name to load; default `train_merged_data` |
| `--exclude` | Immediate child directory names to ignore; default `all_data` |

### Example

```bash
bash 02_combined_dataset.sh \
  --base_dir /absolute/path/pretrain_data \
  --save_path /absolute/path/pretrain_data/all_data \
  --subdir_name train_merged_data \
  --exclude all_data
```

### Output

The combined dataset is saved in `/absolute/path/pretrain_data/all_data`.

## Step 3 — Pretrain the Model
The reported training objective is:

```text
loss = 0.5 × reconstruction_loss_view_1
     + 0.5 × reconstruction_loss_view_2
     + contrastive_loss
```

Reconstruction uses mean squared error on the selected masked feature values. It predicts continuous morphology values rather than categorical feature tokens.

### Configure the GPU launcher

The supplied Bash script currently launches DeepSpeed with:

```bash
deepspeed --include=localhost:1,2,3 --master_port=$RADOM_PORT 03_pretrain_model.py
```

### Key arguments and presets

| Argument | Supplied Bash preset / meaning |
|---|---|
| `--train_dataset_path` | Arrow dataset from Step 1 or Step 2 |
| `--output_dir` | Experiment directory for checkpoints, logs, and the final model |
| `--deepspeed` | Absolute path to the ZeRO-2 configuration |
| `--hidden_size`, `--n_layers`, `--num_attention_heads` | 256, 8, and 8 |
| `--feature_name_vocab_size` | 7366; must cover the feature IDs in the dataset |
| `--max_seq_len`, `--max_length` | 1024 and 1024 |
| `--esm2_input_dim` | 5120, persisted in the model configuration; this entry does not load an ESM2 pickle |
| `--mlm_probability`, `--do_mlm` | 0.3 and `True` |
| `--mask_value`, `--pad_value` | -100 and 100 |
| `--sampling`, `--keep_first_n_tokens` | `True` and 0 |
| `--n_routed_experts`, `--num_experts_per_tok` | 8 routed experts, selecting 4 per token |
| `--num_train_epochs` | 1 |
| `--per_device_train_batch_size` | 64 |
| `--gradient_accumulation_steps` | 3 |
| `--learning_rate`, `--lr_scheduler_type` | `1e-3` and `cosine` |
| `--warmup_ratio`, `--seed` | 0.1 and 42 |
| `--bf16`, `--report_to` | `True` and `tensorboard` |
| `--do_eval` | `False`; no validation dataset is supplied by this entry |

### Example

After configuring the launcher GPUs, run:

```bash
bash 03_pretrain_model.sh \
  --train_dataset_path /absolute/path/pretrain_data/all_data \
  --output_dir /absolute/path/output_model/pretrain_run_01 \
  --deepspeed /fse/home/wangyongyan/Project/CellPainting/DeSCORT/other_model/DeSCORT/configs/deepspeed/zero2.json \
  --feature_name_vocab_size 7366 \
  --max_seq_len 1024 \
  --max_length 1024 \
  --mlm_probability 0.3 \
  --num_train_epochs 1 \
  --per_device_train_batch_size 64 \
  --gradient_accumulation_steps 3 \
  --learning_rate 1e-3 \
  --bf16 True \
  --seed 42 \
  --overwrite_output_dir False
```

### Output

```text
output_model/pretrain_run_01/
├── models/                  # Final model weights and configuration
│   ├── config.json
│   └── model.safetensors    # May be sharded depending on model size/settings
├── checkpoint-*/            # Intermediate Trainer checkpoints
├── runs/                    # Default TensorBoard logging location
└── trainer_state.json
```

## Step 4 — Extract Pretrained Embeddings (`04_run_emb.sh`)

Load the representation model saved in Step 3 and extract one embedding per retained input profile. This step reads normalized **Parquet profiles**, not the Arrow training dataset. It does not update the model weights.

The Python entry `04_run_emb.py` contains its own argument parser and extraction loop. It recursively finds input Parquet files, optionally filters plate paths, keeps labels present in the label mapping, selects features using the feature vocabulary/list, and calls `model.get_latent_embedding()` in evaluation mode with gradients disabled.

### Key arguments

| Argument | Description |
|---|---|
| `--source_data_paths` | One or more directories containing normalized Parquet profiles |
| `--gene2index_pkl` | Label mapping matching the selected metadata column |
| `--feature2index_pkl` | The feature vocabulary used for the model inputs |
| `--need_feature_list_pkl` | Selected feature list; use the feature definition appropriate to the checkpoint |
| `--pretrained_model_path` | Step 3's final `models/` directory or a compatible representation checkpoint |
| `--save_path` | Root directory for embedding Parquet outputs |
| `--key_gene_column` | Perturbation metadata column used for label filtering |
| `--need_plate_list` | Optional filename/path substrings selecting plates; omitted means all matching files |
| `--metadata_parts` | Two input-path component indices: batch first, type second; Python default `-2 -5` |
| `--device` | Device such as `cuda:0` or `cpu`; the entry falls back to CPU when CUDA is unavailable |
| `--batch_size` | Number of profiles per inference batch; Bash preset 64 |
| `--max_length` | Feature sequence length; set it to suit the checkpoint and feature list |
| `--mlm_probability` | Masking probability; use 0.0 with MLM disabled for embedding extraction |
| `--sampling` | Enable random feature sampling; omitted means disabled (default `False`) |
| `--log_file`, `--log_level` | Log destination and verbosity |

### Example

For a directory layout such as:

```text
/absolute/path/profiles/
└── Crispr/
    └── Batch01/
        └── plate01.parquet
```

Run from the `02_pretrain` directory:

```bash
bash 04_run_emb.sh \
  --source_data_paths /absolute/path/profiles \
  --gene2index_pkl /absolute/path/label_vocab/my_dataset_label2index_161.pkl \
  --feature2index_pkl /absolute/path/feature_vocab/feature2index_vocabulary_7366.pkl \
  --need_feature_list_pkl /absolute/path/feature_intersection_union/my_dataset_union_feature_894.pkl \
  --pretrained_model_path /absolute/path/output_model/pretrain_run_01/models \
  --save_path /absolute/path/embeddings/pretrain_run_01 \
  --key_gene_column Metadata_Foci_Barcode_MatchedTo_GeneCode \
  --metadata_parts -2 -3 \
  --device cuda:0 \
  --batch_size 64 \
  --max_length 1024 \
  --mlm_probability 0.0 \
  --nproc 1 \
  --verbosity 1 \
  --log_file /absolute/path/logs/pretrain_embeddings.log \
  --log_level INFO
```

For the illustrated input path, `--metadata_parts -2 -3` selects `Batch01` and `Crispr`, respectively. These indices control the output directory structure; change them if your input layout differs. They do not read batch/type values from Parquet metadata columns. For CPU extraction, replace `--device cuda:0` with `--device cpu`. To select plates, append an option such as `--need_plate_list plate01`.

### Output

The output path is:

```text
<save_path>/<type>/<batch>/<input_file_stem>.parquet
```

For the example above:

```text
embeddings/pretrain_run_01/
└── Crispr/
    └── Batch01/
        └── plate01.parquet
```
