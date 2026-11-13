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

Complete [data preprocessing](../01_preprocess_data/README.md) first. Prepare normalized numerical features, consistent label and feature mappings, and the feature list required by the experiment.

Activate the DeSCORT environment and enter this tutorial directory:

```bash
conda activate DeSCORT
cd /absolute/path/tutorials/02_pretrain
```


Inspect script arguments:

```bash
bash 01_process_data.sh --help
bash 02_combined_dataset.sh --help
bash 03_pretrain_model.sh --help
bash 04_run_emb.sh --help
```

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
| `--need_gene_list` | Optional pickle defining a label subset |
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


### Key arguments and presets

| Argument | Supplied Bash preset / meaning |
|---|---|
| `--train_dataset_path` | Arrow dataset from Step 1 or Step 2 |
| `--output_dir` | Experiment directory for checkpoints, logs, and the final model |
| `--deepspeed` | `/absolute/path/deepspeed/zero2.json` |
| `--hidden_size` | 256 |
| `--n_layers` | 8 |
| `--num_attention_heads` | 8 |
| `--feature_name_vocab_size` | 7366; must cover the feature IDs in the dataset |
| `--max_seq_len` | 1024 |
| `--max_length` | 1024 |
| `--esm2_input_dim` | 5120, persisted in the model configuration; this entry does not load an ESM2 pickle |
| `--mlm_probability` | 0.3 |
| `--do_mlm` | `True` |
| `--mask_value` | -100 |
| `--pad_value` | 100 |
| `--sampling` | `True` |
| `--keep_first_n_tokens` | 0 |
| `--n_routed_experts` | 8 routed experts |
| `--num_experts_per_tok` | Select 4 experts per token |
| `--num_train_epochs` | 1 |
| `--per_device_train_batch_size` | 64 |
| `--gradient_accumulation_steps` | 3 |
| `--learning_rate` | `1e-3` |
| `--lr_scheduler_type` | `cosine` |
| `--warmup_ratio` | 0.1 |
| `--seed` | 42 |
| `--bf16` | `True` |
| `--report_to` | `tensorboard` |
| `--do_eval` | `False`; no validation dataset is supplied by this entry |

### Example

After configuring the launcher GPUs, run:

```bash
bash 03_pretrain_model.sh \
  --train_dataset_path /absolute/path/pretrain_data/all_data \
  --output_dir /absolute/path/output_model/pretrain_run_01 \
  --deepspeed /absolute/path/deepspeed/zero2.json \
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

## Step 4 — Extract CLS Embeddings

### Key arguments and presets

| Argument | Current Bash setting or Python behavior |
|---|---|
| `--source_data_paths` | `/absolute/path/raw_data/well_level_new/normalization_clip_data`; normalized Parquet directory |
| `--gene2index_pkl` | `gene2index_21611.pkl`; labels are matched against the selected metadata column |
| `--feature2index_pkl` | `feature2index_vocabulary_7366.pkl` |
| `--need_feature_list_pkl` | `xxx_union_feature_N.pkl`; the filename does not guarantee the actual feature count |
| `--pretrained_model_path` | `/absolute/path/pretrain/models` |
| `--save_path` | `/absolute/path/02_pretrain_data/` |
| `--key_gene_column` | `Metadata_Foci_Barcode_MatchedTo_GeneCode` |
| `--need_plate_list` | Empty in Bash; all discovered files are used unless filtered on the command line |
| `--metadata_parts` | Not passed by Bash; Python defaults to `-2 -5`, interpreted as batch index followed by type index |
| `--device` | Bash passes `cuda:0`; falls back to CPU when CUDA is unavailable |
| `--batch_size` | 64 |
| `--max_length` | Bash passes the length of the selected xxx_union_feature_N.pkl list |
| `--do_mlm` | Bash sets `do_mlm=false` and omits the flag; Python defaults to `False` |
| `--mlm_probability` | 0.0 |
| `--sampling` | Bash sets `sampling=false` and omits the flag, but the current Python default is **`False`** |
| `--nproc` | 1 |
| `--verbosity` | 1 |
| `--log_file` | Absolute path ending in `tutorials/02_pretrain/logs/pretrain_embeddings.log` |
| `--log_level` | `INFO` |

Keep MLM disabled for this extraction entry. Its default collator/model input path uses the unmasked first-view features.


### Example

For this input layout:

```text
/absolute/path/profiles/
└── Crispr/
    └── Batch01/
        └── plate01.parquet
```

Run with explicit absolute paths:

```bash
bash 04_run_emb.sh \
  --source_data_paths /absolute/path/profiles \
  --gene2index_pkl /absolute/path/label_vocab/my_dataset_label2index.pkl \
  --feature2index_pkl /absolute/path/feature_vocab/feature2index_vocabulary_7366.pkl \
  --need_feature_list_pkl /absolute/path/feature_intersection_union/my_dataset_union_feature.pkl \
  --pretrained_model_path /absolute/path/output_model/pretrain_run_01/models \
  --save_path /absolute/path/embeddings/pretrain_run_01 \
  --key_gene_column Metadata_Foci_Barcode_MatchedTo_GeneCode \
  --metadata_parts -2 -3 \
  --device cuda:0 \
  --batch_size 64 \
  --max_length 3505 \
  --mlm_probability 0.0 \
  --nproc 1 \
  --verbosity 1 \
  --log_file /absolute/path/logs/pretrain_embeddings.log \
  --log_level INFO
```

To display the available options without extracting embeddings:

### Output

The writer uses this path structure:

```text
<save_path>/<type>/<batch>/<input_file_stem>.parquet
```

The example produces:

```text
embeddings/pretrain_run_01/
└── Crispr/
    └── Batch01/
        └── plate01.parquet
```
