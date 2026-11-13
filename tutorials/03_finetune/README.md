# 03 — Representation Fine-tuning

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
          │               02_finetune               │
          │       Load a pretrained checkpoint      │
          │   Continue reconstruction + contrastive │
          │                learning                 │
          └────────────────────┬────────────────────┘
                               │
                               ▼
                      output_dir/models/
                               │
                               ▼
          ┌─────────────────────────────────────────┐
          │               03_run_emb                │
          │    Extract fine-tuned CLS embeddings    │
          │     Save metadata + embedding columns   │
          └────────────────────┬────────────────────┘
                               │
                               ▼
                     Embedding Parquet files
```

## Before You Start

Complete [data preprocessing](../01_preprocess_data/README.md) and prepare a representation checkpoint from [pretraining](../02_pretrain/README.md). Fine-tuning continues the masked reconstruction and contrastive objectives on task-specific profiles; this entry does not train a supervised classifier.

Prepare normalized Parquet files, a label mapping matching the metadata column, the feature vocabulary used by the checkpoint, and a selected feature list. Preserve the checkpoint's feature-to-ID assignments: matching vocabulary sizes alone does not ensure compatible inputs.

Activate the environment and enter this directory:

```bash
conda activate DeSCORT
cd /absolute/path/tutorials/03_finetune
```


## Step 1 — Prepare Arrow Datasets

### Key arguments

| Argument | Description |
|---|---|
| `--source_data_paths` | One or more normalized Parquet directories; searched recursively |
| `--train_output_dir` | Directory for per-file Arrow dataset shards |
| `--train_save_dir` | Directory for the merged fine-tuning dataset |
| `--gene2index_pkl` | Label-to-integer mapping matching the selected metadata column |
| `--feature2index_pkl` | Feature-name-to-integer vocabulary compatible with the checkpoint; ID 0 is reserved for padding |
| `--need_feature_list_pkl` | Selected feature list; features are intersected with the vocabulary and available columns |
| `--key_gene_column` | Metadata column containing perturbation labels |
| `--need_gene_list` | Optional pickle containing a label subset; omitted means all keys in the label mapping |
| `--train_data_names` | Optional filename/path substrings; omitted or empty means no filename filtering |
| `--nproc` | Processing/serialization workers; default 1 |
| `--merge_nproc` | Shard-loading workers; default 2 |
| `--verbosity` | Logging verbosity; default 1 |

### Example

```bash
bash 01_process_data.sh \
  --source_data_paths /absolute/path/profiles/Crispr \
  --train_output_dir /absolute/path/finetune_data/Crispr/train_tmp_data \
  --train_save_dir /absolute/path/finetune_data/Crispr/train_merged_data \
  --gene2index_pkl /absolute/path/label_vocab/my_dataset_label2index.pkl \
  --feature2index_pkl /absolute/path/feature_vocab/feature2index_vocabulary_7366.pkl \
  --need_feature_list_pkl /absolute/path/feature_intersection_union/my_dataset_union_feature.pkl \
  --key_gene_column Metadata_Foci_Barcode_MatchedTo_GeneCode \
  --nproc 1 \
  --merge_nproc 2 \
  --verbosity 1
```

Append `--need_gene_list /absolute/path/label_vocab/selected_genes.pkl` to restrict labels, or `--train_data_names CP186N` to restrict the selected plates.


### Output

Each merged row contains:

| Field | Content |
|---|---|
| `Perturb_Gene` | Perturbation label |
| `Feature_expression` | Float32 morphological feature values |
| `Feature_name_index` | Integer feature IDs aligned with the values |
| `Perturb_Gene_Index` | Integer perturbation label ID |

```text
finetune_data/
└── Crispr/
    ├── train_tmp_data/
    │   └── <parent>_<input_file>_0_index/
    └── train_merged_data/
        ├── dataset_info.json
        ├── state.json
        └── data-*.arrow
```

The merged dataset is the input to Step 2. To combine multiple prepared datasets, see the optional combination step in the [pretraining tutorial](../02_pretrain/README.md#step-2--combine-prepared-datasets-optional).

## Step 2 — Fine-tune the Pretrained Model

The entry loads `DeSCORTForContrastiveLM` from the pretrained model directory and continues training on the dataset from Step 1. The reported objective is:

```text
loss = 0.5 × reconstruction_loss_view_1
     + 0.5 × reconstruction_loss_view_2
     + contrastive_loss
```


### Key arguments and presets

| Argument | Supplied Bash preset / meaning |
|---|---|
| `--train_dataset_path` | Merged Arrow dataset from Step 1 |
| `--pretrianed_model_path` | Pretrained representation checkpoint; preserve this exact spelling for the current CLI |
| `--output_dir` | Directory for fine-tuning checkpoints, logs, and the final model |
| `--deepspeed` | `/absolute/path/deepspeed/zero2.json` |
| `--max_length` | 1024; input sequence length used by the collator |
| `--mlm_probability` | 0.3 |
| `--do_mlm` | `True` |
| `--mask_value` | -100 |
| `--pad_value` | 100 |
| `--sampling` | `True`; sample features when their count exceeds `max_length` |
| `--keep_first_n_tokens` | 0 |
| `--num_train_epochs` | 10 |
| `--per_device_train_batch_size` | 32 |
| `--gradient_accumulation_steps` | 3 |
| `--learning_rate` | `1e-3` |
| `--lr_scheduler_type` | `cosine` |
| `--warmup_ratio` | 0.1 |
| `--seed` | 42 |
| `--bf16` | `True` |
| `--report_to` | `tensorboard` |
| `--do_eval` | `False`; this entry supplies no validation dataset |

The Python entry constructs a configuration from model arguments, but its `from_pretrained()` call loads the saved checkpoint configuration without passing that new configuration. Therefore, flags such as `--hidden_size`, `--n_layers`, and `--feature_name_vocab_size` do not change the loaded architecture. The checkpoint must already support the feature IDs used by your dataset.

### Example

```bash
bash 02_finetune.sh \
  --train_dataset_path /absolute/path/finetune_data/Crispr/train_merged_data \
  --pretrianed_model_path /absolute/path/output_model/pretrain_run_01/models \
  --output_dir /absolute/path/output_model/finetune_run_01 \
  --deepspeed /absolute/path/deepspeed/zero2.json \
  --max_length 1024 \
  --do_mlm True \
  --mlm_probability 0.3 \
  --sampling True \
  --num_train_epochs 10 \
  --per_device_train_batch_size 32 \
  --gradient_accumulation_steps 3 \
  --learning_rate 1e-3 \
  --bf16 True \
  --seed 42 \
  --overwrite_output_dir False
```

Use BF16 on supported hardware. This entry calls `train(resume_from_checkpoint=False)`; loading the pretrained model starts a new fine-tuning run rather than restoring an interrupted Trainer run.

### Output

```text
output_model/finetune_run_01/
├── models/
│   ├── config.json
│   └── model.safetensors    # May be sharded depending on model size/settings
├── checkpoint-*/
├── runs/                   # Default TensorBoard logging location
└── trainer_state.json
```



## Step 3 — Extract Fine-tuned CLS Embeddings

`03_run_emb.py` provides argument parsing and a main function. It reads normalized Parquet, filters rows using the label mapping, selects features, and uses `get_latent_embedding()` to extract the final-layer CLS representation. One view's embedding is exported.

### Key arguments and presets

| Argument | Current Bash setting or Python behavior |
|---|---|
| `--source_data_paths` | One or more normalized Parquet directories; searched recursively |
| `--gene2index_pkl` | Explicitly specified label mapping |
| `--feature2index_pkl` | Feature vocabulary compatible with the checkpoint |
| `--need_feature_list_pkl` | Selected feature list |
| `--pretrained_model_path` | Model directory; override the Bash pretraining preset with the fine-tuned `models/` directory |
| `--save_path` | Embedding output root; use a separate fine-tuning directory |
| `--key_gene_column` | `Metadata_Foci_Barcode_MatchedTo_GeneCode` |
| `--need_plate_list` | Optional filename/path substring filters; empty means all discovered files |
| `--metadata_parts` | Python default `-2 -5`; path indices for batch followed by type |
| `--device` | Bash passes `cuda:2`; Python defaults to `cuda:0`; CPU is used if CUDA is unavailable |
| `--batch_size` | 64 |
| `--max_length` | Bash passes 3505; Python defaults to 1024 |
| `--do_mlm` | `False` |
| `--mlm_probability` | 0.0 |
| `--sampling` | `False` in both the supplied Bash preset and Python default; append `--sampling` to enable |
| `--nproc` | 1 |
| `--verbosity` | 1 |
| `--log_file` | Override the pretraining log path with a path for this run |
| `--log_level` | `INFO` |

Keep MLM disabled for this extraction entry: its embedding method expects the unmasked collator input fields. When the selected feature count exceeds `max_length`, sampling selects a random subset if enabled; otherwise it retains the first `max_length` features. Shorter sequences are padded, and sequences of exactly that length are retained. Sampling has no selection effect when all features fit.

### Example

For this input layout:

```text
/absolute/path/profiles/
└── Crispr/
    └── Batch01/
        └── plate01.parquet
```

Run with explicit paths to the fine-tuned checkpoint and output directory:

```bash
bash 03_run_emb.sh \
  --source_data_paths /absolute/path/profiles \
  --gene2index_pkl /absolute/path/label_vocab/my_dataset_label2index.pkl \
  --feature2index_pkl /absolute/path/feature_vocab/feature2index_vocabulary_7366.pkl \
  --need_feature_list_pkl /absolute/path/feature_intersection_union/my_dataset_union_feature.pkl \
  --pretrained_model_path /absolute/path/output_model/finetune_run_01/models \
  --save_path /absolute/path/embeddings/finetune_run_01 \
  --key_gene_column Metadata_Foci_Barcode_MatchedTo_GeneCode \
  --metadata_parts -2 -3 \
  --device cuda:0 \
  --batch_size 64 \
  --max_length 3505 \
  --mlm_probability 0.0 \
  --nproc 1 \
  --verbosity 1 \
  --log_file /absolute/path/logs/finetune_embeddings.log \
  --log_level INFO
```

The example retains the supplied Bash sequence length. Choose `max_length` for your experiment and available memory. The explicit `--metadata_parts -2 -3` matches the example layout: `Batch01` is the parent directory and `Crispr` is its parent.

### Output

The writer saves:

```text
<save_path>/<type>/<batch>/<input_file_stem>.parquet
```

The example produces:

```text
embeddings/finetune_run_01/
└── Crispr/
    └── Batch01/
        └── plate01.parquet
```
