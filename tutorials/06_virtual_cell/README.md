# 06 — Virtual Cell Perturbation Generation

## Pipeline Overview

```text
             Normalized profiles + perturbation conditions
                                  │
                                  ▼
              ┌───────────────────────────────────────┐
              │            01_process_data            │
              │ Select training and test profiles     │
              │ Pair perturbations with controls      │
              └───────────────────┬───────────────────┘
                                  │
                                  ▼
              ┌───────────────────────────────────────┐
              │              02_finetune              │
              │ Fine-tune a perturbation predictor    │
              │ using control profiles + conditions   │
              └───────────────────┬───────────────────┘
                                  │
                                  ▼
                         finetune/models/
                                  │
                                  ▼
              ┌───────────────────────────────────────┐
              │              03_inference             │
              │ Generate perturbed morphology         │
              │ Export truth, prediction, train mean  │
              └───────────────────┬───────────────────┘
                                  │
                                  ▼
              ┌───────────────────────────────────────┐
              │     04_calculate_vcc2026_metric        │
              │ Validate alignment and calculate      │
              │ model and baseline morphology metrics │
              └───────────────────┬───────────────────┘
                                  │
                                  ▼
                       model_vs_baseline.csv
```

## Before You Start

Cell-line transfer and unseen-gene generation follow the same workflow: prepare paired data, fine-tune a perturbation predictor, generate perturbed profiles, and evaluate predictions. Their main difference is the dataset and the definition of the training/test split.

| Task | Training/test definition | CRISPR entry directory |
|---|---|---|
| Unseen-gene generation | Separate non-control gene sets for training and testing | `unseen_gene_generation/Crispr/` |
| Cell-line transfer | Select training and test data according to the source/target cell-line experiment | `cell_line_transfer/Crispr/` |

Both tasks also provide `Compound/` entries. This tutorial gives **one example only: CRISPR unseen-gene generation**. The cell-line-transfer workflow uses its corresponding entries and datasets; directory names alone do not enforce data separation.

This task predicts continuous **Cell Painting morphological features**, rather than CLS embeddings or gene expression. It does not require running `run_emb` first. Use a checkpoint compatible with `DeSCORTForPerturbPrediction` and the selected architecture and condition vectors.

## Step 1 — Prepare Paired Training and Test Datasets

### Required resources

| Resource | Purpose |
|---|---|
| Normalized Parquet files | Morphological values, perturbation labels, and within-file controls |
| `gene2index_pkl` / `index2gene_pkl` | Consistent gene-name/integer mappings |
| `esm2_dict_pkl` | Gene-name-to-ESM2 dictionary used to filter the gene lists |
| `index2esm2_pkl` | Gene-ID-to-ESM2 dictionary used by the model; must agree with the name-based dictionary |
| `feature2index_pkl` / `index2feature_pkl` | Consistent feature-name/integer mappings |
| `need_feature_list_pkl` | Selected morphological features, shared across training and inference |
| Training/test gene-list pickles | Explicit non-overlapping non-control gene sets |


### Key arguments

| Argument | Meaning |
|---|---|
| `--source_data_paths` | One or more normalized Parquet directories, searched recursively |
| `--train_output_dir`, `--test_output_dir` | Separate temporary shard directories |
| `--train_save_dir`, `--test_save_dir` | Separate merged Hugging Face Dataset directories |
| `--gene2index_pkl`, `--esm2_dict_pkl` | Gene mapping and name-based condition dictionary |
| `--feature2index_pkl`, `--need_feature_list_pkl` | Feature vocabulary and selected feature list |
| `--train_gene_list_pkl`, `--test_gene_list_pkl` | Training/test gene-list pickles; this example supplies both |
| `--key_gene_column` | `Metadata_Foci_Barcode_MatchedTo_GeneCode` |
| `--control_key_word` | `nontargeting`; controls must exist within each processed file |
| `--train_data_names`, `--test_data_names` | Optional filename/path substring filters for each split |
| `--nproc`, `--merge_nproc`, `--verbosity` | Defaults 1, 2, and 1 |
| `--check-only` | Inspect resource loading and gene-list selection without building datasets |

### Example

```bash
bash 01_process_data_unseen_gene_generation.sh \
  --source_data_paths /fse/DC/CP_data/feature_data/20210422_6W_CP257/raw_data/well_level_new/normalization_clip_data \
  --train_output_dir /absolute/path/virtual_cell/unseen_crispr_50genes/tmp/train \
  --train_save_dir /absolute/path/virtual_cell/unseen_crispr_50genes/data/train \
  --test_output_dir /absolute/path/virtual_cell/unseen_crispr_50genes/tmp/test \
  --test_save_dir /absolute/path/virtual_cell/unseen_crispr_50genes/data/test \
  --gene2index_pkl /absolute/path/ESM2_files/gene2index_21611.pkl \
  --esm2_dict_pkl /absolute/path/ESM2_files/ESM2_pert_features_with_nontargeting.pkl \
  --feature2index_pkl /absolute/path/feature_vocab/feature2index_vocabulary_7366.pkl \
  --need_feature_list_pkl /absolute/path/selected_features/intersection_columns_1107.pkl \
  --train_gene_list_pkl /absolute/path/few_shot/gene_50_train.pkl \
  --test_gene_list_pkl /absolute/path/few_shot/gene_50_test.pkl \
  --key_gene_column Metadata_Foci_Barcode_MatchedTo_GeneCode \
  --control_key_word nontargeting \
  --train_data_names CP257A CP257B CP257D CP257F CP257H \
  --test_data_names CP257A CP257B CP257D CP257F CP257H \
  --nproc 1 \
  --merge_nproc 2 \
  --verbosity 1
```



### Output

```text
virtual_cell/unseen_crispr_50genes/
├── tmp/
│   ├── train/
│   └── test/
└── data/
    ├── train/
    └── test/
```

Each dataset row contains:

| Field | Content |
|---|---|
| `Perturb_Gene` | Perturbation gene name |
| `Perturb_expression` | Observed perturbed morphology, used as the target |
| `Control_expression` | Paired control morphology |
| `Feature_name_index` | Feature IDs aligned with both vectors |
| `Perturb_Gene_Index` | Gene ID used to retrieve the condition vector |

Use `datasets.load_from_disk()` to inspect the saved datasets. Verify that the non-control gene sets do not overlap and that paired vectors and feature IDs have matching lengths and consistent order. Use fresh temporary directories for each experiment.

## Step 2 — Fine-tune the Perturbation Predictor

The entry loads an initialization checkpoint and trains a conditional predictor using control morphology and gene ESM2 vectors. The supplied loss weights are:

```text
loss = 1.0 × mean_loss
     + 1.0 × direction_loss
     + 0.1 × marginal_quantile_loss
```


### Key arguments and presets

| Argument | Supplied Bash preset / meaning |
|---|---|
| `--train_dataset_path` | Training Dataset from Step 1; test data are not passed to training |
| `--pretrianed_model_path` | Initialization checkpoint; retain this exact CLI spelling |
| `--index2esm2_pkl` | Gene-ID-to-ESM2 dictionary |
| `--output_dir` | New fine-tuning experiment directory |
| `--gene_column` | `Perturb_Gene`; used for same-gene training batches |
| `--hidden_size`, `--n_layers`, `--num_attention_heads` | 256, 8, and 8 |
| `--n_routed_experts`, `--num_experts_per_tok` | 8 routed experts, selecting 4 per token |
| `--esm2_input_dim` | 5120; must match the condition vectors |
| `--max_seq_len`, `--max_length` | 1107 and 1107 |
| `--sampling`, `--pad_value` | `False` and 100 |
| `--num_train_epochs`, `--per_device_train_batch_size` | 50 and 64 |
| `--gradient_accumulation_steps` | 1 |
| `--learning_rate`, `--lr_scheduler_type` | `2e-4` and `cosine` |
| `--warmup_ratio`, `--seed` | 0.1 and 42 |
| `--bf16`, `--do_eval` | `True` and `False` |
| `--save_strategy`, `--save_total_limit` | `epoch` and 5 |

### Example

```bash
GPU_IDS=0,1 bash 02_finetune.sh \
  --train_dataset_path /absolute/path/virtual_cell/unseen_crispr_50genes/data/train \
  --pretrianed_model_path /absolute/path/initialization_checkpoint/models \
  --index2esm2_pkl /absolute/path/ESM2_files/Index2ESM2_pert_features_with_nontargeting_19791.pkl \
  --output_dir /absolute/path/virtual_cell/unseen_crispr_50genes/finetune \
  --esm2_input_dim 5120 \
  --max_length 1107 \
  --sampling False \
  --num_train_epochs 50 \
  --per_device_train_batch_size 64 \
  --gradient_accumulation_steps 1 \
  --learning_rate 2e-4 \
  --bf16 True \
  --seed 42 \
  --overwrite_output_dir False
```

### Output

```text
virtual_cell/unseen_crispr_50genes/finetune/
├── models/
│   ├── config.json
│   └── model.safetensors    # May be sharded depending on settings
├── checkpoint-*/
├── runs/
└── trainer_state.json
```


## Step 3 — Generate Morphology for Unseen Genes

The model receives paired control morphology and the test gene's ESM2 condition vector. Observed test perturbation morphology is exported as ground truth; it is not the input used to generate predictions.

### Key arguments and presets

| Argument | Meaning |
|---|---|
| `--pretrained_model_path` | Fine-tuned `models/` directory from Step 2 |
| `--index2gene_pkl`, `--index2esm2_pkl`, `--index2feature_pkl` | Inverse mappings and condition vectors consistent with Step 1 |
| `--train_dataset_path` | Training Dataset, used to build the train-mean baseline |
| `--test_dataset_path` | Held-out-gene Dataset to predict |
| `--output_dir` | Directory for the three H5AD files |
| `--prediction_filename` | Model prediction filename; set explicitly below |
| `--ground_truth_filename`, `--mean_baseline_filename` | Truth and baseline filenames |
| `--device`, `--batch_size`, `--max_length` | Logical device, inference batch size, and sequence length |
| `--no_sampling` | Preserve the preset's disabled sampling behavior |
| `--gene_column`, `--control_label` | Output label column `Perturb_Gene` and control label `nontargeting` |

### Example

```bash
CUDA_VISIBLE_DEVICES=0 bash 03_run_inference.sh \
  --pretrained_model_path /absolute/path/virtual_cell/unseen_crispr_50genes/finetune/models \
  --index2gene_pkl /absolute/path/label_vocab/index2gene_21611.pkl \
  --index2esm2_pkl /absolute/path/ESM2_files/Index2ESM2_pert_features_with_nontargeting_19791.pkl \
  --index2feature_pkl /absolute/path/feature_vocab/index2feature_vocabulary_7366.pkl \
  --train_dataset_path /absolute/path/virtual_cell/unseen_crispr_50genes/data/train \
  --test_dataset_path /absolute/path/virtual_cell/unseen_crispr_50genes/data/test \
  --output_dir /absolute/path/virtual_cell/unseen_crispr_50genes/inference \
  --prediction_filename prediction_data.h5ad \
  --ground_truth_filename ground_truth.h5ad \
  --mean_baseline_filename perturb_mean_data.h5ad \
  --device cuda:0 \
  --batch_size 64 \
  --max_length 1107 \
  --pad_value 100 \
  --no_sampling \
  --gene_column Perturb_Gene \
  --control_label nontargeting \
  --log_file /absolute/path/virtual_cell/unseen_crispr_50genes/logs/inference.log \
  --log_level INFO
```

`CUDA_VISIBLE_DEVICES=0` selects physical GPU 0; `--device cuda:0` selects the first visible GPU. Use `--device cpu` for CPU execution. The exporter collects the complete test matrices in memory, so reducing batch size does not remove the memory requirement of the final outputs.

Training and inference must use the same feature axis. With sampling disabled, sequences longer than `max_length` are truncated; shorter sequences are padded. This exporter expects the complete original feature-name axis to match the returned matrix, so choose a length covering all selected features and keep the feature order consistent. Padding positions are excluded from exported morphology.

### Output

| File | Contents of `X` |
|---|---|
| `ground_truth.h5ad` | Observed test perturbation morphology, followed by paired controls |
| `prediction_data.h5ad` | Predicted test perturbation morphology, followed by the same controls |
| `perturb_mean_data.h5ad` | Repeated train-mean baseline, followed by the same controls |


## Step 4 — Evaluate Predictions and the Train-mean Baseline

This entry calculates seven raw VCC2026-style metrics for continuous morphology. Differential-feature testing uses Wilcoxon/Mann–Whitney tests with BH correction. Effect sizes are perturbation-minus-control mean differences; `lfc` in the metric names does not mean taking log2 fold changes of morphology values.

### Key arguments

| Argument | Meaning |
|---|---|
| `--ground_truth_path`, `--prediction_path`, `--baseline_path` | Three H5AD files from Step 3 |
| `--outdir`, `--cache_dir` | Metric output and reusable DE-cache directories |
| `--pert_col`, `--control_name` | `Perturb_Gene` and `nontargeting` |
| `--dedup_controls` | `true`; deduplicate identical control vectors |
| `--strict_alignment` | `true`; require matching sample IDs/order and perturbation labels |
| `--copy_truth_controls` | `false`; do not replace the supplied controls |
| `--fdr_scope` | `per_pert`; correct across features within each perturbation |
| `--n_jobs`, `--blas_threads` | 8 and 4 in the supplied preset |
| `--reuse_cache`, `--show_progress` | `true` and `true` |
| `--check-only` | Validate input alignment and run a small synthetic MWU check without calculating experiment metrics |

### Example

```bash
bash 04_calculate_vcc2026_metric.sh \
  --ground_truth_path /absolute/path/virtual_cell/unseen_crispr_50genes/inference/ground_truth.h5ad \
  --prediction_path /absolute/path/virtual_cell/unseen_crispr_50genes/inference/prediction_data.h5ad \
  --baseline_path /absolute/path/virtual_cell/unseen_crispr_50genes/inference/perturb_mean_data.h5ad \
  --outdir /absolute/path/virtual_cell/unseen_crispr_50genes/metrics \
  --cache_dir /absolute/path/virtual_cell/unseen_crispr_50genes/de_cache \
  --pert_col Perturb_Gene \
  --control_name nontargeting \
  --dedup_controls true \
  --strict_alignment true \
  --copy_truth_controls false \
  --fdr_scope per_pert \
  --n_jobs 8 \
  --blas_threads 4 \
  --scipy_checks 100 \
  --reuse_cache true \
  --show_progress true
```


### Output metrics

| Metric | Interpretation |
|---|---|
| `pds_cosine` | Perturbation retrieval using cosine distance between morphology effects; higher is better |
| `mse` | Mean-squared error of perturbation mean morphology; lower is better |
| `expr_mse_unbiased_capped_norm` | Noise-corrected, normalized, capped error; lower is better |
| `de_wilcoxon_direction_fidelity_yield_raw` | Direction agreement and yield of significant effects; higher is better |
| `de_wilcoxon_direction_reach_raw` | Effect coverage under the direction-agreement criterion; higher is better |
| `de_wilcoxon_sig_jaccard` | Jaccard similarity of significant feature sets; higher is better |
| `de_wilcoxon_lfc_nmae` | Normalized absolute error of significant morphology effects; lower is better |

These are raw metrics, not uniformly scaled scores relative to replicate upper bounds and baselines. Inspect `valid_metric_gene_counts` alongside the summary; undefined effects can produce NaN values and should not automatically be replaced with zero.

```text
virtual_cell/unseen_crispr_50genes/
├── tmp/
│   ├── train/
│   └── test/
├── data/
│   ├── train/
│   └── test/
├── finetune/models/
├── inference/
│   ├── ground_truth.h5ad
│   ├── prediction_data.h5ad
│   └── perturb_mean_data.h5ad
├── logs/inference.log
├── de_cache/
└── metrics/
    ├── input_provenance.json
    ├── cli_parameters.json
    ├── model_metrics.csv
    ├── model.parquet
    ├── cell_eval/
    │   ├── results.csv
    │   └── agg_results.csv
    ├── run_manifest.json
    ├── baseline/
    │   ├── baseline_metrics.csv
    │   ├── baseline.parquet
    │   └── run_manifest.json
    └── model_vs_baseline.csv
```
