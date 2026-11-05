# 04 — Morphological Retrieval mAP

## Pipeline Overview

```text
               Embedding or normalized feature Parquet files
                                    │
                                    ▼
             ┌─────────────────────────────────────────┐
             │             01_run_mAP                  │
             │   Select metadata and numerical columns │
             │     Optional: aggregate within wells    │
             └────────────────────┬────────────────────┘
                                  │
                                  ▼
             ┌─────────────────────────────────────────┐
             │        Choose a retrieval metric        │
             │   replicates_activity / sister_activity │
             │           / sister_consistency          │
             └────────────────────┬────────────────────┘
                                  │
                                  ▼
             ┌─────────────────────────────────────────┐
             │     Rank profiles by cosine similarity  │
             │     Calculate AP and aggregate to mAP   │
             │       Estimate significance with copairs│
             └────────────────────┬────────────────────┘
                                  │
                                  ▼
                  <save_path>/<seed>/ap_scores_dict.pickle
```

## Before You Start

Before calculating mAP for DeSCORT representations, **run the embedding extraction step first** to generate embedding Parquet files:

- For a pretrained model, run `bash 04_run_emb.sh` in `tutorials/02_pretrain/`, following [Step 4 of the pretraining tutorial](../02_pretrain/README.md#step-4--extract-cls-embeddings).
- For a fine-tuned model, run `bash 03_run_emb.sh` in `tutorials/03_finetune/`, following [Step 3 of the fine-tuning tutorial](../03_finetune/README.md#step-3--extract-fine-tuned-cls-embeddings).

Then set the mAP argument `--source_data_paths` to the directory containing the generated embedding Parquet files. The mAP script evaluates these existing embeddings; it does not load a model or extract embeddings.

For a morphological baseline, use normalized feature Parquet files from [preprocessing](../01_preprocess_data/README.md); this baseline does not require embedding extraction.

Activate the environment and inspect the available arguments:

```bash
conda activate DeSCORT
cd /fse/home/wangyongyan/Project/CellPainting/DeSCORT/other_model/DeSCORT/tutorials/04_mAP
bash 01_run_mAP.sh --help
```

## Step 1 — Prepare Profiles and Metadata

### Required input columns

For the **CRISPR** example, each Parquet file contains:

| Column | Purpose |
|---|---|
| `Metadata_Foci_Barcode_MatchedTo_GeneCode` | Perturbation label, including the negative-control label |
| `Metadata_Foci_Barcode_MatchedTo_Barcode` | sgRNA barcode; used to define replicate groups for `replicates_activity` |
| `Metadata_Foci_well` | Well identifier used in the supplied aggregation preset |
| `emb_0`, `emb_1`, ... | Numerical embedding dimensions; alternatively, normalized morphological features |


## Step 2 — Calculate Retrieval mAP

### Choose the metric

| `--mAP_type` | Positive profiles | Negative profiles | Result grouping |
|---|---|---|---|
| `replicates_activity` | Other profiles with the same gene and sgRNA barcode | Negative-control profiles | Gene and barcode |
| `sister_activity` | Other profiles with the same gene, pooling barcodes | Negative-control profiles | Gene |
| `sister_consistency` | Other profiles with the same gene, pooling barcodes | Profiles from other non-control genes | Gene |

For every query, the helper ranks profiles by descending cosine similarity and excludes the query itself. Average precision is the mean precision at the ranks occupied by positive profiles; copairs then aggregates query AP values within the result group. Higher mAP indicates stronger retrieval under the chosen positive/negative definition. Sister metrics do not explicitly require positives to have a different barcode.

With `--sample_method random`, negative profiles are sampled without replacement up to `min(group_size × n_number, available_negatives)`.

With `all`, every available negative profile is used. For `sister_consistency`, controls are removed before retrieval and negative samples come from other genes.

### Key arguments and presets

| Argument | Supplied Bash preset / meaning |
|---|---|
| `--mAP_type` | `sister_consistency`; select one of the three metrics above |
| `--source_data_paths` | One or more directories searched recursively for Parquet files; Bash currently points to pretraining embeddings |
| `--need_feature_list_pkl` | Selected feature list; required even when intersection is disabled |
| `--feature2index_pkl` | Morphological feature vocabulary; required even when intersection is disabled |
| `--save_path` | Root directory for per-seed result files |
| `--perturbation_gene_key` | `Metadata_Foci_Barcode_MatchedTo_GeneCode` |
| `--sgRNA_barcode_key` | `Metadata_Foci_Barcode_MatchedTo_Barcode`; used by replicate activity |
| `--negative_control_list` | `nontargeting`; one or more exact control labels |
| `--choose_type` | `well-level`; currently parsed but does not control aggregation |
| `--do_groupby_mean` | Enabled by Bash; direct Python execution defaults to disabled |
| `--need_columns` | Metadata columns retained before aggregation |
| `--group_by_columns` | Gene, barcode, and well in the Bash preset; required by the parser even without aggregation |
| `--do_intersect_features` | Disabled in Bash and Python |
| `--need_plate_list` | Optional path-substring filters; Bash preset is empty |
| `--random_seeds` | Bash passes `42`; Python defaults to `42 2026 2025 2024 2023` |
| `--sample_method`, `--n_number` | `random` and 2 |
| `--n_jobs` | 20 parallel gene-level workers; reduce to fit CPU and memory resources |
| `--log_file`, `--log_level` | Experiment log path and `INFO` |


### Example

Evaluate the fine-tuned embeddings produced in the previous tutorial:

```bash
bash 01_run_mAP.sh \
  --mAP_type sister_consistency \
  --source_data_paths /absolute/path/embeddings/finetune_run_01 \
  --need_feature_list_pkl /absolute/path/feature_intersection_union/my_dataset_union_feature.pkl \
  --feature2index_pkl /absolute/path/feature_vocab/feature2index_vocabulary_7366.pkl \
  --save_path /absolute/path/map_results/finetune_run_01/sister_consistency \
  --perturbation_gene_key Metadata_Foci_Barcode_MatchedTo_GeneCode \
  --sgRNA_barcode_key Metadata_Foci_Barcode_MatchedTo_Barcode \
  --negative_control_list nontargeting \
  --choose_type well-level \
  --do_groupby_mean \
  --need_columns Metadata_Foci_Barcode_MatchedTo_GeneCode Metadata_Foci_Barcode_MatchedTo_Barcode Metadata_Foci_well \
  --group_by_columns Metadata_Foci_Barcode_MatchedTo_GeneCode Metadata_Foci_Barcode_MatchedTo_Barcode Metadata_Foci_well \
  --random_seeds 42 2026 2025 2024 2023 \
  --sample_method random \
  --n_number 2 \
  --n_jobs 4 \
  --log_file /absolute/path/logs/finetune_sister_consistency.log \
  --log_level INFO
```

For activity evaluation, change `--mAP_type` to `sister_activity` or `replicates_activity` and choose a separate `--save_path` and log file. Ensure negative controls are present in the input embeddings; upstream label filtering may already have excluded them.

To evaluate a morphological baseline, replace `--source_data_paths` with the normalized feature directories and use a separate result root. Keep the aggregation, controls, metric, and negative-sampling settings consistent when comparing representations.

### Output

```text
map_results/finetune_run_01/sister_consistency/
├── 42/
│   └── ap_scores_dict.pickle
├── 2026/
│   └── ap_scores_dict.pickle
├── 2025/
│   └── ap_scores_dict.pickle
├── 2024/
│   └── ap_scores_dict.pickle
└── 2023/
    └── ap_scores_dict.pickle
```
