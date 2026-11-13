# 01 — Preprocess Data

This directory contains the complete **data preprocessing pipeline** for Cell Painting data used in the DeSCORT framework. The pipeline takes raw CSV/Parquet files and produces cleaned, normalized, feature-selected data ready for downstream pretraining and fine-tuning.

## Typical Workflow

Run the parameterized examples above in order from this tutorial directory:

1. CSV to Parquet
2. Preprocess Data
3. Build Label Vocabulary
4. Feature Intersection & Union
5. Extract Union Feature Data


## Pipeline Overview

```text
                                  Raw CSV files
                                        │
                                        ▼
                   ┌─────────────────────────────────────────┐
                   │              01_csv2parquet             │
                   │         CSV → Parquet conversion        │
                   └────────────────────┬────────────────────┘
                                        │
                                        ▼
                   ┌─────────────────────────────────────────┐
                   │            02_preprocess_data           │
                   │    Aggregation → Normalization → Clip   │
                   │           → Feature selection           │
                   └────────────────────┬────────────────────┘
                                        │
                  ┌─────────────────────┴─────────────────────┐
                  │                                           │
                  ▼                                           ▼
┌───────────────────────────────────┐    ┌─────────────────────────────────────────┐
│        03_make_label_vocab        │    │   04_select_intersection_union_feature  │
│       Build label vocabulary      │    │      Compute shared & union feature     │
│    (label2index / index2label)    │    │          sets across all plates         │
│                                   │    │                                         │
└───────────────────────────────────┘    └────────────────────┬────────────────────┘
                                                              │
                                                              ▼
                                         ┌─────────────────────────────────────────┐
                                         │        05_get_union_feature_data        │
                                         │     Filter to union features + save     │
                                         └─────────────────────────────────────────┘
```

This tutorial directory:

```bash
cd /absolute/path/tutorials/01_preprocess_data
```

## Step-by-Step Guide

### Step 1 — Convert CSV to Parquet

Converts raw Cell Painting CSV feature files into the more efficient Parquet format.

**Key arguments:**

| Argument    | Description                              |
| ----------- | ---------------------------------------- |
| `--csv_dir` | Input directory containing CSV files     |
| `--out_dir` | Output directory for Parquet files       |
| `--pattern` | File matching pattern (default: `*.csv`) |

**Example:**

```bash
bash 01_csv2parquet.sh \
  --csv_dir /absolute/path/csv_raw_data \
  --out_dir /absolute/path/raw_data \
  --pattern "*.csv" \
  --log_file /absolute/path/logs/csv2parquet.log
```

***

### Step 2 — Preprocess data

For each plate/Parquet file:

1. **Median aggregation** — Aggregates single-cell features to well-level by median (optional, controlled by `--data_level`)
2. **Normalization** — Supports `standardize`, `robustize`, `mad_robustize`, and `spherize` methods. Control samples can be selected via `--normalize_samples_query` (e.g., `"Metadata_Foci_Barcode_MatchedTo_GeneCode == 'nontargeting'"`)
3. **Percentile clipping** — Clips feature values to `[lower_percentile, upper_percentile]` (default: 1–99)
4. **Feature selection** — Removes low-variance features, highly correlated features (`--corr_threshold`), columns with excessive NAs (`--na_cutoff`), and blocklisted features

**Key arguments:**

| Argument                      | Description                                                                               |
| ----------------------------- | ----------------------------------------------------------------------------------------- |
| `--source_data_paths`         | One or more directories of Parquet files                                                  |
| `--data_level`                | Start with the `well` or `sc` aggregation level                                            |
| `--normalize_method`          | Normalization method (default: `mad_robustize`)                                           |
| `--normalize_samples_query`   | Select normalization controls with a Pandas query expression; the default control label is `nontargeting`                                 |
| `--clip_lower_percentile`     | Lower clipping percentile; default `1`                                                |
| `--clip_upper_percentile`     | Upper clipping percentile; default `99`                                               |
| `--feature_select_operations` | Operations: `variance_threshold`, `correlation_threshold`, `drop_na_columns`, `blocklist` |
| `--corr_threshold`            | Correlation threshold for feature removal (default: 0.9)                                  |
| `--na_cutoff`                 | Maximum NA fraction before dropping a column (default: 0.05)                              |
| `--need_columns`              | Columns used as aggregation strata (metadata columns)                                     |
| `--do_map_channel_names`      | Map legacy channel names (DAPI→DNA, Phalloidin→AGP, ConA→ER, Mito→Mito)                   |

**Example:**

```bash
bash 02_preprocess_data.sh \
  --source_data_paths /absolute/path/raw_data \
  --clip_output_path /absolute/path/normalization_clip_data \
  --normalization_select_feature_output_path /absolute/path/normalization_select_feature_data \
  --data_level sc \
  --need_columns Metadata_Foci_Barcode_MatchedTo_GeneCode Metadata_Foci_Barcode_MatchedTo_Barcode Metadata_Foci_well Metadata_Foci_plate \
  --normalize_method mad_robustize \
  --normalize_samples_query "Metadata_Foci_Barcode_MatchedTo_GeneCode == 'nontargeting'" \
  --clip_lower_percentile 1 \
  --clip_upper_percentile 99 \
  --feature_select_operations variance_threshold correlation_threshold drop_na_columns blocklist \
  --corr_threshold 0.9 \
  --na_cutoff 0.05 \
  --do_map_channel_names \
  --log_file /absolute/path/logs/preprocess_data.log
```

***

### Step 3 — Build the label vocabulary

Collects unique label values from a specified metadata column across all Parquet files and builds a bidirectional vocabulary (`label2index` and `index2label`) saved as pickle files.

**Key arguments:**

| Argument              | Description                                                              |
| --------------------- | ------------------------------------------------------------------------ |
| `--source_data_paths` | One or more directories of Parquet files                                 |
| `--label_column`      | Metadata column containing labels; default `Metadata_Foci_Barcode_MatchedTo_GeneCode` |
| `--output_dir`        | Directory to save vocabulary pickle files                                |
| `--output_prefix`     | Prefix for output filenames                                              |

**Output:**

* `{prefix}_label2index_{N}.pkl` — maps label → integer index
* `{prefix}_index2label_{N}.pkl` — maps integer index → label

**Example:**

```bash
bash 03_make_label_vocab.sh \
  --source_data_paths /absolute/path/data1 /absolute/path/data2 \
  --label_column Metadata_Foci_Barcode_MatchedTo_GeneCode \
  --output_dir /absolute/path/label_vocab \
  --output_prefix my_dataset \
  --log_file /absolute/path/logs/make_label_vocab.log
```

***

### Step 4 — Feature intersection and union

Computes the **intersection** (features shared across ALL plates) and the **union** (features present in ANY plate) of feature columns. Optionally filters by a pre-existing feature vocabulary to retain only known features.

**Key arguments:**

| Argument              | Description                                                          |
| --------------------- | -------------------------------------------------------------------- |
| `--source_data_paths` | One or more directories of Parquet files                             |
| `--vocab_file`        | Optional pickle file containing a `feature2index` dict for filtering |
| `--output_dir`        | Directory to save output pickle files                                |
| `--output_prefix`     | Prefix for output filenames                                          |

**Output:**

* `{prefix}_union_feature_{N}.pkl` — set of all feature columns across plates
* `{prefix}_intersect_feature_{N}.pkl` — set of feature columns shared by all plates

**Example:**

```bash
bash 04_select_intersection_union_feature.sh \
  --source_data_paths /absolute/path/normalization_select_feature_data \
  --vocab_file /absolute/path/feature2index_vocabulary_7366.pkl \
  --output_dir /absolute/path/feature_intersection_union \
  --output_prefix my_dataset \
  --log_file /absolute/path/logs/select_features.log
```

***

### Step 5 — Extract union-feature data (optional)

Filter each Parquet using the feature set from Step 4, separate metadata and feature columns, and save the result. The retained features can differ between plates after preprocessing; this step uses their union for analysis.

**Key arguments:**

| Argument              | Description                                                          |
| --------------------- | -------------------------------------------------------------------- |
| `--source_data_paths` | One or more directories of Parquet files                             |
| `--feature_list_pkl`  | Path to the union/intersection feature set pickle file (from Step 4) |
| `--output_path`       | Directory to save filtered Parquet files                             |

**Example:**

```bash
bash 05_get_union_feature_data.sh \
  --source_data_paths /absolute/path/normalization_clip_data \
  --feature_list_pkl /absolute/path/feature_intersection_union/my_dataset_union_feature_N.pkl \
  --output_path /absolute/path/normalization_clip_data_select_feature \
  --log_file /absolute/path/logs/get_union_feature_data.log
```



## Output Directory Structure

After running the full pipeline, expect a directory layout like:

```
output/
├── raw_data/                          # Step 1: CSV → Parquet
├── normalization_clip_data/           # Step 2: normalized & clipped
├── normalization_select_feature_data/ # Step 2: feature-selected
├── label_vocab/                       # Step 3: label vocabularies
│   ├── vocab_label2index_{N}.pkl
│   └── vocab_index2label_{N}.pkl
├── feature_intersection_union/        # Step 4: feature sets
│   ├── vocab_union_feature_{N}.pkl
│   └── vocab_intersect_feature_{N}.pkl
└── normalization_clip_data_select_feature/  # Step 5: filtered data
    └── {batch_name}/
        └── {plate_name}.parquet
```
