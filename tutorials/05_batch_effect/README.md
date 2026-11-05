# 05 — Batch Effect Correction and Evaluation

## Pipeline Overview

```text
              Matched compound CSV + normalized Parquet files
                                  │
                                  ▼
                  ┌─────────────────────────────────┐
                  │     00_preprocess_batch_data    │
                  │ Load plates, clean, build PCA   │
                  │ neighbors, UMAP, and clustering │
                  └───────────────┬─────────────────┘
                                  │
                   ┌──────────────┴──────────────┐
                   ▼                             ▼
           Input AnnData H5AD             Input feature Parquet
                   │                             │
                   ▼                             ▼
       ┌───────────────────────┐     ┌─────────────────────────┐
       │    01_batch_method    │     │ 02_batch_method_R_clean │
       │ Seven Python adapters │     │ FastMNN, Seurat CCA/RPCA │
       └───────────┬───────────┘     └────────────┬────────────┘
                   │                             │
                   ▼                             ▼
          Corrected H5AD files          Corrected Parquet files
                   │                             │
                   │                  Convert to matched H5AD
                   │                             │
                   └──────────────┬──────────────┘
                                  ▼
                  ┌─────────────────────────────────┐
                  │      03_calculate_metrics       │
                  │ scIB + LISI + retrieval mAP     │
                  └───────────────┬─────────────────┘
                                  ▼
                     <method>_scib_metric_result.pkl
```

## Before You Start

Prepare normalized morphological profiles from [preprocessing](../01_preprocess_data/README.md). To include DeSCORT representations in the comparison, first extract embeddings with `04_run_emb.sh` in [pretraining](../02_pretrain/README.md) or `03_run_emb.sh` in [fine-tuning](../03_finetune/README.md), then prepare an AnnData file containing those embeddings and matching sample metadata.

Start with `00_preprocess_batch_data.ipynb` to prepare the initial `MAD_adata.h5ad` and `MAD_adata.parquet`. Run the Python adapters and R notebook on these matched inputs, then evaluate their H5AD results. Converting corrected R outputs to H5AD remains a separate preparation step.

Activate the Python environment and inspect the entry points:

```bash
conda activate DeSCORT
cd /fse/home/wangyongyan/Project/CellPainting/DeSCORT/other_model/DeSCORT/tutorials/05_batch_effect
bash run.sh --help
bash run.sh 01_batch_method.py --help
bash run.sh 03_calculate_metrics.py --help
```

Use the existing Bash wrapper to launch Python entries. Replace every `/absolute/path/...` placeholder below with your actual absolute path. Follow the [installation instructions](../../docs/installation.md) for the base environment.

Method-specific Python dependencies include BBKNN, harmonypy, Scanorama, the Scanpy MNN backend. Evaluation additionally imports scIB and uses its kBET backend. The R notebook requires an R Jupyter kernel and the packages `arrow`, `batchelor`, `SingleCellExperiment`, `Seurat`, `dplyr`, and `Matrix`. These method-specific dependencies may require additional installation beyond the base DeSCORT environment. The notebook uses Seurat's `layer` interface; use a compatible Seurat installation.

## Step 0 — Prepare Batch Evaluation Data

Open [00_preprocess_batch_data.ipynb](00_preprocess_batch_data.ipynb) with the DeSCORT Python kernel and run its cells in order. Use this tutorial directory as the working directory: the notebook imports repository utilities through the relative path `../../`.

### Input paths and settings

| Notebook variable | Purpose |
|---|---|
| `hit_meta_path` | Matched JUMP-Target-2 compound/plate CSV; must contain `Metadata_Source` and `Metadata_Plate` |
| `DATA_ROOT` | Root containing `<source>/normalization_clip_data/` directories |
| `save_path` | Output H5AD file for the Python batch-correction methods |
| `parquet_save_path` | Output Parquet file for the R methods |


Replace the path assignments in the corresponding notebook cells. For example:

```python
# Metadata-loading cell
hit_meta_path = "/absolute/path/batch_inputs/JUMP-Target-2-Compound-Plate.csv"

# Plate-loading cell
DATA_ROOT = "/absolute/path/cpg0016_jump"

# Save cell
save_path = "/absolute/path/batch_inputs/MAD_adata.h5ad"
parquet_save_path = "/absolute/path/batch_inputs/MAD_adata.parquet"
```


### Output

```text
batch_inputs/
├── MAD_adata.h5ad      # Float32 profiles, obs metadata, PCA, graph, UMAP, clusters
└── MAD_adata.parquet   # Cleaned profiles and metadata for R
```


## Step 1 — Prepare Inputs and Run Python Methods

### Required input data

| AnnData field | Purpose |
|---|---|
| `adata.X` | Finite numerical profile matrix, with samples in rows and features in columns |
| `adata.obs_names` | Unique sample identifiers shared across methods |
| `adata.obs["Metadata_Source"]` | Batch/source labels in the supplied preset |
| `adata.obs["Metadata_JCP2022"]` | Biological perturbation labels required by the current evaluation entry |
| `adata.obsm["X_pca"]` | PCA coordinates needed by methods such as Harmony and Scanorama |
| `adata.obs["Metadata_Cluster"]` | Clustering labels required for evaluation |


### Key arguments

| Argument | Meaning |
|---|---|
| `--data-path` | Input H5AD file; the preset filename is `MAD_adata.h5ad` |
| `--save-path` | Output directory for corrected H5AD files |
| `--batch-key` | Batch column in `obs`; default `Metadata_Source` |

### Example

```bash
bash run.sh 01_batch_method.py \
  --data-path /absolute/path/batch_inputs/MAD_adata.h5ad \
  --save-path /absolute/path/batch_results/python_methods \
  --batch-key Metadata_Source
```

### Methods and outputs

The entry runs all seven adapters on separate copies of the input:

| Method | Output filename | Representation used by the evaluation preset |
|---|---|---|
| Sphering | `Sphering_adata.h5ad` | `X` |
| BBKNN | `Bbknn_adata.h5ad` | `obsm["X_umap"]` |
| ComBat | `Combat_adata.h5ad` | `X` |
| Harmony | `Harmony_adata.h5ad` | `obsm["X_pca_harmony"]` |
| Scanorama | `Scanorama_adata.h5ad` | `obsm["X_scanorama"]` |
| MNN | `Mnn_adata.h5ad` | `X` |

BBKNN corrects the neighbor graph; its original `X` and PCA are not a corrected feature matrix. The current evaluation preset uses UMAP derived from the BBKNN graph for embedding-based metrics. This evaluates its UMAP representation and should be reported as such when comparing it with higher-dimensional representations.


## Step 2 — Run R Methods and Prepare Their H5AD Files

### Configure the notebook

Open [02_batch_method_R_clean.ipynb](02_batch_method_R_clean.ipynb) with an R kernel. In its configuration cell, replace the historical server paths with your experiment paths:

```r
input_file <- "/absolute/path/batch_inputs/MAD_adata.parquet"
output_dir <- "/absolute/path/batch_results/r_methods"
batch_col <- "Metadata_Source"
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
```

Set `input_file` to the `MAD_adata.parquet` saved in Step 0, matching the H5AD used in Step 1. Columns beginning with `Metadata_` are metadata; all other columns are features. Run the notebook cells in order, including both helper functions before the Seurat sections.

### Methods and outputs

| Method | Processing | Output filename |
|---|---|---|
| FastMNN | `batchelor::fastMNN`; saves corrected reduced dimensions | `Fast_MNN.parquet` |
| Seurat CCA | Build per-batch objects, find CCA anchors, integrate | `Seurat_cca_adata.parquet` |
| Seurat RPCA | Build per-batch objects, find RPCA anchors, integrate | `Seurat_rpca_adata.parquet` |

The Seurat helper replaces underscores in feature names with hyphens for processing and replaces them back on export. It removes the first three Seurat metadata columns when writing results. Check the exported metadata and sample IDs, especially if your input feature names already contain hyphens or the Seurat metadata structure differs.

### Convert results for evaluation

`03_calculate_metrics.py` searches only for `.h5ad` files. It will not read the Parquet outputs directly. Before Step 3:

1. Read each corrected Parquet table and separate its metadata from numerical corrected features.
2. Recover and validate unique sample IDs. Check the correspondence between corrected rows and metadata, including any row reordering during integration.
3. Build an AnnData object with corrected numerical features in `X` and aligned metadata in `obs`.
4. Compute neighbors and clustering from the corrected representation, saving the cluster column as `Metadata_Cluster`.
5. Save with method prefixes matching the evaluation mapping, for example:

```text
batch_results/r_h5ad/
├── Fast_adata.h5ad
├── Seuratcca_adata.h5ad
└── Seuratrpca_adata.h5ad
```

These conversion steps are required preparation; they are not executed by the supplied notebook or evaluation entry. In particular, `Seurat_cca_adata.h5ad` would be parsed as method `Seurat`, which is absent from the default mapping. Use `Seuratcca_adata.h5ad` or provide an appropriate mapping.

## Step 3 — Calculate Batch and Biological Metrics

### Key arguments

| Argument | Meaning |
|---|---|
| `--data-path` | Directory recursively searched for H5AD files |
| `--save-path` | Directory for per-method metric dictionaries |
| `--method-embeddings-json` | Optional JSON file replacing the entire method-to-representation mapping |

### Current evaluation settings

These settings are fixed inside `03_calculate_metrics.py`, rather than exposed as CLI arguments:

| Setting | Current value |
|---|---|
| Batch column | `Metadata_Source` |
| Biological label column | `Metadata_JCP2022` |
| Cluster column | `Metadata_Cluster` |
| mAP perturbation column | `Metadata_JCP2022` |
| Negative-control label | `JCP2022_033924` |
| Negative sampling | `random` |
| mAP helper defaults | Seed 42, negative multiplier 2, 10 parallel workers |

For another dataset, update these settings in the evaluation script before running. Passing a different `--batch-key` to Step 1 does not change Step 3. The negative-control label must be present, and both mAP calculations need valid repeated perturbation groups.

### Example

With files collected in `/absolute/path/batch_results/evaluation_h5ad`:

```bash
bash run.sh 03_calculate_metrics.py \
  --data-path /absolute/path/batch_results/evaluation_h5ad \
  --save-path /absolute/path/batch_results/metrics
```

For a custom representation mapping, append:

```text
--method-embeddings-json /absolute/path/batch_results/method_embeddings.json
```

### Output metrics

| Saved field | Meaning |
|---|---|
| `graph_connectivity` | scIB graph connectivity within biological labels |
| `silhouette_batch` | Batch silhouette score returned by scIB |
| `leiden_NMI`, `leiden_ARI` | Agreement between clustering and biological labels |
| `silhouette_label` | Biological-label silhouette score |
| `kBET` | kBET score returned by scIB |
| `LISI_label`, `LISI_batch` | Scaled direct-representation cLISI and iLISI scores from the local implementation |
| `with_control_mAP_mean` | Mean sister-activity mAP over valid perturbation groups |
| `with_control_mAP_p_value_mean` | Mean mAP among groups with corrected p-value below 0.05; this is not a mean p-value |
| `with_control_hit_gene_number` | Number of those significant activity groups |
| `with_other_mAP_mean` | Mean sister-consistency mAP over valid perturbation groups |
| `with_other_mAP_p_value_mean` | Mean mAP among significant consistency groups |
| `with_other_hit_gene_number` | Number of those significant consistency groups |

The scIB fields are retained when returned by the installed scIB implementation. The mAP significance values come from helper calls performed separately for each perturbation; they do not provide one multiple-testing correction over the full experiment. See the [mAP tutorial](../04_mAP/README.md) for the retrieval definitions.

```text
batch_results/metrics/
├── MAD_scib_metric_result.pkl
├── DeSCORT_scib_metric_result.pkl
├── Bbknn_scib_metric_result.pkl
├── Combat_scib_metric_result.pkl
├── Harmony_scib_metric_result.pkl
└── ...
```

Each pickle contains one Python dictionary of summary metrics. The entry does not save the full per-perturbation mAP tables. It catches failures per file and continues, so check the error messages and output files before interpreting the final completion message.
