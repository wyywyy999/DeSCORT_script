# 08 — Compound, CRISPR, and ORF Cross-Modality Analysis

This tutorial prepares three perturbation types for DeSCORT fine-tuning, extracts embeddings, compares them with original morphology, and retrieves representative microscopy fields.

## Workflow

```text
Normalized morphology + label/feature vocabularies
                       │
                 01_process_data
                       │
               02_combined_dataset
                       │
                  03_finetune
                       │
                   04_run_emb
                       │
           ┌───────────┴────────────┐
        05_umap                06_compare
  Profiles, cosine, UMAP    Pooled-profile comparison
                           + built-in temporal MRR/Hit
                                   │
                              Candidate queries
                                   │
                              07_get_image
                       Five-channel microscopy fields
```

UMAP is a visualization, not the space used for retrieval scores. The comparison workflow calculates cosine distances in the configured representation. Image queries are explicitly chosen in the image launcher's arrays; Step 7 does not automatically select candidates from Step 6.

## Setup and Current Examples

Follow the [installation guide](../../docs/installation.md), activate the environment, and enter this directory:

```bash
conda activate DeSCORT
cd /fse/home/wangyongyan/Project/CellPainting/DeSCORT/other_model/DeSCORT/tutorials/08_three_type
```

Configure experiment paths and settings in each `.sh` file. Additional command-line arguments can override the options supplied by the launcher. Resource paths point to local datasets and checkpoints; these resources are not bundled with the package.

| Scripts | Current preset |
|---|---|
| 01–04 | U2OS, 300 epochs; the active preprocessing/extraction input is Orf |
| 05–06 | A549, 150 epochs; Original and DeSCORT representations |
| 07 | Six selected A549 compound/CRISPR queries in batch `2020_11_04_CPJUMP1` |

These are separate example configurations. For one end-to-end experiment, align the cell line, checkpoint, perturbation directories, epoch labels, metadata manifests, and output paths across the launchers. An epoch argument identifies existing model outputs; it does not generate a new model version.

Steps 3–4 use GPU resources. Steps 5–7 operate on existing profiles or images without loading a DeSCORT checkpoint. Analysis code is implemented in [`descort.multimodality`](../../src/descort/multimodality); tutorial Python entries retain the main workflow and import its helpers.

## Steps 01–04 — Prepare Data, Fine-Tune, and Extract Embeddings

These steps follow the existing preprocessing and training workflows; use their dedicated guides for detailed arguments and model configuration.

| Entry | Purpose | Related guide |
|---|---|---|
| [01_process_data.sh](01_process_data.sh) | Convert each perturbation type into indexed training datasets | [Data preparation](../01_preprocess_data/README.md), [training-data processing](../02_pretrain/README.md) |
| [02_combined_dataset.sh](02_combined_dataset.sh) | Concatenate processed Compound/Crispr/Orf datasets | [Pretraining workflow](../02_pretrain/README.md) |
| [03_finetune.sh](03_finetune.sh) | Fine-tune an existing checkpoint on the combined dataset | [Fine-tuning](../03_finetune/README.md) |
| [04_run_emb.sh](04_run_emb.sh) | Extract metadata-preserving embedding Parquets | [Embedding extraction](../03_finetune/README.md#step-3--extract-fine-tuned-cls-embeddings) |

Prepare each intended modality separately before combining datasets. The active Step 1 launcher processes Orf only; the merger does not enforce that all three perturbation types exist. For downstream analysis, organize extracted embeddings as:

```text
embedding_root/
├── Compound/<batch>/<plate>.parquet
├── Crispr/<batch>/<plate>.parquet
└── Orf/<batch>/<plate>.parquet
```

Preserve `Metadata_Treatment`, compound names/IDs, CRISPR guide/sample IDs, and plate/well metadata. Check the GPU selection, external DeepSpeed configuration path, checkpoint, and output directories in Steps 3–4 before running.

## Step 05 — UMAP and Cross-Modality Similarity

[05_umap.py](05_umap.py) uses helpers in [multimodality/umap.py](../../src/descort/multimodality/umap.py). It supports model-only, Original-only, and paired analysis.

### Inputs and aggregation

| Input | Requirements |
|---|---|
| Model input | Compound/Crispr/Orf subdirectories containing consistent numerical `emb_N` columns |
| Original input | Three independently configured modality directories with normalized morphology Parquets |
| Labels | `Metadata_Treatment` |
| Compound metadata | `Metadata_Pert_IName` |
| CRISPR metadata | `Metadata_Target_Sequence` and `Metadata_Broad_Sample` |
| Original feature list | Candidate feature pickle; only numerical non-metadata features common to all input files are retained |

All selected plates, batches, and times are pooled within each modality. Compound profiles are well means per Treatment/name. CRISPR profiles are well means per Treatment/guide followed by a median across guide means. ORF profiles are well means per Treatment.

PCA is fitted independently for model and Original profiles. Cosine distances and rankings are calculated in PCA or native feature space, before UMAP. Independently fitted UMAP coordinates cannot be compared as a shared coordinate system.

### Configuration and commands

| Setting | Default |
|---|---|
| `input_mode` / `UMAP_INPUT_MODE` | `both`; alternatives: `model`, `original` |
| `representation`, `pca_variance` | `pca95`, 0.95; `native` skips PCA |
| `top_n` | 20 highlighted targets |
| `n_neighbors`, `min_dist`, `umap_metric` | 15, 0.5, `cosine` |
| `seed`, `threads` | 0, 4 |
| `figure_formats`, `dpi` | `png pdf`, 450; also supports `svg` and `tiff` |
| `output_dir` | `06_umap/outputs/A549_150epochs/` |

```bash
bash 05_umap.sh --dry-run
bash 05_umap.sh
UMAP_INPUT_MODE=model bash 05_umap.sh
UMAP_INPUT_MODE=original bash 05_umap.sh
bash 05_umap.sh --figure-formats png pdf svg tiff
```

The dry-run checks schemas and metadata; full numerical validation occurs during analysis. The script records excluded features and repaired CRISPR DMSO annotations. It does not silently impute invalid measurement values.

### Outputs

| Relative to the output root | Contents |
|---|---|
| `{model,original}/results/perturbation_profiles.parquet` | Aggregated profiles with input counts |
| `{model,original}/results/source_file_manifest.csv` | Source files and metadata-repair counts |
| `{model,original}/results/selected_feature_columns.csv` | Features actually used |
| `{model,original}/results/gene_pairwise_similarity_pca95.csv` | Per-gene similarities and rankings |
| `{model,original}/results/umap_coordinates_pca95.csv` | Profile identities and UMAP coordinates |
| `{model,original}/figures/` | Five figure families per input, in the selected formats |
| `comparison/` | Profile/pair membership audits and paired similarity tables |
| `run_config.json` | Actual run settings |

The five figures cover closest Compound–CRISPR targets in UMAP, pairwise distances, Compound–ORF similarity ranking, average pairwise similarity distributions, and the Compound–ORF-ranked UMAP. The mean-three-pair-ranked UMAP is no longer generated. Figure-check JSON/log files use temporary storage; analysis summaries and configuration JSON files are retained outside the figure directory.

## Step 06 — Compare Representations and Calculate MRR / Hit

[06_compare.py](06_compare.py) uses [multimodality/comparison.py](../../src/descort/multimodality/comparison.py) and [multimodality/temporal_retrieval.py](../../src/descort/multimodality/temporal_retrieval.py). Temporal retrieval is built into the package; no external `05_MRR_hit.py` is required.

### Stages and inputs

| Stage | Input and analysis |
|---|---|
| `comparison` | Model embedding Parquets or an existing model profile bundle, plus an aggregated Original bundle; compares distances, complete mismatched-target backgrounds, retrieval, rankings, and cosine signs |
| `temporal` | Original/model `selected_plates.csv` manifests, embedding root, feature list/vocabulary, and reference target sets; computes drug→gene ranks separately by time combination |

`model_profiles_dir` can point to Step 5's `model/` output and takes precedence over raw model aggregation. `original_profiles_dir` accepts Step 5's `original/` output or a compatible legacy bundle containing `results/perturbation_profiles.parquet` and `selected_feature_columns.csv`. The default Original bundle is the existing `old_version_orignal` analysis.

Temporal manifests come from the corresponding mAP result roots. The reference seed's `ap_scores_dict.pickle` supplies target gene sets, not the AP scores used to calculate MRR. Original and model well identities are checked before retrieval.

### Configuration and commands

| Setting | Default |
|---|---|
| `stages` | `comparison temporal` |
| `cell_line`, `model_epochs` | A549, 150 |
| `pca_variance`, `top_k` | 0.95, `20 50` |
| `bootstraps`, `seed`, `threads` | 5000, 20261012, 4 |
| `target_modalities` | `crispr orf` |
| `batch_scope`, `temporal_representation`, `reference_seed` | `primary`, `pca95`, 42 |
| `figure_formats`, `dpi` | `png pdf`, 600; also supports `svg` and `tiff` |
| `output_dir` | `07_compare_results/A549_150epochs/` |

```bash
bash 06_compare.sh --dry-run
bash 06_compare.sh
bash 06_compare.sh --stages comparison
bash 06_compare.sh --stages temporal
bash 06_compare.sh --figure-formats png pdf svg tiff
```

Changing experiments requires updating both the profile and temporal inputs in the launcher. `--stages comparison` does not require running temporal retrieval, and vice versa. Only `status: complete` in `run_manifest.json` indicates that all selected stages completed successfully.

### Two retrieval definitions

| Property | Pooled-profile comparison | Time-stratified retrieval |
|---|---|---|
| Time handling | All input times pooled | Four time combinations per target modality |
| PCA | Each representation's full profile set | Query and candidate profiles jointly, separately for each method/time comparison |
| Query unit | Compound profile or CRISPR target | Compound Broad Sample ID |
| Directions | Compound→CRISPR, Compound→ORF, CRISPR→ORF | Compound→CRISPR and Compound→ORF |
| Weighting | Compound queries averaged within gene, then genes weighted equally | Drug IDs weighted equally within a time comparison |
| Tie rule | Average tied rank | Worst tied rank |
| Correct target | Matching gene | Best annotated target for a multi-target drug |
| Statistical unit | Paired non-control genes | Paired drug IDs after averaging their four time comparisons |

MRR averages reciprocal target ranks. Hit@K measures whether a correct target is retrieved within K, under the respective weighting rule. The temporal stage's legacy `Recall_at_K` columns represent Hit@K, not the fraction of all annotated targets recovered. The combined summary records `workflow`, `unit`, and `tie_rule`; do not average its two workflows into one MRR.

### Current figure numbering

| Number | File stem | Contents |
|---|---|---|
| 1 | `figure1_paired_gene_distance_distributions` | Matched gene-distance distributions |
| 2 | `figure2_paired_distance_scatter` | Paired Original/model distances |
| 3 | `figure3_true_target_retrieval_MRR` | Pooled-profile MRR |
| 4 | `figure4_true_target_retrieval_Hit` | Hit@1/5/10/20, with gene-bootstrap intervals |
| 5 | `figure5_top20_top50_overlap` | Top-k ranking agreement |
| 6 | `figure6_up_down_compound_crispr_venn` | Overlap of cosine-sign classes |

The other seven plotting routines have been removed; their statistical tables remain. Historical output folders may retain earlier exports or formats until explicitly cleaned.

### Saved results

| Relative to `07_compare_results/A549_150epochs/` | Contents |
|---|---|
| `REPORT_zh.md` | Integrated report |
| `retrieval_summary_all_workflows.csv` | Both retrieval definitions, explicitly identified |
| `run_manifest.json` | Parameters, stage status, and failure/completion information |
| `profile_comparison/REPORT_zh.md` | Pooled-profile report |
| `profile_comparison/results/profile_input_alignment.csv` | Profile membership and input-well count differences |
| `profile_comparison/results/paired_gene_distances.csv` | Per-gene paired distances |
| `profile_comparison/results/true_target_retrieval_summary.csv` | Three-direction MRR/Hit summaries |
| `profile_comparison/results/profile_Hit_plot_summary.csv` | Hit means and bootstrap intervals |
| `profile_comparison/results/up_down_classification.csv` | Cosine-sign assignments |
| `profile_comparison/figures/` | Six current figure families |
| `temporal_retrieval/` | Time-specific ranks, candidate evidence, summaries, and three retrieval figures |
| `{profile_comparison,temporal_retrieval}/qa/` | Alignment, PDF text, and collision checks |

Existing A549 pooled-profile results:

![Pooled-profile MRR](07_compare_results/A549_150epochs/profile_comparison/figures/figure3_true_target_retrieval_MRR.png)

![Pooled-profile Hit](07_compare_results/A549_150epochs/profile_comparison/figures/figure4_true_target_retrieval_Hit.png)

These are saved figures, not results newly calculated by reading this document.

## Step 07 — Retrieve Five-Channel Microscopy Fields

[07_get_image.py](07_get_image.py) uses [multimodality/get_image.py](../../src/descort/multimodality/get_image.py). It matches perturbation annotations to `load_data.csv`, resolves public S3 image URLs, downloads the five original TIFF channels, and renders color PNGs.

Configure the query arrays in [07_get_image.sh](07_get_image.sh):

```bash
compound_queries=(thiostrepton anisomycin CCG50014)
crispr_queries=(FOXM1 TUBB3 AGER)
orf_queries=()
```

Set an unused query type to `()`. Gene names match annotated Treatment targets; compound names match `Pert_IName`, ignoring case and punctuation. Compound targets do not imply direct gene editing. For example, `CCG50014` matches `CCG-50014`; DMSO can be requested as a compound query.

| Setting | Default / meaning |
|---|---|
| `annotation`, `index_dir` | Existing JUMP pilot annotation and well-index cache |
| `load_data_dir` | Root containing `<batch>/<plate>/load_data.csv` |
| `cell_line`, `batch` | A549, `2020_11_04_CPJUMP1` |
| `plate`, `well`, `timepoint` | Empty: no additional restriction |
| `site`, `max_fields` | 1, 1 per query |
| `all_sites`, `all_fields` | `false`; enable to expand the selected wells/sites |
| `low_percentile`, `high_percentile`, `dpi` | 1, 99.8, 600 |
| `output_dir` | `07_get_image_results/` |

```bash
# Inspect the selected fields before downloading TIFFs.
bash 07_get_image.sh --dry-run

# Download/render the configured queries.
bash 07_get_image.sh

# All wells and sites satisfying the current filters.
bash 07_get_image.sh --all-sites --all-fields
```

The dry-run writes a download plan without fetching image TIFFs or rendering. Missing image-location CSVs may still be downloaded. `--sync-metadata` refreshes location metadata; `--rebuild-index` rebuilds the well-index cache. The script reuses existing TIFFs and does not copy the large annotation table into the tutorial directory.

Each selected field is saved under:

```text
07_get_image_results/<query>_<type>/<cell_line>/<batch>/time_<hours>h/<plate>/<well>/site_<NN>/
├── raw/                         # Five original TIFF channels
├── DNA_color_<hours>h.png        # Also ER, RNA, AGP, Mito
├── merged_rgb_<hours>h.png
├── channels_and_merged_<hours>h.png
└── manifest.json
```

Query roots also contain `matching_wells.csv`, `download_plan.csv`, and, after rendering, `run_summary.json`.

Channels are DNA blue, ER green, RNA yellow, AGP red, and Mito magenta. Each field/channel is independently percentile-scaled before equal-weight additive RGB merging. Native channel/composite PNGs retain the original pixel dimensions; DPI metadata does not add microscope resolution. The 12×8-inch montage is rendered at the requested DPI. This is full-field visualization without single-cell cropping or illumination correction; independently stretched brightness is not suitable for direct quantitative comparison between samples.

Step 7 requires AWS CLI and Python packages pandas, NumPy, tifffile, Pillow, and matplotlib. Public image downloads use unsigned S3 requests. Edit the query arrays for single-query runs; trailing launcher arguments are intended for filters, output paths, and display settings.

## Interpretation and Reproducibility

Review input manifests and matched identities before comparing representations. The existing A549 Original data contain additional CRISPR DMSO wells, which can affect PCA even when controls are excluded from the final statistics.

Profile statistics use paired genes, separate BH test families, and observed-gene bootstrap intervals. Temporal statistics use paired drug IDs after averaging repeated time comparisons. Neither unit represents independent culture or training replicates. `up`/`down` denotes cosine distance below/above 1, not gene-expression regulation or target activation/inhibition. Ranking overlap measures agreement, not biological correctness.

These examples reuse data involved in fine-tuning. Treat candidates and statistical comparisons as exploratory; establish generalization or mechanisms with independent evidence. Saved manifests describe historical runs, whereas Bash configuration edits affect future runs only.
