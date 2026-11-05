# A549 DMEM — Complete Output File Guide

This guide documents the existing A549 DMEM results in `DeSCORT/tutorials/07_biological_new/result/`. It is based on the actual saved files and current analysis code, rather than a newly executed experiment.

## Final directory structure

The historical results use these directories:

```text
result/
├── 01_umap_analysis/
│   └── cpg0021_A549_DMEM/
│       ├── all.log
│       ├── all_parameters.json
│       ├── gene_sets_used.json
│       ├── library_cache/
│       ├── model/
│       │   ├── results/
│       │   └── figures/{ANNOTATED,GO_BP,GO_CC,GO_MF,KEGG}/
│       ├── original/
│       │   ├── results/
│       │   └── figures/{ANNOTATED,GO_BP,GO_CC,GO_MF,KEGG}/
│       ├── neighbor_queries/{results,figures,qa}/
│       └── comparison/
├── 02_CORUM_STRING/
│   └── cpg0021_A549_DMEM/
│       ├── results/
│       │   ├── model/
│       │   └── original/
│       ├── figures/
│       ├── qa/
│       └── <stage>.log, <stage>_parameters.json
├── 03_de_cluster/
│   └── hit_10_cluster/
│       ├── all.log
│       ├── all_parameters.json
│       ├── results/
│       │   └── profiles/{model,original}/
│       ├── figures/
│       └── qa/
└── 04_cluster_enrichment/
    ├── analysis.log, parameters.json
    ├── analysis_p.adjust.log, parameters_p.adjust.json
    ├── results/, figures/, qa/
    └── results_p.adjust/, figures_p.adjust/, qa_p.adjust/
```

`model` means DeSCORT and `original` means normalized morphological features. Some historical tables label DeSCORT as `LayerNorm`. Step 3's A549 hit-list run is called `hit_10_cluster`, not `cpg0021_A549_DMEM`. Step 4's parameter files point to that run and select clusters 2, 3, and 4. Other sibling directories, including HeLa and all-gene 50/100-cluster experiments, are outside this example.

The [README commands](README.md) choose new output paths. Their correspondence to the historical folders is:

| Step | Historical folder below `result/` | README example output below `/absolute/path/biological/A549_DMEM/` |
|---|---|---|
| 1 | `01_umap_analysis/cpg0021_A549_DMEM/` | `01_umap/` |
| 2 | `02_CORUM_STRING/cpg0021_A549_DMEM/` | `02_CORUM_STRING/` |
| 3 | `03_de_cluster/hit_10_cluster/` | `03_de_cluster/` |
| 4 | `04_cluster_enrichment/` | `04_cluster_enrichment/` |

## How to read the inventory

CSV files are tables; Parquet files are binary tables or matrices readable with `pandas.read_parquet()`. JSON stores parameters, annotations, summaries, or audit records. PNG/PDF/SVG versions of the same figure contain the same scientific panel in different formats. Label-coordinate CSVs describe display labels, not additional observations.

The inventory groups repeated pathway images and QA files by filename pattern. Every existing file in the four selected result folders is covered by a row below. File counts and schemas can change with input data, annotations, stage, queries, and figure settings.

The historical directory contains accumulated calculation, redraw, and external figure-review outputs. `check`, `query`, and redraw stages produce additional stage-specific files. Offline `--gene-sets-json` runs may not create `library_cache/`. A single `--stage all` run does not automatically reproduce every historical review, font audit, collision audit, or manually added README. The inventories below explicitly distinguish these records.

## Step 1 — Pathway UMAP and gene profiles

All paths in this section are relative to `01_umap_analysis/cpg0021_A549_DMEM/`.

### Run root

| File or pattern | Contents |
|---|---|
| `all.log` | Stage execution log, containing processing messages, paths, and any warnings/errors. |
| `all_parameters.json` | Resolved command-line parameters for this calculation or redraw stage, including actual input/output paths. |
| `gene_sets_used.json` | Full database → pathway → member-gene annotation mapping used by Step 1; reusable as an offline input. |

### `comparison/`

| File or pattern | Contents |
|---|---|
| `all_strict_pathways.csv` | Paired model/original pathway distances after the strict gene-set comparison rules, including distance changes and model-better flags. |
| `by_database_summary.csv` | Per-database paired-pathway summary, model-better fraction, and Wilcoxon comparison. |
| `common_pathways_gene_set_check.csv` | Common pathway pairs with member lists and same_genes flags; inspect this before interpreting paired distances. |
| `comparison_summary.json` | Comparison scope, excluded databases, strict/top/manual subset counts, ranking settings, and source result directories. |
| `manual_selected_pathways.csv` | Paired comparison rows for the manually specified database/pathway list. |
| `selected_pathway_model_vs_original_by_database.alignment.json` | Panel-layout alignment audit: measured geometry, tolerances, findings, and verdict; not a biological score. |
| `top_strict_pathways.csv` | Ranked subset of strict pathway pairs selected by the run's top-fraction/ranking parameters. |
| `manual_selected_boxplot.{pdf,png}` | Paired model/original comparison for the manual pathway list. |
| `selected_pathway_model_vs_original_boxplot.{pdf,png}` | Paired distance comparison for the selected strict-pathway subset. |
| `selected_pathway_model_vs_original_by_database.{pdf,png}` | Paired pathway-distance comparisons split by annotation database. |

### `library_cache/`

| File or pattern | Contents |
|---|---|
| `GO_Biological_Process_2025.json` | Cached Enrichr library mapping pathway names to member genes; supports annotation reuse without refetching. |
| `GO_Cellular_Component_2025.json` | Cached Enrichr library mapping pathway names to member genes; supports annotation reuse without refetching. |
| `GO_Molecular_Function_2025.json` | Cached Enrichr library mapping pathway names to member genes; supports annotation reuse without refetching. |
| `KEGG_2021_Human.json` | Cached Enrichr library mapping pathway names to member genes; supports annotation reuse without refetching. |

### `model/figures/`

| File or pattern | Contents |
|---|---|
| `01_GO_KEGG_distance_distributions.alignment.json` | Panel-layout alignment audit: measured geometry, tolerances, findings, and verdict; not a biological score. |
| `selected_plotted_pathways_combined_UMAP_direct_labels.labels.csv` | Display-label placement records for the combined pathway UMAP. |
| `selected_plotted_pathways_combined_UMAP_selection.csv` | Pathways selected for the combined direct-label UMAP panel. |
| `00_global_hit_gene_UMAP.{pdf,png}` | Global selected-gene UMAP in the corresponding representation. |
| `01_GO_KEGG_distance_distributions.{pdf,png}` | Within-pathway distance distributions grouped by annotation database. |
| `selected_plotted_pathways_combined_UMAP_direct_labels.{pdf,png}` | Combined pathway overlay on the gene UMAP with direct labels. |

### `model/figures/ANNOTATED/`

| File pattern | Contents |
|---|---|
| `<rank>_<pathway>.{png,pdf,svg}` | Individual pathway overlay on the global gene UMAP; member genes are highlighted. The rank/name and number of exported formats depend on the run. This historical directory contains 100 image files. |

### `model/figures/GO_BP/`

| File pattern | Contents |
|---|---|
| `<rank>_<pathway>.{png,pdf,svg}` | Individual pathway overlay on the global gene UMAP; member genes are highlighted. The rank/name and number of exported formats depend on the run. This historical directory contains 350 image files. |

### `model/figures/GO_CC/`

| File pattern | Contents |
|---|---|
| `<rank>_<pathway>.{png,pdf,svg}` | Individual pathway overlay on the global gene UMAP; member genes are highlighted. The rank/name and number of exported formats depend on the run. This historical directory contains 72 image files. |

### `model/figures/GO_MF/`

| File pattern | Contents |
|---|---|
| `<rank>_<pathway>.{png,pdf,svg}` | Individual pathway overlay on the global gene UMAP; member genes are highlighted. The rank/name and number of exported formats depend on the run. This historical directory contains 62 image files. |

### `model/figures/KEGG/`

| File pattern | Contents |
|---|---|
| `<rank>_<pathway>.{png,pdf,svg}` | Individual pathway overlay on the global gene UMAP; member genes are highlighted. The rank/name and number of exported formats depend on the run. This historical directory contains 88 image files. |

### `model/results/`

| File or pattern | Contents |
|---|---|
| `GO_KEGG_distance_thresholds.csv` | Per-database distance threshold and percentile used to select compact GO/KEGG pathways. |
| `all_gene_profiles.parquet` | Aggregated gene-level profiles before the final hit/valid-gene subset; index identifies genes and numerical columns hold the representation features. |
| `excluded_requested_features.csv` | Requested morphology features not used, for example because they are unavailable in the input schema. |
| `features_used.csv` | Names of the numerical features actually retained from the input. |
| `global_hit_gene_UMAP.csv` | Gene identifiers and their global UMAP1/UMAP2 display coordinates. |
| `hit_gene_PCA.csv` | Gene identifiers and PCA component scores used for pathway-distance analysis. |
| `hit_gene_profiles.parquet` | Selected gene-level profile matrix used in this analysis, after the relevant hit selection and validity checks. |
| `missing_hit_genes.csv` | Requested hit genes unavailable in the corresponding gene profiles. |
| `pathway_distance_df_model.csv` | DeSCORT pathway-level member genes, gene/pair counts, distance summary statistics, and selection threshold. |
| `pathway_gene_UMAP_coordinates.csv` | Database, pathway, gene, and display coordinates for pathway-associated genes, with pathway size and distance/rank information. |
| `plotted_pathway_gene_dict.json` | Database-to-pathway mapping of the gene members retained for plotting. |
| `rendered_pathways.csv` | Pathway rows actually selected for individual figures; includes distance statistics and rank. |
| `run_manifest.json` | Input references, effective gene/feature QC, PCA dimensions/explained variance, pathway counts, and run settings. |
| `selected_pathway_df_model.csv` | DeSCORT pathways retained after pathway-size and compactness selection, with members and distance ranks. |
| `valid_pathway_summary.csv` | Per-database valid-pathway counts and median pathway size/distance. |

### `neighbor_queries/figures/`

| File or pattern | Contents |
|---|---|
| `figure7_query_example_ATP6V0C.{pdf,png}` | Nearest-neighbor query figure for the named gene, comparing representations and biological support. |
| `figure7_query_example_PSMA1.{pdf,png}` | Nearest-neighbor query figure for the named gene, comparing representations and biological support. |

### `neighbor_queries/qa/`

| File or pattern | Contents |
|---|---|
| `figure7_query_example_ATP6V0C.alignment.json` | Panel-layout alignment audit: measured geometry, tolerances, findings, and verdict; not a biological score. |
| `figure7_query_example_ATP6V0C.alignment.svg` | Visual overlay of the panel-layout audit for checking alignment; not a new scientific figure. |
| `figure7_query_example_PSMA1.alignment.json` | Panel-layout alignment audit: measured geometry, tolerances, findings, and verdict; not a biological score. |
| `figure7_query_example_PSMA1.alignment.svg` | Visual overlay of the panel-layout audit for checking alignment; not a new scientific figure. |

### `neighbor_queries/results/`

| File or pattern | Contents |
|---|---|
| `figure7_manifest.json` | Neighbor-query run provenance, query genes, selected profiles, representation settings, and output records. |
| `gene_neighbor_rank.csv` | Query/neighbor gene pairs, rank, similarities, clusters, and CORUM/STRING/GO/KEGG support annotations. |
| `gene_neighbor_rank.parquet` | Parquet version of the annotated neighbor-ranking table. |
| `model_gene_PCA.parquet` | DeSCORT gene-level PCA coordinates used for nearest-neighbor analysis. |
| `original_gene_PCA.parquet` | Original-morphology gene-level PCA coordinates used for nearest-neighbor analysis. |
| `query_ATP6V0C.csv` | Annotated neighbor table for the gene named in the filename, with ranks, similarities, cluster identities, and database support. |
| `query_PSMA1.csv` | Annotated neighbor table for the gene named in the filename, with ranks, similarities, cluster identities, and database support. |
| `query_all_genes.csv` | Combined exported query-neighbor results; the actual query scope is recorded in the run settings and query columns, not inferred from this filename alone. |
| `query_missing_genes.csv` | Requested query genes unavailable for neighbor export. |

### `original/figures/`

| File or pattern | Contents |
|---|---|
| `01_GO_KEGG_distance_distributions.alignment.json` | Panel-layout alignment audit: measured geometry, tolerances, findings, and verdict; not a biological score. |
| `selected_plotted_pathways_combined_UMAP_direct_labels.labels.csv` | Display-label placement records for the combined pathway UMAP. |
| `selected_plotted_pathways_combined_UMAP_selection.csv` | Pathways selected for the combined direct-label UMAP panel. |
| `00_global_hit_gene_UMAP.{pdf,png}` | Global selected-gene UMAP in the corresponding representation. |
| `01_GO_KEGG_distance_distributions.{pdf,png}` | Within-pathway distance distributions grouped by annotation database. |
| `selected_plotted_pathways_combined_UMAP_direct_labels.{pdf,png}` | Combined pathway overlay on the gene UMAP with direct labels. |

### `original/figures/ANNOTATED/`

| File pattern | Contents |
|---|---|
| `<rank>_<pathway>.{png,pdf,svg}` | Individual pathway overlay on the global gene UMAP; member genes are highlighted. The rank/name and number of exported formats depend on the run. This historical directory contains 100 image files. |

### `original/figures/GO_BP/`

| File pattern | Contents |
|---|---|
| `<rank>_<pathway>.{png,pdf,svg}` | Individual pathway overlay on the global gene UMAP; member genes are highlighted. The rank/name and number of exported formats depend on the run. This historical directory contains 350 image files. |

### `original/figures/GO_CC/`

| File pattern | Contents |
|---|---|
| `<rank>_<pathway>.{png,pdf,svg}` | Individual pathway overlay on the global gene UMAP; member genes are highlighted. The rank/name and number of exported formats depend on the run. This historical directory contains 72 image files. |

### `original/figures/GO_MF/`

| File pattern | Contents |
|---|---|
| `<rank>_<pathway>.{png,pdf,svg}` | Individual pathway overlay on the global gene UMAP; member genes are highlighted. The rank/name and number of exported formats depend on the run. This historical directory contains 62 image files. |

### `original/figures/KEGG/`

| File pattern | Contents |
|---|---|
| `<rank>_<pathway>.{png,pdf,svg}` | Individual pathway overlay on the global gene UMAP; member genes are highlighted. The rank/name and number of exported formats depend on the run. This historical directory contains 88 image files. |

### `original/results/`

| File or pattern | Contents |
|---|---|
| `GO_KEGG_distance_thresholds.csv` | Per-database distance threshold and percentile used to select compact GO/KEGG pathways. |
| `all_gene_profiles.parquet` | Aggregated gene-level profiles before the final hit/valid-gene subset; index identifies genes and numerical columns hold the representation features. |
| `excluded_requested_features.csv` | Requested morphology features not used, for example because they are unavailable in the input schema. |
| `features_used.csv` | Names of the numerical features actually retained from the input. |
| `global_hit_gene_UMAP.csv` | Gene identifiers and their global UMAP1/UMAP2 display coordinates. |
| `hit_gene_PCA.csv` | Gene identifiers and PCA component scores used for pathway-distance analysis. |
| `hit_gene_profiles.parquet` | Selected gene-level profile matrix used in this analysis, after the relevant hit selection and validity checks. |
| `missing_hit_genes.csv` | Requested hit genes unavailable in the corresponding gene profiles. |
| `pathway_distance_df_original.csv` | Original-morphology counterpart of the pathway-distance table. |
| `pathway_gene_UMAP_coordinates.csv` | Database, pathway, gene, and display coordinates for pathway-associated genes, with pathway size and distance/rank information. |
| `plotted_pathway_gene_dict.json` | Database-to-pathway mapping of the gene members retained for plotting. |
| `rendered_pathways.csv` | Pathway rows actually selected for individual figures; includes distance statistics and rank. |
| `run_manifest.json` | Input references, effective gene/feature QC, PCA dimensions/explained variance, pathway counts, and run settings. |
| `selected_pathway_df_original.csv` | Original-morphology counterpart of the selected-pathway table. |
| `valid_pathway_summary.csv` | Per-database valid-pathway counts and median pathway size/distance. |

## Step 2 — CORUM/STRING validation

All paths in this section are relative to `02_CORUM_STRING/cpg0021_A549_DMEM/`.

### Run root

| File or pattern | Contents |
|---|---|
| `corum-plot.log` | Stage execution log, containing processing messages, paths, and any warnings/errors. |
| `corum-plot_parameters.json` | Resolved command-line parameters for this calculation or redraw stage, including actual input/output paths. |
| `string-plot.log` | Stage execution log, containing processing messages, paths, and any warnings/errors. |
| `string-plot_parameters.json` | Resolved command-line parameters for this calculation or redraw stage, including actual input/output paths. |

### `figures/`

| File or pattern | Contents |
|---|---|
| `CORUM_background_corrected_distribution.{pdf,png,svg}` | Distribution of background-corrected CORUM pair similarities. |
| `CORUM_paired_model_minus_original.{pdf,png}` | Paired changes in CORUM similarities between representations. |
| `PCA_explained_variance.{pdf,png}` | PCA explained-variance comparison for the two representations. |
| `STRING_confidence_background_corrected.{pdf,png}` | STRING confidence versus background-corrected similarity. |
| `STRING_paired_model_minus_original.{pdf,png}` | Paired changes in STRING pair similarities. |
| `STRING_similarity_quantiles_boxen.{pdf,png}` | Distribution of STRING scores by saved similarity bins. |
| `STRING_similarity_quantiles_line.{pdf,png,svg}` | STRING score summary curve over the saved similarity intervals. |

### `qa/`

| File or pattern | Contents |
|---|---|
| `CORUM_distribution.alignment.json` | Panel-layout alignment audit: measured geometry, tolerances, findings, and verdict; not a biological score. |
| `CORUM_distribution.collision-audit.json` | Historical figure-review record for text/object collisions; JSON/log records findings and PDF visualizes review output. External review artifact, not guaranteed from the four analysis commands. |
| `CORUM_distribution.review.md` | Historical explanatory notes or manual/automated figure-review notes; not automatically generated by the four analysis steps. |
| `CORUM_distribution.source-audit.json` | Historical figure-source/preflight inspection report or helper code used during review; not part of the core calculation outputs. |
| `CORUM_distribution.source-preflight.py` | Historical figure-source/preflight inspection report or helper code used during review; not part of the core calculation outputs. |
| `CORUM_distribution.text-audit.json` | Historical text/font inspection report or execution log. External review artifact, not an analysis metric. |
| `STRING_line.collision-audit.json` | Historical figure-review record for text/object collisions; JSON/log records findings and PDF visualizes review output. External review artifact, not guaranteed from the four analysis commands. |
| `STRING_line.review.md` | Historical explanatory notes or manual/automated figure-review notes; not automatically generated by the four analysis steps. |
| `STRING_line.source-audit.json` | Historical figure-source/preflight inspection report or helper code used during review; not part of the core calculation outputs. |
| `STRING_line.source-preflight.py` | Historical figure-source/preflight inspection report or helper code used during review; not part of the core calculation outputs. |
| `STRING_line.text-audit.json` | Historical text/font inspection report or execution log. External review artifact, not an analysis metric. |

### `results/`

| File or pattern | Contents |
|---|---|
| `CORUM_background_corrected.csv` | Per-gene-pair and per-representation raw cosine similarity, background mean, and background-corrected similarity. |
| `CORUM_background_pairs.csv` | Gene pairs used as the CORUM non-associated background. |
| `CORUM_paired_model_minus_original.csv` | Matched CORUM pairs with DeSCORT-minus-original raw and corrected similarities. |
| `CORUM_positive_pairs.csv` | Gene pairs classified as CORUM positives under the retained complex definitions. |
| `STRING_background_corrected.csv` | STRING pair identities, confidence scores/intervals, representation, raw similarity, background, corrected similarity, and similarity-bin assignments. |
| `STRING_edges.csv` | Retained STRING interaction pairs, combined scores, and confidence intervals. |
| `STRING_paired_change_by_model_quantile.csv` | Binned counts and mean/median paired similarity changes for STRING pairs. |
| `STRING_paired_model_minus_original.csv` | Matched STRING pairs with DeSCORT-minus-original similarities, STRING score, and bin assignment. |
| `STRING_similarity_intervals_summary.csv` | Fixed similarity-interval boundaries, counts, means, medians, and standard errors by representation, used for the line plot. |
| `STRING_similarity_quantiles_summary.csv` | Saved similarity-bin summary with counts, means, medians, and standard errors by representation; interpret its bins using the saved run settings. |
| `common_hit_genes.csv` | Historical common-selected-gene list; retained alongside common_selected_genes.csv for compatibility. |
| `common_selected_genes.csv` | Final shared valid gene list used for the representation comparison. |
| `input_qc.json` | Input records, common/requested genes, finite-profile checks, selection mode, and order of PCA fitting versus selection. |
| `metrics.csv` | Representation-level metrics including PCA dimensions/variance, CORUM enrichment/AUROC, and STRING association statistics. |
| `missing_requested_hits.csv` | Requested hit genes absent from one or both input representations. |
| `nonfinite_hits_removed.csv` | Genes removed because at least one representation contains nonfinite profile values. |
| `pair_universe.json` | Retained genes/complexes and counts/definitions of CORUM and STRING positive/background pair pools. |
| `paired_change_summary.csv` | Aggregate differences between DeSCORT and original similarities and retrieval/association metrics. |
| `validation.json` | Saved numerical checks of background calculations, verification subset size, background-pair counts, and zero-norm rows. |

### `results/model/`

| File or pattern | Contents |
|---|---|
| `all_gene_profiles.parquet` | Aggregated gene-level profiles before the final hit/valid-gene subset; index identifies genes and numerical columns hold the representation features. |
| `excluded_requested_features.csv` | Requested morphology features not used, for example because they are unavailable in the input schema. |
| `features_used.csv` | Names of the numerical features actually retained from the input. |
| `hit_PCA_L2_profiles.parquet` | PCA scores for the selected shared genes after row L2 normalization; used for pairwise cosine calculations. |
| `hit_gene_profiles.parquet` | Selected gene-level profile matrix used in this analysis, after the relevant hit selection and validity checks. |

### `results/original/`

| File or pattern | Contents |
|---|---|
| `all_gene_profiles.parquet` | Aggregated gene-level profiles before the final hit/valid-gene subset; index identifies genes and numerical columns hold the representation features. |
| `excluded_requested_features.csv` | Requested morphology features not used, for example because they are unavailable in the input schema. |
| `features_used.csv` | Names of the numerical features actually retained from the input. |
| `hit_PCA_L2_profiles.parquet` | PCA scores for the selected shared genes after row L2 normalization; used for pairwise cosine calculations. |
| `hit_gene_profiles.parquet` | Selected gene-level profile matrix used in this analysis, after the relevant hit selection and validity checks. |

## Step 3 — A549 hit-list clustering (10 clusters)

All paths in this section are relative to `03_de_cluster/hit_10_cluster/`.

### Run root

| File or pattern | Contents |
|---|---|
| `all.log` | Stage execution log, containing processing messages, paths, and any warnings/errors. |
| `all_parameters.json` | Resolved command-line parameters for this calculation or redraw stage, including actual input/output paths. |

### `figures/`

| File or pattern | Contents |
|---|---|
| `figure6b_case_umap.labels.csv` | Display-label positions and identifiers exported for this figure. |
| `figure8b_specified_pathways_umap.labels.csv` | Display-label positions and identifiers exported for this figure. |
| `figure1_cluster_sizes.{pdf,png,svg}` | Number of genes assigned to each cluster. |
| `figure1_unsupervised_clusters.{pdf,png,svg}` | UMAP of gene representations colored by unsupervised clusters. |
| `figure3_differential_feature_heatmap_clustered.{pdf,png,svg}` | Clustered heatmap of selected differential morphology effect sizes. |
| `figure3b_differential_feature_dotplot_down.{pdf,png,svg}` | Dotplot of selected negative-direction morphology effects across clusters. |
| `figure3b_differential_feature_dotplot_up.{pdf,png,svg}` | Dotplot of selected positive-direction morphology effects across clusters. |
| `figure4_down_channel_mix.{pdf,png,svg}` | Channel composition of significant negative-direction features. |
| `figure4_feature_composition_purity.{pdf,png,svg}` | Compartment/channel composition and purity of significant features. |
| `figure4_up_channel_mix.{pdf,png,svg}` | Channel composition of significant positive-direction features. |
| `figure5_cluster_pathway_validation.{pdf,png,svg}` | Cluster-level biological/pathway validation panels. |
| `figure6a_case_comparison.{pdf,png,svg}` | Selected biological case similarities compared with original morphology and random references. |
| `figure6b_case_umap.{pdf,png,svg}` | Selected biological case members highlighted on the gene UMAP. |
| `figure6c_cluster_pathway_distance_by_database.{pdf,png,svg}` | Biological-unit model/original distances split by database group. |
| `figure6d_cluster_pathway_distance_overall.{pdf,png,svg}` | Overall paired biological-unit distance comparison. |
| `figure8a_specified_pathway_comparison.{pdf,png,svg}` | Model/original and random-reference similarities for the specified pathways. |
| `figure8b_specified_pathways_umap.{pdf,png,svg}` | Specified pathway members highlighted on the gene UMAP. |

### `qa/`

| File or pattern | Contents |
|---|---|
| `figure1_cluster_sizes.alignment.json` | Panel-layout alignment audit: measured geometry, tolerances, findings, and verdict; not a biological score. |
| `figure1_cluster_sizes.alignment.svg` | Visual overlay of the panel-layout audit for checking alignment; not a new scientific figure. |
| `figure1_unsupervised_clusters.alignment.json` | Panel-layout alignment audit: measured geometry, tolerances, findings, and verdict; not a biological score. |
| `figure1_unsupervised_clusters.alignment.svg` | Visual overlay of the panel-layout audit for checking alignment; not a new scientific figure. |
| `figure3_differential_feature_heatmap_clustered.alignment.json` | Panel-layout alignment audit: measured geometry, tolerances, findings, and verdict; not a biological score. |
| `figure3_differential_feature_heatmap_clustered.alignment.svg` | Visual overlay of the panel-layout audit for checking alignment; not a new scientific figure. |
| `figure3b_differential_feature_dotplot_down.alignment.json` | Panel-layout alignment audit: measured geometry, tolerances, findings, and verdict; not a biological score. |
| `figure3b_differential_feature_dotplot_down.alignment.svg` | Visual overlay of the panel-layout audit for checking alignment; not a new scientific figure. |
| `figure3b_differential_feature_dotplot_up.alignment.json` | Panel-layout alignment audit: measured geometry, tolerances, findings, and verdict; not a biological score. |
| `figure3b_differential_feature_dotplot_up.alignment.svg` | Visual overlay of the panel-layout audit for checking alignment; not a new scientific figure. |
| `figure4_down_channel_mix.alignment.json` | Panel-layout alignment audit: measured geometry, tolerances, findings, and verdict; not a biological score. |
| `figure4_down_channel_mix.alignment.svg` | Visual overlay of the panel-layout audit for checking alignment; not a new scientific figure. |
| `figure4_feature_composition_purity.alignment.json` | Panel-layout alignment audit: measured geometry, tolerances, findings, and verdict; not a biological score. |
| `figure4_feature_composition_purity.alignment.svg` | Visual overlay of the panel-layout audit for checking alignment; not a new scientific figure. |
| `figure4_up_channel_mix.alignment.json` | Panel-layout alignment audit: measured geometry, tolerances, findings, and verdict; not a biological score. |
| `figure4_up_channel_mix.alignment.svg` | Visual overlay of the panel-layout audit for checking alignment; not a new scientific figure. |
| `figure5_cluster_pathway_validation.alignment.json` | Panel-layout alignment audit: measured geometry, tolerances, findings, and verdict; not a biological score. |
| `figure5_cluster_pathway_validation.alignment.svg` | Visual overlay of the panel-layout audit for checking alignment; not a new scientific figure. |
| `figure6a_case_comparison.alignment.json` | Panel-layout alignment audit: measured geometry, tolerances, findings, and verdict; not a biological score. |
| `figure6a_case_comparison.alignment.svg` | Visual overlay of the panel-layout audit for checking alignment; not a new scientific figure. |
| `figure6b_case_umap.alignment.json` | Panel-layout alignment audit: measured geometry, tolerances, findings, and verdict; not a biological score. |
| `figure6b_case_umap.alignment.svg` | Visual overlay of the panel-layout audit for checking alignment; not a new scientific figure. |
| `figure6c_cluster_pathway_distance_by_database.alignment.json` | Panel-layout alignment audit: measured geometry, tolerances, findings, and verdict; not a biological score. |
| `figure6c_cluster_pathway_distance_by_database.alignment.svg` | Visual overlay of the panel-layout audit for checking alignment; not a new scientific figure. |
| `figure6d_cluster_pathway_distance_overall.alignment.json` | Panel-layout alignment audit: measured geometry, tolerances, findings, and verdict; not a biological score. |
| `figure6d_cluster_pathway_distance_overall.alignment.svg` | Visual overlay of the panel-layout audit for checking alignment; not a new scientific figure. |
| `figure8a_specified_pathway_comparison.alignment.json` | Panel-layout alignment audit: measured geometry, tolerances, findings, and verdict; not a biological score. |
| `figure8a_specified_pathway_comparison.alignment.svg` | Visual overlay of the panel-layout audit for checking alignment; not a new scientific figure. |
| `figure8b_specified_pathways_umap.alignment.json` | Panel-layout alignment audit: measured geometry, tolerances, findings, and verdict; not a biological score. |
| `figure8b_specified_pathways_umap.alignment.svg` | Visual overlay of the panel-layout audit for checking alignment; not a new scientific figure. |

### `results/`

| File or pattern | Contents |
|---|---|
| `analysis_summary.json` | Analysis summary. Step 3 combines data provenance, clustering, differential-feature and biological-support counts; Step 4 records requested clusters and enrichment selection/plotting counts. |
| `biological_co_clustering_support.csv` | Biological edge co-clustering rates, random expectations, fold enrichment, and permutation results for both representations. |
| `biological_pair_support.csv` | Biological positive-pair cosine summaries, all-pair backgrounds, AUROC, and top-neighbor support comparisons. |
| `biological_resource_summary.json` | Counts of retained CORUM complexes, STRING400 edges, pathways, derived pairs, and annotation-snapshot notes. |
| `case_selection_config.json` | Rules used to select biological case studies, including ranking, minimum overlap, databases, and one-case-per-cluster setting. |
| `cluster_STRING400_support.csv` | Within-cluster STRING400 edge counts/density and size-matched random-reference enrichment/p-values. |
| `cluster_biological_distance_comparison.csv` | Per-cluster biological units and their model/original within-unit distances, member genes, and model-better flags. |
| `cluster_biological_distance_summary.csv` | Per-database-group aggregation of biological-unit distances, improvement fractions, and one-sided Wilcoxon tests. |
| `cluster_case_candidates.csv` | Candidate cluster/biological-term cases, overlaps, similarity-based ranking, and resource enrichment statistics. |
| `cluster_case_gene_pairs.csv` | Gene pairs within selected case studies and their DeSCORT/original cosine similarities. |
| `cluster_case_studies.csv` | Selected case studies with member genes, overlaps, observed/random similarities, empirical comparisons, and selection ranks. |
| `cluster_differential_features.csv` | Cluster-versus-rest original-morphology tests: group means, mean difference, Cohen's d, raw P, and within-cluster BH-adjusted P. |
| `cluster_differential_features.parquet` | Parquet version of the differential-feature test table. |
| `cluster_feature_composition.csv` | Composition of significant up/down features by compartment/channel, compared with the feature background. |
| `cluster_feature_purity.csv` | Dominant feature category, purity, normalized entropy, and divergence from background for each cluster/direction/category type. |
| `cluster_pathway_complex_enrichment.csv` | Enrichment results for model/original clusters: database, term, cluster size, overlap genes/count, odds ratio, raw P, and database-wise BH-adjusted P. Input to Step 4. |
| `cluster_top_biological_terms.csv` | Leading enrichment terms for each cluster/database and representation. |
| `data_representation_summary.csv` | Tabular record of aggregation, selected genes/features, PCA dimensions/variance, and representation settings. |
| `differential_dotplot_feature_selection.csv` | Features selected for up/down dotplots, their source clusters/ranks, and source effect sizes/significance. |
| `differential_dotplot_values.csv` | Selected-feature effects and significance across target clusters, plus plotting order and labels. |
| `differential_feature_heatmap_matrix.csv` | Cluster-by-feature Cohen's d matrix for the selected differential morphology heatmap. |
| `differential_feature_heatmap_matrix_clustered.csv` | The same cluster-by-feature heatmap matrix reordered according to the displayed clustered heatmap. |
| `embedding_cluster_selection_extended.csv` | Candidate k values and multi-seed silhouette/stability/size statistics, ranking, and selected-k flag. |
| `embedding_cluster_umap.csv` | Gene UMAP coordinates and DeSCORT cluster assignments used for the cluster map. |
| `figure8_selected_pathway_comparison.csv` | Specified pathway member sets, model/original and random-reference similarities, empirical P values, and selection ranks. |
| `gene_cluster_assignment.csv` | Gene identifiers with DeSCORT embedding_cluster and original-feature original_cluster assignments. |
| `gene_neighbor_rank.csv` | Query/neighbor gene pairs, rank, similarities, clusters, and CORUM/STRING/GO/KEGG support annotations. |
| `gene_neighbor_rank.parquet` | Parquet version of the annotated neighbor-ranking table. |
| `gene_top20_database_support.csv` | Per-gene, per-representation top-neighbor support counts/fractions for each biological database. |
| `gene_top20_database_support.parquet` | Parquet version of the per-gene top-neighbor support summary. |
| `input_provenance.json` | Source input records, plate/feature selection, aggregation, final common genes, missing/nonfinite exclusions, and selection mode. |
| `query_PSMA1.csv` | Annotated neighbor table for the gene named in the filename, with ranks, similarities, cluster identities, and database support. |
| `query_all_genes.csv` | Combined exported query-neighbor results; the actual query scope is recorded in the run settings and query columns, not inferred from this filename alone. |
| `query_example_PSMA1.csv` | Annotated neighbor table for the gene named in the filename, with ranks, similarities, cluster identities, and database support. |
| `query_missing_genes.csv` | Requested query genes unavailable for neighbor export. |
| `selected_cluster_compactness.csv` | Selected k, within-cluster similarities, all-pair backgrounds, cross-space compactness changes, and silhouette scores. |
| `selected_genes.csv` | Final selected common finite gene identifiers used for clustering and comparison. |
| `significant_differential_features.csv` | Differential features passing the configured significance/effect-size criteria, with direction and compartment/channel annotation. |

### `results/profiles/`

| File or pattern | Contents |
|---|---|
| `model_gene_profiles.parquet` | Aggregated DeSCORT gene profile matrix before final shared-hit selection. |
| `model_hit_gene_profiles.parquet` | Final selected shared finite DeSCORT gene profile matrix. |
| `original_gene_profiles.parquet` | Aggregated original-morphology gene profile matrix before final shared-hit selection. |
| `original_hit_gene_profiles.parquet` | Final selected shared finite original-morphology gene profile matrix. |

### `results/profiles/model/`

| File or pattern | Contents |
|---|---|
| `excluded_requested_features.csv` | Requested morphology features not used, for example because they are unavailable in the input schema. |
| `features_used.csv` | Names of the numerical features actually retained from the input. |

### `results/profiles/original/`

| File or pattern | Contents |
|---|---|
| `excluded_requested_features.csv` | Requested morphology features not used, for example because they are unavailable in the input schema. |
| `features_used.csv` | Names of the numerical features actually retained from the input. |

## Step 4 — Cluster enrichment (raw and adjusted)

All paths in this section are relative to `04_cluster_enrichment/`.

### Run root

| File or pattern | Contents |
|---|---|
| `analysis.log` | Stage execution log, containing processing messages, paths, and any warnings/errors. |
| `analysis_p.adjust.log` | Stage execution log, containing processing messages, paths, and any warnings/errors. |
| `parameters.json` | Resolved command-line parameters for this calculation or redraw stage, including actual input/output paths. |
| `parameters_p.adjust.json` | Resolved command-line parameters for this calculation or redraw stage, including actual input/output paths. |

### `figures/`

| File or pattern | Contents |
|---|---|
| `figure_cluster2_enrichment_bubble_data.csv` | Terms eligible under this plotting mode; includes GeneRatio, Count, raw/adjusted P, transformed color values, and plotted flag. |
| `figure_cluster3_enrichment_bubble_data.csv` | Terms eligible under this plotting mode; includes GeneRatio, Count, raw/adjusted P, transformed color values, and plotted flag. |
| `figure_cluster4_enrichment_bubble_data.csv` | Terms eligible under this plotting mode; includes GeneRatio, Count, raw/adjusted P, transformed color values, and plotted flag. |
| `figure_cluster2_enrichment_bubble.{pdf,png,svg}` | Enrichment bubble plot for this cluster. X = GeneRatio; area = Count; color = −log10(raw P) or −log10(adjusted P), according to the directory/mode. Adjusted-mode no-hit runs can export a no-significant-enrichment panel. |
| `figure_cluster3_enrichment_bubble.{pdf,png,svg}` | Enrichment bubble plot for this cluster. X = GeneRatio; area = Count; color = −log10(raw P) or −log10(adjusted P), according to the directory/mode. Adjusted-mode no-hit runs can export a no-significant-enrichment panel. |
| `figure_cluster4_enrichment_bubble.{pdf,png,svg}` | Enrichment bubble plot for this cluster. X = GeneRatio; area = Count; color = −log10(raw P) or −log10(adjusted P), according to the directory/mode. Adjusted-mode no-hit runs can export a no-significant-enrichment panel. |

### `figures_p.adjust/`

| File or pattern | Contents |
|---|---|
| `README.md` | Historical explanatory notes or manual/automated figure-review notes; not automatically generated by the four analysis steps. |
| `figure_cluster2_enrichment_bubble_data.csv` | Terms eligible under this plotting mode; includes GeneRatio, Count, raw/adjusted P, transformed color values, and plotted flag. |
| `figure_cluster3_enrichment_bubble_data.csv` | Terms eligible under this plotting mode; includes GeneRatio, Count, raw/adjusted P, transformed color values, and plotted flag. |
| `figure_cluster4_enrichment_bubble_data.csv` | Terms eligible under this plotting mode; includes GeneRatio, Count, raw/adjusted P, transformed color values, and plotted flag. |
| `figure_cluster2_enrichment_bubble.{pdf,png,svg}` | Enrichment bubble plot for this cluster. X = GeneRatio; area = Count; color = −log10(raw P) or −log10(adjusted P), according to the directory/mode. Adjusted-mode no-hit runs can export a no-significant-enrichment panel. |
| `figure_cluster3_enrichment_bubble.{pdf,png,svg}` | Enrichment bubble plot for this cluster. X = GeneRatio; area = Count; color = −log10(raw P) or −log10(adjusted P), according to the directory/mode. Adjusted-mode no-hit runs can export a no-significant-enrichment panel. |
| `figure_cluster4_enrichment_bubble.{pdf,png,svg}` | Enrichment bubble plot for this cluster. X = GeneRatio; area = Count; color = −log10(raw P) or −log10(adjusted P), according to the directory/mode. Adjusted-mode no-hit runs can export a no-significant-enrichment panel. |

### `qa/`

| File or pattern | Contents |
|---|---|
| `bubble_blue_red_review.md` | Historical explanatory notes or manual/automated figure-review notes; not automatically generated by the four analysis steps. |
| `figure_cluster2_enrichment_bubble.alignment.json` | Panel-layout alignment audit: measured geometry, tolerances, findings, and verdict; not a biological score. |
| `figure_cluster2_enrichment_bubble.alignment.svg` | Visual overlay of the panel-layout audit for checking alignment; not a new scientific figure. |
| `figure_cluster3_enrichment_bubble.alignment.json` | Panel-layout alignment audit: measured geometry, tolerances, findings, and verdict; not a biological score. |
| `figure_cluster3_enrichment_bubble.alignment.svg` | Visual overlay of the panel-layout audit for checking alignment; not a new scientific figure. |
| `figure_cluster4_enrichment_bubble.alignment.json` | Panel-layout alignment audit: measured geometry, tolerances, findings, and verdict; not a biological score. |
| `figure_cluster4_enrichment_bubble.alignment.svg` | Visual overlay of the panel-layout audit for checking alignment; not a new scientific figure. |

### `qa/figure_cluster2_enrichment_bubble/`

| File or pattern | Contents |
|---|---|
| `collision.log` | Historical figure-review record for text/object collisions; JSON/log records findings and PDF visualizes review output. External review artifact, not guaranteed from the four analysis commands. |
| `collisions.json` | Historical figure-review record for text/object collisions; JSON/log records findings and PDF visualizes review output. External review artifact, not guaranteed from the four analysis commands. |
| `text.log` | Historical text/font inspection report or execution log. External review artifact, not an analysis metric. |

### `qa/figure_cluster3_enrichment_bubble/`

| File or pattern | Contents |
|---|---|
| `collision.log` | Historical figure-review record for text/object collisions; JSON/log records findings and PDF visualizes review output. External review artifact, not guaranteed from the four analysis commands. |
| `collisions.json` | Historical figure-review record for text/object collisions; JSON/log records findings and PDF visualizes review output. External review artifact, not guaranteed from the four analysis commands. |
| `text.log` | Historical text/font inspection report or execution log. External review artifact, not an analysis metric. |

### `qa/figure_cluster4_enrichment_bubble/`

| File or pattern | Contents |
|---|---|
| `collision.log` | Historical figure-review record for text/object collisions; JSON/log records findings and PDF visualizes review output. External review artifact, not guaranteed from the four analysis commands. |
| `collisions.json` | Historical figure-review record for text/object collisions; JSON/log records findings and PDF visualizes review output. External review artifact, not guaranteed from the four analysis commands. |
| `collisions.pdf` | Historical figure-review record for text/object collisions; JSON/log records findings and PDF visualizes review output. External review artifact, not guaranteed from the four analysis commands. |
| `text.log` | Historical text/font inspection report or execution log. External review artifact, not an analysis metric. |

### `qa_p.adjust/`

| File or pattern | Contents |
|---|---|
| `QA_NOTES.md` | Historical explanatory notes or manual/automated figure-review notes; not automatically generated by the four analysis steps. |
| `figure_cluster2_enrichment_bubble.alignment.json` | Panel-layout alignment audit: measured geometry, tolerances, findings, and verdict; not a biological score. |
| `figure_cluster2_enrichment_bubble.collision-audit.json` | Historical figure-review record for text/object collisions; JSON/log records findings and PDF visualizes review output. External review artifact, not guaranteed from the four analysis commands. |
| `figure_cluster2_enrichment_bubble.font-audit.json` | Historical text/font inspection report or execution log. External review artifact, not an analysis metric. |
| `figure_cluster3_enrichment_bubble.alignment.json` | Panel-layout alignment audit: measured geometry, tolerances, findings, and verdict; not a biological score. |
| `figure_cluster3_enrichment_bubble.alignment.svg` | Visual overlay of the panel-layout audit for checking alignment; not a new scientific figure. |
| `figure_cluster3_enrichment_bubble.collision-audit.json` | Historical figure-review record for text/object collisions; JSON/log records findings and PDF visualizes review output. External review artifact, not guaranteed from the four analysis commands. |
| `figure_cluster3_enrichment_bubble.font-audit.json` | Historical text/font inspection report or execution log. External review artifact, not an analysis metric. |
| `figure_cluster4_enrichment_bubble.alignment.json` | Panel-layout alignment audit: measured geometry, tolerances, findings, and verdict; not a biological score. |
| `figure_cluster4_enrichment_bubble.alignment.svg` | Visual overlay of the panel-layout audit for checking alignment; not a new scientific figure. |
| `figure_cluster4_enrichment_bubble.collision-audit.json` | Historical figure-review record for text/object collisions; JSON/log records findings and PDF visualizes review output. External review artifact, not guaranteed from the four analysis commands. |
| `figure_cluster4_enrichment_bubble.font-audit.json` | Historical text/font inspection report or execution log. External review artifact, not an analysis metric. |
| `source_preflight.json` | Historical figure-source/preflight inspection report or helper code used during review; not part of the core calculation outputs. |

### `results/`

| File or pattern | Contents |
|---|---|
| `analysis_summary.json` | Analysis summary. Step 3 combines data provenance, clustering, differential-feature and biological-support counts; Step 4 records requested clusters and enrichment selection/plotting counts. |
| `cluster2_DeSCORT_enrichment_significant.csv` | Existing database-wise BH-significant terms for this DeSCORT cluster, selected using --fdr; includes overlap/effect/significance columns. |
| `cluster2_analysis_summary.json` | Per-cluster selected/displayed counts, FDR scope, plotting mode/ranking, and source enrichment-file record. |
| `cluster3_DeSCORT_enrichment_significant.csv` | Existing database-wise BH-significant terms for this DeSCORT cluster, selected using --fdr; includes overlap/effect/significance columns. |
| `cluster3_analysis_summary.json` | Per-cluster selected/displayed counts, FDR scope, plotting mode/ranking, and source enrichment-file record. |
| `cluster4_DeSCORT_enrichment_significant.csv` | Existing database-wise BH-significant terms for this DeSCORT cluster, selected using --fdr; includes overlap/effect/significance columns. |
| `cluster4_analysis_summary.json` | Per-cluster selected/displayed counts, FDR scope, plotting mode/ranking, and source enrichment-file record. |

### `results_p.adjust/`

| File or pattern | Contents |
|---|---|
| `analysis_summary.json` | Analysis summary. Step 3 combines data provenance, clustering, differential-feature and biological-support counts; Step 4 records requested clusters and enrichment selection/plotting counts. |
| `cluster2_DeSCORT_enrichment_significant.csv` | Existing database-wise BH-significant terms for this DeSCORT cluster, selected using --fdr; includes overlap/effect/significance columns. |
| `cluster2_analysis_summary.json` | Per-cluster selected/displayed counts, FDR scope, plotting mode/ranking, and source enrichment-file record. |
| `cluster3_DeSCORT_enrichment_significant.csv` | Existing database-wise BH-significant terms for this DeSCORT cluster, selected using --fdr; includes overlap/effect/significance columns. |
| `cluster3_analysis_summary.json` | Per-cluster selected/displayed counts, FDR scope, plotting mode/ranking, and source enrichment-file record. |
| `cluster4_DeSCORT_enrichment_significant.csv` | Existing database-wise BH-significant terms for this DeSCORT cluster, selected using --fdr; includes overlap/effect/significance columns. |
| `cluster4_analysis_summary.json` | Per-cluster selected/displayed counts, FDR scope, plotting mode/ranking, and source enrichment-file record. |

## Which outputs to inspect first

1. **Input and scope:** `run_manifest.json`, `input_qc.json`, `input_provenance.json`, and parameter files establish which genes, features, plates, and annotations were actually analyzed.
2. **Pathway compactness:** Step 1's `comparison/all_strict_pathways.csv` and `by_database_summary.csv` compare matching pathway member sets.
3. **Known interactions:** Step 2's `results/metrics.csv` and corrected pair tables show CORUM/STRING support.
4. **Cluster interpretation:** Step 3's assignments, differential-feature tests, and enrichment table connect genes, morphology, and biological annotations.
5. **Presentation:** Inspect `figures/` together with its plotting-data CSVs. For Step 4, check whether the plot came from raw or adjusted mode before interpreting significance.

These files describe the historical saved experiment. They do not establish that a new run with different inputs or annotation versions will have the same counts, cluster IDs, filenames for individual pathways, or numerical results.
