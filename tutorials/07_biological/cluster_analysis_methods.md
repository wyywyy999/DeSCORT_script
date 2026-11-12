# Methods for cluster-based biological interpretation

Source: `03_de_cluster.py/.sh`, `04_cluster_enrichment.py/.sh`, and their imported `descort.biological` modules. Drafted on 2026-11-11. This document describes existing code; it does not run or modify the analysis.

## Core manuscript draft

### Gene-level profile construction and unsupervised clustering

DeSCORT embeddings and preprocessed Cell Painting measurements were aggregated into gene-level profiles by first averaging observations within each gene–guide group and then taking the feature-wise median across guides targeting the same gene. Nontargeting controls were excluded. Analyses were restricted to hit genes present in both representations with finite values across all retained features. For the historical A549 DMEM analysis, this yielded 1,087 genes represented by 256 DeSCORT embedding dimensions and 1,357 retained original morphology features. The embeddings were generated using the A549-fine-tuned, approximately one-epoch layer-normalized DeSCORT checkpoint, as documented in the separate provenance audit.

Principal component analysis (PCA) was fitted separately to each representation, retaining at least 95% of the variance. PCA used centered input profiles without additional feature standardization; the resulting scores were not row-normalized before clustering. Genes were partitioned using Euclidean k-means in DeSCORT PCA space, with a fixed number of ten clusters and 20 initializations per run. Clustering was repeated with five consecutive random seeds beginning at 20260926. Stability was quantified using pairwise adjusted Rand indices (ARIs), and separation was summarized by cosine silhouette scores. The run with the highest mean ARI relative to the other runs was retained as the representative partition. Original-feature PCA profiles were independently clustered into the same number of clusters for comparison. UMAP was used only for visualization of DeSCORT PCA profiles, with cosine distance, 10 neighbors, minimum distance 0.05, and random seed 42; UMAP coordinates were not used as clustering inputs.

### Differential morphological features

To characterize DeSCORT-derived clusters, each original morphology feature was compared between gene-level profiles within a cluster and those in all remaining clusters using a two-sided pooled-variance Student's t-test. The unit of analysis was the aggregated gene profile. P values were adjusted across features separately within each cluster using the Benjamini–Hochberg procedure. Cluster-associated differential features used for composition summaries were required to satisfy adjusted P < 0.05, absolute Cohen's d >= 0.5, and an absolute mean difference >= 0.2 in the input feature scale. Cohen's d was calculated as the within-cluster minus remaining-gene mean difference divided by the pooled standard deviation. Positive and negative effects were summarized separately.

### Pathway and protein-complex over-representation analysis

Biological annotations were obtained from local GO Biological Process, Cellular Component and Molecular Function, KEGG, and CORUM membership tables. GO and KEGG memberships were read from the fixed local pathway snapshot, while CORUM memberships were restricted to human complexes. All memberships were intersected with the analyzed gene universe; GO/KEGG terms required at least five analyzed members and CORUM complexes at least two. For each cluster and annotation term, over-representation was assessed using a one-sided Fisher's exact test against the full analyzed gene universe. The historical A549 analysis therefore used the 1,087 retained hit genes, rather than the whole genome, as the background. Tests were performed separately for DeSCORT and original-feature clustering. Benjamini–Hochberg correction was applied jointly across all cluster–term tests within each annotation database and representation, with adjusted P < 0.05 used to define significant enrichment.

### Enrichment visualization: use the paragraph matching the actual figure

**Raw-P mode, current shell default:** For the selected clusters, up to ten annotation terms with nominal P < 0.05 were displayed in ascending order of nominal P, with ties ordered by descending GeneRatio, database and term name. GeneRatio was defined as the number of overlapping genes divided by cluster size. Bubble area represented overlap count and color represented -log10(nominal P). The existing database-wise BH-adjusted results were retained for significance tables; visualization did not recalculate multiple-testing correction. Nominally selected plotted terms were not necessarily BH-significant.

**Adjusted-P mode, only if `--p-value-mode adjusted` was used:** Up to ten terms with database-wise BH-adjusted P < 0.05 were displayed in ascending adjusted P order, with ties ordered by descending GeneRatio, database and term name. GeneRatio was defined as overlap count divided by cluster size; bubble area represented overlap count and color represented -log10(adjusted P). No multiple-testing correction was recalculated during plotting.

The two paragraphs are alternatives, not two descriptions to combine for the same figure. Historical raw figures are in `figures/`; adjusted figures are in `figures_p.adjust/`. The shell selects clusters 2, 3 and 4. These are selected illustrative clusters, not automatically the three most enriched clusters.

## Add only when the corresponding results are shown

| Analysis shown in paper | Methods details required |
|---|---|
| Compartment/channel composition or purity | Features categorized from names into Cells/Cytoplasm/Nuclei and channel categories; up/down considered separately; category proportions, dominant proportion as purity, normalized entropy, and Jensen–Shannon distance from all retained feature categories as applicable |
| Differential heatmap | The heatmap helper selects FDR-significant features, takes the three largest absolute Cohen's d values per cluster, unions candidates, and retains at most 45 features by their maximum absolute d; heatmap values are Cohen's d. This selection does not additionally require the 0.5/0.2 thresholds used for composition summaries |
| Differential dotplots | Per-cluster up/down candidates satisfy all three differential-feature thresholds; default five per direction per cluster. Describe dot encoding using the actual plotted output |
| Biological-pair validation | Within-resource gene pairs defined by CORUM co-membership, STRING score >=400, or shared GO/KEGG membership; PCA-space cosine similarities; AUROC against sampled non-resource pairs; top-20 neighbor enrichment relative to pair prevalence where reported; 200 gene-label permutations for similarity support |
| Biological co-clustering | Fraction of resource-supported pairs assigned to the same cluster; compare with 200 permutations of cluster labels using (1 + number of null statistics >= observed)/(B+1) |
| Within-cluster STRING support | Internal STRING >=400 edge density compared with 500 size-matched random gene sets, using the same finite-permutation P-value correction. The 500 here is hard-coded and is not the shell's 200 `--cluster-permutations` setting |
| Cluster biological-unit distance comparison | Define units as cluster–term overlaps with at least six genes, plus within-cluster STRING connected components meeting that size; compare mean pairwise cosine distances of the same members in the two representations; one-sided paired Wilcoxon signed-rank tests, alternative Original > DeSCORT, separately by resource and overall |
| Selected case studies | Minimum six overlapping genes; candidates must have higher mean cosine similarity in DeSCORT than original features; candidates ranked by DeSCORT mean cosine similarity, at most one per cluster, retaining up to ten; 10,000 random-pair reference resamplings where shown. Because selection uses the compared representations, cases are illustrative, not independently selected validation examples |
| Manually specified pathways | Report the explicit pathway list, fixed gene universe, representation comparison, and 10,000 reference resamplings where shown; do not describe these pathways as independently discovered |
| Gene-neighbor query | Cosine similarity on PCA scores, self-neighbors excluded, top 50 exported and top 20 evaluated; specify queried genes (default PSMA1) and supporting annotation criteria |

## Statistical distinctions that must remain explicit

1. The workflow has **differential morphology**, not RNA differential expression. The t-test compares gene-profile groups between a cluster and its complement, not perturbed cells against nontargeting controls.
2. Enrichment uses **cluster-member perturbation genes** as the foreground. Differential morphology features are not converted into a gene list for enrichment.
3. Enrichment is **over-representation analysis (ORA)**, not ranked-list GSEA. GO terms, KEGG pathways and CORUM complexes are distinct annotation categories.
4. The t-test uses pooled variance (Student's t-test), not Welch's t-test or Wilcoxon rank-sum.
5. Differential-feature BH correction is **within cluster across features**. Enrichment BH correction is **within database across all clusters and terms, separately per representation**.
6. The preset fixes k=10 (`K_CANDIDATES=[N_CLUSTERS]`). Computing silhouette and ARI does not constitute a search over multiple cluster numbers.
7. Clustering uses raw PCA scores and Euclidean k-means; cosine is used for silhouette, neighbors, resource similarities, and UMAP. Do not describe this as cosine k-means or clustering in UMAP space.
8. `04_cluster_enrichment` reads the enrichment table produced by Step 03. It does not re-cluster, re-test enrichment, or recalculate BH.
9. A raw-P bubble plot is not evidence that every displayed term passed FDR <0.05. In raw mode, the BH-significant export can contain fewer terms than the plot.

## Provenance and missing manuscript inputs

- The historical A549 parameter/result files are under `DeSCORT/tutorials/07_biological_new/result/03_de_cluster/hit_10_cluster`; the packaged `other_model/DeSCORT/tutorials/07_biological/result` preset directory has no existing results at the time of this audit. Do not substitute current defaults for run-specific metadata without checking.
- Historical A549 PCA retained 82 DeSCORT components and 294 original-feature components. These are dataset-specific outputs, not fixed method parameters.
- The local GO/KEGG membership snapshot has 948 GO/KEGG terms; the memberships were previously checked to match the pathway run using `GO_Biological_Process_2025`, `GO_Cellular_Component_2025`, `GO_Molecular_Function_2025`, and `KEGG_2021_Human`. It contains hit-restricted memberships rather than a newly downloaded whole-genome annotation universe.
- CORUM and STRING release numbers remain unverified from the available local tables. Use the frozen-file checksums in `figure3_provenance_audit.json`; do not invent release labels. STRING400 denotes a score threshold, not a STRING release.
- The manuscript should identify which clusters/terms were displayed and whether the final figures use raw or adjusted P. Both historical modes exist, so final panel selection cannot be inferred solely from the wrapper default.
- Describe how the hit list was originally defined elsewhere in Methods; this clustering code consumes an existing list and does not discover the hits itself.
- If claiming comparative statistical evidence from features used to derive the learned representations, describe these tests as post-clustering characterization rather than independent replication.

## Evidence map

| Draft claim | Implementation |
|---|---|
| Guide means, gene median | `src/descort/biological/utils.py:aggregate` |
| Shared hit universe and finite-value checks | `de_cluster.py:load_profiles` |
| Separate PCA; fixed k; stability-based representative seed | `fit_pca95`, `configure`, `select_embedding_clustering`, and tutorial Step 03 |
| Student t-test and three characterization thresholds | `differential_features`, `feature_composition` |
| Fisher ORA and database-wide BH | `load_biological_resources`, `enrich_gene_sets` |
| Raw/adjusted plot conventions | `cluster_enrichment.py:prepare_enrichment_plot_data`, `plot_cluster_enrichment_bubble`, and tutorial Step 04 |

Terminology: gene-level profile; differential morphological feature; over-representation analysis (ORA); Benjamini–Hochberg-adjusted P; Cohen's d; adjusted Rand index (ARI). Method descriptions follow the supplied code, without external bibliographic or software-version claims.
