#!/usr/bin/env python3
"""Run the de_cluster workflow using functions from descort.biological."""

# Support both an installed package and direct execution from this checkout.
from pathlib import Path as _Path
import sys as _sys
_SOURCE = _Path(__file__).resolve().parents[2] / "src"
if str(_SOURCE) not in _sys.path:
    _sys.path.insert(0, str(_SOURCE))

from pathlib import Path
import sys


import json
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from descort.biological import de_cluster as analysis
from descort.biological.de_cluster import (
    all_pair_mean,
    argument_parser,
    build_case_studies_all_resources,
    build_cluster_biological_distance_table,
    build_neighbor_table,
    build_selected_pathway_comparison,
    check_inputs,
    co_cluster_support,
    configure,
    differential_features,
    differential_heatmap_matrix,
    enrich_gene_sets,
    evaluate_pair_support,
    feature_composition,
    fit_pca95,
    load_biological_resources,
    load_profiles,
    plot_case_comparison,
    plot_case_umap,
    plot_cluster_biological_distance_by_database,
    plot_cluster_biological_distance_overall,
    plot_cluster_map,
    plot_differential_dotplots,
    plot_differential_heatmap,
    plot_feature_purity,
    plot_pathway_validation,
    plot_selected_pathway_comparison,
    plot_selected_pathway_umap,
    query_gene_neighbors,
    select_embedding_clustering,
    string_cluster_support,
    summarize_cluster_biological_distances,
    top20_database_support_table,
    topk_cosine,
    within_cluster_similarity,
)


def main() -> None:
    print('1/10 Loading and aggregating profiles ...', flush=True)
    model_df, original_df, provenance = load_profiles()
    genes = model_df.index.to_numpy(dtype=str)
    n = len(genes)
    model_raw = model_df.to_numpy(dtype=np.float32)
    original_raw = original_df.to_numpy(dtype=np.float32)

    print('2/10 Fitting independent PCA95 representations ...', flush=True)
    model_scores, model_pca = fit_pca95(model_df)
    original_scores, original_pca = fit_pca95(original_df)
    # Keep raw PCA scores as the representation; cosine normalization is
    # applied only inside pairwise cosine calculations.
    model_z = model_scores
    original_z = original_scores
    provenance.update({
        'model_pca95_components': int(model_pca.n_components_),
        'model_pca95_explained_variance': float(model_pca.explained_variance_ratio_.sum()),
        'original_pca95_components': int(original_pca.n_components_),
        'original_pca95_explained_variance': float(original_pca.explained_variance_ratio_.sum()),
        'pca_scaling': 'none; input morphology was already MAD-normalized',
        'pca_row_normalization_for_similarity_and_clustering': 'none; raw PCA scores',
    })
    pd.DataFrame([provenance]).to_csv(analysis.RESULTS / 'data_representation_summary.csv', index=False)

    print('3/10 Computing exact nearest neighbours ...', flush=True)
    model_nn, model_nn_sim = topk_cosine(model_z, analysis.TOP_K_EXPORT)
    original_nn, original_nn_sim = topk_cosine(original_z, analysis.TOP_K_EXPORT)

    print('4/10 Loading CORUM, STRING, GO and KEGG ...', flush=True)
    pair_sets, records, resource_summary = load_biological_resources(genes)
    (analysis.RESULTS / 'biological_resource_summary.json').write_text(
        json.dumps(resource_summary, ensure_ascii=False, indent=2), encoding='utf-8'
    )

    print('5/10 Comparing representations against biological pairs and random labels ...', flush=True)
    pair_support = evaluate_pair_support(
        model_z, original_z, pair_sets, model_nn, original_nn
    )
    pair_support.to_csv(analysis.RESULTS / 'biological_pair_support.csv', index=False)

    print('6/10 Selecting embedding-only unsupervised clustering ...', flush=True)
    model_labels, cluster_selection, selected_k = select_embedding_clustering(model_z)
    original_labels = KMeans(
        n_clusters=selected_k, random_state=analysis.SEED, n_init=analysis.ARGS.kmeans_n_init
    ).fit_predict(original_z)
    cluster_selection.to_csv(analysis.RESULTS / 'embedding_cluster_selection_extended.csv', index=False)
    cluster_assignment = pd.DataFrame({
        'gene': genes,
        'embedding_cluster': model_labels,
        'original_cluster': original_labels,
    })
    cluster_assignment.to_csv(analysis.RESULTS / 'gene_cluster_assignment.csv', index=False)

    model_co = co_cluster_support(model_labels, pair_sets, n, 'DeSCORT')
    original_co = co_cluster_support(original_labels, pair_sets, n, 'Original')
    co_cluster = pd.concat([model_co, original_co], ignore_index=True)
    co_cluster.to_csv(analysis.RESULTS / 'biological_co_clustering_support.csv', index=False)

    model_within, within_pairs = within_cluster_similarity(model_z, model_labels)
    original_cross_within, _ = within_cluster_similarity(original_z, model_labels)
    compactness = pd.DataFrame([{
        'selected_k': selected_k,
        'within_cluster_pairs': within_pairs,
        'model_within_cosine': model_within,
        'model_all_pair_cosine': all_pair_mean(model_z),
        'model_compactness_delta': model_within - all_pair_mean(model_z),
        'original_within_cosine_for_embedding_clusters': original_cross_within,
        'original_all_pair_cosine': all_pair_mean(original_z),
        'original_cross_space_compactness_delta': original_cross_within - all_pair_mean(original_z),
        'model_silhouette_cosine': silhouette_score(model_z, model_labels, metric='cosine'),
        'original_silhouette_of_embedding_labels': silhouette_score(
            original_z, model_labels, metric='cosine'
        ),
    }])
    compactness.to_csv(analysis.RESULTS / 'selected_cluster_compactness.csv', index=False)

    print('7/10 Differential morphology and feature purity ...', flush=True)
    differential = differential_features(original_df, model_labels)
    differential.to_parquet(analysis.RESULTS / 'cluster_differential_features.parquet', index=False)
    differential.to_csv(analysis.RESULTS / 'cluster_differential_features.csv', index=False)
    significant, composition, purity = feature_composition(
        differential, original_df.columns
    )
    heatmap_matrix = differential_heatmap_matrix(differential)

    print('8/10 Cluster pathway/complex enrichment and STRING support ...', flush=True)
    enrichment_model = enrich_gene_sets(model_labels, genes, records, 'DeSCORT')
    enrichment_original = enrich_gene_sets(original_labels, genes, records, 'Original')
    enrichment = pd.concat([enrichment_model, enrichment_original], ignore_index=True)
    enrichment.to_csv(analysis.RESULTS / 'cluster_pathway_complex_enrichment.csv', index=False)
    string_support = string_cluster_support(
        model_labels, genes, pair_sets['STRING400']
    )
    string_support.to_csv(analysis.RESULTS / 'cluster_STRING400_support.csv', index=False)

    top_terms = (
        enrichment_model.sort_values(['p_adj_BH_within_database', 'p_value'])
        .groupby(['cluster', 'database'], as_index=False)
        .head(3)
    )
    top_terms.to_csv(analysis.RESULTS / 'cluster_top_biological_terms.csv', index=False)

    cases, case_pairs = build_case_studies_all_resources(
        enrichment_model, differential, model_z, original_z, genes,
        pair_sets, model_labels,
    )
    (analysis.RESULTS / 'case_selection_config.json').write_text(
        json.dumps({
            'selection_metric': analysis.CASE_SELECTION_METRIC,
            'top_n': analysis.CASE_SELECTION_TOP_N,
            'minimum_case_overlap': analysis.CASE_SELECTION_MIN_OVERLAP,
            'require_model_mean_cosine_gt_original': True,
            'databases': analysis.CASE_SELECTION_DATABASES,
            'one_case_per_cluster': analysis.CASE_SELECTION_ONE_PER_CLUSTER,
            'string_threshold': analysis.STRING_THRESHOLD,
            'selection_scope': 'DeSCORT enrichment plus within-cluster STRING400 edges',
        }, ensure_ascii=False, indent=2), encoding='utf-8'
    )
    cases.to_csv(analysis.RESULTS / 'cluster_case_studies.csv', index=False)
    case_pairs.to_csv(analysis.RESULTS / 'cluster_case_gene_pairs.csv', index=False)
    cluster_biological_distances = build_cluster_biological_distance_table(
        enrichment_model, model_z, original_z, genes, pair_sets, model_labels,
        min_genes=analysis.CASE_SELECTION_MIN_OVERLAP,
    )
    cluster_biological_distance_summary = summarize_cluster_biological_distances(
        cluster_biological_distances
    )
    cluster_biological_distances.to_csv(
        analysis.RESULTS / 'cluster_biological_distance_comparison.csv', index=False
    )
    cluster_biological_distance_summary.to_csv(
        analysis.RESULTS / 'cluster_biological_distance_summary.csv', index=False
    )
    selected_pathways = build_selected_pathway_comparison(
        model_z, original_z, genes
    )
    selected_pathways.to_csv(
        analysis.RESULTS / 'figure8_selected_pathway_comparison.csv', index=False
    )

    print('9/10 Building queryable neighbour database ...', flush=True)
    top20_support = top20_database_support_table(
        genes, model_nn, original_nn, pair_sets, top_n=analysis.TOP_K_EVAL
    )
    neighbor_table = build_neighbor_table(
        genes, model_labels, model_nn, model_nn_sim,
        original_z, pair_sets, records,
    )
    queried_genes = analysis.export_gene_queries(
        neighbor_table, analysis.ARGS.query_gene, analysis.ARGS.query_top_n)
    example_gene = queried_genes[0]

    if not analysis.ARGS.skip_figures:
        from umap import UMAP
        print('10/10 Rendering publication-style figures ...', flush=True)
        umap_xy = UMAP(
            n_neighbors=analysis.ARGS.umap_n_neighbors, min_dist=analysis.ARGS.umap_min_dist, metric='cosine',
            random_state=analysis.ARGS.umap_seed, n_jobs=1,
        ).fit_transform(model_z)
        pd.DataFrame({
            'gene': genes,
            'UMAP1': umap_xy[:, 0],
            'UMAP2': umap_xy[:, 1],
            'cluster': model_labels,
        }).to_csv(analysis.RESULTS / 'embedding_cluster_umap.csv', index=False)
        plot_cluster_map(umap_xy, model_labels, cluster_selection)
        plot_differential_heatmap(heatmap_matrix)
        plot_differential_dotplots(
            differential, original_df, model_labels, n_top=analysis.ARGS.dotplot_top_n
        )
        plot_feature_purity(composition, purity)
        plot_pathway_validation(co_cluster, enrichment)
        plot_case_comparison(cases)
        plot_case_umap(cases, umap_xy, genes)
        plot_cluster_biological_distance_by_database(
            cluster_biological_distances, cluster_biological_distance_summary
        )
        plot_cluster_biological_distance_overall(
            cluster_biological_distances, cluster_biological_distance_summary
        )
        plot_selected_pathway_comparison(selected_pathways)
        plot_selected_pathway_umap(selected_pathways, umap_xy, genes)

    summary = {
        **provenance,
        **resource_summary,
        'selected_k': selected_k,
        'significant_cluster_feature_tests': int(len(significant)),
        'significant_unique_original_features': int(significant['feature'].nunique()),
        'median_compartment_purity': float(
            purity.loc[purity['category_type'] == 'compartment', 'purity'].median()
        ),
        'median_channel_purity': float(
            purity.loc[purity['category_type'] == 'channel', 'purity'].median()
        ),
        'high_compartment_purity_cluster_directions_ge_0_60': int(
            (
                purity.loc[purity['category_type'] == 'compartment', 'purity']
                >= 0.60
            ).sum()
        ),
        'STRING400_supported_clusters_empirical_p_lt_0_05': int(
            (string_support['empirical_p'] < 0.05).sum()
        ),
        'significant_DeSCORT_enrichment_tests': int(
            (enrichment_model['p_adj_BH_within_database'] < 0.05).sum()
        ),
        'significant_Original_enrichment_tests': int(
            (enrichment_original['p_adj_BH_within_database'] < 0.05).sum()
        ),
        'case_studies_retained': int(len(cases)),
        'cluster_biological_distance_units': int(len(cluster_biological_distances)),
        'cluster_biological_distance_model_better_fraction_overall': float(
            cluster_biological_distance_summary.loc[
                cluster_biological_distance_summary['database_group'] == 'Overall',
                'model_better_fraction',
            ].iloc[0]
        ),
        'figure8_selected_pathways_retained': int(len(selected_pathways)),
        'neighbor_rows': int(len(neighbor_table)),
        'top20_database_support_model_mean': float(
            top20_support.loc[top20_support['representation'] == 'DeSCORT', 'database_supported_n'].mean()
        ),
        'top20_database_support_original_mean': float(
            top20_support.loc[top20_support['representation'] == 'Original', 'database_supported_n'].mean()
        ),
        'query_example_gene': example_gene,
        'query_genes': queried_genes,
    }
    (analysis.RESULTS / 'analysis_summary.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8'
    )
    print('Completed:', analysis.OUT)

def cli():
    args=argument_parser().parse_args()
    configure(args)
    if args.stage=='query':
        table=pd.read_parquet(analysis.RESULTS/'gene_neighbor_rank.parquet')
        found=analysis.export_gene_queries(table,args.query_gene,args.query_top_n)
        print('Saved queries:', ', '.join(found))
        return
    if args.stage=='plot':
        analysis.redraw_revised_figures()
        return
    check_inputs(args)
    if args.stage=='check':return
    from sklearn import set_config
    set_config(working_memory=256)
    with threadpool_limits(limits=args.threads):
        main()


if __name__ == '__main__':
    cli()
