"""Integrated biological interpretation of A549 DMEM perturbation embeddings.

Scientific question
-------------------
Do clusters learned solely from DeSCORT hit-gene embeddings recover coherent
original Cell Painting phenotypes and independent CORUM, STRING, GO and KEGG
biology beyond original-morphology and randomized-label controls?

Pipeline
--------
raw wells -> guide mean -> gene median -> hit-gene intersection -> independent
PCA95 representations -> embedding-only unsupervised clustering -> differential
morphology -> compartment/channel purity -> biological validation -> concrete
case studies selected by CORUM FDR -> queryable nearest-neighbour database.
"""

from __future__ import annotations

from collections import defaultdict
from itertools import combinations
import json
import re
import textwrap
import os
import pickle
import sys
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/tmp/mpl_integrated_embedding_cluster')
os.environ.setdefault('NUMBA_CACHE_DIR', '/tmp/numba_integrated_embedding_cluster')
Path(os.environ['MPLCONFIGDIR']).mkdir(parents=True, exist_ok=True)
Path(os.environ['NUMBA_CACHE_DIR']).mkdir(parents=True, exist_ok=True)

import duckdb
import matplotlib as mpl
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, leaves_list, linkage
from scipy.spatial.distance import jensenshannon
from scipy.stats import fisher_exact, t as t_dist, wilcoxon
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score,
    roc_auc_score,
    silhouette_score,
)
from statsmodels.stats.multitest import multipletests



# Command-line configuration. All algorithm/plot functions below are retained
# from the integrated notebook, with paths and runtime settings exposed here.
import argparse
import logging
from threadpoolctl import threadpool_limits

from . import utils as shared_io
from .utils import PROJECT, TUTORIAL_DIR, RESOURCE_ROOT, WORKSPACE
HERE = TUTORIAL_DIR
ROOT = WORKSPACE

def argument_parser():
    defaults = shared_io.input_defaults()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    for name in ['model_input', 'original_input', 'model_feature_prefix', 'model_feature_dim',
                 'gene_column', 'guide_column', 'control_key_word', 'threads', 'memory_limit']:
        value = getattr(defaults, name)
        p.add_argument('--' + name.replace('_', '-'), type=Path if isinstance(value, Path) else type(value), default=value)
    p.add_argument('--original-features-pkl', type=Path, default=RESOURCE_ROOT/'feature_intersection_union/cpg0021_A549/cpg0021_A549_union_feature_1382.pkl')
    p.add_argument('--hit-file', type=lambda s: Path(s) if s.strip() else None, nargs='?', const=None)
    p.add_argument('--hit-column', default='Gene')
    p.add_argument('--corum-file', type=Path, default=ROOT/'LLLM/common_files/CORUM_humanComplexes.txt')
    p.add_argument('--string-file', type=Path, default=ROOT/'LLLM/common_files/STRING_data.csv')
    p.add_argument('--pathway-snapshot', type=Path, default=PROJECT/'tutorials/07_biological/GO_KEGG_ANNOTATED_pathway_UMAP/results/all_pathway_distances.csv')
    p.add_argument('--selected-pathways-json', type=Path, default=HERE/'03_de_cluster_selected_pathways.json')
    p.add_argument('--model-gene-profiles', type=Path)
    p.add_argument('--original-gene-profiles', type=Path)
    p.add_argument('--output-dir', type=Path, default=HERE/'result/03_de_cluster')
    p.add_argument('--dataset-name', default='A549 DMEM')
    p.add_argument('--plates', nargs='*', default=defaults.plates)
    p.add_argument('--pca-variance', type=float, default=.95)
    p.add_argument('--n-clusters', type=int, default=10)
    p.add_argument('--stability-seeds', type=int, default=5)
    p.add_argument('--kmeans-n-init', type=int, default=20)
    p.add_argument('--top-k-export', type=int, default=50)
    p.add_argument('--top-k-eval', type=int, default=20)
    p.add_argument('--pair-permutations', type=int, default=200)
    p.add_argument('--cluster-permutations', type=int, default=200)
    p.add_argument('--case-random-repeats', type=int, default=10000)
    p.add_argument('--pathway-random-repeats', type=int, default=10000)
    p.add_argument('--case-top-n', type=int, default=10)
    p.add_argument('--case-min-overlap', type=int, default=6)
    p.add_argument('--case-one-per-cluster', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--query-gene', nargs='+', default=['PSMA1'], help='One or more gene symbols separated by spaces')
    p.add_argument('--query-top-n', type=int, default=20)
    p.add_argument('--de-fdr', type=float, default=.05)
    p.add_argument('--min-cohens-d', type=float, default=.5)
    p.add_argument('--min-mean-difference', type=float, default=.2)
    p.add_argument('--dotplot-top-n', type=int, default=5)
    p.add_argument('--umap-n-neighbors', type=int, default=10)
    p.add_argument('--umap-min-dist', type=float, default=.05)
    p.add_argument('--umap-seed', type=int, default=42)
    p.add_argument('--pair-chunk-size', type=int, default=50000)
    p.add_argument('--neighbor-block-size', type=int, default=256)
    p.add_argument('--seed', type=int, default=20260926)
    p.add_argument('--figure-formats', nargs='+', type=str.lower, choices=['png','pdf','svg'], default=['pdf','svg','png'])
    p.add_argument('--png-dpi', type=int, default=400)
    p.add_argument('--skip-figures', action='store_true')
    p.add_argument('--stage', choices=['all','check','query','plot'], default='all')
    return p


def configure(args):
    global ARGS, SEED, OUT, RESULTS, FIGURES, QA, PROFILES, PLATES, GENE, GUIDE
    global MODEL_FEATURES, MODEL_PATH, ORIGINAL_PATH, FEATURE_FILE, HITS_FILE
    global CORUM_FILE, STRING_FILE, PATHWAY_SNAPSHOT, N_CLUSTERS, K_CANDIDATES
    global TOP_K_EXPORT, TOP_K_EVAL, PAIR_PERMUTATIONS, CLUSTER_PERMUTATIONS
    global STRING_THRESHOLD, CASE_SELECTION_METRIC, CASE_SELECTION_TOP_N
    global CASE_SELECTION_MIN_OVERLAP, CASE_SELECTION_ONE_PER_CLUSTER, CASE_SELECTION_DATABASES
    global QUERY_EXAMPLE_GENE, SELECTED_PATHWAYS_FIG8, FIG8_RANDOM_REPEATS
    if not 0 < args.pca_variance < 1 or not 0 < args.de_fdr < 1:
        raise ValueError('PCA variance and DE FDR must be in (0,1)')
    if args.stability_seeds < 2 or args.n_clusters < 2:
        raise ValueError('At least 2 stability seeds and 2 clusters are required')
    for name in ['kmeans_n_init','top_k_export','top_k_eval','pair_permutations','cluster_permutations',
                 'case_random_repeats','pathway_random_repeats','case_top_n','query_top_n','dotplot_top_n','threads','pair_chunk_size','neighbor_block_size']:
        if getattr(args,name) < 1: raise ValueError(f'{name} must be positive')
    if args.case_min_overlap < 2 or args.min_cohens_d < 0 or args.min_mean_difference < 0:
        raise ValueError('Invalid case size or effect threshold')
    if args.top_k_eval > args.top_k_export:
        raise ValueError('top-k-eval must not exceed top-k-export')
    args.query_gene = list(dict.fromkeys(str(g).strip().upper() for g in
                           ([args.query_gene] if isinstance(args.query_gene, str) else args.query_gene) if str(g).strip()))
    if not args.query_gene: raise ValueError('At least one query gene is required')
    ARGS=args; SEED=args.seed; OUT=args.output_dir.resolve()
    RESULTS=OUT/'results'; FIGURES=OUT/'figures'; QA=OUT/'qa'; PROFILES=RESULTS/'profiles'
    for folder in [RESULTS, FIGURES, QA, PROFILES]:folder.mkdir(parents=True,exist_ok=True)
    logging.basicConfig(level=logging.INFO,format='%(asctime)s | %(levelname)s | %(message)s',
                        handlers=[logging.StreamHandler(),logging.FileHandler(OUT/f'{args.stage}.log')])
    logging.getLogger('fontTools').setLevel(logging.WARNING)
    shared_io.save_json(OUT/f'{args.stage}_parameters.json',vars(args))
    PLATES=args.plates; GENE=args.gene_column; GUIDE=args.guide_column
    MODEL_FEATURES=[f'{args.model_feature_prefix}{i}' for i in range(args.model_feature_dim)]
    MODEL_PATH=args.model_input; ORIGINAL_PATH=args.original_input; FEATURE_FILE=args.original_features_pkl
    HITS_FILE=args.hit_file; CORUM_FILE=args.corum_file; STRING_FILE=args.string_file; PATHWAY_SNAPSHOT=args.pathway_snapshot
    N_CLUSTERS=args.n_clusters; K_CANDIDATES=[N_CLUSTERS]
    TOP_K_EXPORT=args.top_k_export; TOP_K_EVAL=args.top_k_eval
    PAIR_PERMUTATIONS=args.pair_permutations; CLUSTER_PERMUTATIONS=args.cluster_permutations
    STRING_THRESHOLD=400
    CASE_SELECTION_METRIC='model_mean_cosine'; CASE_SELECTION_TOP_N=args.case_top_n
    CASE_SELECTION_MIN_OVERLAP=args.case_min_overlap; CASE_SELECTION_ONE_PER_CLUSTER=args.case_one_per_cluster
    CASE_SELECTION_DATABASES=['CORUM','STRING400','GO_BP','GO_CC','GO_MF','KEGG']
    QUERY_EXAMPLE_GENE=args.query_gene[0]; SELECTED_PATHWAYS_FIG8=[tuple(v) for v in json.loads(args.selected_pathways_json.read_text())]
    FIG8_RANDOM_REPEATS=args.pathway_random_repeats


def check_inputs(args):
    for key in ['model','original']:
        shared_io.inspect_input(shared_io.get_files(getattr(args,key+'_input'),args.plates),key,args)
        cached=getattr(args,key+'_gene_profiles')
        if cached and not cached.is_file():raise FileNotFoundError(cached)
    shared_io.resolve_hit_file(args.hit_file)
    for path in [args.corum_file,args.string_file,args.pathway_snapshot,args.selected_pathways_json]:
        if not path.is_file():raise FileNotFoundError(path)
    columns=pd.read_csv(args.pathway_snapshot,nrows=0).columns
    if not {'database','pathway','genes'} <= set(columns):raise ValueError('Pathway snapshot needs database, pathway, genes columns')
    print('Input checks passed.',flush=True)



from .audit_panel_alignment import require_matplotlib_panel_alignment

mpl.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans', 'sans-serif'],
    'font.size': 7,
    'axes.labelsize': 8,
    'axes.titlesize': 9,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 7,
    'axes.spines.right': False,
    'axes.spines.top': False,
    'axes.linewidth': 0.8,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'svg.fonttype': 'none',
    'savefig.facecolor': 'white',
})
sns.set_theme(style='ticks', rc=mpl.rcParams)

COLORS = {
    'DeSCORT': '#BC3C28',
    'Original': '#0072B5',
    'Random': '#A7ADB2',
    'Cells': '#D95F5F',
    'Cytoplasm': '#4AAE8A',
    'Nuclei': '#4C78A8',
    'Other': '#B8BDC2',
}
DATABASE_COLORS = {
    'CORUM': '#7A5195',
    'STRING400': '#EF8354',
    'GO_BP': '#3B82A0',
    'GO_CC': '#59A14F',
    'GO_MF': '#E3A43B',
    'KEGG': '#B45F82',
}


def panel_label(ax, label: str, *, inside: bool = False, offset=None) -> None:
    if offset is None:
        offset = (4, -4) if inside else (-22, 8)
    ax.annotate(
        label, xy=(0, 1), xycoords='axes fraction', xytext=offset,
        textcoords='offset points', ha='left', va='top' if inside else 'bottom',
        fontsize=9, fontweight='bold', annotation_clip=False,
        bbox=(dict(facecolor='white', edgecolor='none', alpha=0.75, pad=0.5)
              if inside else None),
    )


def save_figure(fig, stem: Path, *, multi_panel: bool = False, exclude_axes=None) -> None:
    """Apply the render-time geometry gate and export editable publication files."""
    fig.canvas.draw()
    require_matplotlib_panel_alignment(
        fig,
        json_out=str(QA / f'{stem.name}.alignment.json'),
        overlay_svg=str(QA / f'{stem.name}.alignment.svg'),
        tolerance_pt=1.5,
        gutter_tolerance_pt=1.5,
        require_panel_labels=multi_panel,
        exclude_axes=exclude_axes or [],
        strict=True,
    )
    for extension in dict.fromkeys(ARGS.figure_formats):
        fig.savefig(str(stem)+'.'+extension, dpi=ARGS.png_dpi, bbox_inches='tight')
    plt.close(fig)







def selected_files(folder: Path) -> list[Path]:
    return shared_io.get_files(folder, PLATES)



def quote_identifier(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def aggregate_gene_profiles(files, features, output_path):
    frame=shared_io.aggregate(files,features,ARGS)
    frame.to_parquet(output_path)
    return frame



def load_profiles() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    frames = {}; inputs = {}
    for key in ['model', 'original']:
        files = shared_io.get_files(getattr(ARGS,key+'_input'),ARGS.plates)
        folder=PROFILES/key;folder.mkdir(exist_ok=True)
        features,info = shared_io.inspect_input(files,key,ARGS,folder)
        cache=getattr(ARGS,key+'_gene_profiles')
        frame=pd.read_parquet(cache) if cache else shared_io.aggregate(files,features,ARGS)
        if not set(features) <= set(frame.columns):raise ValueError(f'{key}: cached profiles lack required features')
        frame=frame[features].copy();frame.index=frame.index.astype(str).str.strip()
        if not frame.index.is_unique:raise ValueError(f'{key}: duplicate gene identifiers')
        frame=frame.loc[frame.index != ARGS.control_key_word].sort_index()
        frame.to_parquet(PROFILES/f'{key}_gene_profiles.parquet')
        frames[key]=frame;info['profile_cache']=str(cache) if cache else None;inputs[key]=info
    model,original=frames['model'],frames['original']
    path=shared_io.resolve_hit_file(HITS_FILE)
    if path is None:
        requested=pd.Index(model.index.intersection(original.index,sort=False),name=GENE)
        selection_mode='all_genes'
    else:
        hit_table=pd.read_excel(path) if path.suffix.lower() in ['.xls','.xlsx'] else pd.read_csv(path)
        requested=pd.Index(hit_table[ARGS.hit_column].dropna().astype(str).str.strip().unique(),name=GENE)
        selection_mode='hit_list'
    common=pd.Index([g for g in requested if g in model.index and g in original.index],name=GENE)
    missing=[g for g in requested if g not in model.index or g not in original.index]
    model=model.loc[common];original=original.loc[common]
    finite=np.isfinite(model.to_numpy()).all(axis=1)&np.isfinite(original.to_numpy()).all(axis=1)
    excluded=common[~finite].tolist()
    model=model.loc[finite].astype('float32');original=original.loc[finite].astype('float32')
    if not model.index.equals(original.index):raise AssertionError('Gene order mismatch')
    if len(model) <= max(N_CLUSTERS,TOP_K_EXPORT):
        raise ValueError(f'{len(model)} selected genes: n-clusters and top-k-export must each be smaller')
    model.to_parquet(PROFILES/'model_hit_gene_profiles.parquet');original.to_parquet(PROFILES/'original_hit_gene_profiles.parquet')
    provenance=dict(aggregation_source='explicit_cache_or_raw_parquet',aggregation='guide mean followed by gene median',
                    plates=PLATES,hit_genes_in_file=len(requested),common_finite_hit_genes=len(model),
                    model_features=model.shape[1],original_features=original.shape[1],
                    excluded_nonfinite_genes=excluded,missing_requested_genes=missing,gene_selection_mode=selection_mode)
    shared_io.save_json(RESULTS/'input_provenance.json',dict(**provenance,inputs=inputs))
    pd.DataFrame({'gene':model.index}).to_csv(RESULTS/'selected_genes.csv',index=False)
    return model,original,provenance




def cosine_normalized(values: np.ndarray) -> np.ndarray:
    """Return row-normalized values only for cosine calculations."""
    values = np.asarray(values, dtype=np.float32)
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)


def fit_pca95(frame: pd.DataFrame) -> tuple[np.ndarray, PCA]:
    pca = PCA(n_components=ARGS.pca_variance, svd_solver='full')
    scores = pca.fit_transform(frame.to_numpy(dtype=np.float32)).astype(np.float32)
    return scores, pca


def topk_cosine(z: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    unit = cosine_normalized(z)
    if len(unit) > 2500:
        from .de_cluster_scalable import topk
        return topk(unit,k,ARGS.neighbor_block_size)
    similarity = unit @ unit.T
    np.fill_diagonal(similarity, -np.inf)
    idx = np.argpartition(-similarity, kth=np.arange(k), axis=1)[:, :k]
    values = np.take_along_axis(similarity, idx, axis=1)
    order = np.argsort(-values, axis=1)
    return np.take_along_axis(idx, order, axis=1), np.take_along_axis(values, order, axis=1)



def cosine_pair_scores(unit, first, second):
    from .de_cluster_scalable import pair_scores
    return pair_scores(unit, first, second, ARGS.pair_chunk_size)


def reference_pair_pools(model_cos, original_cos):
    if len(model_cos) <= 2500:
        first, second, _ = all_pair_codes(len(model_cos))
        return cosine_pair_scores(model_cos, first, second), cosine_pair_scores(original_cos, first, second)
    from .de_cluster_scalable import all_pair_pool
    folder=RESULTS/'pair_cache'
    return (all_pair_pool(model_cos,folder,'model',ARGS.neighbor_block_size),
            all_pair_pool(original_cos,folder,'original',ARGS.neighbor_block_size))


def reference_resample_means(pool, rng, repeats, count):
    from .de_cluster_scalable import resampled_means
    return resampled_means(pool, rng, repeats, count)


def all_pair_mean(z: np.ndarray) -> float:
    unit = cosine_normalized(z).astype(np.float64)
    n = len(unit)
    return float((np.square(unit.sum(axis=0, dtype=np.float64)).sum() - n) / (n * (n - 1)))


def all_pair_codes(n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    first, second = np.triu_indices(n, 1)
    return first, second, first.astype(np.int64) * n + second.astype(np.int64)


def pair_codes(first, second, n: int) -> np.ndarray:
    first = np.asarray(first, dtype=np.int64)
    second = np.asarray(second, dtype=np.int64)
    low = np.minimum(first, second)
    high = np.maximum(first, second)
    return low * n + high


def load_biological_resources(genes: np.ndarray) -> tuple[dict, dict, dict]:
    universe = set(genes)
    gene_to_idx = {gene: i for i, gene in enumerate(genes)}
    n = len(genes)

    corum = pd.read_csv(CORUM_FILE, sep='\t')
    if 'Organism' in corum:
        corum = corum[corum['Organism'].astype(str).str.contains('Human', case=False, na=False)]
    corum_records = []
    corum_codes = set()
    for _, row in corum.iterrows():
        members = [
            x.strip() for x in str(row.get('subunits(Gene name)', '')).split(';')
            if x.strip() in universe
        ]
        members = sorted(set(members))
        if len(members) < 2:
            continue
        name = str(row.get('ComplexName', 'CORUM complex'))
        corum_records.append((name, members))
        for gene_a, gene_b in combinations(members, 2):
            corum_codes.add(int(pair_codes(gene_to_idx[gene_a], gene_to_idx[gene_b], n)))

    string_codes = set()
    for chunk in pd.read_csv(
        STRING_FILE,
        usecols=['protein1', 'protein2', 'combined_score'],
        chunksize=500_000,
    ):
        chunk = chunk.loc[
            (chunk['combined_score'] >= STRING_THRESHOLD)
            & chunk['protein1'].isin(universe)
            & chunk['protein2'].isin(universe)
        ]
        for gene_a, gene_b in chunk[['protein1', 'protein2']].itertuples(index=False, name=None):
            if gene_a != gene_b:
                string_codes.add(int(pair_codes(gene_to_idx[gene_a], gene_to_idx[gene_b], n)))

    pathway_source = pd.read_csv(PATHWAY_SNAPSHOT)
    pathway_source = pathway_source[
        pathway_source['database'].isin(['GO_BP', 'GO_CC', 'GO_MF', 'KEGG'])
    ].copy()
    pathway_records = defaultdict(list)
    pathway_codes = defaultdict(set)
    for row in pathway_source.itertuples(index=False):
        members = sorted({x for x in str(row.genes).split(';') if x in universe})
        if len(members) < 5:
            continue
        pathway_records[row.database].append((str(row.pathway), members))
        for gene_a, gene_b in combinations(members, 2):
            pathway_codes[row.database].add(
                int(pair_codes(gene_to_idx[gene_a], gene_to_idx[gene_b], n))
            )

    pair_sets = {
        'CORUM': corum_codes,
        'STRING400': string_codes,
        **dict(pathway_codes),
    }
    records = {'CORUM': corum_records, **dict(pathway_records)}
    resource_summary = {
        'CORUM_complexes': len(corum_records),
        'CORUM_pairs': len(corum_codes),
        'STRING400_pairs': len(string_codes),
        **{f'{database}_pathways': len(values) for database, values in pathway_records.items()},
        **{f'{database}_pairs': len(values) for database, values in pathway_codes.items()},
        'pathway_snapshot_note': (
            'All valid hit-gene pathways in the local snapshot were used; '
            'distance/rank columns were ignored.'
        ),
    }
    return pair_sets, records, resource_summary


def codes_to_indices(codes: set[int], n: int) -> tuple[np.ndarray, np.ndarray]:
    values = np.fromiter(codes, dtype=np.int64)
    return values // n, values % n


def empirical_pair_p(
    z: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
    observed: float,
    repeats: int,
    seed: int,
) -> float:
    rng = np.random.default_rng(seed)
    null = np.empty(repeats, dtype=np.float64)
    z_cos = cosine_normalized(z)
    n = len(z_cos)
    for i in range(repeats):
        perm = rng.permutation(n)
        null[i] = cosine_pair_scores(z_cos, perm[first], perm[second]).mean()
    return float((1 + np.sum(null >= observed)) / (repeats + 1))


def evaluate_pair_support(
    model_z: np.ndarray,
    original_z: np.ndarray,
    pair_sets: dict[str, set[int]],
    model_nn: np.ndarray,
    original_nn: np.ndarray,
) -> pd.DataFrame:
    n = len(model_z)
    model_cos = cosine_normalized(model_z)
    original_cos = cosine_normalized(original_z)
    if n > 2500:
        from .de_cluster_scalable import evaluate_support
        rows=evaluate_support(model_cos,original_cos,pair_sets,model_nn,original_nn,
                              TOP_K_EVAL,PAIR_PERMUTATIONS,SEED,empirical_pair_p,ARGS.pair_chunk_size)
        return pd.DataFrame(rows)
    all_first, all_second, all_codes = all_pair_codes(n)
    all_model = cosine_pair_scores(model_cos, all_first, all_second)
    all_original = cosine_pair_scores(original_cos, all_first, all_second)
    model_top_codes = set(
        pair_codes(
            np.repeat(np.arange(n), TOP_K_EVAL),
            model_nn[:, :TOP_K_EVAL].ravel(),
            n,
        ).tolist()
    )
    original_top_codes = set(
        pair_codes(
            np.repeat(np.arange(n), TOP_K_EVAL),
            original_nn[:, :TOP_K_EVAL].ravel(),
            n,
        ).tolist()
    )
    rng = np.random.default_rng(SEED)
    rows = []
    for index, (database, codes) in enumerate(pair_sets.items()):
        valid = set(codes) & set(all_codes.tolist())
        if not valid:
            continue
        first, second = codes_to_indices(valid, n)
        model_positive = cosine_pair_scores(model_cos, first, second)
        original_positive = cosine_pair_scores(original_cos, first, second)
        negative_mask = ~np.isin(all_codes, np.fromiter(valid, dtype=np.int64))
        negative_pool = np.flatnonzero(negative_mask)
        sample_n = min(len(valid), len(negative_pool), 200_000)
        negative_idx = rng.choice(negative_pool, sample_n, replace=False)
        model_auc = roc_auc_score(
            np.r_[np.ones(len(model_positive)), np.zeros(sample_n)],
            np.r_[model_positive, all_model[negative_idx]],
        )
        original_auc = roc_auc_score(
            np.r_[np.ones(len(original_positive)), np.zeros(sample_n)],
            np.r_[original_positive, all_original[negative_idx]],
        )
        prevalence = len(valid) / len(all_codes)
        model_top_rate = len(model_top_codes & valid) / max(1, len(model_top_codes))
        original_top_rate = len(original_top_codes & valid) / max(1, len(original_top_codes))
        rows.append({
            'database': database,
            'positive_pairs': len(valid),
            'all_pair_prevalence': prevalence,
            'model_positive_mean_cosine': float(model_positive.mean()),
            'original_positive_mean_cosine': float(original_positive.mean()),
            'model_all_pair_mean_cosine': float(all_model.mean()),
            'original_all_pair_mean_cosine': float(all_original.mean()),
            'model_delta_vs_all_pairs': float(model_positive.mean() - all_model.mean()),
            'original_delta_vs_all_pairs': float(original_positive.mean() - all_original.mean()),
            'model_AUROC': float(model_auc),
            'original_AUROC': float(original_auc),
            'model_top20_edge_rate': model_top_rate,
            'original_top20_edge_rate': original_top_rate,
            'model_top20_fold': model_top_rate / prevalence,
            'original_top20_fold': original_top_rate / prevalence,
            'model_permutation_p': empirical_pair_p(
                model_z, first, second, float(model_positive.mean()),
                PAIR_PERMUTATIONS, SEED + index,
            ),
            'original_permutation_p': empirical_pair_p(
                original_z, first, second, float(original_positive.mean()),
                PAIR_PERMUTATIONS, SEED + 100 + index,
            ),
        })
    return pd.DataFrame(rows)


def build_cluster_biological_distance_table(
    enrichment_model: pd.DataFrame,
    model_z: np.ndarray,
    original_z: np.ndarray,
    genes: np.ndarray,
    pair_sets: dict[str, set[int]],
    model_labels: np.ndarray,
    min_genes: int = 6,
) -> pd.DataFrame:
    """Paired within-unit cosine distances using identical genes in both spaces.

    CORUM/GO/KEGG units are DeSCORT-cluster overlaps from enrichment_model.
    STRING units are connected components (>= min_genes) of the STRING400
    graph restricted to each DeSCORT cluster. This makes STRING comparable
    with pathway/complex gene sets without treating individual edges as pathways.
    """
    gene_to_idx = {str(gene): i for i, gene in enumerate(genes)}
    model_cos = cosine_normalized(model_z)
    original_cos = cosine_normalized(original_z)
    rows = []

    def append_unit(database_group, database, cluster, unit_name, member_indices, n_supported_edges=np.nan):
        member_indices = np.array(sorted(set(map(int, member_indices))), dtype=int)
        if len(member_indices) < min_genes:
            return
        local_i, local_j = np.triu_indices(len(member_indices), k=1)
        first, second = member_indices[local_i], member_indices[local_j]
        model_distance = 1.0 - cosine_pair_scores(model_cos, first, second)
        original_distance = 1.0 - cosine_pair_scores(original_cos, first, second)
        rows.append({
            'database_group': database_group,
            'database': database,
            'cluster': int(cluster),
            'unit_name': str(unit_name),
            'n_genes': int(len(member_indices)),
            'n_pairs': int(len(first)),
            'n_supported_edges': n_supported_edges,
            'genes': ';'.join(map(str, genes[member_indices])),
            'mean_distance_model': float(model_distance.mean()),
            'mean_distance_original': float(original_distance.mean()),
            'delta_distance_original_minus_model': float(original_distance.mean() - model_distance.mean()),
            'model_better': bool(model_distance.mean() < original_distance.mean()),
        })

    term_rows = enrichment_model[
        enrichment_model['database'].isin(['CORUM', 'GO_BP', 'GO_CC', 'GO_MF', 'KEGG'])
        & (enrichment_model['overlap_n'] >= min_genes)
    ]
    for row in term_rows.itertuples(index=False):
        member_indices = [
            gene_to_idx[gene] for gene in str(row.overlap_genes).split(';')
            if gene in gene_to_idx
        ]
        database_group = str(row.database)
        append_unit(database_group, row.database, row.cluster, row.term, member_indices)

    # Build STRING400 connected components separately within each DeSCORT cluster.
    adjacency = defaultdict(set)
    string_codes = pair_sets.get('STRING400', set())
    for code in string_codes:
        first, second = divmod(int(code), len(genes))
        if first == second or model_labels[first] != model_labels[second]:
            continue
        adjacency[first].add(second)
        adjacency[second].add(first)
    for cluster in sorted(np.unique(model_labels)):
        cluster_nodes = {int(i) for i in np.where(model_labels == cluster)[0] if int(i) in adjacency}
        components = []
        unseen = set(cluster_nodes)
        while unseen:
            start = unseen.pop()
            component = {start}
            stack = [start]
            while stack:
                node = stack.pop()
                neighbours = adjacency[node] & unseen
                unseen.difference_update(neighbours)
                component.update(neighbours)
                stack.extend(neighbours)
            if len(component) >= min_genes:
                components.append(sorted(component))
        components.sort(key=lambda values: (-len(values), [str(genes[i]) for i in values]))
        for component_rank, component in enumerate(components, start=1):
            component_set = set(component)
            supported_edges = sum(
                1 for first in component for second in adjacency[first]
                if first < second and second in component_set
            )
            append_unit(
                'STRING', 'STRING400', cluster,
                f'Cluster {int(cluster)} STRING component {component_rank}',
                component, n_supported_edges=int(supported_edges),
            )

    result = pd.DataFrame(rows)
    if result.empty:
        return result
    return result.sort_values(
        ['database_group', 'cluster', 'mean_distance_model', 'unit_name']
    ).reset_index(drop=True)


def summarize_cluster_biological_distances(distance_table: pd.DataFrame) -> pd.DataFrame:
    """Summarize paired DeSCORT/Original unit distances and paired Wilcoxon tests."""
    rows = []
    groups = ['CORUM', 'STRING', 'GO_BP', 'GO_CC', 'GO_MF', 'KEGG', 'Overall']
    for group in groups:
        subset = distance_table if group == 'Overall' else distance_table[distance_table['database_group'] == group]
        if subset.empty:
            continue
        model_values = subset['mean_distance_model'].to_numpy(dtype=float)
        original_values = subset['mean_distance_original'].to_numpy(dtype=float)
        differences = original_values - model_values
        if len(subset) >= 2 and not np.allclose(differences, 0):
            statistic, pvalue = wilcoxon(
                original_values, model_values, alternative='greater', method='auto'
            )
        else:
            statistic, pvalue = np.nan, np.nan
        rows.append({
            'database_group': group,
            'n_units': int(len(subset)),
            'mean_distance_model': float(model_values.mean()),
            'mean_distance_original': float(original_values.mean()),
            'median_distance_model': float(np.median(model_values)),
            'median_distance_original': float(np.median(original_values)),
            'mean_delta_original_minus_model': float(differences.mean()),
            'model_better_fraction': float(np.mean(model_values < original_values)),
            'wilcoxon_statistic_original_gt_model': float(statistic),
            'wilcoxon_p_one_sided': float(pvalue),
        })
    return pd.DataFrame(rows)


def select_embedding_clustering(model_z: np.ndarray) -> tuple[np.ndarray, pd.DataFrame, int]:
    seeds = [SEED + offset for offset in range(ARGS.stability_seeds)]
    rows = []
    labels_by_k = {}
    for k in K_CANDIDATES:
        label_runs = [
            KMeans(n_clusters=k, random_state=seed, n_init=ARGS.kmeans_n_init).fit_predict(model_z)
            for seed in seeds
        ]
        silhouette_values = [
            silhouette_score(model_z, labels, metric='cosine')
            for labels in label_runs
        ]
        pairwise_ari = [
            adjusted_rand_score(label_runs[i], label_runs[j])
            for i in range(len(label_runs))
            for j in range(i + 1, len(label_runs))
        ]
        mean_ari_by_run = [
            np.mean([
                adjusted_rand_score(label_runs[i], label_runs[j])
                for j in range(len(label_runs)) if i != j
            ])
            for i in range(len(label_runs))
        ]
        representative_run = int(np.argmax(mean_ari_by_run))
        representative = label_runs[representative_run]
        counts = np.bincount(representative)
        rows.append({
            'k': k,
            'silhouette_cosine_mean_5_seeds': float(np.mean(silhouette_values)),
            'silhouette_cosine_sd_5_seeds': float(np.std(silhouette_values)),
            'stability_pairwise_ARI_mean_5_seeds': float(np.mean(pairwise_ari)),
            'stability_pairwise_ARI_sd_5_seeds': float(np.std(pairwise_ari)),
            'representative_seed': seeds[representative_run],
            'min_cluster_size': int(counts.min()),
            'median_cluster_size': float(np.median(counts)),
            'max_cluster_size': int(counts.max()),
        })
        labels_by_k[k] = representative
    table = pd.DataFrame(rows)
    table['selection_eligible'] = table['k'].between(2, len(model_z) - 1)
    table['rank_silhouette'] = np.nan
    table['rank_stability'] = np.nan
    eligible = table['selection_eligible']
    table.loc[eligible, 'rank_silhouette'] = table.loc[eligible, 'silhouette_cosine_mean_5_seeds'].rank(ascending=False)
    table.loc[eligible, 'rank_stability'] = table.loc[eligible, 'stability_pairwise_ARI_mean_5_seeds'].rank(ascending=False)
    table['internal_composite_rank'] = table[['rank_silhouette', 'rank_stability']].mean(axis=1)
    ranked = table.loc[eligible].sort_values(
        [
            'internal_composite_rank',
            'stability_pairwise_ARI_mean_5_seeds',
            'silhouette_cosine_mean_5_seeds',
        ],
        ascending=[True, False, False],
    )
    selected_k = int(ranked.iloc[0]['k'])
    table['selected'] = table['k'].eq(selected_k)
    return labels_by_k[selected_k], table, selected_k


def same_cluster_expected(labels: np.ndarray) -> float:
    counts = np.bincount(labels)
    n = len(labels)
    return float(np.sum(counts * (counts - 1)) / (n * (n - 1)))


def co_cluster_support(
    labels: np.ndarray,
    pair_sets: dict[str, set[int]],
    n: int,
    representation: str,
) -> pd.DataFrame:
    expected = same_cluster_expected(labels)
    rng = np.random.default_rng(SEED + (0 if representation == 'DeSCORT' else 1000))
    rows = []
    for database, codes in pair_sets.items():
        if not codes:
            continue
        first, second = codes_to_indices(codes, n)
        observed = float(np.mean(labels[first] == labels[second]))
        null = np.empty(CLUSTER_PERMUTATIONS, dtype=np.float64)
        for i in range(CLUSTER_PERMUTATIONS):
            perm = rng.permutation(labels)
            null[i] = np.mean(perm[first] == perm[second])
        rows.append({
            'representation': representation,
            'database': database,
            'edge_n': len(codes),
            'same_cluster_rate': observed,
            'random_expected_rate': expected,
            'fold_over_random': observed / expected,
            'permutation_p': float((1 + np.sum(null >= observed)) / (len(null) + 1)),
            'permutation_mean_rate': float(null.mean()),
        })
    return pd.DataFrame(rows)


def within_cluster_similarity(z: np.ndarray, labels: np.ndarray) -> tuple[float, int]:
    total = 0.0
    pairs = 0
    for cluster in np.unique(labels):
        block = cosine_normalized(z[labels == cluster])
        n = len(block)
        if n < 2:
            continue
        total += (np.square(block.sum(axis=0, dtype=np.float64)).sum() - n) / 2
        pairs += n * (n - 1) // 2
    return float(total / pairs), int(pairs)


def differential_features(
    original: pd.DataFrame, labels: np.ndarray,
) -> pd.DataFrame:
    x = original.to_numpy(dtype=np.float32)
    rows = []
    for cluster in np.unique(labels):
        mask = labels == cluster
        x_a, x_b = x[mask], x[~mask]
        n_a, n_b = len(x_a), len(x_b)
        mean_a, mean_b = x_a.mean(axis=0), x_b.mean(axis=0)
        var_a = x_a.var(axis=0, ddof=1)
        var_b = x_b.var(axis=0, ddof=1)
        pooled = np.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2))
        pooled = np.maximum(pooled, 1e-12)
        mean_diff = mean_a - mean_b
        effect = mean_diff / pooled
        se = pooled * np.sqrt(1 / n_a + 1 / n_b)
        t_stat = mean_diff / np.maximum(se, 1e-12)
        p_value = 2 * t_dist.sf(np.abs(t_stat), n_a + n_b - 2)
        p_adj = multipletests(p_value, method='fdr_bh')[1]
        rows.append(pd.DataFrame({
            'cluster': int(cluster),
            'cluster_size': n_a,
            'feature': original.columns.astype(str),
            'cluster_mean': mean_a,
            'rest_mean': mean_b,
            'mean_difference': mean_diff,
            'cohens_d': effect,
            'p_value': p_value,
            'p_adj_BH_within_cluster': p_adj,
        }))
    return pd.concat(rows, ignore_index=True)


def compartment(feature: str) -> str:
    for value in ('Cells', 'Cytoplasm', 'Nuclei'):
        if feature.startswith(value):
            return value
    return 'Other'


def channel(feature: str) -> str:
    for value in ('DNA', 'Mito', 'AGP', 'ER', 'RNA', 'WGA'):
        if f'_{value}_' in feature or feature.endswith(f'_{value}'):
            return value
    return 'Other'


def feature_composition(
    differential: pd.DataFrame,
    all_features: pd.Index,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    significant = differential[
        (differential['p_adj_BH_within_cluster'] < ARGS.de_fdr)
        & (differential['cohens_d'].abs() >= ARGS.min_cohens_d)
        & (differential['mean_difference'].abs() >= ARGS.min_mean_difference)
    ].copy()
    significant['direction'] = np.where(significant['cohens_d'] > 0, 'Up', 'Down')
    significant['compartment'] = significant['feature'].map(compartment)
    significant['channel'] = significant['feature'].map(channel)
    significant.to_csv(RESULTS / 'significant_differential_features.csv', index=False)

    background_comp = pd.Series(all_features.map(compartment)).value_counts(normalize=True)
    background_channel = pd.Series(all_features.map(channel)).value_counts(normalize=True)
    composition_rows = []
    purity_rows = []
    for (cluster, direction), group in significant.groupby(['cluster', 'direction']):
        for category_type, column, background in [
            ('compartment', 'compartment', background_comp),
            ('channel', 'channel', background_channel),
        ]:
            counts = group[column].value_counts()
            proportions = counts / counts.sum()
            categories = sorted(set(background.index) | set(counts.index))
            p = np.array([proportions.get(x, 0.0) for x in categories], dtype=float)
            q = np.array([background.get(x, 0.0) for x in categories], dtype=float)
            for category in categories:
                composition_rows.append({
                    'cluster': int(cluster),
                    'direction': direction,
                    'category_type': category_type,
                    'category': category,
                    'feature_n': int(counts.get(category, 0)),
                    'percentage': float(100 * proportions.get(category, 0.0)),
                    'background_percentage': float(100 * background.get(category, 0.0)),
                })
            entropy = -np.sum(p[p > 0] * np.log(p[p > 0]))
            normalized_entropy = entropy / np.log(max(2, len(categories)))
            purity_rows.append({
                'cluster': int(cluster),
                'direction': direction,
                'category_type': category_type,
                'significant_feature_n': len(group),
                'dominant_category': str(proportions.idxmax()),
                'purity': float(proportions.max()),
                'normalized_entropy': float(normalized_entropy),
                'jensen_shannon_distance_vs_all_features': float(jensenshannon(p, q)),
            })
    composition = pd.DataFrame(composition_rows)
    purity = pd.DataFrame(purity_rows)
    composition.to_csv(RESULTS / 'cluster_feature_composition.csv', index=False)
    purity.to_csv(RESULTS / 'cluster_feature_purity.csv', index=False)
    return significant, composition, purity


def differential_heatmap_matrix(
    differential: pd.DataFrame,
    max_features: int = 45,
) -> pd.DataFrame:
    significant = differential[differential['p_adj_BH_within_cluster'] < ARGS.de_fdr].copy()
    significant['abs_d'] = significant['cohens_d'].abs()
    candidates = (
        significant.sort_values('abs_d', ascending=False)
        .drop_duplicates(['cluster', 'feature'])
        .groupby('cluster', group_keys=False)
        .head(3)
    )
    feature_score = candidates.groupby('feature')['abs_d'].max().sort_values(ascending=False)
    selected = feature_score.head(max_features).index
    matrix = differential[differential['feature'].isin(selected)].pivot(
        index='cluster', columns='feature', values='cohens_d'
    )
    matrix = matrix.loc[sorted(matrix.index)]
    matrix.to_csv(RESULTS / 'differential_feature_heatmap_matrix.csv')
    return matrix


def enrich_gene_sets(
    labels: np.ndarray,
    genes: np.ndarray,
    records: dict[str, list[tuple[str, list[str]]]],
    representation: str,
) -> pd.DataFrame:
    universe = set(genes)
    rows = []
    for database, terms in records.items():
        database_rows = []
        for cluster in np.unique(labels):
            cluster_genes = set(genes[labels == cluster])
            for term, members in terms:
                members_set = set(members) & universe
                overlap = cluster_genes & members_set
                a = len(overlap)
                b = len(cluster_genes) - a
                c = len(members_set) - a
                d = len(universe) - a - b - c
                odds, p_value = fisher_exact([[a, b], [c, d]], alternative='greater')
                database_rows.append({
                    'representation': representation,
                    'database': database,
                    'cluster': int(cluster),
                    'cluster_size': len(cluster_genes),
                    'term': term,
                    'term_genes_in_universe': len(members_set),
                    'overlap_n': a,
                    'overlap_genes': ';'.join(sorted(overlap)),
                    'odds_ratio': float(odds),
                    'p_value': float(p_value),
                })
        if database_rows:
            frame = pd.DataFrame(database_rows)
            frame['p_adj_BH_within_database'] = multipletests(
                frame['p_value'], method='fdr_bh'
            )[1]
            rows.append(frame)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True).sort_values(
        ['p_adj_BH_within_database', 'p_value']
    )


def string_cluster_support(
    labels: np.ndarray,
    genes: np.ndarray,
    string_codes: set[int],
) -> pd.DataFrame:
    n = len(genes)
    adjacency = [set() for _ in range(n)]
    for code in string_codes:
        i, j = divmod(int(code), n)
        adjacency[i].add(j)
        adjacency[j].add(i)
    rng = np.random.default_rng(SEED)
    rows = []
    for cluster in np.unique(labels):
        indices = np.flatnonzero(labels == cluster)
        observed = sum(
            1 for i in indices for j in adjacency[i]
            if j > i and labels[j] == cluster
        )
        possible = len(indices) * (len(indices) - 1) // 2
        density = observed / max(1, possible)
        null = np.empty(500, dtype=float)
        for repeat in range(len(null)):
            sample = rng.choice(n, len(indices), replace=False)
            sample_set = set(sample.tolist())
            edges = sum(
                1 for i in sample for j in adjacency[i]
                if j > i and j in sample_set
            )
            null[repeat] = edges / max(1, possible)
        rows.append({
            'cluster': int(cluster),
            'cluster_size': len(indices),
            'observed_internal_STRING400_edges': int(observed),
            'observed_edge_density': density,
            'random_edge_density_mean': float(null.mean()),
            'fold_over_size_matched_random': density / max(null.mean(), 1e-12),
            'empirical_p': float((1 + np.sum(null >= density)) / (len(null) + 1)),
        })
    return pd.DataFrame(rows)


def build_case_studies(
    enrichment: pd.DataFrame,
    differential: pd.DataFrame,
    model_z: np.ndarray,
    original_z: np.ndarray,
    genes: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Build eligible CORUM cluster-term candidates. The selection ranking
    # is computed below from the mean DeSCORT cosine of all within-case
    # gene pairs; CORUM FDR is retained as a deterministic tie-breaker.
    corum = enrichment[enrichment['database'] == 'CORUM'].copy()
    corum = corum[corum['overlap_n'] >= CASE_SELECTION_MIN_OVERLAP].copy()
    if corum.empty:
        return pd.DataFrame(), pd.DataFrame()
    corum['p_adj_BH_within_database'] = pd.to_numeric(
        corum['p_adj_BH_within_database'], errors='coerce'
    )
    corum = corum[np.isfinite(corum['p_adj_BH_within_database'])].copy()
    gene_to_idx = {gene: i for i, gene in enumerate(genes)}
    model_cos = cosine_normalized(model_z)
    original_cos = cosine_normalized(original_z)
    random_model, random_original = reference_pair_pools(model_cos, original_cos)

    candidates = []
    for _, candidate in corum.iterrows():
        case_genes = [
            gene for gene in str(candidate['overlap_genes']).split(';')
            if gene in gene_to_idx
        ]
        gene_pairs = list(combinations(case_genes, 2))
        if not gene_pairs:
            continue
        first = np.array([gene_to_idx[a] for a, _ in gene_pairs], dtype=int)
        second = np.array([gene_to_idx[b] for _, b in gene_pairs], dtype=int)
        model_similarity = cosine_pair_scores(model_cos, first, second)
        original_similarity = cosine_pair_scores(original_cos, first, second)
        model_mean_cosine = float(model_similarity.mean())
        original_mean_cosine = float(original_similarity.mean())
        if not model_mean_cosine > original_mean_cosine:
            return
        candidates.append({
            'candidate': candidate,
            'case_genes': case_genes,
            'gene_pairs': gene_pairs,
            'model_similarity': model_similarity,
            'original_similarity': original_similarity,
            'model_mean_cosine': model_mean_cosine,
        })
    candidates.sort(
        key=lambda item: (
            -item['model_mean_cosine'],
            float(item['candidate']['p_adj_BH_within_database']),
            -int(item['candidate']['overlap_n']),
            str(item['candidate']['term']),
        )
    )
    selected = []
    used_clusters = set()
    for item in candidates:
        cluster_id = int(item['candidate']['cluster'])
        if CASE_SELECTION_ONE_PER_CLUSTER and cluster_id in used_clusters:
            continue
        selected.append(item)
        used_clusters.add(cluster_id)
        if len(selected) >= CASE_SELECTION_TOP_N:
            break
    if not selected:
        return pd.DataFrame(), pd.DataFrame()
    rng = np.random.default_rng(SEED + 8)
    case_rows = []
    pair_rows = []
    for selection_rank, item in enumerate(selected, start=1):
        best = item['candidate']
        complex_name = str(best['term'])
        theme = complex_name
        case_genes = item['case_genes']
        gene_pairs = item['gene_pairs']
        model_similarity = item['model_similarity']
        original_similarity = item['original_similarity']
        repeats = ARGS.case_random_repeats
        model_null = reference_resample_means(random_model, rng, repeats, len(gene_pairs))
        original_null = reference_resample_means(random_original, rng, repeats, len(gene_pairs))
        cluster = int(best['cluster'])
        top_features = (
            differential[differential['cluster'] == cluster]
            .assign(abs_d=lambda x: x['cohens_d'].abs())
            .nlargest(5, 'abs_d')
        )
        case_rows.append({
            'case_selection_rank': int(selection_rank),
            'case_selection_metric': CASE_SELECTION_METRIC,
            'case_selection_value': float(item['model_mean_cosine']),
            'cluster': cluster,
            'theme': theme,
            'CORUM_complex': complex_name,
            'overlap_n': len(case_genes),
            'overlap_genes': ';'.join(case_genes),
            'gene_pair_n': len(gene_pairs),
            'model_mean_cosine': float(model_similarity.mean()),
            'original_mean_cosine': float(original_similarity.mean()),
            'random_model_mean_cosine': float(random_model.mean()),
            'random_original_mean_cosine': float(random_original.mean()),
            'model_delta_vs_random': float(model_similarity.mean() - random_model.mean()),
            'original_delta_vs_random': float(original_similarity.mean() - random_original.mean()),
            'model_empirical_p': float(
                (1 + np.sum(model_null >= model_similarity.mean())) / (repeats + 1)
            ),
            'original_empirical_p': float(
                (1 + np.sum(original_null >= original_similarity.mean())) / (repeats + 1)
            ),
            'CORUM_odds_ratio': float(best['odds_ratio']),
            'CORUM_FDR': float(best['p_adj_BH_within_database']),
            'top_original_features': '; '.join(
                f'{row.feature} (d={row.cohens_d:.2f})'
                for _, row in top_features.iterrows()
            ),
        })
        for (gene_a, gene_b), model_value, original_value in zip(
            gene_pairs, model_similarity, original_similarity
        ):
            pair_rows.append({
                'case_selection_rank': int(selection_rank),
                'cluster': cluster,
                'theme': theme,
                'complex': complex_name,
                'gene1': gene_a,
                'gene2': gene_b,
                'model_cosine': float(model_value),
                'original_cosine': float(original_value),
            })
    return pd.DataFrame(case_rows), pd.DataFrame(pair_rows)


def build_case_studies_all_resources(
    enrichment: pd.DataFrame,
    differential: pd.DataFrame,
    model_z: np.ndarray,
    original_z: np.ndarray,
    genes: np.ndarray,
    pair_sets: dict[str, set[int]],
    model_labels: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select high-cosine cases from CORUM, STRING, GO and KEGG."""
    n = len(genes)
    gene_to_idx = {gene: i for i, gene in enumerate(genes)}
    model_cos = cosine_normalized(model_z)
    original_cos = cosine_normalized(original_z)
    random_model, random_original = reference_pair_pools(model_cos, original_cos)

    candidates = []
    def add_candidate(database, term, cluster, case_genes, resource_fdr=np.nan, resource_odds=np.nan):
        case_genes = list(dict.fromkeys(g for g in case_genes if g in gene_to_idx))
        if len(case_genes) < CASE_SELECTION_MIN_OVERLAP:
            return
        gene_pairs = list(combinations(case_genes, 2))
        if not gene_pairs:
            return
        first = np.array([gene_to_idx[a] for a, _ in gene_pairs], dtype=int)
        second = np.array([gene_to_idx[b] for _, b in gene_pairs], dtype=int)
        model_similarity = cosine_pair_scores(model_cos, first, second)
        original_similarity = cosine_pair_scores(original_cos, first, second)
        model_mean_cosine = float(model_similarity.mean())
        original_mean_cosine = float(original_similarity.mean())
        if not model_mean_cosine > original_mean_cosine:
            return
        candidates.append({
            'database': str(database),
            'term': str(term),
            'cluster': int(cluster),
            'case_genes': case_genes,
            'gene_pairs': gene_pairs,
            'model_similarity': model_similarity,
            'original_similarity': original_similarity,
            'model_mean_cosine': model_mean_cosine,
            'resource_fdr': float(resource_fdr) if np.isfinite(resource_fdr) else np.nan,
            'resource_odds': float(resource_odds) if np.isfinite(resource_odds) else np.nan,
        })

    # Term-based candidates: cluster enrichment supplies the member genes.
    term_databases = [db for db in CASE_SELECTION_DATABASES if db != 'STRING400']
    eligible = enrichment[
        enrichment['database'].isin(term_databases)
        & (enrichment['overlap_n'] >= CASE_SELECTION_MIN_OVERLAP)
    ].copy()
    if not eligible.empty:
        eligible['p_adj_BH_within_database'] = pd.to_numeric(
            eligible['p_adj_BH_within_database'], errors='coerce'
        )
        eligible = eligible[
            np.isfinite(eligible['p_adj_BH_within_database'])
        ]
        for _, row in eligible.iterrows():
            members = [
                gene for gene in str(row['overlap_genes']).split(';')
                if gene in gene_to_idx
            ]
            add_candidate(
                row['database'], row['term'], int(row['cluster']), members,
                row['p_adj_BH_within_database'], row['odds_ratio'],
            )

    # STRING has pair-level evidence rather than named term sets. Each
    # within-cluster STRING400 edge is therefore one two-gene case.
    string_codes = pair_sets.get('STRING400', set())
    for code in sorted(string_codes):
        first, second = divmod(int(code), n)
        if first >= n or second >= n or model_labels[first] != model_labels[second]:
            continue
        gene_a, gene_b = str(genes[first]), str(genes[second])
        add_candidate(
            'STRING400', f'{gene_a} – {gene_b}', int(model_labels[first]),
            [gene_a, gene_b],
        )

    candidates.sort(
        key=lambda item: (
            -item['model_mean_cosine'],
            item['resource_fdr'] if np.isfinite(item['resource_fdr']) else 1.0,
            -len(item['case_genes']), item['database'], item['term'],
        )
    )
    pd.DataFrame([
        {
            'candidate_rank_by_model_cosine': rank,
            'database': item['database'],
            'term': item['term'],
            'cluster': item['cluster'],
            'overlap_n': len(item['case_genes']),
            'gene_pair_n': len(item['gene_pairs']),
            'model_mean_cosine': item['model_mean_cosine'],
            'resource_FDR': item['resource_fdr'],
            'resource_odds_ratio': item['resource_odds'],
        }
        for rank, item in enumerate(candidates, start=1)
    ]).to_csv(RESULTS / 'cluster_case_candidates.csv', index=False)
    selected = []
    used_clusters = set()
    for item in candidates:
        if CASE_SELECTION_ONE_PER_CLUSTER and item['cluster'] in used_clusters:
            continue
        selected.append(item)
        used_clusters.add(item['cluster'])
        if len(selected) >= CASE_SELECTION_TOP_N:
            break
    if not selected:
        return pd.DataFrame(), pd.DataFrame()

    rng = np.random.default_rng(SEED + 8)
    case_rows, pair_rows = [], []
    for selection_rank, item in enumerate(selected, start=1):
        database = item['database']
        term = item['term']
        case_genes = item['case_genes']
        gene_pairs = item['gene_pairs']
        model_similarity = item['model_similarity']
        original_similarity = item['original_similarity']
        repeats = ARGS.case_random_repeats
        model_null = reference_resample_means(random_model, rng, repeats, len(gene_pairs))
        original_null = reference_resample_means(random_original, rng, repeats, len(gene_pairs))
        cluster = int(item['cluster'])
        top_features = (
            differential[differential['cluster'] == cluster]
            .assign(abs_d=lambda x: x['cohens_d'].abs())
            .nlargest(5, 'abs_d')
        )
        case_rows.append({
            'case_selection_rank': int(selection_rank),
            'case_selection_metric': CASE_SELECTION_METRIC,
            'case_selection_value': float(item['model_mean_cosine']),
            'cluster': cluster,
            'database': database,
            'case_database': database,
            'term': term,
            'case_term': term,
            'theme': f'{database}: {term}',
            'CORUM_complex': term,
            'overlap_n': len(case_genes),
            'overlap_genes': ';'.join(case_genes),
            'gene_pair_n': len(gene_pairs),
            'model_mean_cosine': float(model_similarity.mean()),
            'original_mean_cosine': float(original_similarity.mean()),
            'random_model_mean_cosine': float(random_model.mean()),
            'random_original_mean_cosine': float(random_original.mean()),
            'model_delta_vs_random': float(model_similarity.mean() - random_model.mean()),
            'original_delta_vs_random': float(original_similarity.mean() - random_original.mean()),
            'model_empirical_p': float((1 + np.sum(model_null >= model_similarity.mean())) / (repeats + 1)),
            'original_empirical_p': float((1 + np.sum(original_null >= original_similarity.mean())) / (repeats + 1)),
            'resource_odds_ratio': item['resource_odds'],
            'resource_FDR': item['resource_fdr'],
            'CORUM_odds_ratio': item['resource_odds'] if database == 'CORUM' else np.nan,
            'CORUM_FDR': item['resource_fdr'] if database == 'CORUM' else np.nan,
            'STRING_threshold': STRING_THRESHOLD if database == 'STRING400' else np.nan,
            'top_original_features': '; '.join(
                f'{row.feature} (d={row.cohens_d:.2f})'
                for _, row in top_features.iterrows()
            ),
        })
        for (gene_a, gene_b), model_value, original_value in zip(
            gene_pairs, model_similarity, original_similarity
        ):
            pair_rows.append({
                'case_selection_rank': int(selection_rank),
                'cluster': cluster,
                'database': database,
                'term': term,
                'theme': f'{database}: {term}',
                'complex': term,
                'gene1': gene_a, 'gene2': gene_b,
                'model_cosine': float(model_value),
                'original_cosine': float(original_value),
            })
    return pd.DataFrame(case_rows), pd.DataFrame(pair_rows)


def build_selected_pathway_comparison(
    model_z: np.ndarray,
    original_z: np.ndarray,
    genes: np.ndarray,
) -> pd.DataFrame:
    """Compare user-specified pathway gene pairs in Model and Original."""
    source = pd.read_csv(PATHWAY_SNAPSHOT, usecols=['database', 'pathway', 'genes'])
    requested = set(SELECTED_PATHWAYS_FIG8)
    source = source[source.apply(
        lambda row: (str(row['database']), str(row['pathway'])) in requested,
        axis=1,
    )].copy()
    gene_to_idx = {gene: i for i, gene in enumerate(genes)}
    model_cos = cosine_normalized(model_z)
    original_cos = cosine_normalized(original_z)
    random_model, random_original = reference_pair_pools(model_cos, original_cos)
    rng = np.random.default_rng(SEED + 80)
    rows = []
    for selection_rank, (database, pathway) in enumerate(SELECTED_PATHWAYS_FIG8, start=1):
        match = source[(source['database'] == database) & (source['pathway'] == pathway)]
        if match.empty:
            print(f'Figure 8 pathway not found: {database} | {pathway}')
            continue
        pathway_genes = [
            gene.strip() for gene in str(match.iloc[0]['genes']).split(';')
            if gene.strip() in gene_to_idx
        ]
        pathway_genes = list(dict.fromkeys(pathway_genes))
        gene_pairs = list(combinations(pathway_genes, 2))
        if not gene_pairs:
            print(f'Figure 8 pathway skipped (<2 genes): {database} | {pathway}')
            continue
        first = np.array([gene_to_idx[a] for a, _ in gene_pairs], dtype=int)
        second = np.array([gene_to_idx[b] for _, b in gene_pairs], dtype=int)
        model_values = cosine_pair_scores(model_cos, first, second)
        original_values = cosine_pair_scores(original_cos, first, second)
        model_null = reference_resample_means(random_model, rng, FIG8_RANDOM_REPEATS, len(gene_pairs))
        original_null = reference_resample_means(random_original, rng, FIG8_RANDOM_REPEATS, len(gene_pairs))
        rows.append({
            'selection_rank': selection_rank,
            'database': database,
            'pathway': pathway,
            'n_genes': len(pathway_genes),
            'n_pairs': len(gene_pairs),
            'genes': ';'.join(pathway_genes),
            'model_mean_cosine': float(model_values.mean()),
            'original_mean_cosine': float(original_values.mean()),
            'random_model_mean_cosine': float(random_model.mean()),
            'random_original_mean_cosine': float(random_original.mean()),
            'model_delta_vs_random': float(model_values.mean() - random_model.mean()),
            'original_delta_vs_random': float(original_values.mean() - random_original.mean()),
            'model_empirical_p': float((1 + np.sum(model_null >= model_values.mean())) / (FIG8_RANDOM_REPEATS + 1)),
            'original_empirical_p': float((1 + np.sum(original_null >= original_values.mean())) / (FIG8_RANDOM_REPEATS + 1)),
        })
    return pd.DataFrame(rows)


def term_membership(
    genes: np.ndarray,
    records: dict[str, list[tuple[str, list[str]]]],
) -> dict[str, dict[str, set[str]]]:
    output = {}
    for database, terms in records.items():
        mapping = defaultdict(set)
        for term, members in terms:
            for gene in members:
                mapping[gene].add(term)
        output[database] = mapping
    return output


def build_neighbor_table(
    genes: np.ndarray,
    labels: np.ndarray,
    model_nn: np.ndarray,
    model_nn_sim: np.ndarray,
    original_z: np.ndarray,
    pair_sets: dict[str, set[int]],
    records: dict[str, list[tuple[str, list[str]]]],
) -> pd.DataFrame:
    n = len(genes)
    memberships = term_membership(genes, records)
    original_cos = cosine_normalized(original_z)
    rows = []
    for i, gene in enumerate(genes):
        for rank, (j, model_similarity) in enumerate(
            zip(model_nn[i, :TOP_K_EXPORT], model_nn_sim[i, :TOP_K_EXPORT]),
            start=1,
        ):
            code = int(pair_codes(i, j, n))
            row = {
                'query_gene': gene,
                'neighbor_gene': genes[j],
                'rank': rank,
                'model_cosine': float(model_similarity),
                'original_cosine': float(np.dot(original_cos[i], original_cos[j])),
                'query_cluster': int(labels[i]),
                'neighbor_cluster': int(labels[j]),
                'same_cluster': bool(labels[i] == labels[j]),
                'CORUM_supported': code in pair_sets.get('CORUM', set()),
                'STRING400_supported': code in pair_sets.get('STRING400', set()),
            }
            for database in ('GO_BP', 'GO_CC', 'GO_MF', 'KEGG'):
                shared = sorted(
                    memberships.get(database, {}).get(gene, set())
                    & memberships.get(database, {}).get(genes[j], set())
                )
                row[f'shared_{database}_n'] = len(shared)
                row[f'shared_{database}_terms'] = '; '.join(shared[:5])
            row['biological_support_count'] = int(row['CORUM_supported']) + int(
                row['STRING400_supported']
            ) + sum(row[f'shared_{database}_n'] > 0 for database in ('GO_BP', 'GO_CC', 'GO_MF', 'KEGG'))
            rows.append(row)
    table = pd.DataFrame(rows)
    table.to_parquet(RESULTS / 'gene_neighbor_rank.parquet', index=False)
    table.to_csv(RESULTS / 'gene_neighbor_rank.csv', index=False)
    return table


def query_gene_neighbors(table: pd.DataFrame, gene: str, top_n: int = 20) -> pd.DataFrame:
    gene = str(gene).strip().upper()
    result = table[table['query_gene'].str.upper() == gene].nsmallest(top_n, 'rank')
    if result.empty:
        raise KeyError(f'Gene not found: {gene}')
    return result.reset_index(drop=True)

def top20_database_support_table(
    genes: np.ndarray,
    model_nn: np.ndarray,
    original_nn: np.ndarray,
    pair_sets: dict[str, set[int]],
    top_n: int | None = None,
) -> pd.DataFrame:
    """Count database-supported neighbours among each representation's top-n list.

    A neighbour is supported when its unordered gene pair occurs in at least
    one of CORUM, STRING400, GO_BP, GO_CC, GO_MF or KEGG.
    """
    if top_n is None:
        top_n = TOP_K_EVAL
    n = len(genes)
    supported_union = set()
    for codes in pair_sets.values():
        supported_union.update(codes)
    rows = []
    for representation, nearest in (('DeSCORT', model_nn), ('Original', original_nn)):
        nearest = np.asarray(nearest)[:, :top_n]
        for query_index, neighbours in enumerate(nearest):
            query_codes = pair_codes(
                np.full(len(neighbours), query_index, dtype=np.int64),
                neighbours, n,
            )
            supported = np.fromiter(
                (int(code) in supported_union for code in query_codes),
                dtype=bool, count=len(query_codes),
            )
            row = {
                'gene': str(genes[query_index]),
                'representation': representation,
                'top_n': int(len(neighbours)),
                'database_supported_n': int(supported.sum()),
                'database_supported_fraction': float(supported.mean()) if len(supported) else 0.0,
            }
            for database, database_codes in pair_sets.items():
                row[f'{database}_supported_n'] = int(
                    sum(int(code) in database_codes for code in query_codes)
                )
            rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(RESULTS / 'gene_top20_database_support.csv', index=False)
    table.to_parquet(RESULTS / 'gene_top20_database_support.parquet', index=False)
    return table


def plot_cluster_map(umap_xy, labels, cluster_table):
    """Export cluster coordinates and sizes as two independent figures."""
    fig, ax = plt.subplots(figsize=(5.2, 4.6), constrained_layout=True)
    cmap = plt.get_cmap('tab20', len(np.unique(labels)))
    ax.scatter(umap_xy[:, 0], umap_xy[:, 1], c=labels, cmap=cmap,
               s=9, alpha=.78, linewidths=0, rasterized=True)
    for cluster in sorted(np.unique(labels)):
        center = np.nanmedian(umap_xy[labels == cluster], axis=0)
        ax.text(*center, str(int(cluster)), ha='center', va='center', fontsize=6.5,
                fontweight='bold', color='white', zorder=5,
                bbox=dict(boxstyle='round,pad=0.18', facecolor='#20262A',
                          edgecolor='white', linewidth=.45, alpha=.88))
    ax.set(xlabel='UMAP 1', ylabel='UMAP 2', title='Embedding-only clusters')
    save_figure(fig, FIGURES/'figure1_unsupervised_clusters')
    counts = pd.Series(labels).value_counts().sort_index()
    selected_k = int(cluster_table.loc[cluster_table['selected'], 'k'].iloc[0])
    fig, ax = plt.subplots(figsize=(5.2, 4.0), constrained_layout=True)
    ax.bar(np.arange(len(counts)), counts.values, color='#73808A', width=.8)
    ax.axhline(counts.mean(), color=COLORS['DeSCORT'], lw=1, ls='--')
    ax.set(xlabel='Cluster', ylabel='Genes', title=f'Cluster sizes (k={selected_k})')
    step = max(1, int(np.ceil(len(counts)/25)))
    ax.set_xticks(np.arange(len(counts))[::step], counts.index.astype(str)[::step], rotation=90)
    save_figure(fig, FIGURES/'figure1_cluster_sizes')


def plot_cluster_selection(cluster_table: pd.DataFrame) -> None:
    """Report five-seed stability for the fixed N_CLUSTERS-cluster partition."""
    table = cluster_table.sort_values('k').copy()
    selected_k = int(table.loc[table['selected'], 'k'].iloc[0])
    x = table['k'].to_numpy(dtype=float)
    eligible = table['selection_eligible'].to_numpy(dtype=bool)
    silhouette = table['silhouette_cosine_mean_5_seeds'].to_numpy(dtype=float)
    stability = table['stability_pairwise_ARI_mean_5_seeds'].to_numpy(dtype=float)
    silhouette[~eligible] = np.nan
    stability[~eligible] = np.nan

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), sharex=True)
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.22, top=0.86, wspace=0.34)
    series = [
        (silhouette, 'Cosine silhouette', '#BC3C28'),
        (stability, 'Pairwise ARI', '#0072B5'),
    ]
    for label, ax, (values, ylabel, color) in zip('ab', axes, series):
        ax.plot(x, values, color=color, lw=1.2, marker='o', ms=3.2,
                mec='white', mew=0.35, zorder=3)
        ax.axvline(selected_k, color='#20262A', lw=0.9, ls='--', zorder=1)
        value_at_selected = values[np.flatnonzero(x == selected_k)[0]]
        ax.scatter([selected_k], [value_at_selected], s=42, color='#20262A',
                   edgecolors='white', linewidths=0.6, zorder=4)
        ax.set(xlabel='Candidate number of clusters (k)', ylabel=ylabel,
               ylim=(0, 1.02), xlim=(1.5, 62))
        ax.set_xticks([2, 5, 10, 15, 20, 29, 40, 60])
        ax.grid(axis='y', color='#E7EBEE', lw=0.7)
        ax.set_axisbelow(True)
        panel_label(ax, label)
    axes[0].set_title('Cluster separation', fontsize=9, fontweight='bold')
    axes[1].set_title('Clustering stability', fontsize=9, fontweight='bold')
    fig.text(0.5, 1.045, f'Fixed {N_CLUSTERS}-cluster stability (k={selected_k})',
             ha='center', va='top', fontsize=10, fontweight='bold')
    save_figure(fig, FIGURES / 'figure1_cluster_selection', multi_panel=True)


def plot_pair_support(pair_support: pd.DataFrame) -> None:
    order = ['CORUM', 'STRING400', 'GO_BP', 'GO_CC', 'GO_MF', 'KEGG']
    display_labels = ['CORUM', 'STRING', 'BP', 'CC', 'MF', 'KEGG']
    frame = pair_support.set_index('database').reindex(order).dropna(how='all').reset_index()
    x = np.arange(len(frame))
    width = 0.34
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.8))
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.27, top=0.94, wspace=0.42)
    metrics = [
        ('model_delta_vs_all_pairs', 'original_delta_vs_all_pairs', 'Cosine Δ vs all pairs'),
        ('model_AUROC', 'original_AUROC', 'AUROC'),
        ('model_top20_fold', 'original_top20_fold', 'Top-20 edge fold'),
    ]
    for label, ax, (model_col, original_col, ylabel) in zip('abc', axes, metrics):
        ax.bar(x - width / 2, frame[model_col], width, color=COLORS['DeSCORT'], label='DeSCORT')
        ax.bar(x + width / 2, frame[original_col], width, color=COLORS['Original'], label='Original')
        if 'AUROC' in ylabel:
            ax.axhline(0.5, color=COLORS['Random'], lw=0.9, ls='--')
        if 'fold' in ylabel:
            ax.axhline(1.0, color=COLORS['Random'], lw=0.9, ls='--')
        ax.set_xticks(x, display_labels[:len(frame)], rotation=0, ha='center', fontsize=5)
        ax.set_ylabel(ylabel)
        panel_label(ax, label)
    axes[0].legend(frameon=False, loc='best')
    save_figure(fig, FIGURES / 'figure2_biological_pair_support', multi_panel=True)


def wrap_display_label(value, width=38):
    """Wrap a long name into two balanced lines without dropping text."""
    value = str(value).replace('_', ' ')
    if len(value) <= width:
        return value
    spaces = [m.start() for m in re.finditer(r'\s+', value)]
    if not spaces:
        middle = len(value)//2
        return value[:middle]+'\n'+value[middle:]
    middle = min(spaces, key=lambda i: abs(i-len(value)/2))
    return value[:middle].rstrip()+'\n'+value[middle:].lstrip()


def plot_differential_heatmap(matrix):
    """Portrait heatmap: morphology features on rows, clusters on columns."""
    values = matrix.T
    limit = max(float(np.nanpercentile(np.abs(values.to_numpy()), 98)), 1e-12)
    height = max(11.4, .34*len(values)+1.5)
    g = sns.clustermap(values, cmap='vlag', center=0, vmin=-limit, vmax=limit,
                      row_cluster=len(values)>1, col_cluster=values.shape[1]>1,
                      method='average', metric='euclidean', figsize=(9.2, height),
                      linewidths=0, xticklabels=True, yticklabels=True,
                      dendrogram_ratio=(.08,.06), tree_kws={'linewidths':.7},
                      cbar_kws={'label':"Cohen's d", 'orientation':'horizontal'}, cbar_pos=(.49,.07,.33,.018))
    rows = g.dendrogram_row.reordered_ind if g.dendrogram_row else list(range(len(values)))
    cols = g.dendrogram_col.reordered_ind if g.dendrogram_col else list(range(values.shape[1]))
    # Left-side feature names use outward tick marks aligned to row centers.
    # Mirror the row tree so its leaves meet the right edge of the heatmap.
    g.ax_heatmap.set_position([.46,.19,.40,.69])
    g.ax_row_dendrogram.set_position([.86,.19,.035,.69])
    g.ax_row_dendrogram.invert_xaxis()
    g.ax_col_dendrogram.set_position([.46,.88,.40,.055])
    g.cax.set_position([.49,.07,.33,.018])
    g.cax.tick_params(labelsize=8)
    g.cax.set_xlabel("Cohen's d", fontsize=10, labelpad=8)
    g.ax_heatmap.set_yticks(np.arange(len(rows))+.5)
    g.ax_heatmap.set_yticklabels([wrap_display_label(v) for v in values.index.to_numpy()[rows]],
                                rotation=0, fontsize=8)
    g.ax_heatmap.yaxis.tick_left()
    g.ax_heatmap.yaxis.set_label_position('left')
    g.ax_heatmap.tick_params(axis='y', left=True, right=False,
                            labelleft=True, labelright=False,
                            direction='out', length=5, width=1, pad=5)
    g.ax_heatmap.set_xticklabels(values.columns.to_numpy()[cols], rotation=0, fontsize=8)
    g.ax_heatmap.set_xlabel('Embedding cluster', fontsize=11, labelpad=8)
    g.ax_heatmap.set_ylabel('Original Cell Painting feature', fontsize=11, labelpad=12)
    g.fig.text(.96,.535,'Differential morphology across embedding clusters',
               rotation=270, va='center', ha='center', fontsize=12)
    # Retain the original cluster-by-feature CSV schema for downstream readers.
    matrix.loc[values.columns.to_numpy()[cols], values.index.to_numpy()[rows]].to_csv(
        RESULTS/'differential_feature_heatmap_matrix_clustered.csv')
    save_figure(g.fig, FIGURES/'figure3_differential_feature_heatmap_clustered',
                exclude_axes=[g.ax_row_dendrogram, g.ax_col_dendrogram, g.cax])


def plot_feature_purity(composition: pd.DataFrame, purity: pd.DataFrame) -> None:
    compartment_categories = ['Cells', 'Cytoplasm', 'Nuclei', 'Other']
    channel_categories = ['DNA', 'Mito', 'AGP', 'ER', 'RNA', 'WGA', 'Other']
    channel_colors = {
        'DNA': '#E15759', 'Mito': '#59A14F', 'AGP': '#F28E2B',
        'ER': '#B07AA1', 'RNA': '#4E79A7', 'WGA': '#76B7B2',
        'Other': '#B8BDC2',
    }
    fig, axes = plt.subplots(2, 4, figsize=(9.6, 5.8))
    fig.subplots_adjust(left=0.06, right=0.99, bottom=0.12, top=0.79,
                        wspace=0.42, hspace=0.70)
    for row_index, direction in enumerate(['Up', 'Down']):
        comp = composition[
            (composition['direction'] == direction)
            & (composition['category_type'] == 'compartment')
        ].copy()
        channel = composition[
            (composition['direction'] == direction)
            & (composition['category_type'] == 'channel')
        ].copy()
        comp_pivot = comp.pivot(
            index='cluster', columns='category', values='percentage'
        ).fillna(0).reindex(columns=compartment_categories, fill_value=0).sort_index()
        channel_pivot = channel.pivot(
            index='cluster', columns='category', values='percentage'
        ).fillna(0).reindex(columns=channel_categories, fill_value=0).reindex(
            index=comp_pivot.index, fill_value=0
        )
        compartment_purity = purity[
            (purity['direction'] == direction)
            & (purity['category_type'] == 'compartment')
        ].set_index('cluster').reindex(comp_pivot.index)
        channel_purity = purity[
            (purity['direction'] == direction)
            & (purity['category_type'] == 'channel')
        ].set_index('cluster').reindex(comp_pivot.index)
        x = np.arange(len(comp_pivot))

        bottom = np.zeros(len(comp_pivot))
        for category in compartment_categories:
            values = comp_pivot[category].to_numpy()
            axes[row_index, 0].bar(
                x, values, bottom=bottom, color=COLORS[category],
                label=category, width=0.82,
            )
            bottom += values
        bottom = np.zeros(len(channel_pivot))
        for category in channel_categories:
            values = channel_pivot[category].to_numpy()
            axes[row_index, 1].bar(
                x, values, bottom=bottom, color=channel_colors[category],
                label=category, width=0.82,
            )
            bottom += values
        for ax, title in zip(
            axes[row_index, :2],
            [f'{direction}: compartment mix', f'{direction}: channel mix'],
        ):
            ax.set(xlabel='Cluster', ylabel='Feature fraction (%)',
                   title=title, ylim=(0, 100))
            # Keep the longer titles clear of the y-axis tick labels.
            ax.title.set_position((0.56, 1.0))
            ax.set_xticks(x[::2], comp_pivot.index.astype(str)[::2], rotation=90)

        axes[row_index, 2].scatter(
            compartment_purity.index, compartment_purity['purity'],
            c=[COLORS.get(x, COLORS['Other']) for x in compartment_purity['dominant_category']],
            s=25, edgecolors='white', linewidths=0.4,
        )
        axes[row_index, 2].set(
            xlabel='Cluster', ylabel='Purity',
            title=f'{direction}: compartment purity', ylim=(0, 1.05),
        )
        axes[row_index, 2].set_xticks(x[::2], comp_pivot.index.astype(str)[::2], rotation=90)

        axes[row_index, 3].scatter(
            channel_purity.index, channel_purity['purity'],
            c=[channel_colors.get(x, channel_colors['Other']) for x in channel_purity['dominant_category']],
            s=25, edgecolors='white', linewidths=0.4,
        )
        axes[row_index, 3].set(
            xlabel='Cluster', ylabel='Purity',
            title=f'{direction}: channel purity', ylim=(0, 1.05),
        )
        axes[row_index, 3].set_xticks(x[::2], comp_pivot.index.astype(str)[::2], rotation=90)
    for label, ax in zip('abcdefgh', axes.ravel()):
        panel_label(ax, label, offset=(-26, 12))
    comp_handles, comp_labels = axes[0, 0].get_legend_handles_labels()
    channel_handles, channel_labels = axes[0, 1].get_legend_handles_labels()
    # Remove legend entries for channels that have no non-zero bar in either direction.
    channel_present = []
    for category in channel_categories:
        present = composition.loc[
            (composition['category_type'] == 'channel')
            & (composition['category'] == category)
            & (composition['percentage'] > 0),
        ]
        if not present.empty:
            channel_present.append(category)
    channel_handles, channel_labels = zip(*[
        (handle, label) for handle, label in zip(channel_handles, channel_labels)
        if label in channel_present
    ]) if channel_present else ([], [])
    fig.legend(
        comp_handles, comp_labels, ncol=4, frameon=False, loc='upper center',
        bbox_to_anchor=(0.25, 0.985), fontsize=6.5,
    )
    fig.legend(
        channel_handles, channel_labels, ncol=max(1, min(7, len(channel_present))), frameon=False, loc='upper center',
        bbox_to_anchor=(0.75, 0.985), fontsize=6.5,
    )
    save_figure(fig, FIGURES / 'figure4_feature_composition_purity', multi_panel=True)
    for direction in ['Up', 'Down']:
        sub = composition[(composition.direction == direction) & (composition.category_type == 'channel')]
        pivot = sub.pivot(index='cluster', columns='category', values='percentage').fillna(0)
        pivot = pivot.reindex(columns=channel_categories, fill_value=0).sort_index()
        present = [c for c in channel_categories if (pivot[c] > 0).any()]
        fig, ax = plt.subplots(figsize=(5.2, 4.0))
        fig.subplots_adjust(left=.14, right=.98, bottom=.16, top=.80)
        x = np.arange(len(pivot)); bottom = np.zeros(len(pivot))
        for category in present:
            ax.bar(x, pivot[category], bottom=bottom, color=channel_colors[category],
                   width=.82, label=category)
            bottom += pivot[category].to_numpy()
        ax.set(xlabel='Cluster', ylabel='Feature fraction (%)', ylim=(0,100),
               title=f'{direction}: channel mix')
        step = max(1, int(np.ceil(len(pivot)/25)))
        ax.set_xticks(x[::step], pivot.index.astype(str)[::step], rotation=90)
        ax.legend(loc='lower center', bbox_to_anchor=(.5,1.12),
                  ncol=max(1,len(present)), frameon=False)
        save_figure(fig, FIGURES/f'figure4_{direction.lower()}_channel_mix')



def plot_pathway_validation(
    co_cluster: pd.DataFrame,
    enrichment: pd.DataFrame,
) -> None:
    """Show the former panel B only: significant cluster–term enrichment tests."""
    order = ['CORUM', 'GO_BP', 'GO_CC', 'GO_MF', 'KEGG']
    labels = ['CORUM', 'BP', 'CC', 'MF', 'KEGG']
    significant = enrichment[enrichment['p_adj_BH_within_database'] < 0.05]
    counts = significant.groupby(['representation', 'database']).size().unstack(fill_value=0)
    counts = counts.reindex(columns=order, fill_value=0)
    count_plot = counts.T
    x = np.arange(len(count_plot))
    width = 0.34
    descort_counts = count_plot['DeSCORT'] if 'DeSCORT' in count_plot else np.zeros(len(count_plot))
    original_counts = count_plot['Original'] if 'Original' in count_plot else np.zeros(len(count_plot))
    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    fig.subplots_adjust(left=0.19, right=0.98, bottom=0.28, top=0.86)
    ax.bar(x - width/2, descort_counts, width, color=COLORS['DeSCORT'], label='DeSCORT')
    ax.bar(x + width/2, original_counts, width, color=COLORS['Original'], label='Original')
    ax.set_xticks(x, labels, rotation=0, ha='center', fontsize=6)
    ax.set(ylabel='Significant cluster–term tests', title='Cluster pathway enrichment')
    ax.legend(frameon=False, loc='upper center', bbox_to_anchor=(.5,-.20), ncol=2, fontsize=7)
    save_figure(fig, FIGURES / 'figure5_cluster_pathway_validation')




def _significance_stars(pvalue: float) -> str:
    if not np.isfinite(pvalue):
        return 'n.s.'
    if pvalue < 1e-4:
        return '****'
    if pvalue < 1e-3:
        return '***'
    if pvalue < 1e-2:
        return '**'
    if pvalue < 5e-2:
        return '*'
    return 'n.s.'


def _draw_paired_distance_panel(
    ax, subset: pd.DataFrame, summary_row: pd.Series, title: str, seed_offset: int = 0,
) -> None:
    model_values = subset['mean_distance_model'].to_numpy(dtype=float)
    original_values = subset['mean_distance_original'].to_numpy(dtype=float)
    values = [model_values, original_values]
    boxplot = ax.boxplot(
        values, positions=[0, 1], widths=0.48, patch_artist=True, showfliers=False,
        medianprops={'color': '#20262A', 'linewidth': 1.25},
        boxprops={'edgecolor': '#505A60', 'linewidth': 0.8},
        whiskerprops={'color': '#505A60', 'linewidth': 0.8},
        capprops={'color': '#505A60', 'linewidth': 0.8},
    )
    for patch, color in zip(boxplot['boxes'], [COLORS['DeSCORT'], COLORS['Original']]):
        patch.set_facecolor(color)
        patch.set_alpha(0.72)
    rng = np.random.default_rng(SEED + 600 + seed_offset)
    jitter_model = np.clip(rng.normal(0, 0.052, len(model_values)), -0.13, 0.13)
    jitter_original = np.clip(rng.normal(1, 0.052, len(original_values)), 0.87, 1.13)
    ax.scatter(jitter_model, model_values, s=10, color=COLORS['DeSCORT'], alpha=0.24, edgecolors='none', rasterized=True, zorder=2)
    ax.scatter(jitter_original, original_values, s=10, color=COLORS['Original'], alpha=0.24, edgecolors='none', rasterized=True, zorder=2)
    ax.scatter([0, 1], [model_values.mean(), original_values.mean()], marker='D', s=30,
               color=[COLORS['DeSCORT'], COLORS['Original']], edgecolor='black', linewidth=0.45, zorder=4)
    combined = np.r_[model_values, original_values]
    ymin, ymax = float(combined.min()), float(combined.max())
    yrange = max(ymax - ymin, 0.05)
    bracket_y = ymax + 0.10 * yrange
    ax.plot([0, 0, 1, 1], [bracket_y - 0.018 * yrange, bracket_y, bracket_y, bracket_y - 0.018 * yrange],
            color='#586168', linewidth=0.8)
    pvalue = float(summary_row['wilcoxon_p_one_sided'])
    ax.text(0.5, bracket_y + 0.080 * yrange, _significance_stars(pvalue), ha='center', va='bottom',
            fontsize=9, fontweight='bold', color='#2F3A40')
    p_label = 'P = NA' if not np.isfinite(pvalue) else f'P = {pvalue:.2g}'
    ax.text(0.5, bracket_y + 0.215 * yrange, p_label, ha='center', va='bottom', fontsize=5.8, color='#59656C')
    ax.set_ylim(ymin - 0.09 * yrange, bracket_y + 0.39 * yrange)
    ax.set_xticks([0, 1], ['DeSCORT', 'Original'])
    ax.set_title(f'{title}  (n={len(subset)})', fontsize=8.5, fontweight='bold', pad=7)
    ax.yaxis.grid(False)
    ax.set_axisbelow(True)
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['left', 'bottom']].set_color('#7A858C')
    ax.tick_params(axis='both', labelsize=7, colors='#34434A')


def plot_cluster_biological_distance_by_database(
    distance_table: pd.DataFrame, distance_summary: pd.DataFrame,
) -> None:
    order = ['CORUM', 'STRING', 'GO_BP', 'GO_CC', 'GO_MF', 'KEGG']
    display_names = {'GO_BP': 'GO BP', 'GO_CC': 'GO CC', 'GO_MF': 'GO MF'}
    fig, axes = plt.subplots(2, 3, figsize=(9.4, 6.0), sharey=False, dpi=300)
    fig.subplots_adjust(left=0.085, right=0.985, bottom=0.10, top=0.90, hspace=0.42, wspace=0.32)
    for index, (ax, group) in enumerate(zip(axes.ravel(), order)):
        subset = distance_table[distance_table['database_group'] == group]
        summary_match = distance_summary[distance_summary['database_group'] == group]
        if subset.empty or summary_match.empty:
            ax.text(0.5, 0.5, 'No eligible units', transform=ax.transAxes, ha='center', va='center')
            ax.set_title(group)
        else:
            _draw_paired_distance_panel(ax, subset, summary_match.iloc[0], display_names.get(group, group), index)
        if index % 3 == 0:
            ax.set_ylabel('Mean within-unit cosine distance', fontsize=7.5)
        panel_label(ax, chr(ord('a') + index), inside=True, offset=(10, -4))
    fig.suptitle('Within-cluster biological compactness by database', fontsize=11, fontweight='bold', y=0.975)
    fig.text(0.5, 0.025, 'Lower distance indicates a more compact biological unit; paired one-sided Wilcoxon test',
             ha='center', va='center', fontsize=6.5, color='#52616A')
    save_figure(fig, FIGURES / 'figure6c_cluster_pathway_distance_by_database', multi_panel=True)


def plot_cluster_biological_distance_overall(
    distance_table: pd.DataFrame, distance_summary: pd.DataFrame,
) -> None:
    summary_match = distance_summary[distance_summary['database_group'] == 'Overall']
    if distance_table.empty or summary_match.empty:
        return
    fig, ax = plt.subplots(figsize=(4.7, 4.7), dpi=300)
    fig.subplots_adjust(left=0.19, right=0.96, bottom=0.20, top=0.86)
    _draw_paired_distance_panel(ax, distance_table, summary_match.iloc[0], 'All biological units', 10)
    ax.set_ylabel('Mean within-unit cosine distance', fontsize=8.5)
    ax.set_title('Overall within-cluster biological compactness', fontsize=10.5, fontweight='bold', pad=10)
    fig.text(0.57, 0.055, 'CORUM + STRING + GO + KEGG',
             ha='center', va='center', fontsize=6.5, color='#52616A')
    save_figure(fig, FIGURES / 'figure6d_cluster_pathway_distance_overall')


def _display_case_database(database: str) -> str:
    return (
        str(database).replace('STRING400', 'STRING')
        .replace('GO_BP', 'GO-BP').replace('GO_CC', 'GO-CC').replace('GO_MF', 'GO-MF')
    )


def _prepare_case_studies(cases: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    import textwrap
    sort_col = 'case_selection_rank' if 'case_selection_rank' in cases.columns else 'cluster'
    case_plot = cases.sort_values(sort_col).reset_index(drop=True).copy()
    labels = []
    for row in case_plot.itertuples():
        database = getattr(row, 'database', 'CORUM')
        term = getattr(row, 'term', getattr(row, 'CORUM_complex', ''))
        labels.append(
            f"{_display_case_database(database)}: "
            + textwrap.fill(str(term).replace('_', ' '), width=34)
        )
    return case_plot, labels


def plot_case_comparison(cases: pd.DataFrame) -> None:
    if cases.empty:
        return
    case_plot, labels = _prepare_case_studies(cases)
    y = np.arange(len(case_plot))
    fig, ax = plt.subplots(figsize=(7.4, max(4.8, 0.46 * len(case_plot) + 2.4)))
    fig.subplots_adjust(left=0.50, right=0.96, bottom=0.16, top=0.90)
    ax.scatter(case_plot['model_delta_vs_random'], y, color=COLORS['DeSCORT'], s=32, zorder=3)
    ax.scatter(case_plot['original_delta_vs_random'], y, color=COLORS['Original'], s=32, zorder=3)
    for yi, row in case_plot.iterrows():
        ax.plot(
            [row['original_delta_vs_random'], row['model_delta_vs_random']], [yi, yi],
            color='#C7CDD1', lw=0.9, zorder=1,
        )
    ax.set_yticks(y, labels)
    ax.margins(y=0.16)
    ax.set(xlabel='Cosine similarity Δ vs all pairs', title='Highest-model-cosine biological cases')
    # Keep the horizontal coordinate anchored at zero without drawing a
    # distracting zero-reference line.
    ax.set_xlim(left=0.0)
    ax.legend(
        handles=[
            Line2D([0], [0], marker='o', color='none', markerfacecolor=COLORS['DeSCORT'], markeredgecolor='none', markersize=6, label='DeSCORT'),
            Line2D([0], [0], marker='o', color='none', markerfacecolor=COLORS['Original'], markeredgecolor='none', markersize=6, label='Original'),
        ],
        frameon=False, loc='upper right', fontsize=7.5, handletextpad=0.35, borderaxespad=0.1,
    )
    ax.spines[['top', 'right']].set_visible(False)
    panel_label(ax, 'a', inside=True, offset=(4, -4))
    save_figure(fig, FIGURES / 'figure6a_case_comparison')


def plot_case_umap(cases: pd.DataFrame, umap_xy: np.ndarray, genes: np.ndarray) -> None:
    if cases.empty:
        return
    import re
    import textwrap
    from scipy.spatial.distance import cdist
    case_plot, _ = _prepare_case_studies(cases)
    gene_to_idx = {gene: i for i, gene in enumerate(genes)}
    records = []
    for _, row in case_plot.iterrows():
        case_genes = [g for g in str(row['overlap_genes']).split(';') if g in gene_to_idx]
        if case_genes:
            database = row.get('database', 'CORUM')
            term = row.get('term', row.get('CORUM_complex', ''))
            records.append((f'{_display_case_database(database)}: {term}', case_genes))
    if not records:
        return
    case_colors = [
        '#3C7D74', '#C17A52', '#6678A5', '#9A6C82', '#76955D',
        '#B45A5A', '#8C8068', '#5E8C9A', '#A58A4A', '#7C6D9A',
    ]
    fig, ax = plt.subplots(figsize=(7.2, 5.5), dpi=300, facecolor='white')
    fig.subplots_adjust(left=0.095, right=0.965, top=0.90, bottom=0.135)
    ax.scatter(
        umap_xy[:, 0], umap_xy[:, 1], s=12, color='#CBD0D4', alpha=0.40,
        edgecolors='none', linewidths=0, rasterized=True, zorder=1,
    )
    anchors = []
    colored_gene_set = set()
    for index, (name, case_genes) in enumerate(records):
        color = case_colors[index]
        xy = umap_xy[[gene_to_idx[g] for g in case_genes]]
        ax.scatter(xy[:, 0], xy[:, 1], s=12, color=color, alpha=0.95, edgecolors='none', rasterized=True, zorder=3)
        colored_gene_set.update(case_genes)
        if len(xy) == 1:
            anchors.append(xy[0])
        else:
            distances = cdist(xy, xy)
            neighbour_count = min(len(xy), max(3, int(np.ceil(0.60 * len(xy)))))
            nearest = np.argsort(distances, axis=1)[:, :neighbour_count]
            compactness = np.take_along_axis(distances, nearest, axis=1).sum(axis=1)
            anchors.append(np.median(xy[nearest[int(np.argmin(compactness))]], axis=0))
    xmin, xmax = umap_xy[:, 0].min(), umap_xy[:, 0].max()
    ymin, ymax = umap_xy[:, 1].min(), umap_xy[:, 1].max()
    ax.set_xlim(xmin - 0.42, xmax + 0.42)
    ax.set_ylim(ymin - 0.38, ymax + 0.38)
    ax.set_xlabel('UMAP 1', fontsize=9)
    ax.set_ylabel('UMAP 2', fontsize=9)
    ax.set_title(f'{ARGS.dataset_name} selected genes', fontsize=14, fontweight='bold', pad=14)
    ax.tick_params(labelsize=8, width=0.7, length=3)
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['left', 'bottom']].set_linewidth(0.7)

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    colored_xy = umap_xy[[gene_to_idx[g] for g in colored_gene_set]]
    colored_pixels = ax.transData.transform(colored_xy)
    pixels_per_point = fig.dpi / 72.0
    directions = [(0, 1), (1, 1), (-1, 1), (1, 0), (-1, 0), (1, -1), (-1, -1), (0, -1)]
    candidate_offsets = [(distance * dx, distance * dy) for distance in (10, 20, 32, 46, 61, 78) for dx, dy in directions]
    accepted_bboxes, label_rows = [], []
    anchor_pixels = ax.transData.transform(np.asarray(anchors))
    if len(anchors) > 1:
        crowding = cdist(anchor_pixels, anchor_pixels)
        np.fill_diagonal(crowding, np.inf)
        placement_order = np.argsort(np.min(crowding, axis=1))
    else:
        placement_order = np.arange(len(anchors))
    for index in placement_order:
        name, case_genes = records[index]
        color = case_colors[index]
        rgb = np.asarray(mpl.colors.to_rgb(color))
        text_color = tuple(np.clip(rgb * 0.76, 0, 1))
        clean_name = re.sub(r'\s*\(CORUM:\s*\d+\)', '', name).replace('_', ' ')
        label = ax.text(anchors[index][0], anchors[index][1], '\n'.join(textwrap.wrap(clean_name, width=22, break_long_words=False, break_on_hyphens=False)), ha='center', va='center', fontsize=7.4, fontweight='semibold', linespacing=1.08, color=text_color, zorder=5)
        best = None
        for dx_pt, dy_pt in candidate_offsets:
            candidate_pixel = anchor_pixels[index] + pixels_per_point * np.array([dx_pt, dy_pt])
            candidate_data = ax.transData.inverted().transform(candidate_pixel)
            label.set_position(candidate_data)
            bbox = label.get_window_extent(renderer).expanded(1.08, 1.14)
            if not (bbox.x0 >= ax.bbox.x0 + 3 and bbox.x1 <= ax.bbox.x1 - 3 and bbox.y0 >= ax.bbox.y0 + 3 and bbox.y1 <= ax.bbox.y1 - 3):
                continue
            overlap = sum(max(0, min(bbox.x1, prior.x1) - max(bbox.x0, prior.x0)) * max(0, min(bbox.y1, prior.y1) - max(bbox.y0, prior.y0)) for prior in accepted_bboxes)
            point_cover = np.count_nonzero((colored_pixels[:, 0] >= bbox.x0) & (colored_pixels[:, 0] <= bbox.x1) & (colored_pixels[:, 1] >= bbox.y0) & (colored_pixels[:, 1] <= bbox.y1))
            distance_pt = np.hypot(dx_pt, dy_pt)
            cost = 35 * overlap + 20 * point_cover + 0.8 * distance_pt
            if best is None or cost < best[0]:
                best = (cost, candidate_data, bbox, overlap, point_cover, distance_pt)
        if best is None:
            best_position = anchors[index]
            label.set_position(best_position)
            bbox = label.get_window_extent(renderer).expanded(1.08, 1.14)
            best = (np.inf, best_position, bbox, np.nan, np.nan, 0)
        label.set_position(best[1])
        accepted_bboxes.append(best[2])
        label_rows.append({'case': name, 'n_genes': len(case_genes), 'anchor_umap1': anchors[index][0], 'anchor_umap2': anchors[index][1], 'label_umap1': best[1][0], 'label_umap2': best[1][1], 'label_shift_pt': best[5], 'label_overlap_px2': best[3], 'colored_points_covered': best[4]})
    pd.DataFrame(label_rows).sort_values('case').to_csv(FIGURES / 'figure6b_case_umap.labels.csv', index=False)
    save_figure(fig, FIGURES / 'figure6b_case_umap')


def _prepare_selected_pathways(selected: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    import textwrap
    selected = selected.sort_values('selection_rank').reset_index(drop=True)
    labels = [
        f"{row.database}: {textwrap.fill(str(row.pathway).replace('_', ' '), width=36)}"
        for row in selected.itertuples()
    ]
    return selected, labels


def plot_selected_pathway_comparison(selected: pd.DataFrame) -> None:
    if selected.empty:
        return
    selected, labels = _prepare_selected_pathways(selected)
    y = np.arange(len(selected))
    fig, ax = plt.subplots(figsize=(7.4, max(4.8, 0.34 * len(selected) + 2.8)))
    fig.subplots_adjust(left=0.50, right=0.96, bottom=0.16, top=0.90)

    ax.scatter(selected['model_delta_vs_random'], y, s=32, color=COLORS['DeSCORT'], zorder=3)
    ax.scatter(selected['original_delta_vs_random'], y, s=32, color=COLORS['Original'], zorder=3)
    for yi, row in selected.iterrows():
        ax.plot(
            [row['original_delta_vs_random'], row['model_delta_vs_random']], [yi, yi],
            color='#C7CDD1', lw=0.9, zorder=1,
        )
    ax.set_yticks(y, labels)
    ax.set(xlabel='Cosine similarity Δ vs all pairs', title='Specified pathway comparison')
    # Anchor the coordinate at zero while omitting a distracting zero line.
    ax.set_xlim(left=0.0)
    ax.legend(
        handles=[
            Line2D([0], [0], marker='o', color='none', markerfacecolor=COLORS['DeSCORT'], markeredgecolor='none', markersize=6, label='DeSCORT'),
            Line2D([0], [0], marker='o', color='none', markerfacecolor=COLORS['Original'], markeredgecolor='none', markersize=6, label='Original'),
        ],
        frameon=False, loc='upper right', fontsize=7.5, handletextpad=0.35, borderaxespad=0.1,
    )
    ax.spines[['top', 'right']].set_visible(False)
    panel_label(ax, 'a', inside=True, offset=(4, -4))
    save_figure(fig, FIGURES / 'figure8a_specified_pathway_comparison')


def plot_selected_pathway_umap(
    selected: pd.DataFrame,
    umap_xy: np.ndarray,
    genes: np.ndarray,
) -> None:
    if selected.empty:
        return
    import re
    import textwrap
    from scipy.spatial.distance import cdist

    selected, _ = _prepare_selected_pathways(selected)
    gene_to_idx = {gene: i for i, gene in enumerate(genes)}
    pathway_rows = []
    for row in selected.itertuples():
        pathway_genes = [g for g in str(row.genes).split(';') if g in gene_to_idx]
        if pathway_genes:
            pathway_rows.append((row.database, row.pathway, pathway_genes))
    if not pathway_rows:
        return

    # Palette and geometry follow old_umap_A549.py.
    pathway_colors = [
        '#3C7D74', '#C17A52', '#6678A5', '#9A6C82', '#76955D', '#B45A5A',
        '#8C8068', '#5E8C9A', '#A58A4A', '#7C6D9A', '#4F8A6D', '#B06F7A',
    ]
    if len(pathway_rows) > len(pathway_colors):
        raise ValueError('More selected pathways than available direct-label colors.')

    fig, ax = plt.subplots(figsize=(7.2, 5.5), dpi=300, facecolor='white')
    fig.subplots_adjust(left=0.095, right=0.965, top=0.90, bottom=0.135)
    ax.scatter(
        umap_xy[:, 0], umap_xy[:, 1], s=12, color='#CBD0D4', alpha=0.40,
        edgecolors='none', linewidths=0, rasterized=True, zorder=1,
    )

    anchors = []
    colored_gene_set = set()
    for index, (database, pathway_name, pathway_genes) in enumerate(pathway_rows):
        color = pathway_colors[index]
        idx = [gene_to_idx[g] for g in pathway_genes]
        xy = umap_xy[idx]
        ax.scatter(
            xy[:, 0], xy[:, 1], s=12, color=color, alpha=0.95,
            edgecolors='none', zorder=3, rasterized=True,
        )
        colored_gene_set.update(pathway_genes)
        if len(xy) == 1:
            anchors.append(xy[0])
        else:
            distances = cdist(xy, xy)
            neighbour_count = min(len(xy), max(3, int(np.ceil(0.60 * len(xy)))))
            nearest = np.argsort(distances, axis=1)[:, :neighbour_count]
            compactness = np.take_along_axis(distances, nearest, axis=1).sum(axis=1)
            medoid = int(np.argmin(compactness))
            anchors.append(np.median(xy[nearest[medoid]], axis=0))

    xmin, xmax = umap_xy[:, 0].min(), umap_xy[:, 0].max()
    ymin, ymax = umap_xy[:, 1].min(), umap_xy[:, 1].max()
    ax.set_xlim(xmin - 0.42, xmax + 0.42)
    ax.set_ylim(ymin - 0.38, ymax + 0.38)
    ax.set_xlabel('UMAP 1', fontsize=9)
    ax.set_ylabel('UMAP 2', fontsize=9)
    ax.set_title(f'{ARGS.dataset_name} selected genes', fontsize=14, fontweight='bold', pad=14)
    ax.tick_params(labelsize=8, width=0.7, length=3)
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['left', 'bottom']].set_linewidth(0.7)

    # Direct labels are placed near the densest local group, as in the old plot.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    colored_genes = list(colored_gene_set)
    colored_xy = umap_xy[[gene_to_idx[g] for g in colored_genes]]
    colored_pixels = ax.transData.transform(colored_xy)
    pixels_per_point = fig.dpi / 72.0
    directions = [(0, 1), (1, 1), (-1, 1), (1, 0), (-1, 0), (1, -1), (-1, -1), (0, -1)]
    candidate_offsets = [
        (distance * dx, distance * dy)
        for distance in (10, 20, 32, 46, 61, 78)
        for dx, dy in directions
    ]
    accepted_bboxes = []
    label_rows = []
    anchor_pixels = ax.transData.transform(np.asarray(anchors))
    if len(anchors) > 1:
        crowding = cdist(anchor_pixels, anchor_pixels)
        np.fill_diagonal(crowding, np.inf)
        placement_order = np.argsort(np.min(crowding, axis=1))
    else:
        placement_order = np.arange(len(anchors))

    for index in placement_order:
        database, pathway_name, pathway_genes = pathway_rows[index]
        color = pathway_colors[index]
        rgb = np.asarray(mpl.colors.to_rgb(color))
        text_color = tuple(np.clip(rgb * 0.76, 0, 1))
        clean_name = re.sub(r'\s*\(GO:\d+\)', '', str(pathway_name)).replace('_', ' ')
        label = ax.text(
            anchors[index][0], anchors[index][1],
            '\n'.join(textwrap.wrap(clean_name, width=22, break_long_words=False, break_on_hyphens=False)),
            ha='center', va='center', fontsize=7.4, fontweight='semibold',
            linespacing=1.08, color=text_color, zorder=5,
        )
        best = None
        for dx_pt, dy_pt in candidate_offsets:
            candidate_pixel = anchor_pixels[index] + pixels_per_point * np.array([dx_pt, dy_pt])
            candidate_data = ax.transData.inverted().transform(candidate_pixel)
            label.set_position(candidate_data)
            bbox = label.get_window_extent(renderer).expanded(1.08, 1.14)
            inside = (
                bbox.x0 >= ax.bbox.x0 + 3 and bbox.x1 <= ax.bbox.x1 - 3 and
                bbox.y0 >= ax.bbox.y0 + 3 and bbox.y1 <= ax.bbox.y1 - 3
            )
            if not inside:
                continue
            overlap = 0.0
            for prior in accepted_bboxes:
                overlap_x = max(0, min(bbox.x1, prior.x1) - max(bbox.x0, prior.x0))
                overlap_y = max(0, min(bbox.y1, prior.y1) - max(bbox.y0, prior.y0))
                overlap += overlap_x * overlap_y
            point_cover = np.count_nonzero(
                (colored_pixels[:, 0] >= bbox.x0) & (colored_pixels[:, 0] <= bbox.x1) &
                (colored_pixels[:, 1] >= bbox.y0) & (colored_pixels[:, 1] <= bbox.y1)
            )
            distance_pt = np.hypot(dx_pt, dy_pt)
            cost = 35 * overlap + 20 * point_cover + 0.8 * distance_pt
            if best is None or cost < best[0]:
                best = (cost, candidate_data, bbox, overlap, point_cover, distance_pt)
        if best is None:
            best_position = anchors[index]
            label.set_position(best_position)
            bbox = label.get_window_extent(renderer).expanded(1.08, 1.14)
            best = (np.inf, best_position, bbox, np.nan, np.nan, 0)
        label.set_position(best[1])
        accepted_bboxes.append(best[2])
        label_rows.append({
            'database': database, 'pathway': pathway_name, 'n_genes': len(pathway_genes),
            'anchor_umap1': anchors[index][0], 'anchor_umap2': anchors[index][1],
            'label_umap1': best[1][0], 'label_umap2': best[1][1],
            'label_shift_pt': best[5], 'label_overlap_px2': best[3],
            'colored_points_covered': best[4],
        })

    label_table = pd.DataFrame(label_rows).sort_values(['database', 'pathway']).reset_index(drop=True)
    label_table.to_csv(FIGURES / 'figure8b_specified_pathways_umap.labels.csv', index=False)
    save_figure(fig, FIGURES / 'figure8b_specified_pathways_umap')






def _dotplot_display_feature(feature: str) -> str:
    """Shorten repeated Cell Painting tokens while keeping feature identity readable."""
    replacements = {
        'Cytoplasm_': 'Cyto_',
        'Correlation_': 'Corr_',
        'RadialDistribution_': 'Radial_',
        'AngularSecondMoment': 'ASM',
        'DifferenceVariance': 'DiffVar',
        'DifferenceEntropy': 'DiffEnt',
        'InverseDifferenceMoment': 'InvDiffMom',
    }
    label = str(feature)
    for old, new in replacements.items():
        label = label.replace(old, new)
    return label


def differential_dotplot_tables(
    differential: pd.DataFrame,
    original: pd.DataFrame,
    labels: np.ndarray,
    n_top: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build long-form dotplot data for per-cluster up/down morphology markers.

    Features are selected independently for every embedding cluster and direction
    using the same thresholds as the composition analysis. Each selected feature
    is then evaluated across all clusters, so the dotplot shows both strength and
    cluster specificity rather than only the source cluster's value.
    """
    required = {
        'cluster', 'feature', 'mean_difference', 'cohens_d',
        'p_adj_BH_within_cluster',
    }
    missing = required.difference(differential.columns)
    if missing:
        raise KeyError(f'Missing differential columns: {sorted(missing)}')
    if n_top < 1:
        raise ValueError('n_top must be at least 1')
    labels = np.asarray(labels, dtype=int)
    if len(original) != len(labels):
        raise ValueError('original rows and cluster labels are not aligned')

    clusters = sorted(differential['cluster'].astype(int).unique())
    significant = differential[
        (differential['p_adj_BH_within_cluster'] < ARGS.de_fdr)
        & (differential['cohens_d'].abs() >= ARGS.min_cohens_d)
        & (differential['mean_difference'].abs() >= ARGS.min_mean_difference)
    ].copy()

    selections = []
    for direction in ('Up', 'Down'):
        for cluster in clusters:
            candidates = significant[significant['cluster'].astype(int) == cluster]
            if direction == 'Up':
                candidates = candidates[candidates['cohens_d'] > 0].nlargest(
                    n_top, 'cohens_d'
                )
            else:
                candidates = candidates[candidates['cohens_d'] < 0].nsmallest(
                    n_top, 'cohens_d'
                )
            for rank, row in enumerate(candidates.itertuples(index=False), start=1):
                selections.append({
                    'direction': direction,
                    'source_cluster': int(cluster),
                    'source_rank': int(rank),
                    'feature': str(row.feature),
                    'source_cohens_d': float(row.cohens_d),
                    'source_mean_difference': float(row.mean_difference),
                    'source_p_adj': float(row.p_adj_BH_within_cluster),
                })
    selection = pd.DataFrame(selections)
    if selection.empty:
        raise RuntimeError('No significant differential features available for dotplot')

    lookup = differential.set_index(['cluster', 'feature'])[
        ['cohens_d', 'mean_difference', 'p_adj_BH_within_cluster']
    ]
    dot_rows = []
    for direction, direction_selection in selection.groupby('direction', sort=False):
        direction_selection = direction_selection.sort_values(
            ['source_cluster', 'source_rank'], kind='stable'
        ).reset_index(drop=True)
        for column_index, selected in direction_selection.iterrows():
            for target_cluster in clusters:
                values = lookup.loc[(target_cluster, selected['feature'])]
                dot_rows.append({
                    'direction': direction,
                    'column_index': int(column_index),
                    'source_cluster': int(selected['source_cluster']),
                    'source_rank': int(selected['source_rank']),
                    'target_cluster': int(target_cluster),
                    'feature': selected['feature'],
                    'feature_label': _dotplot_display_feature(selected['feature']),
                    'cohens_d': float(values['cohens_d']),
                    'mean_difference': float(values['mean_difference']),
                    'p_adj_BH_within_cluster': float(
                        values['p_adj_BH_within_cluster']
                    ),
                })
    dot_table = pd.DataFrame(dot_rows)
    selection.to_csv(RESULTS / 'differential_dotplot_feature_selection.csv', index=False)
    dot_table.to_csv(RESULTS / 'differential_dotplot_values.csv', index=False)
    return dot_table, selection


def plot_differential_dotplots(
    differential: pd.DataFrame,
    original: pd.DataFrame,
    labels: np.ndarray,
    n_top: int = 5,
) -> None:
    """Render ordered up/down bubble heatmaps without clustering."""
    dot_table, selection = differential_dotplot_tables(
        differential, original, labels, n_top=n_top
    )
    cluster_order = sorted(dot_table['target_cluster'].astype(int).unique())
    effect_limit = float(np.nanpercentile(np.abs(differential['cohens_d']), 98))
    effect_limit = max(effect_limit, 0.5)

    for direction in ('Up', 'Down'):
        selected = (
            selection[selection['direction'] == direction]
            .sort_values(['source_cluster', 'source_rank'], kind='stable')
            .reset_index(drop=True)
        )
        frame = dot_table[dot_table['direction'] == direction].copy()
        column_map = {
            (int(row.source_cluster), int(row.source_rank), str(row.feature)): idx
            for idx, row in enumerate(selected.itertuples(index=False))
        }
        frame['column_index'] = [
            column_map[(int(row.source_cluster), int(row.source_rank), str(row.feature))]
            for row in frame.itertuples(index=False)
        ]
        column_order = np.arange(len(selected), dtype=int)
        effect_matrix = frame.pivot(
            index='target_cluster', columns='column_index', values='cohens_d'
        ).reindex(index=cluster_order, columns=column_order)
        feature_labels = [
            _dotplot_display_feature(feature)
            for feature in selected['feature'].astype(str)
        ]
        width = max(13.0, 0.19 * len(selected) + 2.4)
        height = max(5.4, 0.20 * len(cluster_order) + 1.4)
        fig, ax = plt.subplots(figsize=(width, height))
        fig.subplots_adjust(left=0.055, right=0.865, bottom=0.39, top=0.86)
        cbar_ax = fig.add_axes([0.91, 0.16, 0.012, 0.27])
        sns.heatmap(
            effect_matrix, ax=ax, cmap='vlag', center=0,
            vmin=-effect_limit, vmax=effect_limit, linewidths=0,
            xticklabels=feature_labels, yticklabels=cluster_order,
            cbar=True, cbar_ax=cbar_ax, cbar_kws={'label': "Cohen's d"},
        )

        ax.set_xticklabels(
            feature_labels, rotation=90, rotation_mode='anchor',
            ha='right', va='center', fontsize=5,
        )
        ax.set_yticklabels([str(x) for x in cluster_order], rotation=0, fontsize=6)
        ax.set_xlabel('Top differential Cell Painting features, grouped by source cluster')
        ax.set_ylabel('Embedding cluster')
        ax.set_title(
            f'Top-{n_top} {direction.lower()}-shifted morphology features per cluster',
            fontsize=9, fontweight='bold', pad=20,
        )

        group_counts = selected.groupby('source_cluster', sort=True).size()
        start = 0
        for source_cluster, count in group_counts.items():
            end = start + int(count)
            if start > 0:
                ax.axvline(start, color='#F3F3F3', lw=0.7, zorder=4)
            ax.plot(
                [start + 0.08, start + 0.08, end - 0.08, end - 0.08],
                [1.006, 1.018, 1.018, 1.006],
                transform=ax.get_xaxis_transform(), color='#303030',
                lw=0.75, clip_on=False,
            )
            ax.text(
                (start + end) / 2, 1.032, str(int(source_cluster)),
                transform=ax.get_xaxis_transform(), ha='center', va='bottom',
                fontsize=6, color='#202020',
            )
            start = end

        cbar_ax.tick_params(labelsize=6)
        cbar_ax.yaxis.label.set_size(7)
        save_figure(
            fig,
            FIGURES / f'figure3b_differential_feature_dotplot_{direction.lower()}',
            exclude_axes=[cbar_ax],
        )


def export_gene_queries(neighbor_table, query_genes, top_n):
    """Save one query per requested gene plus a combined table and missing-gene audit."""
    queries, missing, found = [], [], []
    for gene in query_genes:
        try:
            query = query_gene_neighbors(neighbor_table, gene, top_n)
        except KeyError:
            missing.append(gene)
            logging.warning('Query gene not found in the analyzed gene set: %s', gene)
            continue
        query.to_csv(RESULTS/f'query_{gene}.csv', index=False)
        query.to_csv(RESULTS/f'query_example_{gene}.csv', index=False)
        queries.append(query); found.append(gene)
    pd.DataFrame({'gene': missing}).to_csv(RESULTS/'query_missing_genes.csv', index=False)
    combined = pd.concat(queries, ignore_index=True) if queries else neighbor_table.iloc[:0].copy()
    combined.to_csv(RESULTS/'query_all_genes.csv', index=False)
    if not found:
        raise ValueError('None of the requested query genes were found; see query_missing_genes.csv')
    return found




def redraw_revised_figures():
    """Redraw the revised panels from saved results without rerunning analysis."""
    coordinates = pd.read_csv(RESULTS/'embedding_cluster_umap.csv')
    plot_cluster_map(coordinates[['UMAP1','UMAP2']].to_numpy(), coordinates.cluster.to_numpy(),
                     pd.read_csv(RESULTS/'embedding_cluster_selection_extended.csv'))
    plot_differential_heatmap(pd.read_csv(RESULTS/'differential_feature_heatmap_matrix.csv', index_col=0))
    plot_feature_purity(pd.read_csv(RESULTS/'cluster_feature_composition.csv'),
                        pd.read_csv(RESULTS/'cluster_feature_purity.csv'))
