#!/usr/bin/env python3
"""Compare cross-modality profiles and run packaged temporal retrieval.

Configure all paths and run options in 06_compare.sh. The pooled-profile and
time-stratified retrieval analyses intentionally retain separate definitions.
"""
from pathlib import Path
import json
import os
import re
import sys
import platform
import hashlib

os.environ.setdefault('MPLCONFIGDIR', '/tmp/mpl_cross_modality_compare')
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import scipy
from scipy.stats import hypergeom, kendalltau, rankdata, spearmanr, wilcoxon
import sklearn
from sklearn.decomposition import PCA


import argparse
import subprocess
import shutil
from threadpoolctl import threadpool_limits

MODALITIES = ("Compound", "Crispr", "Orf")
CONTROL_NAMES = {"nontargeting", "non-targeting", "control", "ctrl", "dmso"}
CRISPR_GUIDE = "Metadata_Target_Sequence"
PROFILE_KEYS = ["Metadata_All_Modality", "Metadata_Treatment", "Metadata_Pert_IName"]


def _read_wells(data_dir: Path):
    data_dir = Path(data_dir).resolve()
    parquet_files = sorted(data_dir.rglob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files under {data_dir}")

    raw_parts, source_manifest = [], []
    embedding_columns = None
    for path in parquet_files:
        relative = path.relative_to(data_dir)
        modality = relative.parts[0]
        if modality not in MODALITIES or len(relative.parts) < 2:
            raise ValueError(f"Unexpected parquet layout: {relative}")
        frame = pd.read_parquet(path)
        features = sorted(
            (column for column in frame if re.fullmatch(r"emb_\d+", str(column))),
            key=lambda column: int(str(column).split("_")[-1]),
        )
        if not features or (embedding_columns is not None and features != embedding_columns):
            raise ValueError(f"Missing or inconsistent embedding columns in {relative}")
        if embedding_columns is None:
            embedding_columns = features

        required = ["Metadata_Treatment"]
        if modality == "Compound":
            required.append("Metadata_Pert_IName")
        elif modality == "Crispr":
            required.extend([CRISPR_GUIDE, "Metadata_Broad_Sample"])
        missing = set(required).difference(frame.columns)
        if missing:
            raise ValueError(f"{relative}: missing metadata {sorted(missing)}")

        work = frame[required + features].copy()
        for column in required:
            values = work[column].astype("string").str.strip()
            if column == "Metadata_Pert_IName":
                is_control = (work["Metadata_Treatment"].astype("string").str.strip()
                              .str.lower().isin(CONTROL_NAMES))
                values = values.where(~is_control, "nontargeting")
            if values.isna().any() or values.eq("").any():
                raise ValueError(f"{relative}: null or empty values in {column}")
            work[column] = values.astype(str)
        if not np.isfinite(work[features].to_numpy(dtype=np.float32)).all():
            raise ValueError(f"{relative}: non-finite embeddings")

        work["Metadata_All_Modality"] = modality
        work["Metadata_Source_Parquet"] = relative.as_posix()
        raw_parts.append(work)
        source_manifest.append({"source_file": relative.as_posix(), "modality": modality,
                                "input_wells": len(work)})

    raw_wells = pd.concat(raw_parts, ignore_index=True)
    manifest = pd.DataFrame(source_manifest)
    if len(raw_wells) != int(manifest["input_wells"].sum()):
        raise AssertionError("Parquet manifest row counts do not match loaded wells")
    return raw_wells, embedding_columns, manifest, parquet_files


def load_model_profiles(data_dir: Path):
    """Pool all plates, then apply the model-profile aggregation used in this project."""
    raw_wells, embedding_columns, manifest, parquet_files = _read_wells(data_dir)
    profile_parts, aggregation_rows = [], []
    for modality in MODALITIES:
        work = raw_wells.loc[raw_wells["Metadata_All_Modality"].eq(modality)].copy()
        if work.empty:
            raise ValueError(f"No wells found for modality {modality}")
        if modality == "Compound":
            keys = ["Metadata_Treatment", "Metadata_Pert_IName"]
            rule = "all-well mean by Treatment + Pert_IName"
        elif modality == "Crispr":
            sample_guides = work[["Metadata_Broad_Sample", CRISPR_GUIDE]].drop_duplicates()
            if (sample_guides.groupby("Metadata_Broad_Sample")[CRISPR_GUIDE].nunique() > 1).any():
                raise ValueError("A Crispr Broad_Sample has multiple guide sequences")
            keys = ["Metadata_Treatment", CRISPR_GUIDE]
            rule = "all-well guide mean by Treatment + Target_Sequence, then Treatment median"
        else:
            keys = ["Metadata_Treatment"]
            rule = "all-well mean by Treatment"

        grouped = work.groupby(keys, sort=True, dropna=False, observed=True)
        intermediate = grouped[embedding_columns].mean().reset_index()
        counts = grouped.agg(
            n_input_wells=("Metadata_Treatment", "size"),
            n_source_files=("Metadata_Source_Parquet", "nunique"),
        ).reset_index()
        intermediate = intermediate.merge(counts, on=keys, validate="one_to_one")

        if modality in ("Compound", "Orf"):
            final = intermediate.copy()
            final["n_first_level_units"] = 1
            if modality == "Orf":
                final["Metadata_Pert_IName"] = final["Metadata_Treatment"]
        else:
            final = (intermediate.groupby("Metadata_Treatment", sort=True, observed=True)
                     [embedding_columns].median().reset_index())
            final["Metadata_Pert_IName"] = final["Metadata_Treatment"]
            guide_counts = (intermediate.groupby("Metadata_Treatment", observed=True)
                            .size().rename("n_first_level_units").reset_index())
            final = final.merge(guide_counts, on="Metadata_Treatment", validate="one_to_one")
            raw_counts = work.groupby("Metadata_Treatment", observed=True).agg(
                n_input_wells=("Metadata_Treatment", "size"),
                n_source_files=("Metadata_Source_Parquet", "nunique"),
            ).reset_index()
            final = final.merge(raw_counts, on="Metadata_Treatment", validate="one_to_one")

        final["Metadata_All_Modality"] = modality
        final["aggregation_rule"] = rule
        final[embedding_columns] = final[embedding_columns].astype(np.float32)
        if int(final["n_input_wells"].sum()) != len(work):
            raise AssertionError(f"{modality}: aggregation lost or duplicated wells")
        profile_parts.append(final)
        aggregation_rows.append({"modality": modality, "input_wells": len(work),
                                 "first_level_units": len(intermediate),
                                 "final_profiles": len(final), "aggregation_rule": rule})

    profiles = pd.concat(profile_parts, ignore_index=True)[
        PROFILE_KEYS + embedding_columns
        + ["n_source_files", "n_input_wells", "n_first_level_units", "aggregation_rule"]
    ].sort_values(PROFILE_KEYS).reset_index(drop=True)
    if profiles.duplicated(PROFILE_KEYS).any():
        raise ValueError("Duplicate final modality/treatment/perturbation profiles")
    if not np.isfinite(profiles[embedding_columns].to_numpy(dtype=np.float32)).all():
        raise ValueError("Non-finite values in aggregated profiles")

    summary = {
        "input_dir": str(Path(data_dir).resolve()),
        "input_files": len(parquet_files),
        "input_wells": int(manifest["input_wells"].sum()),
        "aggregation": "all plates pooled; Crispr guide mean -> gene median; Compound Treatment+Pert_IName mean; Orf Treatment mean",
        "aggregation_by_modality": aggregation_rows,
        "crispr_guide_column": CRISPR_GUIDE,
        "feature_columns": len(embedding_columns),
        "final_profiles": len(profiles),
    }
    return profiles, summary, embedding_columns, manifest, parquet_files


def pair_identities(profiles: pd.DataFrame) -> pd.DataFrame:
    """Create Gene/Compound matching rows; distances are recomputed by the notebook."""
    records = []
    for gene, group in profiles.groupby("Metadata_Treatment", sort=True, observed=True):
        compounds = group.loc[group["Metadata_All_Modality"].eq("Compound")]
        crispr = group.loc[group["Metadata_All_Modality"].eq("Crispr")]
        orf = group.loc[group["Metadata_All_Modality"].eq("Orf")]
        if compounds.empty or len(crispr) != 1 or len(orf) != 1:
            continue
        for compound in compounds["Metadata_Pert_IName"]:
            records.append({
                "Gene": gene,
                "Compound": compound,
                "Crispr": crispr.iloc[0]["Metadata_Pert_IName"],
                "Orf": orf.iloc[0]["Metadata_Pert_IName"],
            })
    pairs = pd.DataFrame.from_records(records)
    if pairs.empty:
        raise ValueError("No matched Compound/Crispr/Orf identities in model input")
    if pairs.duplicated(["Gene", "Compound"]).any():
        raise ValueError("Duplicate Compound/Gene identities in model input")
    return pairs.sort_values(["Gene", "Compound"]).reset_index(drop=True)


def sha256_file_set(files, root: Path) -> str:
    """Hash a file set reproducibly without loading whole parquet files into RAM."""
    digest = hashlib.sha256()
    root = Path(root).resolve()
    for path in sorted(map(Path, files)):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def calculate_top_k_overlap(rankings, comparisons, k_step=5, k_max=100):
    """Compare ranked gene lists on one shared universe at configured thresholds."""
    if k_step < 1 or k_max < k_step or k_max % k_step:
        raise ValueError("Overlap curve max must be a positive multiple of the step")
    universe = set.intersection(*(set(values) for values in rankings.values()))
    if len(universe) < k_max:
        raise ValueError(f"Top-k overlap through {k_max} requires at least {k_max} shared genes; found {len(universe)}")
    lists = {name: [gene for gene in values if gene in universe] for name, values in rankings.items()}
    if any(len(values) != len(universe) for values in lists.values()):
        raise ValueError("Rankings must contain unique gene identifiers")
    rows = []
    for label, left, right in comparisons:
        for k in range(k_step, k_max + 1, k_step):
            shared = sorted(set(lists[left][:k]) & set(lists[right][:k]))
            rows.append(dict(comparison=label, top_k=k, intersection_n=len(shared),
                             overlap_fraction=len(shared) / k, universe_n=len(universe),
                             chance_expected_fraction=k / len(universe), shared_genes=";".join(map(str, shared))))
    return pd.DataFrame(rows)


def style_profile_comparison_axis(fig, ax, title, ylabel):
    """Use identical physical dimensions and typography for paired comparison plots."""
    fig.set_size_inches(7.2, 4.2)
    fig.subplots_adjust(left=.11, right=.98, bottom=.18, top=.80)
    ax.set_title(title, fontsize=10, weight="bold", pad=28)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.tick_params(labelsize=8, width=.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for spine in ax.spines.values():
        spine.set_linewidth(.8)
    ax.grid(axis="y", color="#E7EBEE", linewidth=.5)
    ax.set_axisbelow(True)


def plot_paired_gene_distance_distributions(table, args):
    """Plot the three modality pairs, without the three-pair mean category."""
    metrics = ["Compound–Crispr", "Compound–Orf", "Crispr–Orf"]
    colors = {"DeSCORT": "#BC3C28", "Original": "#0072B5"}
    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=args.dpi)
    for j, metric in enumerate(metrics):
        frame = table.loc[table.metric.eq(metric)].sort_values("Gene")
        for method, offset in [("DeSCORT", -.20), ("Original", .20)]:
            values, position = frame[method].to_numpy(), j + offset
            boxes = ax.boxplot([values], positions=[position], widths=.32, patch_artist=True,
                showfliers=False, medianprops={"color": "black", "linewidth": 1.1},
                whiskerprops={"color": "#303B40", "linewidth": .8},
                capprops={"color": "#303B40", "linewidth": .8})
            boxes["boxes"][0].set(facecolor=colors[method], alpha=.85, edgecolor="#303B40", linewidth=.8)
            jitter = (np.mod((np.arange(len(values)) + .5) * .61803398875, 1) - .5) * .18
            ax.scatter(position + jitter, values, s=6, c=colors[method], alpha=.25,
                       linewidths=0, rasterized=True, zorder=3)
            ax.scatter([position], [values.mean()], marker="D", s=21, color=colors[method],
                       edgecolors="black", linewidths=.5, zorder=4)
    ax.set_xticks(np.arange(3), [metric.replace("Compound–", "Compound–\n") for metric in metrics])
    ax.set_ylim(0, 2)
    style_profile_comparison_axis(fig, ax,
        f"Matched cross-modality distances (n={table.Gene.nunique()} genes)",
        "Cosine distance (lower = closer)")
    ax.legend(handles=[Patch(facecolor=colors[method], alpha=.85, label=method) for method in colors],
              loc="lower center", bbox_to_anchor=(.5, 1.01), ncol=2, fontsize=8, frameon=False)
    return fig


def plot_top_k_overlap(curves, args, title):
    """Draw model/original gene-ranking agreement at configured thresholds."""
    fig, ax = plt.subplots(figsize=(7.2, 4.3), dpi=args.dpi)
    for (label, frame), color in zip(curves.groupby("comparison", sort=False),
                                    ["#BC3C28", "#0072B5", "#6E875E"]):
        ax.plot(frame.top_k, frame.overlap_fraction, color=color, linewidth=1.3,
                marker="o", markersize=3, label=label)
    thresholds = sorted(curves.top_k.unique())
    padding = (thresholds[1] - thresholds[0]) * .4 if len(thresholds) > 1 else 2
    ax.set(xlim=(thresholds[0] - padding, thresholds[-1] + padding), ylim=(0, 1.02), xlabel="Top-k list size", ylabel="Shared fraction")
    ax.set_xticks(thresholds)
    ax.set_xlabel("Top-k list size", fontsize=8)
    style_profile_comparison_axis(fig, ax, title, "Shared fraction")
    ax.legend(loc="lower center", bbox_to_anchor=(.5, 1.01), ncol=3, fontsize=8, frameon=False)
    return fig


def prepare_top_k_overlap(gene_distances, args):
    """Rank per-gene mean distances; lower distance equals higher similarity."""
    rankings, comparisons = {}, []
    metrics = {"Compound–CRISPR": "Compound–Crispr",
               "Compound–ORF": "Compound–Orf", "CRISPR–ORF": "Crispr–Orf"}
    for label, metric in metrics.items():
        for method in ["DeSCORT", "Original"]:
            frame = gene_distances.loc[gene_distances.metric.eq(metric)].copy()
            frame = frame.loc[~frame.Gene.astype(str).str.strip().str.lower().isin(CONTROL_NAMES)]
            if frame.Gene.duplicated().any() or not np.isfinite(frame[method]).all():
                raise ValueError(f"Duplicate genes or nonfinite distances: {method}, {metric}")
            rankings[f"{method}:{metric}"] = frame.sort_values([method, "Gene"]).Gene.tolist()
        comparisons.append((label, f"DeSCORT:{metric}", f"Original:{metric}"))
    curves = calculate_top_k_overlap(rankings, comparisons,
                                    args.overlap_curve_step, args.overlap_curve_max)
    return plot_top_k_overlap(curves, args, "Model–Original top-k ranking agreement"), curves


def run_comparison(args):
    MODEL_EPOCHS = args.model_epochs
    REPRESENTATION = f'PCA{args.pca_variance * 100:g}'
    ROOT = args.output_dir / 'profile_comparison'
    MODEL_DATA_DIR = args.model_data_dir
    INPUTS = {'DeSCORT': args.model_profiles_dir or args.model_data_dir,
              'Original': args.original_profiles_dir}
    RESULTS, FIGURES, QA = (ROOT / x for x in ('results', 'figures', 'qa'))
    for directory in (RESULTS, FIGURES, QA):
        directory.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(args.qa_tools_dir))
    from audit_panel_alignment import require_matplotlib_panel_alignment
    CONTROL_NAMES = {'nontargeting', 'non-targeting', 'control', 'ctrl', 'dmso'}
    TOP_K = tuple(args.top_k)
    BOOTSTRAPS = args.bootstraps
    SEED = args.seed
    PAIR_FIELDS = {'Compound–Crispr': 'Distance_to_CRISPR', 'Compound–Orf': 'Distance_to_ORF', 'Crispr–Orf': 'Distance_CRISPR_ORF'}
    PAIRS = list(PAIR_FIELDS)
    MEAN3 = 'Three-pair mean'
    DISTANCE_METRICS = PAIRS + [MEAN3]
    RANK_CRITERIA = ['Closest Compound–Crispr', 'Mean Compound–Crispr', 'Mean Compound–Orf', 'Crispr–Orf', 'Three-pair mean']
    MAIN_CRITERIA = ['Closest Compound–Crispr', 'Mean Compound–Orf', 'Three-pair mean']
    COLORS = {'DeSCORT': '#BC3C28', 'Original': '#0072B5'}
    FIGURE_LOG = []

    def control_mask(values):
        return pd.Series(values, index=getattr(values, 'index', None)).astype(str).str.strip().str.lower().isin(CONTROL_NAMES)

    def fdr_bh(values):
        p = np.asarray(values, dtype=float)
        result = np.full(len(p), np.nan)
        valid = np.flatnonzero(np.isfinite(p))
        order = valid[np.argsort(p[valid], kind='stable')]
        if len(order):
            q = p[order] * len(order) / np.arange(1, len(order) + 1)
            result[order] = np.minimum(1, np.minimum.accumulate(q[::-1])[::-1])
        return result

    def bootstrap_mean(values, seed=SEED):
        values = np.asarray(values, dtype=float)
        rng = np.random.default_rng(seed)  # Resample observed genes; no artificial measurements.
        means = values[rng.integers(0, len(values), size=(BOOTSTRAPS, len(values)))].mean(axis=1)
        return np.quantile(means, [0.025, 0.975])

    def paired_summary(table, positive_direction):
        rows = []
        for index, (metric, frame) in enumerate(table.groupby('metric', sort=False)):
            a, b = frame['DeSCORT'].to_numpy(), frame['Original'].to_numpy()
            delta = (b - a) if positive_direction == 'closer' else (a - b)
            statistic, p = (0.0, 1.0) if np.allclose(delta, 0, atol=1e-12) else wilcoxon(delta, alternative='two-sided', zero_method='wilcox', method='auto')
            lo, hi = bootstrap_mean(delta, SEED + index)
            nonzero = delta[np.abs(delta) > 1e-12]
            signed_ranks = rankdata(np.abs(nonzero))
            rank_biserial = float(np.dot(np.sign(nonzero), signed_ranks) / signed_ranks.sum()) if len(nonzero) else 0.0
            rows.append({'metric': metric, 'n_genes': len(frame), 'mean_DeSCORT': a.mean(), 'mean_Original': b.mean(),
                         'median_DeSCORT': np.median(a), 'median_Original': np.median(b),
                         'mean_improvement': delta.mean(), 'median_improvement': np.median(delta),
                         'mean_improvement_ci95_low': lo, 'mean_improvement_ci95_high': hi,
                         'fraction_DeSCORT_better': np.mean(delta > 1e-12), 'fraction_tied': np.mean(np.abs(delta) <= 1e-12),
                         'wilcoxon_statistic': statistic, 'p_two_sided': p, 'signed_rank_biserial': rank_biserial,
                         'positive_improvement_means': positive_direction})
        result = pd.DataFrame(rows)
        result['q_BH'] = fdr_bh(result['p_two_sided'])
        return result

    source_summaries, pairs_by_method, profiles_by_method = {}, {}, {}
    vectors_by_method, distances_by_method, transform_rows, input_hashes = {}, {}, [], {}
    for method, directory in INPUTS.items():
        print(f'Loading {method} profiles: {directory}', flush=True)
        archived_distances = False
        if method == 'DeSCORT' and args.model_profiles_dir is None:
            profiles, source_summaries[method], features, source_manifest, source_files = load_model_profiles(directory)
            input_hashes[method] = sha256_file_set(source_files, directory)
            source_manifest.to_csv(RESULTS / 'DeSCORT_source_file_manifest.csv', index=False)
        else:
            profiles, source_summaries[method], features, profile_path = load_profile_bundle(directory, method)
            input_hashes[method] = hashlib.sha256(profile_path.read_bytes()).hexdigest()
        profiles = profiles.reset_index(drop=True)
        pairs = pair_identities(profiles)
        if pairs[['Gene', 'Compound']].isna().any().any() or pairs.duplicated(['Gene', 'Compound']).any():
            raise ValueError(f'{method}: missing or duplicate gene/compound identifiers')
        raw = profiles[features].to_numpy(dtype=np.float32)
        if not len(features) or not np.isfinite(raw).all():
            raise ValueError(f'{method}: invalid aggregated features')
        pca = PCA(n_components=args.pca_variance, svd_solver='full', whiten=False)
        vectors = pca.fit_transform(raw).astype(np.float64)
        retained = float(pca.explained_variance_ratio_.sum())
        pd.DataFrame({'component': np.arange(1, pca.n_components_ + 1),
                      'explained_variance_ratio': pca.explained_variance_ratio_,
                      'cumulative_variance_ratio': pca.explained_variance_ratio_.cumsum()}).to_csv(
                          RESULTS / f'{method}_explained_variance.csv', index=False)
        norm = np.linalg.norm(vectors, axis=1)
        if (norm <= 1e-12).any():
            raise ValueError(f'{method}: zero analysis vector')
        unit = vectors / norm[:, None]  # Cosine formula only; no additional learned transform.
        distances = np.clip(1 - unit @ unit.T, 0, 2)
        keys = ['Metadata_All_Modality', 'Metadata_Treatment', 'Metadata_Pert_IName']
        if profiles.duplicated(keys).any():
            raise ValueError(f'{method}: duplicate profile identity')
        index = {tuple(row): i for i, row in enumerate(profiles[keys].to_numpy())}
        source_errors = []
        for row_idx, row in enumerate(pairs.itertuples(index=False)):
            ci, ki, oi = index[('Compound', row.Gene, row.Compound)], index[('Crispr', row.Gene, row.Gene)], index[('Orf', row.Gene, row.Gene)]
            values = [distances[ci, ki], distances[ci, oi], distances[ki, oi]]
            if archived_distances:
                source_errors.extend(abs(np.array(values) - np.array([row.Distance_to_CRISPR, row.Distance_to_ORF, row.Distance_CRISPR_ORF])))
            pairs.loc[row_idx, list(PAIR_FIELDS.values())] = values
        if archived_distances and max(source_errors) > 2e-5:
            raise ValueError(f'{method}: PCA95 does not reconstruct updated source distances')
        transform_rows.append({'method': method, 'representation': REPRESENTATION, 'n_features': len(features),
                               'n_components': vectors.shape[1], 'retained_variance': retained,
                               'n_profiles': len(profiles),
                               'source_PCA95_max_distance_difference': max(source_errors) if source_errors else np.nan})
        pd.concat([profiles[keys], pd.DataFrame(vectors, columns=[f'axis_{i}' for i in range(vectors.shape[1])])], axis=1).to_parquet(
            RESULTS / f'{method}_analysis_vectors.parquet', index=False)
        pairs.to_csv(RESULTS / f'{method}_recomputed_compound_pairs.csv', index=False)
        profiles_by_method[method], pairs_by_method[method] = profiles, pairs
        vectors_by_method[method], distances_by_method[method] = vectors, distances
    pd.DataFrame(transform_rows).to_csv(RESULTS / 'representation_audit.csv', index=False)
    print('Representation:', REPRESENTATION)


    compound_pairs = pairs_by_method['DeSCORT'].merge(pairs_by_method['Original'], on=['Gene', 'Compound'],
        how='outer', suffixes=('_DeSCORT', '_Original'), indicator=True, validate='one_to_one')
    compound_pairs[['Gene', 'Compound', '_merge']].to_csv(RESULTS / 'compound_alignment_audit.csv', index=False)
    if not compound_pairs['_merge'].eq('both').all():
        raise ValueError('The two representations do not contain identical gene/compound pairs; see alignment audit')
    for field in ('Crispr', 'Orf'):
        if not compound_pairs[f'{field}_DeSCORT'].eq(compound_pairs[f'{field}_Original']).all():
            raise ValueError(f'Mismatched {field} target identifiers')
    compound_pairs['is_control'] = control_mask(compound_pairs['Gene'])
    compound_pairs.to_csv(RESULTS / 'matched_compound_pair_distances.csv', index=False)

    PROFILE_KEYS = ['Metadata_All_Modality', 'Metadata_Treatment', 'Metadata_Pert_IName']
    count_columns = PROFILE_KEYS + ['n_source_files', 'n_input_wells']
    profile_audit = profiles_by_method['DeSCORT'][count_columns].merge(profiles_by_method['Original'][count_columns],
        on=PROFILE_KEYS, how='outer', suffixes=('_DeSCORT', '_Original'), indicator=True, validate='one_to_one')
    if not profile_audit['_merge'].eq('both').all():
        raise ValueError('Aggregated profile identities differ')
    profile_audit['well_count_difference_Original_minus_DeSCORT'] = profile_audit.n_input_wells_Original - profile_audit.n_input_wells_DeSCORT
    profile_audit.to_csv(RESULTS / 'profile_input_alignment.csv', index=False)
    print('Aligned compound–gene pairs:', len(compound_pairs))
    print('Profiles with different input well counts:')
    print(profile_audit.loc[profile_audit.well_count_difference_Original_minus_DeSCORT.ne(0)].to_string(index=False))

    gene_distances = {}
    for method, frame in pairs_by_method.items():
        for gene, group in frame.groupby('Gene'):
            if np.ptp(group['Distance_CRISPR_ORF']) > 1e-7:
                raise ValueError(f'{method}/{gene}: duplicated Crispr–Orf distances disagree')
        gene_distances[method] = frame.groupby('Gene')[list(PAIR_FIELDS.values())].mean().rename(columns={v: k for k, v in PAIR_FIELDS.items()})
        gene_distances[method][MEAN3] = gene_distances[method][PAIRS].mean(axis=1)
    if set(gene_distances['DeSCORT'].index) != set(gene_distances['Original'].index):
        raise ValueError('Gene universes differ')
    all_genes = sorted(gene_distances['DeSCORT'].index)
    analysis_genes = [gene for gene in all_genes if gene.strip().lower() not in CONTROL_NAMES]
    gene_rows = []
    for metric in DISTANCE_METRICS:
        for gene in all_genes:
            model = gene_distances['DeSCORT'].at[gene, metric]
            original = gene_distances['Original'].at[gene, metric]
            gene_rows.append({'Gene': gene, 'metric': metric, 'DeSCORT': model, 'Original': original,
                             'improvement_Original_minus_DeSCORT': original - model, 'is_control': gene.lower() in CONTROL_NAMES})
    gene_distance_table = pd.DataFrame(gene_rows)
    gene_distance_table.to_csv(RESULTS / 'paired_gene_distances.csv', index=False)
    primary = gene_distance_table.loc[~gene_distance_table.is_control].copy()
    distance_stats = paired_summary(primary, 'closer')
    distance_stats.to_csv(RESULTS / 'paired_distance_statistics.csv', index=False)
    paired_summary(gene_distance_table, 'closer').to_csv(RESULTS / 'paired_distance_statistics_including_control.csv', index=False)
    gene_distance_table.loc[gene_distance_table.is_control].to_csv(RESULTS / 'control_distances.csv', index=False)
    print(distance_stats[['metric', 'n_genes', 'mean_DeSCORT', 'mean_Original', 'mean_improvement', 'fraction_DeSCORT_better', 'q_BH']].to_string(index=False))

    rank_rows = []
    for method, frame in pairs_by_method.items():
        ranking_values = gene_distances[method].copy()
        ranking_values['Closest Compound–Crispr'] = frame.groupby('Gene')['Distance_to_CRISPR'].min()
        ranking_values['Mean Compound–Crispr'] = ranking_values['Compound–Crispr']
        ranking_values['Mean Compound–Orf'] = ranking_values['Compound–Orf']
        for include_controls in (True, False):
            scope = 'including_control' if include_controls else 'perturbations_only'
            for criterion in RANK_CRITERIA:
                selected = ranking_values if include_controls else ranking_values.loc[analysis_genes]
                ranked = selected[criterion].rename('distance').rename_axis('Gene').reset_index().sort_values(['distance', 'Gene'], kind='stable')
                ranked['rank'] = np.arange(1, len(ranked) + 1)
                ranked['method'], ranked['scope'], ranked['criterion'] = method, scope, criterion
                rank_rows.append(ranked)
    rankings = pd.concat(rank_rows, ignore_index=True)
    rankings.to_csv(RESULTS / 'all_gene_rankings.csv', index=False)

    def ranked_genes(method, criterion, scope):
        return rankings.loc[rankings.method.eq(method) & rankings.criterion.eq(criterion) & rankings.scope.eq(scope)].sort_values('rank').Gene.tolist()

    rank_source_audit = []
    for method, directory in INPUTS.items():
        if method != 'Original':
            continue
        if not (directory / 'results/gene_rank_pca95.csv').is_file() or args.pca_variance != 0.95:
            continue
        archived_closest = pd.read_csv(directory / 'results/gene_rank_pca95.csv').sort_values('gene_rank').Gene.tolist()
        archived_means = pd.read_csv(directory / 'results/gene_pairwise_similarity_pca95.csv')
        archived_rankings = {
            'Closest Compound–Crispr': archived_closest,
            'Mean Compound–Orf': archived_means.sort_values('compound_orf_rank').Gene.tolist(),
            'Three-pair mean': archived_means.sort_values('mean_all_pairs_rank').Gene.tolist(),
        }
        for criterion, archived in archived_rankings.items():
            rebuilt = ranked_genes(method, criterion, 'including_control')
            same = archived == rebuilt
            rank_source_audit.append({'method': method, 'criterion': criterion, 'matches_archived_ranking': same})
            # Compare rankings against the current Original PCA95 baseline.
    pd.DataFrame(rank_source_audit).to_csv(RESULTS / 'archived_ranking_validation.csv', index=False)

    overlap_rows, membership_rows, curve_rows, rank_correlation_rows = [], [], [], []
    for scope in ('including_control', 'perturbations_only'):
        for criterion in RANK_CRITERIA:
            model, original = (ranked_genes(m, criterion, scope) for m in ('DeSCORT', 'Original'))
            n = len(model)
            rank_m, rank_o = ({g: i + 1 for i, g in enumerate(values)} for values in (model, original))
            shared = sorted(set(model).intersection(original))
            rho = spearmanr([rank_m[g] for g in shared], [rank_o[g] for g in shared]).statistic
            tau = kendalltau([rank_m[g] for g in shared], [rank_o[g] for g in shared]).statistic
            rank_correlation_rows.append({'scope': scope, 'criterion': criterion, 'n_genes': n, 'spearman_rho': rho, 'kendall_tau': tau})
            for requested_k in TOP_K:
                k = min(requested_k, n)
                a, b = set(model[:k]), set(original[:k])
                common = sorted(a & b)
                overlap_rows.append({'scope': scope, 'criterion': criterion, 'top_k': requested_k, 'actual_k': k, 'universe_n': n,
                                     'intersection_n': len(common), 'overlap_fraction': len(common) / k,
                                     'jaccard': len(common) / len(a | b), 'chance_expected_overlap_n': k * k / n,
                                     'fold_over_chance': len(common) / (k * k / n),
                                     'hypergeom_p_enrichment': hypergeom.sf(len(common) - 1, n, k, k),
                                     'shared_genes': ';'.join(common), 'DeSCORT_only_genes': ';'.join(sorted(a - b)),
                                     'Original_only_genes': ';'.join(sorted(b - a))})
                for gene in sorted(a | b):
                    membership_rows.append({'scope': scope, 'criterion': criterion, 'top_k': requested_k, 'Gene': gene,
                                            'DeSCORT_rank': rank_m[gene], 'Original_rank': rank_o[gene],
                                            'in_DeSCORT_top_k': gene in a, 'in_Original_top_k': gene in b,
                                            'rank_gain_Original_minus_DeSCORT': rank_o[gene] - rank_m[gene]})
            for k in range(1, min(100, n) + 1):
                overlap = len(set(model[:k]) & set(original[:k]))
                curve_rows.append({'scope': scope, 'criterion': criterion, 'top_k': k, 'intersection_n': overlap,
                                   'overlap_fraction': overlap / k, 'chance_expected_fraction': k / n})
    overlaps = pd.DataFrame(overlap_rows)
    overlaps['hypergeom_q_BH'] = fdr_bh(overlaps.hypergeom_p_enrichment)
    overlaps.to_csv(RESULTS / 'top20_top50_overlap.csv', index=False)
    pd.DataFrame(membership_rows).to_csv(RESULTS / 'top20_top50_gene_membership.csv', index=False)
    overlap_curve = pd.DataFrame(curve_rows)
    overlap_curve.to_csv(RESULTS / 'top_k_overlap_curves.csv', index=False)
    rank_correlations = pd.DataFrame(rank_correlation_rows)
    rank_correlations.to_csv(RESULTS / 'rank_correlations.csv', index=False)
    consensus_rows = []
    for method in INPUTS:
        for scope in ('including_control', 'perturbations_only'):
            for k in TOP_K:
                lists = [set(ranked_genes(method, c, scope)[:k]) for c in ['Mean Compound–Crispr', 'Mean Compound–Orf', 'Crispr–Orf']]
                common = sorted(set.intersection(*lists))
                consensus_rows.append({'method': method, 'scope': scope, 'top_k': k,
                                       'three_pair_intersection_n': len(common), 'genes': ';'.join(common)})
    pd.DataFrame(consensus_rows).to_csv(RESULTS / 'within_method_three_pair_consensus.csv', index=False)
    print(overlaps.loc[overlaps.scope.eq('perturbations_only') & overlaps.criterion.isin(MAIN_CRITERIA),
                       ['criterion', 'top_k', 'intersection_n', 'overlap_fraction', 'jaccard']].to_string(index=False))

    # Use the same recomputed vectors/matrix for matched and mismatched distances.
    specificity_rows, retrieval_rows, reconstruction_rows, geometry_rows = [], [], [], []
    for method, profiles in profiles_by_method.items():
        vectors, distances = vectors_by_method[method], distances_by_method[method]
        profile_index = {tuple(row): i for i, row in enumerate(profiles[PROFILE_KEYS].to_numpy())}
        reconstruction_rows.append(next(row for row in transform_rows if row['method'] == method))
        for modality in ('Compound', 'Crispr', 'Orf'):
            idx = profiles.index[profiles.Metadata_All_Modality.eq(modality) & profiles.Metadata_Treatment.isin(analysis_genes)].to_numpy()
            matrix = distances[np.ix_(idx, idx)]
            distinct_genes = profiles.loc[idx, 'Metadata_Treatment'].to_numpy()
            valid = distinct_genes[:, None] != distinct_genes[None, :]
            geometry_rows.append({'method': method, 'modality': modality, 'mean_different_target_distance': float(matrix[valid].mean())})
        partner_specs = [('Compound–Crispr', 'Compound', 'Crispr'), ('Compound–Orf', 'Compound', 'Orf'), ('Crispr–Orf', 'Crispr', 'Orf')]
        for pair, source_modality, target_modality in partner_specs:
            target_indices = np.array([profile_index[(target_modality, gene, gene)] for gene in analysis_genes])
            sources = profiles.loc[profiles.Metadata_All_Modality.eq(source_modality) & profiles.Metadata_Treatment.isin(analysis_genes)]
            for gene, group in sources.groupby('Metadata_Treatment', sort=True):
                target_position = analysis_genes.index(gene)
                matrix = distances[np.ix_(group.index.to_numpy(), target_indices)].astype(float)
                matched = matrix[:, target_position]
                unmatched = np.delete(matrix, target_position, axis=1)
                true_rank = np.array([rankdata(row, method='average')[target_position] for row in matrix])
                specificity_rows.append({'method': method, 'Gene': gene, 'metric': pair,
                                         'matched_distance': matched.mean(), 'mean_mismatched_distance': unmatched.mean(),
                                         'background_gap': (unmatched.mean(axis=1) - matched).mean(),
                                         'matched_percentile': ((unmatched > matched[:, None]).mean(axis=1)).mean(),
                                         'n_source_profiles': len(group), 'n_mismatched_targets_per_profile': len(analysis_genes) - 1})
                retrieval_rows.append({'method': method, 'Gene': gene, 'metric': pair, 'mean_target_rank': true_rank.mean(),
                                       'MRR': (1 / true_rank).mean(), 'Hit_at_1': (true_rank <= 1).mean(),
                                       'Hit_at_5': (true_rank <= 5).mean(), 'Hit_at_10': (true_rank <= 10).mean(),
                                       'Hit_at_20': (true_rank <= 20).mean(), 'n_candidate_targets': len(analysis_genes)})
    pd.DataFrame(reconstruction_rows).to_csv(RESULTS / 'distance_matrix_representation_audit.csv', index=False)
    pd.DataFrame(geometry_rows).to_csv(RESULTS / 'within_modality_geometry.csv', index=False)
    specificity = pd.DataFrame(specificity_rows)
    gap_mean = specificity.groupby(['method', 'Gene'])[['matched_distance', 'mean_mismatched_distance', 'background_gap', 'matched_percentile']].mean().reset_index().assign(metric=MEAN3)
    specificity = pd.concat([specificity, gap_mean], ignore_index=True)
    specificity.to_csv(RESULTS / 'matched_vs_mismatched_target_background.csv', index=False)
    gap_paired = specificity.pivot(index=['Gene', 'metric'], columns='method', values='background_gap').reset_index()
    gap_stats = paired_summary(gap_paired, 'larger matched–unmatched specificity gap')
    gap_stats.to_csv(RESULTS / 'paired_specificity_statistics.csv', index=False)
    retrieval = pd.DataFrame(retrieval_rows)
    retrieval.to_csv(RESULTS / 'true_target_retrieval_per_gene.csv', index=False)
    retrieval_paired = retrieval.pivot(index=['Gene', 'metric'], columns='method', values='MRR').reset_index()
    retrieval_stats = paired_summary(retrieval_paired, 'higher MRR')
    retrieval_stats.to_csv(RESULTS / 'paired_retrieval_MRR_statistics.csv', index=False)
    retrieval_summary = retrieval.groupby(['method', 'metric'])[['MRR', 'Hit_at_1', 'Hit_at_5', 'Hit_at_10', 'Hit_at_20', 'mean_target_rank']].mean().reset_index()
    retrieval_summary['n_genes'] = len(analysis_genes)
    retrieval_summary.to_csv(RESULTS / 'true_target_retrieval_summary.csv', index=False)
    comparison_overview = distance_stats.loc[distance_stats.metric.isin(PAIRS), ['metric', 'mean_DeSCORT', 'mean_Original']].rename(
        columns={'mean_DeSCORT': 'distance_DeSCORT', 'mean_Original': 'distance_Original'})
    comparison_overview = comparison_overview.merge(gap_stats[['metric', 'mean_DeSCORT', 'mean_Original']].rename(
        columns={'mean_DeSCORT': 'specificity_DeSCORT', 'mean_Original': 'specificity_Original'}), on='metric', validate='one_to_one')
    comparison_overview = comparison_overview.merge(retrieval_stats[['metric', 'mean_DeSCORT', 'mean_Original']].rename(
        columns={'mean_DeSCORT': 'MRR_DeSCORT', 'mean_Original': 'MRR_Original'}), on='metric', validate='one_to_one')
    comparison_overview.to_csv(RESULTS / 'comparison_overview.csv', index=False)
    print('Matched-target specificity (larger is better):')
    print(gap_stats[['metric', 'mean_DeSCORT', 'mean_Original', 'fraction_DeSCORT_better', 'q_BH']].to_string(index=False))
    print('True-target retrieval among', len(analysis_genes), 'candidate genes:')
    print(retrieval_summary.to_string(index=False))

    mpl.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': [args.font_family, 'DejaVu Sans'], 'font.size': 8,
                         'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
                         'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': 0.8,
                         'legend.frameon': False, 'savefig.facecolor': 'white'})

    def save_figure(fig, name, multi=False, exclude_axes=None):
        stem = FIGURES / name
        fig.canvas.draw()
        require_matplotlib_panel_alignment(fig, json_out=str(QA / f'{name}.alignment.json'),
            tolerance_pt=1.5, gutter_tolerance_pt=1.5, require_panel_labels=multi,
            exclude_axes=exclude_axes or [], strict=True)
        exported = export_figure(fig, stem, QA, args)
        FIGURE_LOG.append(str(exported))
        # Batch export: no interactive display.
        plt.close(fig)

    overlap_fig, pair_overlap_curves = prepare_top_k_overlap(gene_distance_table, args)
    overlap_stem = f"top_k_overlap_curves_pca{args.pca_variance * 100:g}"
    pair_overlap_curves.to_csv(RESULTS / f"{overlap_stem}.csv", index=False)
    save_figure(overlap_fig, overlap_stem)

    def short_label(metric):
        return metric.replace('Compound–', 'Compound–\n').replace('Three-pair mean', 'Three-pair\nmean')

    fig = plot_paired_gene_distance_distributions(primary, args)
    save_figure(fig, 'figure1_paired_gene_distance_distributions')
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.8), dpi=300, constrained_layout=True)
    for i, (ax, pair) in enumerate(zip(axes, PAIRS)):
        frame = primary.loc[primary.metric.eq(pair)]
        ax.scatter(frame.Original, frame.DeSCORT, s=8, color='#527B8D', alpha=0.55, linewidths=0, rasterized=True)
        ax.plot([0, 2], [0, 2], color='#A6ADB0', linewidth=0.8, zorder=0)
        ax.set(xlim=(0, 2), ylim=(0, 2), xticks=[0, 1, 2], yticks=[0, 1, 2], xlabel='Original distance', ylabel='DeSCORT distance')
        ax.set_title(pair, fontsize=8, pad=7)
        ax.text(0.025, 0.98, chr(97 + i), transform=ax.transAxes, weight='bold', fontsize=9, va='top')
    save_figure(fig, 'figure2_paired_distance_scatter', multi=True)

    fig, ax = plt.subplots(figsize=(7.2, 3.9), dpi=300)
    for method, offset in [('DeSCORT', -0.18), ('Original', 0.18)]:
        means, lows, highs = [], [], []
        for index, pair in enumerate(PAIRS):
            values = retrieval.loc[retrieval.method.eq(method) & retrieval.metric.eq(pair), 'MRR'].to_numpy()
            mean, ci = values.mean(), bootstrap_mean(values, SEED + index)
            means.append(mean); lows.append(mean - ci[0]); highs.append(ci[1] - mean)
        ax.bar(np.arange(3) + offset, means, width=0.32, color=COLORS[method], alpha=0.75, label=method)
        ax.errorbar(np.arange(3) + offset, means, yerr=[lows, highs], fmt='none', color='#26323A', capsize=3, linewidth=0.8)
    chance_mrr = (1 / np.arange(1, len(analysis_genes) + 1)).sum() / len(analysis_genes)
    ax.axhline(chance_mrr, color='#7E8A90', linewidth=0.9, label='Uniform-random rank')
    ax.set_xticks(np.arange(3), [short_label(x) for x in PAIRS])
    ax.set_ylabel('Mean reciprocal target rank (higher = better)')
    ax.set_title(f'True-target retrieval ({len(analysis_genes)} candidate genes)', fontsize=10, weight='bold', pad=28)
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.01), ncol=3, fontsize=7)
    ax.set_ylim(bottom=0)
    ax.grid(axis='y', color='#E7EBEE', linewidth=0.5); ax.set_axisbelow(True)
    fig.tight_layout()
    save_figure(fig, 'figure3_true_target_retrieval_MRR')
    FIGURE_LOG.append(str(plot_profile_hits(retrieval, ROOT, args)))

    fig, ax = plt.subplots(figsize=(7.2, 3.7), dpi=300)
    for k, offset, color in [(TOP_K[0], -0.18, '#507C91'), (TOP_K[1], 0.18, '#89B8B0')]:
        frame = overlaps.loc[overlaps.scope.eq('perturbations_only') & overlaps.top_k.eq(k)].set_index('criterion').loc[MAIN_CRITERIA]
        ax.bar(np.arange(3) + offset, frame.overlap_fraction, width=0.32, color=color, label=f'Top {k}')
    ax.set_xticks(np.arange(3), ['Closest\nCompound–Crispr', 'Mean\nCompound–Orf', 'Three-pair\nmean'])
    ax.set_ylim(0, 1)
    ax.set_ylabel('Shared genes / top-k list size')
    ax.set_title('DeSCORT–Original ranking overlap', fontsize=10, weight='bold', pad=27)
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.01), ncol=2, fontsize=8)
    ax.grid(axis='y', color='#E7EBEE', linewidth=0.5); ax.set_axisbelow(True)
    fig.tight_layout()
    save_figure(fig, 'figure5_top20_top50_overlap')
    changes = primary.loc[primary.metric.eq(MEAN3)].sort_values('improvement_Original_minus_DeSCORT', ascending=False)
    changes.to_csv(RESULTS / 'three_pair_mean_gene_changes.csv', index=False)
    examples = pd.concat([changes.head(10).assign(selection='largest_Original_minus_DeSCORT_difference'),
                          changes.tail(10).assign(selection='smallest_Original_minus_DeSCORT_difference')])
    examples.to_csv(RESULTS / 'balanced_top10_change_examples.csv', index=False)
    # Follow the reference notebook's exact d<1 / d>1 convention. These labels
    # describe positive/negative cosine similarity in the current PCA95 spaces;
    # they do not measure gene-expression changes. d==1 is retained as neutral.
    from matplotlib_venn import venn2

    UP_DOWN_THRESHOLD = 1.0
    UP_DOWN_MAIN_PAIR = 'Compound–Crispr'
    UP_DOWN_MAIN_SCOPE = 'including_control'  # Matches the reference's full pair set.
    SIGN_ORDER = ['up', 'down', 'neutral']

    def cosine_sign(values):
        values = np.asarray(values, dtype=float)
        if not np.isfinite(values).all():
            raise ValueError('Nonfinite cosine distances in up/down analysis')
        return np.select([values < UP_DOWN_THRESHOLD, values > UP_DOWN_THRESHOLD],
                         ['up', 'down'], default='neutral')

    up_down_rows = []
    for pair, field in PAIR_FIELDS.items():
        frame = compound_pairs.copy() if pair != 'Crispr–Orf' else compound_pairs.drop_duplicates('Gene').copy()
        for row in frame.to_dict('records'):
            compound = row['Compound'] if pair != 'Crispr–Orf' else ''
            up_down_rows.append({'analysis_level': 'pair', 'metric': pair,
                                'entity_id': json.dumps([row['Gene'], compound], ensure_ascii=False),
                                'Gene': row['Gene'], 'Compound': compound, 'is_control': row['is_control'],
                                'DeSCORT': row[f'{field}_DeSCORT'], 'Original': row[f'{field}_Original']})
    for row in gene_distance_table.loc[gene_distance_table.metric.isin(PAIRS)].to_dict('records'):
        up_down_rows.append({'analysis_level': 'gene_mean', 'metric': row['metric'],
                            'entity_id': json.dumps([row['Gene']], ensure_ascii=False),
                            'Gene': row['Gene'], 'Compound': '', 'is_control': row['is_control'],
                            'DeSCORT': row['DeSCORT'], 'Original': row['Original']})
    up_down_classification = pd.DataFrame(up_down_rows)
    for method in INPUTS:
        up_down_classification[f'sign_{method}'] = cosine_sign(up_down_classification[method])
    up_down_classification['cosine_similarity_DeSCORT'] = 1 - up_down_classification.DeSCORT
    up_down_classification['cosine_similarity_Original'] = 1 - up_down_classification.Original
    up_down_classification['same_sign'] = up_down_classification.sign_DeSCORT.eq(up_down_classification.sign_Original)
    if up_down_classification.duplicated(['analysis_level', 'metric', 'entity_id']).any():
        raise ValueError('Duplicate entities in up/down table')
    up_down_classification.to_csv(RESULTS / 'up_down_classification.csv', index=False)

    sign_summary_rows, sign_transition_rows, sign_agreement_rows = [], [], []
    common_member_tables, intersection_statistics_rows = [], []
    sign_top_rows, sign_top_membership_rows = [], []
    for (level, pair), full_frame in up_down_classification.groupby(['analysis_level', 'metric'], sort=False):
        for scope in ('including_control', 'perturbations_only'):
            frame = full_frame if scope == 'including_control' else full_frame.loc[~full_frame.is_control]
            common_metadata = {'analysis_level': level, 'scope': scope, 'metric': pair}
            n = len(frame)
            if not n:
                continue
            marginals = {}
            for method in INPUTS:
                marginals[method] = frame[f'sign_{method}'].value_counts().reindex(SIGN_ORDER, fill_value=0)
            agreement = int(frame.same_sign.sum())
            chance_agreement = float(np.dot(marginals['DeSCORT'], marginals['Original']) / n**2)
            sign_agreement_rows.append({**common_metadata, 'n_entities': n,
                                       'same_sign_n': agreement, 'changed_sign_n': n - agreement,
                                       'agreement_fraction': agreement / n,
                                       'cohen_kappa': (agreement / n - chance_agreement) / (1 - chance_agreement) if chance_agreement < 1 else np.nan})
            for original_sign in SIGN_ORDER:
                for model_sign in SIGN_ORDER:
                    transitions = frame.loc[frame.sign_Original.eq(original_sign) & frame.sign_DeSCORT.eq(model_sign)]
                    sign_transition_rows.append({**common_metadata, 'Original_sign': original_sign,
                                                 'DeSCORT_sign': model_sign, 'n_entities': len(transitions),
                                                 'fraction_of_universe': len(transitions) / n})
            for sign in SIGN_ORDER:
                model_set = set(frame.loc[frame.sign_DeSCORT.eq(sign), 'entity_id'])
                original_set = set(frame.loc[frame.sign_Original.eq(sign), 'entity_id'])
                common, union = model_set & original_set, model_set | original_set
                sign_summary_rows.append({**common_metadata, 'sign': sign, 'universe_n': n,
                                          'DeSCORT_count': len(model_set), 'Original_count': len(original_set),
                                          'intersection_n': len(common), 'union_n': len(union),
                                          'jaccard': len(common) / len(union) if union else np.nan,
                                          'intersection_fraction_of_DeSCORT': len(common) / len(model_set) if model_set else np.nan,
                                          'intersection_fraction_of_Original': len(common) / len(original_set) if original_set else np.nan,
                                          'DeSCORT_only_n': len(model_set - original_set),
                                          'Original_only_n': len(original_set - model_set)})
                members = frame.loc[frame.entity_id.isin(common)].copy()
                members['scope'], members['common_sign'] = scope, sign
                members['mean_distance_two_methods'] = members[['DeSCORT', 'Original']].mean(axis=1)
                members['difference_Original_minus_DeSCORT'] = members.Original - members.DeSCORT
                members = members.sort_values(['mean_distance_two_methods', 'Gene', 'Compound'], ascending=[sign != 'down', True, True])
                common_member_tables.append(members)
                for method in INPUTS:
                    values = members[method]
                    intersection_statistics_rows.append({**common_metadata, 'common_sign': sign, 'method': method,
                        'n_entities': len(members), 'n_unique_genes': members.Gene.nunique(),
                        'mean_distance': values.mean(), 'median_distance': values.median(), 'std_distance': values.std(),
                        'min_distance': values.min(), 'max_distance': values.max(),
                        'mean_paired_difference_Original_minus_DeSCORT': (members.Original - members.DeSCORT).mean()})
                if sign == 'neutral':
                    continue
                # Rank within each method's own sign set; down is decreasing distance.
                ranked = {}
                for method in INPUTS:
                    selected = frame.loc[frame[f'sign_{method}'].eq(sign)].sort_values(
                        [method, 'Gene', 'Compound'], ascending=[sign == 'up', True, True])
                    ranked[method] = selected.entity_id.tolist()
                for k in TOP_K:
                    selected_model, selected_original = set(ranked['DeSCORT'][:k]), set(ranked['Original'][:k])
                    top_common, top_union = selected_model & selected_original, selected_model | selected_original
                    model_k, original_k = len(selected_model), len(selected_original)
                    expected = len(common) * (model_k / len(model_set)) * (original_k / len(original_set)) if model_set and original_set else 0.0
                    sign_top_rows.append({**common_metadata, 'sign': sign, 'requested_top_k': k,
                                         'actual_DeSCORT_k': model_k, 'actual_Original_k': original_k,
                                         'intersection_n': len(top_common), 'jaccard': len(top_common) / len(top_union) if top_union else np.nan,
                                         'intersection_fraction_of_DeSCORT': len(top_common) / model_k if model_k else np.nan,
                                         'intersection_fraction_of_Original': len(top_common) / original_k if original_k else np.nan,
                                         'chance_expected_overlap_given_sign_sets': expected,
                                         'shared_genes': ';'.join(sorted(frame.loc[frame.entity_id.isin(top_common), 'Gene'].unique()))})
                    rank_maps = {m: {identifier: index + 1 for index, identifier in enumerate(ranked[m])} for m in INPUTS}
                    for member in frame.loc[frame.entity_id.isin(top_union)].to_dict('records'):
                        sign_top_membership_rows.append({**common_metadata, 'sign': sign, 'requested_top_k': k,
                            'Gene': member['Gene'], 'Compound': member['Compound'], 'entity_id': member['entity_id'],
                            'DeSCORT_sign_rank': rank_maps['DeSCORT'].get(member['entity_id'], np.nan),
                            'Original_sign_rank': rank_maps['Original'].get(member['entity_id'], np.nan),
                            'in_DeSCORT_top_k': member['entity_id'] in selected_model,
                            'in_Original_top_k': member['entity_id'] in selected_original,
                            'DeSCORT': member['DeSCORT'], 'Original': member['Original']})

    sign_summary = pd.DataFrame(sign_summary_rows)
    sign_transitions = pd.DataFrame(sign_transition_rows)
    sign_agreement = pd.DataFrame(sign_agreement_rows)
    common_sign_members = pd.concat(common_member_tables, ignore_index=True)
    intersection_statistics = pd.DataFrame(intersection_statistics_rows)
    sign_top_overlap = pd.DataFrame(sign_top_rows)
    for filename, table in [
        ('up_down_set_overlap_summary.csv', sign_summary), ('up_down_sign_transitions.csv', sign_transitions),
        ('up_down_sign_agreement.csv', sign_agreement), ('up_down_intersection_members.csv', common_sign_members),
        ('up_down_intersection_distance_statistics.csv', intersection_statistics),
        ('up_down_top20_top50_overlap.csv', sign_top_overlap),
        ('up_down_top20_top50_membership.csv', pd.DataFrame(sign_top_membership_rows))]:
        table.to_csv(RESULTS / filename, index=False)

    # A gene can have both positive- and negative-cosine Compound profiles. Retain
    # this ambiguity explicitly instead of presenting a gene-mean label as unanimous.
    mixed_gene_rows = []
    for (pair, gene), frame in up_down_classification.loc[
            up_down_classification.analysis_level.eq('pair') & up_down_classification.metric.ne('Crispr–Orf')].groupby(['metric', 'Gene']):
        for method in INPUTS:
            counts = frame[f'sign_{method}'].value_counts().reindex(SIGN_ORDER, fill_value=0)
            mixed_gene_rows.append({'metric': pair, 'Gene': gene, 'method': method, 'n_compounds': len(frame),
                                   'n_up_compounds': counts['up'], 'n_down_compounds': counts['down'],
                                   'n_neutral_compounds': counts['neutral'],
                                   'mixed_up_and_down': bool(counts['up'] and counts['down']),
                                   'up_compounds': ';'.join(frame.loc[frame[f'sign_{method}'].eq('up'), 'Compound']),
                                   'down_compounds': ';'.join(frame.loc[frame[f'sign_{method}'].eq('down'), 'Compound'])})
    mixed_gene_signs = pd.DataFrame(mixed_gene_rows)
    mixed_gene_signs.to_csv(RESULTS / 'up_down_gene_mixed_compound_signs.csv', index=False)
    for sign in ('up', 'down'):
        common_sign_members.loc[common_sign_members.analysis_level.eq('pair') &
            common_sign_members.scope.eq(UP_DOWN_MAIN_SCOPE) & common_sign_members.metric.eq(UP_DOWN_MAIN_PAIR) &
            common_sign_members.common_sign.eq(sign)].to_csv(RESULTS / f'up_down_common_{sign}_compound_crispr_pairs.csv', index=False)
    for (level, scope, pair), summary_group in sign_summary.groupby(['analysis_level', 'scope', 'metric']):
        assert summary_group.DeSCORT_count.sum() == summary_group.universe_n.iloc[0]
        assert summary_group.Original_count.sum() == summary_group.universe_n.iloc[0]
        tr = sign_transitions.loc[sign_transitions.analysis_level.eq(level) & sign_transitions.scope.eq(scope) & sign_transitions.metric.eq(pair)]
        assert tr.n_entities.sum() == summary_group.universe_n.iloc[0]
    print('Up/down (pair level, including control):')
    print(sign_summary.loc[sign_summary.analysis_level.eq('pair') & sign_summary.scope.eq('including_control') &
        sign_summary.sign.ne('neutral'), ['metric', 'sign', 'universe_n', 'DeSCORT_count', 'Original_count', 'intersection_n', 'jaccard']].to_string(index=False))

    main_sign_pairs = up_down_classification.loc[up_down_classification.analysis_level.eq('pair') &
                                               up_down_classification.metric.eq(UP_DOWN_MAIN_PAIR)].copy()
    if UP_DOWN_MAIN_SCOPE == 'perturbations_only':
        main_sign_pairs = main_sign_pairs.loc[~main_sign_pairs.is_control]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.6), dpi=300, constrained_layout=True)
    for index, (ax, sign) in enumerate(zip(axes, ('up', 'down'))):
        sets = [set(main_sign_pairs.loc[main_sign_pairs[f'sign_{m}'].eq(sign), 'entity_id']) for m in INPUTS]
        venn = venn2(sets, set_labels=None, set_colors=tuple(COLORS.values()), alpha=0.55, ax=ax)
        for label in list(venn.set_labels or []) + list(venn.subset_labels or []):
            if label is not None:
                label.set_fontsize(8)
        for subset, count in [('10', len(sets[0] - sets[1])), ('01', len(sets[1] - sets[0])), ('11', len(sets[0] & sets[1]))]:
            label = venn.get_label_by_id(subset)
            if label is not None and count == 0:
                label.set_visible(False)
            elif label is not None and subset in ('10', '01') and count / max(1, len(sets[0] | sets[1])) < 0.06:
                # Thin exclusive regions cannot fit a readable count inside.
                # Place the count outside the circles and connect to its true region.
                anchor = label.get_position()
                label.set_visible(False)
                side = -1 if subset == '10' else 1
                ax.annotate(str(count), xy=anchor, xytext=(side * 0.83, 0.16),
                            ha='center', va='bottom', fontsize=8,
                            arrowprops={'arrowstyle': '-', 'color': '#555555', 'linewidth': 0.7})
        for x, method, members in zip((0.25, 0.75), INPUTS, sets):
            ax.text(x, -0.02, f'{method} (n={len(members)})', transform=ax.transAxes, ha='center', va='top', fontsize=7)
        ax.set_xlim(-1.1, 1.1); ax.set_ylim(-0.85, 0.85); ax.set_aspect('equal', adjustable='box')
        ax.set_title(f'{sign.capitalize()} (distance {"<" if sign == "up" else ">"} {UP_DOWN_THRESHOLD:g})', fontsize=9, pad=8)
        ax.text(0.025, 0.98, chr(97 + index), transform=ax.transAxes, fontsize=9, weight='bold', va='top')
    fig.suptitle(f'Compound–Crispr sign overlap ({len(main_sign_pairs)} matched pairs)', fontsize=10, weight='bold')
    save_figure(fig, 'figure6_up_down_compound_crispr_venn', multi=True)

    def markdown_table(frame, columns):
        out = frame[columns].copy()
        for column in out:
            if pd.api.types.is_float_dtype(out[column]):
                out[column] = out[column].map(lambda v: f'{v:.3g}' if 'q_' in column or 'p_' in column else f'{v:.3f}')
        return '| ' + ' | '.join(out.columns) + ' |\n| ' + ' | '.join(['---'] * len(out.columns)) + ' |\n' + '\n'.join('| ' + ' | '.join(map(str, row)) + ' |' for row in out.to_numpy())

    report = [
        f'# {args.cell_line} {MODEL_EPOCHS} epochs vs Original',
        '## 输入与分析空间',
        f'Model input: {INPUTS["DeSCORT"]}; Original profiles: {INPUTS["Original"]}.',
        f'{len(compound_pairs)} matched Compound–gene pairs; {len(analysis_genes)} noncontrol genes. '
        f'Each representation has an independently fitted PCA retaining {args.pca_variance:.1%} variance, '
        'without whitening or additional standardization. All distances and ranks are recomputed.',
        markdown_table(pd.DataFrame(transform_rows), ['method', 'n_features', 'n_components', 'retained_variance']),
        '聚合全部 plate/batch/时间：Compound 按 Treatment+名称均值；CRISPR 按 guide 均值后按基因中位数；ORF 按基因均值。',
        'Profile 与 Compound–gene 身份必须一一对应。well 数差异保存在 results/profile_input_alignment.csv；'
        '即使主检验排除对照，输入对照差异仍可能影响 PCA。',
        markdown_table(profile_audit.loc[profile_audit.well_count_difference_Original_minus_DeSCORT.ne(0)],
            ['Metadata_All_Modality', 'Metadata_Treatment', 'n_input_wells_DeSCORT', 'n_input_wells_Original',
             'well_count_difference_Original_minus_DeSCORT']),
        '## 匹配距离与特异性',
        '同基因多个 Compound 先取均值，每个非对照基因等权。匹配距离越小越近；background_gap=全部错配靶点平均距离−匹配距离，越大越好。',
        markdown_table(distance_stats, ['metric', 'mean_DeSCORT', 'mean_Original', 'mean_improvement', 'q_BH']),
        markdown_table(gap_stats, ['metric', 'mean_DeSCORT', 'mean_Original', 'mean_improvement', 'q_BH']),
        '## 聚合 profile 的 MRR / Hit',
        f'候选是 {len(analysis_genes)} 个非对照基因，正确靶点并列采用平均名次。'
        '先对每个 Compound 的正确基因计算 1/rank 与 Hit，再在同基因内平均、跨基因等权。'
        '分别分析 Compound→CRISPR、Compound→ORF、CRISPR→ORF。',
        markdown_table(retrieval_summary, ['method', 'metric', 'n_genes', 'MRR', 'Hit_at_1', 'Hit_at_5', 'Hit_at_10', 'Hit_at_20']),
        '## 排名重合与余弦方向',
        '保存全部排名、Top-k 交集及随机集合期望、排名相关和距离变化两端的基因例子。'
        'up/down 分别表示余弦距离小于/大于 1，等于 1 单列；不是基因表达上调/下调。'
        'up/down 同时保存含对照和去对照、配对级和基因均值级、交集成员和类别转换。',
        '## 统计与局限',
        f'双侧基因配对 Wilcoxon；距离、背景 gap、MRR 为独立 BH 检验族。'
        f'均值差和 MRR 的区间使用 {BOOTSTRAPS} 次观察基因 bootstrap 的百分位 95% CI；'
        '它不是独立培养或训练重复的置信区间。箱线图显示 Q1–Q3、中位数、1.5 IQR 内须，点为所有观察单位。'
        '未使用 UMAP 距离，不将较小绝对距离直接视为更准确的生物关系。结果仅作复用训练数据的探索性比较。',
        '## 图形',
    ]
    for image in FIGURE_LOG:
        prefix = '' if Path(image).suffix.lower() == '.pdf' else '!'
        report.extend([f'### {Path(image).stem}', f'{prefix}[{Path(image).stem}](figures/{Path(image).name})'])
    (ROOT / 'REPORT_zh.md').write_text('\n\n'.join(report) + '\n', encoding='utf-8')
    manifest = {'n_noncontrol_genes': len(analysis_genes), 'n_labels_including_control': len(all_genes),
                'n_matched_compound_pairs': len(compound_pairs), 'top_k': list(TOP_K), 'bootstrap_iterations': BOOTSTRAPS,
                'representation': REPRESENTATION, 'model_epochs': MODEL_EPOCHS,
                'model_data_dir': str(MODEL_DATA_DIR), 'original_data_dir': str(INPUTS['Original']),
                'input_profile_sha256': input_hashes,
                'input_aggregation': {m: source_summaries[m]['aggregation'] for m in INPUTS},
                'seed': SEED, 'comparison_unit': 'gene; equal-weight compound means inside each gene',
                'figure_formats': list(dict.fromkeys(getattr(args, 'figure_formats', ['png', 'pdf']))),
                'up_down_threshold': UP_DOWN_THRESHOLD, 'up_down_main_scope': UP_DOWN_MAIN_SCOPE,
                'up_down_definition': 'up: cosine distance<1; down: distance>1; neutral: distance==1',
                'python_version': platform.python_version(), 'numpy_version': np.__version__, 'pandas_version': pd.__version__,
                'scipy_version': scipy.__version__, 'sklearn_version': sklearn.__version__, 'figures': [Path(p).name for p in FIGURE_LOG]}
    (RESULTS / 'analysis_manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')
    print('Analysis complete. Figures:', len(FIGURE_LOG), 'Report:', ROOT / 'REPORT_zh.md')


def load_profile_bundle(directory, method):
    directory = Path(directory)
    result_dir = directory / 'results' if (directory / 'results').is_dir() else directory
    profile_path = result_dir / 'perturbation_profiles.parquet'
    profiles = pd.read_parquet(profile_path)
    feature_path = result_dir / 'selected_feature_columns.csv'
    if feature_path.is_file():
        features = pd.read_csv(feature_path).feature.tolist()
    elif method == 'DeSCORT':
        features = sorted((c for c in profiles if re.fullmatch(r'emb_\d+', str(c))),
                          key=lambda c: int(c.split('_')[-1]))
    else:
        raise FileNotFoundError(feature_path)
    if not features or set(features) - set(profiles):
        raise ValueError(f'{directory}: missing selected features')
    required = PROFILE_KEYS + ['n_source_files', 'n_input_wells']
    if set(required) - set(profiles):
        raise ValueError(f'{directory}: missing profile identity/count fields')
    summaries = list(result_dir.glob('summary_*.json'))
    summary = json.loads(summaries[0].read_text()) if summaries else {}
    summary['aggregation'] = summary.get('aggregation', 'preaggregated profiles; see input bundle')
    summary['final_profiles'] = len(profiles)
    return profiles, summary, features, profile_path



def plot_profile_hits(retrieval, root, args):
    """Plot existing per-gene Hit@K with the same weighting as pooled MRR."""
    sys.path.insert(0, str(args.qa_tools_dir))
    from audit_panel_alignment import require_matplotlib_panel_alignment
    methods = ['DeSCORT', 'Original']
    pairs = ['Compound–Crispr', 'Compound–Orf', 'Crispr–Orf']
    colors = {'DeSCORT': '#BC3C28', 'Original': '#0072B5'}
    if retrieval.duplicated(['method', 'metric', 'Gene']).any():
        raise ValueError('Duplicate per-gene retrieval identity')
    counts = retrieval.n_candidate_targets.unique()
    if len(counts) != 1 or counts[0] < 1:
        raise ValueError('Inconsistent retrieval candidate counts')
    candidates = int(counts[0])
    mpl.rcParams.update({'font.family': 'sans-serif',
        'font.sans-serif': [args.font_family, 'DejaVu Sans'], 'font.size': 8,
        'pdf.fonttype': 42, 'svg.fonttype': 'none',
        'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.3), dpi=300)
    rows = []
    for panel, (ax, k) in enumerate(zip(axes.flat, [1, 5, 10, 20])):
        for method, offset in [('DeSCORT', -.18), ('Original', .18)]:
            means, lows, highs = [], [], []
            for index, pair in enumerate(pairs):
                frame = retrieval.loc[retrieval.method.eq(method) & retrieval.metric.eq(pair)].sort_values('Gene')
                values = frame[f'Hit_at_{k}'].to_numpy(dtype=float)
                if not len(values) or not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
                    raise ValueError(f'Invalid Hit values: {method}, {pair}, {k}')
                mean = float(values.mean())
                rng = np.random.default_rng(args.seed + index)
                # Resample observed genes, retaining the existing within-gene compound mean.
                boot = values[rng.integers(0, len(values), size=(args.bootstraps, len(values)))].mean(axis=1)
                lo, hi = np.quantile(boot, [.025, .975])
                means.append(mean); lows.append(mean-lo); highs.append(hi-mean)
                rows.append(dict(method=method, metric=pair, top_k=k, n_genes=len(values),
                    n_candidate_targets=candidates, mean_hit=mean, ci95_low=lo, ci95_high=hi))
            ax.bar(np.arange(3)+offset, means, width=.32, color=colors[method], alpha=.75, label=method)
            ax.errorbar(np.arange(3)+offset, means, yerr=[lows, highs], fmt='none',
                        color='#26323A', capsize=3, linewidth=.8)
        ax.axhline(min(k/candidates, 1.), color='#7E8A90', linewidth=.8,
                   linestyle='--', label='Uniform-random rank')
        ax.set_xticks(np.arange(3), ['Compound–\nCrispr', 'Compound–\nOrf', 'Crispr–Orf'])
        ax.set_ylabel(f'Hit@{k} (higher = better)')
        ax.set_title(f'Hit@{k}', fontsize=9, pad=9)
        ax.set_ylim(bottom=0)
        ax.text(-.13, 1.06, chr(97+panel), transform=ax.transAxes,
                weight='bold', fontsize=9, va='bottom')
        ax.grid(axis='y', color='#E7EBEE', linewidth=.5); ax.set_axisbelow(True)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.5,.99),
               ncol=3, frameon=False, fontsize=8)
    fig.subplots_adjust(left=.105, right=.985, bottom=.09, top=.87, wspace=.32, hspace=.58)
    figure_dir, qa = root/'figures', root/'qa'
    figure_dir.mkdir(parents=True, exist_ok=True); qa.mkdir(parents=True, exist_ok=True)
    name = 'figure4_true_target_retrieval_Hit'
    fig.canvas.draw()
    require_matplotlib_panel_alignment(fig, json_out=str(qa/f'{name}.alignment.json'),
        tolerance_pt=1.5, gutter_tolerance_pt=1.5, require_panel_labels=True, strict=True)
    exported = export_figure(fig, figure_dir/name, qa, args)
    plt.close(fig)
    pd.DataFrame(rows).to_csv(root/'results/profile_Hit_plot_summary.csv', index=False)
    return exported


def export_figure(fig, stem, qa_dir, args):
    """Export selected formats and use a temporary PDF for QA when necessary."""
    from tempfile import TemporaryDirectory
    formats = list(dict.fromkeys(getattr(args, 'figure_formats', ['png', 'pdf'])))
    if not formats or set(formats) - {'png', 'pdf', 'svg', 'tiff'}:
        raise ValueError('Choose one or more of png, pdf, svg, tiff')
    paths = {
        'png': stem.with_suffix('.png'), 'pdf': stem.with_suffix('.pdf'),
        'svg': stem.with_suffix('.svg'), 'tiff': stem.with_suffix('.tiff'),
    }
    # Preserve the same exact canvas for these two comparison figures.
    fixed_canvas = (stem.name == 'figure1_paired_gene_distance_distributions'
                    or stem.name.startswith('top_k_overlap_curves_pca'))
    for extension in formats:
        options = {'bbox_inches': None if fixed_canvas else 'tight'}
        if extension in {'png', 'tiff'}:
            options['dpi'] = args.dpi
        fig.savefig(paths[extension], **options)
    if 'pdf' in formats:
        audit_pdf(stem.with_suffix('.pdf'), qa_dir, args)
    else:
        with TemporaryDirectory(prefix='descort_compare_pdf_') as temporary:
            temporary_pdf = Path(temporary) / f'{stem.name}.pdf'
            fig.savefig(temporary_pdf, bbox_inches=None if fixed_canvas else 'tight')
            audit_pdf(temporary_pdf, qa_dir, args)
    return paths['png' if 'png' in formats else formats[0]]


def audit_pdf(pdf, qa_dir, args):
    qa_dir.mkdir(parents=True, exist_ok=True)
    for tool, extra in [
        ('audit_pdf_text.py', ['--min-pt', '5', '--json']),
        ('audit_figure_collisions.py', ['--json-out', str(qa_dir / f'{pdf.stem}.collision.json')]),
    ]:
        run = subprocess.run([sys.executable, str(args.qa_tools_dir / tool), str(pdf), *extra],
                             capture_output=True, text=True)
        (qa_dir / f'{pdf.stem}.{tool.removesuffix(".py")}.log').write_text(run.stdout + run.stderr)
        if run.returncode:
            raise RuntimeError(f'PDF audit failed: {pdf.name}, {tool}; see {qa_dir}')


def run_temporal(args):
    """Execute the packaged temporal analysis; no external script is required."""
    from contextlib import redirect_stdout
    from .temporal_retrieval import run
    class Tee:
        def __init__(self, console, logfile):
            self.console, self.logfile = console, logfile
        def write(self, value):
            self.console.write(value)
            return self.logfile.write(value)
        def flush(self):
            self.console.flush()
            self.logfile.flush()
    print('Running built-in temporal retrieval stage', flush=True)
    with (args.output_dir / 'temporal_retrieval.log').open('w') as logfile:
        with redirect_stdout(Tee(sys.stdout, logfile)):
            run(args)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stages', nargs='+', choices=['comparison', 'temporal'], required=True)
    p.add_argument('--model-data-dir', type=Path)
    p.add_argument('--model-profiles-dir', type=Path, help='Optional existing 06_umap model bundle; overrides raw model aggregation')
    p.add_argument('--original-profiles-dir', type=Path, required=True, help='Original aggregated bundle produced by 06_umap or the legacy workflow')
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--cell-line', required=True)
    p.add_argument('--model-epochs', type=int, required=True)
    p.add_argument('--pca-variance', type=float, required=True)
    p.add_argument('--top-k', type=int, nargs=2, required=True)
    p.add_argument('--overlap-curve-step', type=int, default=5, help='Step between Top-k curve markers')
    p.add_argument('--overlap-curve-max', type=int, default=100, help='Largest Top-k curve threshold')
    p.add_argument('--bootstraps', type=int, required=True)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--threads', type=int, required=True)
    p.add_argument('--dpi', type=int, required=True)
    p.add_argument('--figure-formats', nargs='+', choices=['png', 'pdf', 'svg', 'tiff'],
                   default=['png', 'pdf'], help='Profile comparison figure formats (default: png pdf)')
    p.add_argument('--font-family', required=True)
    p.add_argument('--font-dir', type=Path)
    p.add_argument('--qa-tools-dir', type=Path, required=True)
    p.add_argument('--original-results', type=Path, required=True)
    p.add_argument('--descort-results', type=Path, required=True)
    p.add_argument('--descort-data-root', type=Path)
    p.add_argument('--raw-feature-list', type=Path, required=True)
    p.add_argument('--feature-vocab', type=Path, required=True)
    p.add_argument('--target-modalities', choices=['crispr', 'orf'], nargs='+', required=True)
    p.add_argument('--batch-scope', choices=['primary', 'all'], required=True)
    p.add_argument('--temporal-representation', choices=['pca95', 'native'], required=True)
    p.add_argument('--reference-seed', type=int, required=True)
    p.add_argument('--descort-top-k', type=int, required=True)
    p.add_argument('--original-worse-than', type=int, required=True)
    p.add_argument('--minimum-rank-gain', type=int, required=True)
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()
    for name, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, name, value.expanduser().resolve())
    if args.overlap_curve_step < 1 or args.overlap_curve_max < args.overlap_curve_step or args.overlap_curve_max % args.overlap_curve_step:
        p.error('overlap-curve-max must be a positive multiple of overlap-curve-step')
    if not 0 < args.pca_variance < 1:
        p.error('pca-variance must be between 0 and 1')
    if min(*args.top_k, args.bootstraps, args.model_epochs, args.threads, args.dpi,
           args.descort_top_k, args.original_worse_than) < 1 or args.minimum_rank_gain < 0:
        p.error('Counts, epochs and DPI must be positive; rank gain must be nonnegative')
    if len(set(args.stages)) != len(args.stages) or len(set(args.target_modalities)) != len(args.target_modalities):
        p.error('Duplicate stages or target modalities')
    if 'comparison' in args.stages and not (args.model_data_dir or args.model_profiles_dir):
        p.error('Comparison needs --model-data-dir or --model-profiles-dir')
    return args


def validate_comparison(args):
    profiles, _, features, _ = load_profile_bundle(args.original_profiles_dir, 'Original')
    if not np.isfinite(profiles[features].to_numpy(dtype=np.float32)).all():
        raise ValueError('Original contains nonfinite profile values')
    if args.model_profiles_dir:
        model, _, model_features, _ = load_profile_bundle(args.model_profiles_dir, 'DeSCORT')
        if not np.isfinite(model[model_features].to_numpy(dtype=np.float32)).all():
            raise ValueError('Model contains nonfinite profile values')
        left = pair_identities(model)[['Gene', 'Compound']]
        right = pair_identities(profiles)[['Gene', 'Compound']]
        membership = left.merge(right, on=['Gene', 'Compound'], how='outer', indicator=True)
        if not membership['_merge'].eq('both').all():
            raise ValueError('Model and Original Compound–gene identities differ')
    else:
        import pyarrow.parquet as pq
        for modality in MODALITIES:
            files = sorted((args.model_data_dir / modality).rglob('*.parquet'))
            if not files:
                raise FileNotFoundError(f'No model Parquets for {modality}')
            for path in files:
                fields = pq.read_schema(path).names
                if 'Metadata_Treatment' not in fields or not any(re.fullmatch(r'emb_\d+', c) for c in fields):
                    raise ValueError(f'Invalid model schema: {path}')
    print(f'Comparison input check passed: Original {len(profiles)} profiles, {len(features)} features', flush=True)


def write_combined_report(args):
    definitions = [
        '# Cross-modality comparison and MRR / Hit',
        f'Input configuration: {args.cell_line}, {args.model_epochs} epochs.',
        '两个阶段采用不同的聚合与检索单位，分别报告，不混成同一个 MRR。',
    ]
    tables = []
    if 'comparison' in args.stages:
        table = pd.read_csv(args.output_dir / 'profile_comparison/results/true_target_retrieval_summary.csv')
        table['workflow'] = 'pooled_profiles'
        table['comparison'] = table.metric
        table['unit'] = 'gene; compound profiles averaged within each gene'
        table['tie_rule'] = 'average rank'
        tables.append(table)
        definitions += [
            '## 聚合 profile 比较',
            '全部时间点合并聚合；各表示对全部 profile 独立拟合 PCA。候选为匹配的非对照基因。'
            '多 Compound 先在同基因内平均，再对基因等权；并列取平均名次。包括 Compound→CRISPR、Compound→ORF、CRISPR→ORF。',
            '[完整比较报告与 6 张图](profile_comparison/REPORT_zh.md)',
            'Profile Hit@K figures: see the profile comparison report for the selected export formats.',
        ]
    if 'temporal' in args.stages:
        table = pd.read_csv(args.output_dir / 'temporal_retrieval/overall_retrieval_summary.csv')
        table = table.rename(columns={'Recall_at_1': 'Hit_at_1', 'Recall_at_5': 'Hit_at_5', 'Recall_at_10': 'Hit_at_10'})
        table['workflow'] = 'time_stratified_drug_retrieval'
        table['unit'] = 'compound ID within each time comparison'
        table['tie_rule'] = 'worst tied rank; best annotated target'
        tables.append(table)
        definitions += [
            '## 分时间点的药物检索（内置时间检索）',
            '保留两个分支各四个时间组合；Compound 按 Broad Sample 聚合，PCA 在每个表示、时间比较的查询与候选上联合拟合。'
            '多注释靶点取最佳已知靶点，并列取最差名次。统计先对同一药物的四个组合求平均，采用药物配对检验。',
            '[时间检索完整说明](temporal_retrieval/README_zh.md)',
            '![MRR and Hit@5](temporal_retrieval/figures/overall_retrieval.png)',
        ]
    pd.concat(tables, ignore_index=True).to_csv(args.output_dir / 'retrieval_summary_all_workflows.csv', index=False)
    definitions += ['## 解读边界',
                    '较小绝对距离、Top-k overlap 和余弦符号一致性不能单独证明靶点识别准确。'
                    '数据参与过微调；结果是描述性/探索性分析，不是独立实验重复、因果关系或泛化验证。',
                    '全部实际参数、执行状态见 run_manifest.json；图形字体、面板对齐、碰撞审查记录在各阶段 qa/ 下。']
    (args.output_dir / 'REPORT_zh.md').write_text('\n\n'.join(definitions) + '\n', encoding='utf-8')
