#!/usr/bin/env python3
"""A549 chemical-gene ranking, MRR/Hit@K summaries and publication figures.

Self-contained merge of 05_rank_chemical_gene_cases.py and
report_20261028/01_plot_overall_retrieval.py. No imports from either script.
Run the matching .sh for the full pipeline, or --plot-only to reuse saved CSVs.
Default: Original vs DeSCORT 150 epochs, primary batch, both branches, PCA95.
No model is trained or loaded; existing Parquet profiles are required.
"""
import argparse
import json
import pickle
import re
import hashlib
import warnings
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pyarrow.types as pat
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from threadpoolctl import threadpool_limits
from scipy.stats import wilcoxon
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

HERE = Path(__file__).resolve().parent
PLOT_METHODS = ("DeSCORT", "Original")
COLORS = {"Original": "#0072B5", "DeSCORT": "#BC3C28"}
TITLE_SIZE = 23
BAR_WIDTH = .16

TARGET = "Metadata_Treatment"
REAGENT = "Metadata_Reagent"
WELL_KEYS = ["Metadata_Batch", "Metadata_Plate", "Metadata_Well"]
IDENTITY = WELL_KEYS + [TARGET, REAGENT]
PLATE_KEYS = ["modality", "period", "time_hours", "batch", "plate"]
METHODS = ("Original", "Foundation", "DeSCORT")
GENETIC_MODALITIES = ("crispr", "orf")
HOURS = {"compound": {"short": 24, "long": 48},
         "crispr": {"short": 96, "long": 144},
         "orf": {"short": 48, "long": 96}}


def read_manifest(directory, target_modalities=GENETIC_MODALITIES):
    files = sorted(directory.glob("*/selected_plates.csv"))
    if not files:
        raise FileNotFoundError(f"No comparison plate manifests under {directory}")
    frame = pd.concat([pd.read_csv(p, dtype=str) for p in files], ignore_index=True)
    frame = frame.loc[frame.modality.isin(["compound", *target_modalities])].copy()
    if frame.groupby(PLATE_KEYS).source_file.nunique().gt(1).any():
        raise ValueError(f"Conflicting plate paths under {directory}")
    return frame.drop_duplicates(PLATE_KEYS).sort_values(PLATE_KEYS).reset_index(drop=True)


def pickle_value(path):
    with path.open("rb") as handle:
        return pickle.load(handle)


def load_profiles(method, modality, period, manifest, features, labels, output):
    rows = manifest.loc[manifest.modality.eq(modality) & manifest.period.eq(period)]
    if rows.empty:
        raise ValueError(f"Missing {method}/{modality}/{period} plates")
    reagent = "Metadata_Target_Sequence" if modality == "crispr" else "Metadata_Broad_Sample"
    parts, audits = [], []
    for row in rows.itertuples(index=False):
        schema = pq.read_schema(row.source_file)
        required = [TARGET, reagent, "Metadata_Well"]
        if modality == "compound":
            required.append("Metadata_Pert_IName")
        missing = set(required + features).difference(schema.names)
        if missing:
            raise ValueError(f"Missing columns {sorted(missing)[:8]}: {row.source_file}")
        frame = pd.read_parquet(row.source_file, columns=required + features)
        before = len(frame)
        frame = frame.loc[frame[TARGET].isin(labels)].copy()
        for column in required:
            if frame[column].isna().any() or frame[column].astype(str).str.strip().eq("").any():
                raise ValueError(f"Missing non-control identity {column}: {row.source_file}")
            frame[column] = frame[column].astype(str)
        frame = pd.concat([frame[required], frame[features].fillna(0)], axis=1)
        frame = frame.groupby(required, dropna=False)[features].mean().reset_index()
        frame = frame.rename(columns={reagent: REAGENT})
        frame["Metadata_Batch"], frame["Metadata_Plate"] = row.batch, row.plate
        parts.append(frame)
        audits.append(dict(method=method, modality=modality, period=period,
                           batch=row.batch, plate=row.plate, source_file=row.source_file,
                           input_rows=before, evaluated_wells=len(frame), n_features=len(features)))
    frame = pd.concat(parts, ignore_index=True).sort_values(WELL_KEYS).reset_index(drop=True)
    if frame.duplicated(WELL_KEYS).any():
        raise ValueError(f"Duplicate well identity: {method}/{modality}/{period}")
    if not np.isfinite(frame[features].to_numpy()).all():
        raise ValueError(f"Nonfinite features: {method}/{modality}/{period}")
    # Different representations must use exactly the same wells and identities.
    reference_path = output / f"{modality}_{period}_aligned_wells.csv"
    identity = IDENTITY + (["Metadata_Pert_IName"] if modality == "compound" else [])
    if method == METHODS[0]:
        frame[identity].to_csv(reference_path, index=False)
    elif not frame[identity].equals(pd.read_csv(reference_path, dtype=str, keep_default_na=False)):
        raise ValueError(f"Well/target/reagent/name mismatch: {method}/{modality}/{period}")
    if modality == "compound":
        groups = frame.groupby(REAGENT, sort=True)
        names = groups.Metadata_Pert_IName.nunique()
        if names.gt(1).any():
            raise ValueError("One compound ID has conflicting compound names")
        metadata = groups.agg(
            compound_name=("Metadata_Pert_IName", "first"),
            annotated_targets=(TARGET, lambda x: "|".join(sorted(set(x)))),
            n_wells=(TARGET, "size"),
        ).reset_index().rename(columns={REAGENT: "compound_id"})
        vectors = groups[features].mean().to_numpy(dtype=np.float64)
    elif modality == "crispr":
        # Equal contribution per guide: average wells within guide, then take
        # a coordinate-wise median across guides for each gene.
        guide_means = frame.groupby([TARGET, REAGENT], sort=True)[features].mean().reset_index()
        groups = guide_means.groupby(TARGET, sort=True)
        vectors = groups[features].median().to_numpy(dtype=np.float64)
        metadata = groups[REAGENT].nunique().rename("n_guides").reset_index().rename(columns={TARGET: "gene"})
    elif modality == "orf":
        # ORF evidence is a separate overexpression profile per target. Average
        # the wells directly; do not apply CRISPR's median-across-guides rule.
        groups = frame.groupby(TARGET, sort=True)
        vectors = groups[features].mean().to_numpy(dtype=np.float64)
        metadata = groups.agg(n_reagents=(REAGENT, "nunique"),
                              n_wells=(REAGENT, "size")).reset_index().rename(columns={TARGET: "gene"})
    else:
        raise ValueError(f"Unsupported modality: {modality}")
    if (np.linalg.norm(vectors, axis=1) == 0).any():
        raise ValueError(f"Zero-norm aggregated profiles: {method}/{modality}/{period}")
    # Preserve unnormalized aggregated vectors: PCA must precede the cosine
    # normalization, matching the order in the earlier PCA95 workflow.
    return metadata, vectors, audits


def select_features(manifests, allowed):
    """Use a separate feature intersection for each retrieval branch.

    Adding ORF must not change the existing Compound/CRISPR feature space.
    """
    result = {}
    for method, manifest in manifests.items():
        schemas = [pq.read_schema(p) for p in manifest.source_file]
        if not schemas:
            raise ValueError(f"No selected plates: {method}")
        numeric = [set(f.name for f in schema if not f.name.startswith("Meta")
                       and (pat.is_floating(f.type) or pat.is_integer(f.type))) for schema in schemas]
        if method == "Original":
            result[method] = sorted(set.intersection(*numeric) & allowed)
        else:
            embeddings = [set(c for c in cols if c.startswith("emb_")) for cols in numeric]
            if any(s != embeddings[0] for s in embeddings):
                raise ValueError(f"Inconsistent embedding dimensions: {method}")
            result[method] = sorted(embeddings[0], key=lambda c: int(c.split("_")[-1]))
        if not result[method]:
            raise ValueError(f"No common numeric features: {method}")
    return result


def rank_vectors(query_vectors, candidate_vectors, representation):
    """Fit one shared PCA per method/comparison, then rank by cosine."""
    pca = None
    if representation == "pca95":
        stacked = np.vstack([query_vectors, candidate_vectors])
        pca = PCA(n_components=0.95, svd_solver="full")
        projected = pca.fit_transform(stacked)
        query_vectors, candidate_vectors = projected[:len(query_vectors)], projected[len(query_vectors):]
    if ((np.linalg.norm(query_vectors, axis=1) == 0).any()
            or (np.linalg.norm(candidate_vectors, axis=1) == 0).any()):
        raise ValueError(f"Zero-norm vectors after {representation}")
    similarities = normalize(query_vectors) @ normalize(candidate_vectors).T
    # Conservative tie handling: a tie spanning positions 1..10 gets rank 10.
    ranks = pd.DataFrame(similarities).rank(axis=1, ascending=False, method="max").to_numpy(dtype=int)
    return similarities, ranks, pca, query_vectors.shape[1]


def mark_candidates(frame, args):
    frame = frame.copy()
    frame["rank_gain_vs_original"] = frame.Original_rank - frame.DeSCORT_rank
    if "Foundation_rank" in frame:
        frame["rank_gain_vs_foundation"] = frame.Foundation_rank - frame.DeSCORT_rank
    frame["passes_exploratory_rule"] = ((frame.DeSCORT_rank <= args.descort_top_k)
        & (frame.Original_rank > args.original_worse_than)
        & (frame.rank_gain_vs_original >= args.minimum_rank_gain))
    return frame


def save_tables(all_pairs, queries, output):
    all_pairs.to_csv(output / "all_drug_gene_ranks.csv", index=False)
    queries.to_csv(output / "annotated_target_retrieval.csv", index=False)
    queries.groupby(["target_modality", "comparison", "method"], as_index=False).agg(
        n_drugs=("compound_id", "size"), MRR=("reciprocal_rank", "mean"),
        Recall_at_1=("recall_at_1", "mean"), Recall_at_5=("recall_at_5", "mean"),
        Recall_at_10=("recall_at_10", "mean"),
    ).to_csv(output / "overall_retrieval_summary.csv", index=False)
    candidates = all_pairs.loc[all_pairs.passes_exploratory_rule].copy()
    candidates = candidates.sort_values(["annotated_pair", "DeSCORT_rank", "rank_gain_vs_original"], ascending=[False, True, False])
    candidates.loc[candidates.annotated_pair].to_csv(output / "A_annotated_candidates.csv", index=False)
    candidates.loc[~candidates.annotated_pair].to_csv(output / "B_unannotated_candidates_NEEDS_LITERATURE.csv", index=False)
    candidates.groupby(["target_modality", "compound_id", "compound_name", "gene", "annotated_pair"], as_index=False).agg(
        n_time_combinations_passing=("comparison", "nunique"),
        combinations=("comparison", lambda s: "|".join(sorted(s))),
    ).to_csv(output / "candidate_time_stability.csv", index=False)
    candidates.assign(
        literature_status="not_reviewed", evidence_type="", citation="", mechanism_notes="",
        same_cell_context="", confound_notes="", final_case_decision="pending",
    ).to_csv(output / "literature_review_template.csv", index=False)
    return candidates


def calculate_branch(args, target, manifests, labels, allowed, output):
    output.mkdir(parents=True, exist_ok=True)
    branch = {m: d.loc[d.modality.isin(["compound", target])].copy() for m, d in manifests.items()}
    features = select_features(branch, allowed)
    profiles, metadata, audit = {}, {}, []
    for modality in ("compound", target):
        for period in ("short", "long"):
            reference_meta = None
            for method in manifests:
                print(f"[{target}] Loading {method}/{modality}/{period}", flush=True)
                meta, vectors, rows = load_profiles(method, modality, period, branch[method], features[method], labels, output)
                if reference_meta is None:
                    reference_meta = meta
                elif not meta.equals(reference_meta):
                    raise ValueError(f"Aggregated profile identities differ: {method}/{modality}/{period}")
                profiles[method, modality, period] = vectors
                metadata[modality, period] = meta
                audit.extend(rows)
    ranked_pairs, query_scores, representation_audit = [], [], []
    for compound_period in ("short", "long"):
        for genetic_period in ("short", "long"):
            comparison = f"compound_{compound_period}_vs_{target}_{genetic_period}"
            compounds = metadata["compound", compound_period]
            genes = metadata[target, genetic_period].gene.tolist()
            base = pd.DataFrame({
                "compound_id": np.repeat(compounds.compound_id.to_numpy(), len(genes)),
                "compound_name": np.repeat(compounds.compound_name.to_numpy(), len(genes)),
                "annotated_targets": np.repeat(compounds.annotated_targets.to_numpy(), len(genes)),
                "gene": np.tile(genes, len(compounds)),
            })
            base["target_modality"] = target
            base["comparison"] = comparison
            base["compound_period"] = compound_period
            base["genetic_period"] = genetic_period
            base["compound_time_hours"] = HOURS["compound"][compound_period]
            base["genetic_time_hours"] = HOURS[target][genetic_period]
            base["n_candidate_genes"] = len(genes)
            base["annotated_pair"] = [g in t.split("|") for g, t in zip(base.gene, base.annotated_targets)]
            for method in manifests:
                query_vectors = profiles[method, "compound", compound_period]
                candidate_vectors = profiles[method, target, genetic_period]
                similarities, ranks, pca, dimensions = rank_vectors(query_vectors, candidate_vectors, args.representation)
                if pca is not None:
                    pd.DataFrame({
                        "component": np.arange(1, pca.n_components_ + 1),
                        "explained_variance_ratio": pca.explained_variance_ratio_,
                        "cumulative_variance_ratio": np.cumsum(pca.explained_variance_ratio_),
                    }).to_csv(output / f"{comparison}_{method}_pca_variance.csv", index=False)
                representation_audit.append(dict(
                    target_modality=target, comparison=comparison, method=method, representation=args.representation,
                    input_dimensions=query_vectors.shape[1], analysis_dimensions=dimensions,
                    explained_variance_ratio=float(pca.explained_variance_ratio_.sum()) if pca is not None else None,
                ))
                base[f"{method}_similarity"] = similarities.ravel()
                base[f"{method}_rank"] = ranks.ravel()
                for i, row in compounds.iterrows():
                    truth = np.array([g in row.annotated_targets.split("|") for g in genes])
                    if not truth.any():
                        raise ValueError(f"No annotated {target} candidate for {row.compound_id}")
                    best = int(ranks[i, truth].min())
                    query_scores.append(dict(target_modality=target, comparison=comparison, method=method,
                        compound_id=row.compound_id, compound_name=row.compound_name, annotated_targets=row.annotated_targets,
                        compound_time_hours=HOURS["compound"][compound_period], genetic_time_hours=HOURS[target][genetic_period],
                        best_annotated_rank=best, reciprocal_rank=1/best,
                        recall_at_1=int(best<=1), recall_at_5=int(best<=5), recall_at_10=int(best<=10)))
            ranked_pairs.append(mark_candidates(base, args))
    all_pairs = pd.concat(ranked_pairs, ignore_index=True)
    queries = pd.DataFrame(query_scores)
    pd.DataFrame(representation_audit).to_csv(output / "representation_audit.csv", index=False)
    pd.DataFrame(audit).to_csv(output / "input_audit.csv", index=False)
    pd.DataFrame([dict(method=m, feature=f) for m, fs in features.items() for f in fs]).to_csv(output / "features_used.csv", index=False)
    for method, manifest in branch.items():
        manifest.assign(method=method).to_csv(output / f"{method}_selected_plates.csv", index=False)
    candidates = save_tables(all_pairs, queries, output)
    print(f"[{target}] {len(all_pairs):,} ranked pairs; A={int(candidates.annotated_pair.sum())}, B={int((~candidates.annotated_pair).sum())}", flush=True)
    return all_pairs, queries, representation_audit, features


def combine_evidence(all_pairs, target_modalities, methods=METHODS):
    """Keep every branch/time/method rank; never take a best rank across branches."""
    identity = ["compound_id", "compound_name", "annotated_targets", "gene", "annotated_pair"]
    values = [f"{method}_rank" for method in methods] + [
        "n_candidate_genes", "rank_gain_vs_original", "passes_exploratory_rule"]
    if "rank_gain_vs_foundation" in all_pairs:
        values.append("rank_gain_vs_foundation")
    summary = None
    for (target, cp, gp), part in all_pairs.groupby(["target_modality", "compound_period", "genetic_period"], sort=True):
        prefix = f"{target}_compound{HOURS['compound'][cp]}h_genetic{HOURS[target][gp]}h"
        block = part[identity + values].rename(columns={c:f"{prefix}_{c}" for c in values})
        if block.duplicated(identity).any():
            raise ValueError(f"Duplicate pair evidence in {prefix}")
        summary = block if summary is None else summary.merge(block, on=identity, how="outer", validate="one_to_one")
    for target in target_modalities:
        counts = all_pairs.loc[all_pairs.target_modality.eq(target)].groupby(identity, as_index=False).passes_exploratory_rule.sum().rename(
            columns={"passes_exploratory_rule":f"n_{target}_time_combinations_passing"})
        summary = summary.merge(counts, on=identity, how="left", validate="one_to_one")
        column = f"n_{target}_time_combinations_passing"
        summary[column] = summary[column].fillna(0).astype(int)
    summary["evidence_modalities"] = summary.apply(
        lambda row: "|".join(m for m in target_modalities if row[f"n_{m}_time_combinations_passing"] > 0), axis=1)
    return summary


def candidate_evidence_overview(all_pairs, evidence, target_modalities, methods=METHODS):
    """Readable candidate overview; detailed time-specific ranks remain separate."""
    identity = ["compound_id", "compound_name", "annotated_targets", "gene", "annotated_pair"]
    counts = [f"n_{m}_time_combinations_passing" for m in target_modalities]
    overview = evidence.loc[evidence.evidence_modalities.ne(""), identity + counts + ["evidence_modalities"]].copy()
    for target in target_modalities:
        data = all_pairs.loc[all_pairs.target_modality.eq(target)]
        for method in methods:
            ranges = data.groupby(identity)[f"{method}_rank"].agg(["min", "max"]).reset_index()
            column = f"{target}_{method}_rank_range"
            ranges[column] = ranges.apply(
                lambda row: str(int(row['min'])) if row['min'] == row['max']
                else f"{int(row['min'])}-{int(row['max'])}", axis=1)
            overview = overview.merge(ranges[identity + [column]], on=identity, how="left", validate="one_to_one")
    return overview


def configure_fonts(font_dir=None, family="Arial"):
    """Use installed/user-supplied fonts; local legacy assets are optional."""
    directories = ([Path(font_dir)] if font_dir else []) + [HERE/"fonts", HERE/"report_20261028/fonts"]
    if font_dir is not None and not Path(font_dir).is_dir():
        raise FileNotFoundError(f"Font directory does not exist: {font_dir}")
    for directory in directories:
        if directory.is_dir():
            for path in sorted(directory.iterdir()):
                if path.suffix.lower() in {".ttf", ".otf"}:
                    font_manager.fontManager.addfont(str(path))
    try:
        resolved = Path(font_manager.findfont(family, fallback_to_default=False))
    except ValueError:
        warnings.warn(f"Font {family!r} is unavailable; using DejaVu Sans. "
                      "Install Arial or pass --font-dir for an exact style match.")
        resolved = Path(font_manager.findfont("DejaVu Sans", fallback_to_default=False))
    actual_family = font_manager.FontProperties(fname=str(resolved)).get_name()
    plt.rcParams.update({
        "font.family": actual_family, "font.size": 15,
        "axes.labelsize": 17, "axes.titlesize": 18,
        "xtick.labelsize": 17, "ytick.labelsize": 14,
        "legend.fontsize": 16, "pdf.fonttype": 42, "ps.fonttype": 42,
        "svg.fonttype": "none", "axes.spines.top": False, "axes.spines.right": False,
    })
    return resolved, actual_family


def bh_adjust(pvalues):
    """Benjamini-Hochberg adjustment across the displayed comparisons."""
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p)
    adjusted = np.minimum.accumulate((p[order]*len(p)/np.arange(1, len(p)+1))[::-1])[::-1]
    result = np.empty_like(adjusted)
    result[order] = np.minimum(adjusted, 1.)
    return result

def significance(value):
    if value < .001:
        return '***'
    if value < .01:
        return '**'
    if value < .05:
        return '*'
    return 'n.s.'

def plot_overall_retrieval(summary, queries, output_dir, *, ranking_dir,
                           font_dir=None, font_family="Arial", descort_epochs=150, dpi=300, export_args=None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ranking_dir = Path(ranking_dir)
    # Ranking can include Foundation; these figures compare DeSCORT and Original.
    summary = summary.loc[summary.method.isin(PLOT_METHODS)].copy()
    queries = queries.loc[queries.method.isin(PLOT_METHODS)].copy()
    if set(summary.method) != set(PLOT_METHODS) or set(queries.method) != set(PLOT_METHODS):
        raise ValueError("Plots require both Original and DeSCORT results")
    modes = [mode for mode in GENETIC_MODALITIES if mode in set(queries.target_modality)]
    if not modes or set(summary.target_modality) != set(modes):
        raise ValueError("Missing or inconsistent plot branches")
    requested_font_family = font_family
    font_path, font_family = configure_fonts(font_dir, font_family)
    keys = ['target_modality', 'comparison', 'method']
    if queries.duplicated(keys+['compound_id']).any():
        raise ValueError('Duplicate method/query identity')
    counts = queries.groupby(['target_modality', 'method', 'compound_id']).comparison.nunique()
    if not counts.eq(4).all():
        raise ValueError('Every drug must have all four time combinations for both methods')
    rebuilt = queries.groupby(keys)[['reciprocal_rank', 'recall_at_5']].mean().rename(
        columns={'reciprocal_rank': 'MRR', 'recall_at_5': 'Recall_at_5'})
    expected = summary.set_index(keys)[['MRR', 'Recall_at_5']]
    pd.testing.assert_frame_equal(rebuilt.sort_index(), expected.sort_index(),
                                  check_exact=False, atol=1e-12, rtol=1e-12)

    # Collapse reused times before testing. A drug ID is one paired unit,
    # not four independent observations. These are exploratory comparisons
    # on reused training data, not independent experimental validation.
    per_drug = queries.groupby(['target_modality', 'compound_id', 'method'], as_index=False).agg(
        reciprocal_rank=('reciprocal_rank', 'mean'),
        hit_at_5=('recall_at_5', 'mean'),
        n_time_combinations=('comparison', 'nunique'))
    tests = []
    for metric, query_column in [('MRR', 'reciprocal_rank'), ('Hit@5', 'hit_at_5')]:
        for mode in modes:
            paired = per_drug.loc[per_drug.target_modality.eq(mode)].pivot(
                index='compound_id', columns='method', values=query_column)
            if paired[list(PLOT_METHODS)].isna().any().any():
                raise ValueError('Unpaired drug IDs')
            # Round differences to avoid floating point artifacts splitting ties.
            diff = np.round((paired.DeSCORT-paired.Original).to_numpy(), 12)
            if np.any(diff):
                result = wilcoxon(diff, alternative='two-sided', zero_method='pratt',
                                  correction=False, method='approx')
                statistic, pvalue = float(result.statistic), float(result.pvalue)
            else:
                statistic, pvalue = 0., 1.
            tests.append({
                'metric': metric, 'target_modality': mode,
                'method_1': 'Original', 'method_2': 'DeSCORT',
                'n_paired_drug_ids': len(paired), 'n_nonzero_pairs': int(np.count_nonzero(diff)),
                'Original_mean': float(paired.Original.mean()),
                'DeSCORT_mean': float(paired.DeSCORT.mean()),
                'mean_paired_difference': float((paired.DeSCORT-paired.Original).mean()),
                'wilcoxon_statistic': statistic, 'p_value': pvalue,
                'test': 'two-sided paired Wilcoxon; Pratt zeros; asymptotic; differences rounded to 12 decimals',
                'unit': 'drug ID after averaging its four time combinations',
                'interpretation': 'exploratory_on_reused_training_data_not_independent_validation',
            })
    stats = pd.DataFrame(tests)
    stats['p_adjusted_BH'] = bh_adjust(stats.p_value)
    stats['significance'] = stats.p_adjusted_BH.map(significance)
    stats.to_csv(output_dir/'overall_retrieval_statistics.csv', index=False)
    per_drug.to_csv(output_dir/'overall_retrieval_per_drug.csv', index=False)

    def export_plot(figure, name):
        from .comparison import export_figure
        import sys
        sys.path.insert(0, str(export_args.qa_tools_dir))
        from audit_panel_alignment import require_matplotlib_panel_alignment
        qa = ranking_dir / 'qa'
        qa.mkdir(parents=True, exist_ok=True)
        figure.canvas.draw()
        require_matplotlib_panel_alignment(figure, json_out=str(qa / f'{name}.alignment.json'),
            require_panel_labels=False, strict=True)
        export_figure(figure, output_dir / name, qa, export_args)

    bars = []
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.9))
    # Group centers were [0, 1]. Keep bars legible while closing the empty
    # interval, and keep the two metric panels separate.
    centers = np.arange(len(modes), dtype=float)*.72
    width, offset = BAR_WIDTH, .13
    def draw_panel(ax, metric, label):
        maximum = 0.
        heights = {}
        for mi, method in enumerate(PLOT_METHODS):
            positions = centers+(2*mi-1)*offset
            values, deviations = [], []
            for mode in modes:
                observed = summary.loc[summary.target_modality.eq(mode) & summary.method.eq(method), metric].to_numpy()
                if len(observed) != 4:
                    raise ValueError('Four time-combination scores required per bar')
                mean, sd = float(observed.mean()), float(observed.std(ddof=1))
                values.append(mean)
                deviations.append(sd)
                heights[(mode, method)] = max(float(observed.max()), mean+sd)
                maximum = max(maximum, heights[(mode, method)])
                bars.append(dict(metric=label, target_modality=mode, method=method,
                                 mean=mean, sd_across_four_times=sd, n_time_combinations=4))
            ax.bar(positions, values, width=width, color=COLORS[method], alpha=.86,
                   label=method, yerr=deviations, capsize=5,
                   error_kw={'ecolor': '#505050', 'elinewidth': 1.3, 'capthick': 1.3}, zorder=2)
            for x, mode, mean in zip(positions, modes, values):
                observed = summary.loc[summary.target_modality.eq(mode) & summary.method.eq(method), metric].to_numpy()
                ax.scatter(x+np.linspace(-.035, .035, 4), observed, s=22,
                           facecolor='white', edgecolor='#505050', linewidth=.7, alpha=.85, zorder=4)
                ax.text(x, heights[(mode, method)]+maximum*.03, f'{mean:.3f}',
                        color='#26323A', weight='bold', fontsize=10,
                        ha='center', va='bottom', zorder=5)
        for center, mode in zip(centers, modes):
            height = max(heights[(mode, m)] for m in PLOT_METHODS)+maximum*.075
            depth = maximum*.035
            ax.plot([center-offset, center-offset, center+offset, center+offset],
                    [height, height+depth, height+depth, height], color='#505050', lw=1.)
            test = stats.loc[stats.metric.eq(label) & stats.target_modality.eq(mode)].iloc[0]
            ax.text(center, height+depth+maximum*.018, test.significance,
                    ha='center', va='bottom', fontsize=17, weight='bold')
        ax.set_xticks(centers, [f'Compound → {mode.upper()}' for mode in modes])
        ax.set_xlim(-.405, float(centers[-1])+.405)
        ax.set_ylim(0, maximum*1.27)
        ax.set_ylabel(label)
        ax.grid(False)
        ax.set_axisbelow(True)
        ax.tick_params(axis='x', length=0, pad=10)
    for ax, metric, label in zip(axes, ['MRR', 'Recall_at_5'], ['MRR', 'Hit@5']):
        draw_panel(ax, metric, label)
    fig.suptitle('A549 chemical–gene retrieval',
                 fontsize=TITLE_SIZE, fontweight='bold', y=.99)
    fig.legend(*axes[0].get_legend_handles_labels(), loc='upper center', bbox_to_anchor=(.5,.925), ncol=2, fontsize=12, frameon=False)
    # No footer/caption is rendered in the image. Its definitions are saved
    # in a separate Markdown file and the report, rather than removed.
    fig.subplots_adjust(left=.075, right=.99, bottom=.15, top=.85, wspace=.16)
    export_plot(fig, 'overall_retrieval')
    plt.close(fig)
    for metric, label, suffix in [('MRR', 'MRR', 'MRR'), ('Recall_at_5', 'Hit@5', 'Hit5')]:
        single, ax = plt.subplots(figsize=(6.8, 5.9))
        draw_panel(ax, metric, label)
        single.suptitle('A549 chemical–gene retrieval', fontsize=TITLE_SIZE, fontweight='bold', y=.99)
        single.legend(*ax.get_legend_handles_labels(), loc='upper center', bbox_to_anchor=(.5,.925), ncol=2, fontsize=12, frameon=False)
        single.subplots_adjust(left=.13, right=.99, bottom=.15, top=.85)
        export_plot(single, f'overall_retrieval_{suffix}')
        plt.close(single)
    pd.DataFrame(bars).drop_duplicates(['metric', 'target_modality', 'method']).to_csv(
        output_dir/'overall_retrieval_bar_summary.csv', index=False)
    date = datetime.now(ZoneInfo("Asia/Shanghai")).date().isoformat()
    metadata = {
        "date": date, "actual_descort_epochs": descort_epochs,
        "figure_title": "A549 chemical–gene retrieval", "title_fontsize_pt": TITLE_SIZE,
        "title_fontweight": "bold", "legend_visible": True,
        "bar_order_left_to_right": list(PLOT_METHODS), "colors": COLORS,
        "bar_width": BAR_WIDTH, "bar_alpha": .86, "x_tick_fontsize_pt": 17,
        "dpi": dpi, "target_modalities": modes,
        "rank_methods_in_input": sorted(pd.read_csv(ranking_dir/"overall_retrieval_summary.csv").method.unique()),
        "plot_methods": list(PLOT_METHODS),
        "font_family_requested": requested_font_family, "font_family": font_family,
        "resolved_font_path": str(font_path),
        "font_sha256": hashlib.sha256(font_path.read_bytes()).hexdigest(),
        "error_bars": "sample SD (ddof=1) across four time-combination aggregate scores; not SE/CI",
        "points": "four time-combination aggregate scores; not independent replicates",
        "test": "two-sided paired Wilcoxon; Pratt zeros; asymptotic; differences rounded to 12 decimals",
        "test_unit": "drug ID after averaging four time combinations; matched methods",
        "n_drug_ids_per_branch": {m:int(per_drug.loc[per_drug.target_modality.eq(m)].compound_id.nunique()) for m in modes},
        "multiple_testing": f"Benjamini-Hochberg across {len(stats)} metric-by-branch comparisons",
        "star_thresholds": "adjusted p: * <0.05; ** <0.01; *** <0.001; otherwise n.s.",
        "limitations": "Exploratory analysis on reused training data; drug IDs can share names, targets and candidate vectors. Not independent wet-lab significance, causality or generalization.",
        "script": {"path":str(Path(__file__).resolve()), "sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
        "inputs": {str(ranking_dir/name):hashlib.sha256((ranking_dir/name).read_bytes()).hexdigest()
                   for name in ["overall_retrieval_summary.csv", "annotated_target_retrieval.csv"]},
    }
    (output_dir/"overall_retrieval_plot_settings.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2)+"\n")
    notes = f"""# MRR / Hit@5 绘图与统计说明

生成日期：{date}。DeSCORT模型版本：{descort_epochs} epochs。

- 运行同目录的`06_compare.sh`：排名、指标汇总、绘图一并完成。
- 柱体和图例顺序设置为DeSCORT在左、Original在右；图中不显示图例。
- 图像：`overall_retrieval_MRR`、`overall_retrieval_Hit5`分别保存PNG/PDF/SVG；另保留合并图`overall_retrieval`。
- PNG为{dpi} dpi；PDF嵌入字体，SVG保留可编辑文字。当前实际字体为{font_family}。
- 颜色：DeSCORT #BC3C28，Original #0072B5；标题23 pt加粗、横坐标17 pt、柱宽0.16。
- 柱高：四个时间组合整体成绩的均值。误差棒：四个成绩的样本标准差（SD，ddof=1），不是SE或置信区间。圆点为四个时间组合。
- p值：每个药物ID先对四个时间组合求平均，再在两种方法之间做双侧配对Wilcoxon检验（Pratt处理零差，正态近似，差值保留12位小数处理浮点并列）。对实际绘制的{len(stats)}项比较统一作BH校正。* <0.05，** <0.01，*** <0.001，均依据校正后的p值。
- `overall_retrieval_statistics.csv`保存原始/校正p值和样本数；`overall_retrieval_per_drug.csv`保存检验数据，`overall_retrieval_bar_summary.csv`保存柱均值和SD。
- 可选Foundation参与排名时，仍只绘制Original与DeSCORT的比较，不将Foundation混入这两种方法的均值或检验。

数据参与过微调，药物ID可能共享药名、靶点和候选向量。四个时间组合不是独立实验重复；这些检验仅用于当前数据的探索性比较，不能作为独立湿实验验证、因果机制或泛化证据。误差棒与p值的统计单位不同。
"""
    (output_dir/"overall_retrieval_plot_README_zh.md").write_text(notes,encoding="utf-8")
    return stats


def find_project_root():
    """Locate the checkout after moving this script; never rely on parents[2]."""
    for directory in HERE.parents:
        if (directory/"common_files").is_dir() and (directory/"tutorials").is_dir():
            return directory
    return HERE


def plot_saved_results(args, output):
    summary_file = output/"overall_retrieval_summary.csv"
    queries_file = output/"annotated_target_retrieval.csv"
    for path in [summary_file, queries_file]:
        if not path.is_file():
            raise FileNotFoundError(f"Saved rankings not found: {path}; run without --plot-only first")
    settings_file = output/"settings.json"
    epochs = json.loads(settings_file.read_text()).get("descort_epochs", args.descort_epochs) if settings_file.exists() else args.descort_epochs
    figure_dir = args.figure_dir or output/"figures"
    stats = plot_overall_retrieval(pd.read_csv(summary_file), pd.read_csv(queries_file), figure_dir,
        ranking_dir=output, font_dir=args.font_dir, font_family=args.font_family,
        descort_epochs=epochs, dpi=args.dpi, export_args=args)
    print(stats[["metric","target_modality","n_paired_drug_ids","p_value","p_adjusted_BH","significance"]].to_string(index=False),flush=True)
    print(f"Figures saved to: {figure_dir}",flush=True)



def run(args):
    """Run time-stratified retrieval directly from comparison's configured arguments."""
    from argparse import Namespace
    values = dict(vars(args))
    values.update(output_dir=args.output_dir / 'temporal_retrieval',
                  figure_dir=args.output_dir / 'temporal_retrieval' / 'figures',
                  descort_epochs=args.model_epochs, representation=args.temporal_representation,
                  methods=['Original', 'DeSCORT'], foundation_results=None)
    args = Namespace(**values)
    for path in (args.descort_results,args.descort_data_root):
        epoch_labels=re.findall(r"(?:^|[/_])(\d+)epochs(?:[/_]|$)",str(path)) if path is not None else []
        if any(int(value)!=args.descort_epochs for value in epoch_labels):
            raise ValueError(f"Epoch label disagrees with model epochs: {path}")
    possible_sources = dict(zip(METHODS, [args.original_results, args.foundation_results, args.descort_results]))
    sources = {m: possible_sources[m] for m in METHODS if m in args.methods}
    manifests = {m: read_manifest(d, args.target_modalities) for m, d in sources.items()}
    if args.descort_data_root is not None:
        data = manifests["DeSCORT"].copy()
        data["manifest_source_file"] = data.source_file
        folders = {"compound": "Compound", "crispr": "Crispr", "orf": "Orf"}
        data["source_file"] = [str(args.descort_data_root / folders[row.modality]
            / row.batch / Path(row.source_file).name) for row in data.itertuples(index=False)]
        manifests["DeSCORT"] = data
    if args.batch_scope == "primary":
        manifests = {m: d.loc[d.batch.eq("2020_11_04_CPJUMP1")].copy() for m, d in manifests.items()}
    reference = set(map(tuple, manifests[METHODS[0]][PLATE_KEYS].to_numpy()))
    for method, manifest in manifests.items():
        if set(map(tuple, manifest[PLATE_KEYS].to_numpy())) != reference:
            raise ValueError(f"Selected plate sets differ: {method}")
        for modality in ("compound", *args.target_modalities):
            for period in ("short", "long"):
                if not ((manifest.modality == modality) & (manifest.period == period)).any():
                    raise ValueError(f"Missing plates for {method}/{modality}/{period}")
        for path in manifest.source_file:
            if not Path(path).is_file():
                raise FileNotFoundError(path)
    label_files = sorted(args.descort_results.glob(f"*/{args.reference_seed}/ap_scores_dict.pickle"))
    label_sets = [set(pd.read_pickle(p).gene.dropna()) for p in label_files]
    if not label_sets or any(s != label_sets[0] for s in label_sets):
        raise ValueError("Missing or inconsistent reference target sets")
    labels = label_sets[0] - {"nontargeting", "DMSO", "dmso"}
    allowed = set(pickle_value(args.raw_feature_list))
    allowed &= set(pickle_value(args.feature_vocab))
    configure_fonts(args.font_dir, args.font_family)
    if args.dry_run:
        print(json.dumps({"methods":list(sources), "target_modalities":args.target_modalities,
            "representation":args.representation, "batch_scope":args.batch_scope,
            "n_genes":len(labels), "plate_records_per_method":{m:len(v) for m,v in manifests.items()},
            "ranking_output":str(args.output_dir),
            "figure_output":str(args.figure_dir or args.output_dir/"figures")},indent=2),flush=True)
        return
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    # Refuse to repurpose the previous CRISPR-only folder as a combined run.
    existing_settings = output / "settings.json"
    if existing_settings.exists():
        previous = json.loads(existing_settings.read_text())
        prior_targets = previous.get("target_modalities", ["crispr"])
        if (set(prior_targets) != set(args.target_modalities)
                or previous.get("representation") != args.representation
                or previous.get("batch_scope") != args.batch_scope
                or previous.get("descort_epochs", 150) != args.descort_epochs
                or set(previous.get("methods", METHODS)) != set(sources)):
            raise ValueError("Output directory belongs to a different analysis; choose a new --output-dir")
    pair_parts, query_parts, rep_audit, branch_features = [], [], [], {}
    with threadpool_limits(limits=args.threads):
        for target in args.target_modalities:
            pairs, queries, reps, features = calculate_branch(args, target, manifests, labels, allowed, output / target)
            pair_parts.append(pairs); query_parts.append(queries); rep_audit.extend(reps)
            branch_features[target] = {m:len(f) for m,f in features.items()}
    all_pairs = pd.concat(pair_parts, ignore_index=True)
    queries = pd.concat(query_parts, ignore_index=True)
    candidates = save_tables(all_pairs, queries, output)
    evidence = combine_evidence(all_pairs, args.target_modalities, methods=tuple(sources))
    evidence.to_csv(output / "gene_evidence_summary.csv", index=False)
    candidate_evidence_overview(all_pairs, evidence, args.target_modalities, methods=tuple(sources)).to_csv(output / "candidate_gene_evidence.csv", index=False)
    pd.DataFrame(rep_audit).to_csv(output / "representation_audit.csv", index=False)
    settings = dict(batch_scope=args.batch_scope, target_modalities=args.target_modalities,
        methods=list(sources), descort_epochs=args.descort_epochs,
        descort_data_root=str(args.descort_data_root) if args.descort_data_root is not None else None,
        sources={m:str(p) for m,p in sources.items()}, features_by_target_modality=branch_features,
        raw_feature_list=str(args.raw_feature_list), feature_vocab=str(args.feature_vocab),
        reference_seed=args.reference_seed, merged_script="06_compare.py",
        representation=args.representation, native_vectors=args.representation == "native",
        pca_fit_scope="one shared query+candidate PCA per method/time comparison; no extra feature standardization" if args.representation == "pca95" else None,
        time_hours=HOURS, retrieval_direction="same-direction cosine similarity (higher is better)",
        aggregation=dict(compound="mean wells per Broad Sample", crispr="mean wells per guide, then median guides per target", orf="mean wells per target"),
        thresholds=dict(descort_top_k=args.descort_top_k, original_worse_than=args.original_worse_than, minimum_rank_gain=args.minimum_rank_gain),
        training_reuse=True, note="Class B is unannotated in these data, not verified. No cross-modality ranks/similarities are combined into a single score.")
    (output / "settings.json").write_text(json.dumps(settings, indent=2)+"\n", encoding="utf-8")
    (output / "README_zh.md").write_text(
        "# A549 药物→CRISPR / Orf 基因检索\n\n"
        "本目录由06_compare.py一次完成排名和绘图；两种方法比较图见figures/（若设置--figure-dir则见指定目录）。\n"
        "原始脚本05_rank_chemical_gene_cases.py及旧绘图脚本均不是运行依赖。\n"
        "本分析复用参与微调的数据，只作已见数据上的探索性分析，不证明泛化或因果机制。\n"
        f"本次分支：{', '.join(args.target_modalities)}；批次：{args.batch_scope}；表示：{args.representation}。\n"
        f"比较方法：{', '.join(sources)}；DeSCORT epochs：{args.descort_epochs}。\n"
        "crispr/ 与 orf/ 分别保存完整排名、A/B候选、时间稳定性、样本核对和实际特征清单。根目录保存带target_modality字段的合并长表。\n"
        f"gene_evidence_summary.csv 按药物ID、名称、目标基因并列展示全部时间组合和{len(sources)}种表示的排名；列名明确标出实际小时数。\n"
        "candidate_gene_evidence.csv 为易读候选总览：分别列出CRISPR/Orf在四个时间组合的排名范围和通过筛选的组合数；不是跨模态合成分数。详细逐时间排名见gene_evidence_summary.csv。\n"
        "该表只保留至少一个分支、一个时间组合通过筛选的配对。evidence_modalities表示哪些分支通过规则，不表示验证或要求两分支一致。\n"
        "每条分支四个时间组合单独排序；CRISPR为96/144h，Orf为48/96h，Compound为24/48h。不混成一个总分，也不取两分支中的最佳排名作为总体成绩。\n"
        "Original使用该分支所选文件、3344名单及词表的数值特征交集；因此添加Orf不会改变CRISPR分支的特征空间。每种表示内部的查询/候选使用相同特征。\n"
        f"先逐孔核对{len(sources)}种表示的Batch/Plate/Well/目标/试剂；Compound还核对名称。原始特征已预处理，不额外进行特征标准化。\n"
        "Compound按Broad Sample取孔均值；CRISPR先按目标+guide取均值再取guide中位数；Orf直接按目标取孔均值。\n"
        "PCA95每种表示、每个时间组合联合药物和候选拟合，再计算余弦相似度；不是UMAP距离。实际PCA维度可能不同。\n"
        "仅检索同方向相似表型。Orf过表达与CRISPR敲除作用不同，不要求同目标都高排名；本程序没有计算相反表型检索，Orf低排名不等于无生物学联系。\n"
        "并列使用最差并列名次；多标签药物的Recall_at_K列实际是至少一个已有注释靶点进入前K的比例。\n"
        "A表为已有注释的改善候选；B表只表示本数据未标注，仍需查完整注释及独立文献。筛选阈值不是显著性检验。\n"
        "时间组合并非独立实验重复。后续需核对guide、重复孔、剂量、毒性和原始图像后再挑生物学实例。\n",
        encoding="utf-8")
    print(f"Total class A rows: {int(candidates.annotated_pair.sum())}")
    print(f"Total class B rows requiring review: {int((~candidates.annotated_pair).sum())}")
    print(f"Rankings saved to: {output}", flush=True)
    plot_saved_results(args, output)
