#!/usr/bin/env python3
"""Cross-modality profiles, cosine rankings and UMAP for model and/or original data.

Configure inputs and analysis parameters in 06_umap.sh. Each input is analyzed
independently; model and original UMAP coordinate systems are not aligned.
"""
from pathlib import Path
import argparse
import json
import os
import pickle
import re
import sys

os.environ.setdefault('NUMBA_CACHE_DIR', '/tmp/numba_descort_cross_modality')
os.environ.setdefault('MPLCONFIGDIR', '/tmp/mpl_descort_cross_modality')
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pyarrow.types as pat
from sklearn.decomposition import PCA
from threadpoolctl import threadpool_limits

MODALITIES = ('Compound', 'Crispr', 'Orf')
CONTROL_NAMES = {'nontargeting', 'non-targeting', 'control', 'ctrl', 'dmso'}
CRISPR_GUIDE = 'Metadata_Target_Sequence'
DMSO_UNIT = '__DMSO_control_not_sgRNA__'
PROFILE_KEYS = ['Metadata_All_Modality', 'Metadata_Treatment', 'Metadata_Pert_IName']


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-data-dir', type=Path, help='Root containing Compound/Crispr/Orf embedding Parquets')
    p.add_argument('--original-data-dir', type=Path, help='Root containing Compound/Crispr/Orf raw Parquets')
    for modality in MODALITIES:
        p.add_argument(f'--original-{modality.lower()}-dir', type=Path,
                       help='Override this modality directory; all three overrides also work without a root')
    p.add_argument('--original-feature-list', type=Path, help='Required for original: morphology feature-list pickle')
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--representation', choices=['pca95', 'native'], required=True)
    p.add_argument('--pca-variance', type=float, required=True)
    p.add_argument('--top-n', type=int, required=True)
    p.add_argument('--n-neighbors', type=int, required=True)
    p.add_argument('--min-dist', type=float, required=True)
    p.add_argument('--umap-metric', choices=['cosine', 'euclidean'], required=True)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--threads', type=int, required=True)
    p.add_argument('--dpi', type=int, required=True)
    p.add_argument('--figure-formats', nargs='+', choices=['png', 'pdf', 'svg', 'tiff'],
                   default=['png', 'pdf'], help='Figure export formats (default: png pdf)')
    p.add_argument('--font-family', required=True)
    p.add_argument('--point-area', type=float, required=True)
    p.add_argument('--qa-tools-dir', type=Path, required=True, help='Directory of alignment/PDF audit tools')
    p.add_argument('--dry-run', action='store_true', help='Check schemas and metadata without PCA/UMAP or writing outputs')
    args = p.parse_args()
    original_overrides = [getattr(args, f'original_{m.lower()}_dir') for m in MODALITIES]
    if not args.model_data_dir and not args.original_data_dir and not any(original_overrides):
        p.error('Provide model input, original input, or both')
    if any(original_overrides) and not args.original_data_dir and not all(original_overrides):
        p.error('Without --original-data-dir, all three original modality directories are required')
    if (args.original_data_dir or any(original_overrides)) and not args.original_feature_list:
        p.error('--original-feature-list is required for original data')
    if not 0 < args.pca_variance < 1 or args.top_n < 1 or args.n_neighbors < 2:
        p.error('Require 0 < pca-variance < 1, top-n >= 1, n-neighbors >= 2')
    if args.min_dist < 0 or args.min_dist > 1 or min(args.threads, args.dpi, args.point_area) <= 0:
        p.error('Require 0 <= min-dist <= 1 and positive threads, dpi, point-area')
    # Interpret every supplied path independently of later directory changes.
    for key, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, key, value.expanduser().resolve())
    return args


def discover(directories):
    entries, seen = [], set()
    for modality, directory in directories.items():
        if not directory.is_dir():
            raise NotADirectoryError(directory)
        files = sorted(directory.rglob('*.parquet'))
        if not files:
            raise FileNotFoundError(f'{modality}: no Parquets under {directory}')
        for path in files:
            if path.resolve() in seen:
                raise ValueError(f'Input file assigned to more than one modality: {path}')
            seen.add(path.resolve())
            entries.append((modality, path, f'{modality}/{path.relative_to(directory).as_posix()}'))
    return entries


def select_features(entries, kind, feature_list):
    requested = None
    if kind == 'original':
        with feature_list.open('rb') as handle:
            vocab = pickle.load(handle)
        if isinstance(vocab, (str, bytes)):
            raise TypeError('Feature-list pickle must be a collection of feature names')
        requested = {str(v) for v in vocab}
    sets, schemas = [], []
    for modality, path, source_id in entries:
        schema = pq.read_schema(path)
        numeric = {field.name for field in schema
                   if pat.is_floating(field.type) or pat.is_integer(field.type)}
        if kind == 'model':
            names = {name for name in schema.names if re.fullmatch(r'emb_\d+', name)}
            if not names or not names.issubset(numeric):
                raise ValueError(f'{path}: missing or nonnumeric emb_N columns')
        else:
            names = {name for name in numeric if not name.startswith('Metadata')}
        sets.append(names)
        schemas.append(dict(source_file=str(path), source_id=source_id, modality=modality,
                            input_wells=pq.ParquetFile(path).metadata.num_rows,
                            numeric_features=len(names)))
    if kind == 'model':
        if any(names != sets[0] for names in sets[1:]):
            raise ValueError('Embedding columns differ between files')
        selected = sorted(sets[0], key=lambda name: int(name.split('_')[-1]))
        exclusions = pd.DataFrame(columns=['feature', 'reason', 'n_files_missing'])
    else:
        common = set.intersection(*sets)
        selected = sorted(name for name in requested & common if not name.startswith('Metadata'))
        exclusions = pd.DataFrame([
            dict(feature=name, reason='metadata_not_measurement' if name.startswith('Metadata')
                 else 'absent_or_nonnumeric_in_source', n_files_missing=sum(name not in fields for fields in sets))
            for name in sorted(requested - set(selected))
        ], columns=['feature', 'reason', 'n_files_missing'])
    if not selected:
        raise ValueError('No common numeric features remain')
    return selected, pd.DataFrame(schemas), exclusions


def clean_metadata(frame, modality, path):
    def strings(column):
        return frame[column].astype('string').str.strip()
    def missing(values):
        return values.isna() | values.str.lower().isin(['', 'nan', 'none', 'na']).fillna(False)
    labels = strings('Metadata_Treatment')
    repair = pd.Series(False, index=frame.index)
    if modality == 'Crispr' and 'Metadata_Pert_IName' in frame:
        repair = missing(labels) & strings('Metadata_Pert_IName').str.lower().eq('dmso').fillna(False)
        labels = labels.mask(repair, 'nontargeting')
    if missing(labels).any():
        raise ValueError(f'{path}: unresolved Metadata_Treatment')
    frame['Metadata_Treatment'] = labels.astype(str)
    if modality == 'Compound':
        names = strings('Metadata_Pert_IName')
        names = names.mask(labels.str.lower().isin(CONTROL_NAMES), 'nontargeting')
        if missing(names).any():
            raise ValueError(f'{path}: missing compound name')
        frame['Metadata_Pert_IName'] = names.astype(str)
    else:
        frame['Metadata_Pert_IName'] = labels.astype(str)
    if modality == 'Crispr':
        for column in [CRISPR_GUIDE, 'Metadata_Broad_Sample']:
            values = strings(column).mask(repair, DMSO_UNIT)
            if missing(values).any():
                raise ValueError(f'{path}: missing guide/Broad sample outside repaired DMSO rows')
            frame[column] = values.astype(str)
    return frame, int(repair.sum())


def read_wells(entries, features, dry_run):
    parts, audits = [], []
    for file_index, (modality, path, source_id) in enumerate(entries, start=1):
        if file_index == 1 or file_index % 10 == 0 or file_index == len(entries):
            print(f"Reading {file_index}/{len(entries)}: {source_id}", flush=True)
        required = ['Metadata_Treatment']
        if modality == 'Compound':
            required.append('Metadata_Pert_IName')
        if modality == 'Crispr':
            required += [CRISPR_GUIDE, 'Metadata_Broad_Sample']
        columns = pq.read_schema(path).names
        if set(required) - set(columns):
            raise ValueError(f'{path}: missing {set(required) - set(columns)}')
        if modality == 'Crispr' and 'Metadata_Pert_IName' in columns:
            required.append('Metadata_Pert_IName')
        frame = pd.read_parquet(path, columns=required + ([] if dry_run else features))
        frame, repaired = clean_metadata(frame, modality, path)
        if not dry_run and not np.isfinite(frame[features].to_numpy(dtype=np.float32)).all():
            raise ValueError(f'{path}: nonfinite feature values; no implicit imputation')
        frame['Metadata_All_Modality'] = modality
        frame['Metadata_Source_Parquet'] = source_id
        parts.append(frame)
        audits.append(dict(source_file=str(path), source_id=source_id, modality=modality,
                           input_wells=len(frame), repaired_crispr_dmso_rows=repaired))
    return pd.concat(parts, ignore_index=True), pd.DataFrame(audits)


def display(value):
    print(value.to_string() if hasattr(value, 'to_string') else value, flush=True)


def analyze(kind, entries, features, raw_wells, manifest, exclusions, args):
    ROOT = args.output_dir / kind
    RESULTS, FIGURES = ROOT / 'results', ROOT / 'figures'
    for directory in [RESULTS, FIGURES]:
        directory.mkdir(parents=True, exist_ok=True)
    embedding_columns = features
    manifest.to_csv(RESULTS / 'source_file_manifest.csv', index=False)
    pd.DataFrame({'feature': features}).to_csv(RESULTS / 'selected_feature_columns.csv', index=False)
    exclusions.to_csv(RESULTS / 'feature_exclusions.csv', index=False)
    profile_parts, intermediate_parts, aggregation_audit = [], [], []
    for modality in MODALITIES:
        print(f"[{kind}] Aggregating {modality}", flush=True)
        work = raw_wells.loc[raw_wells.Metadata_All_Modality.eq(modality)].copy()
        if work.empty:
            raise ValueError(f"No input wells for {modality}")
        if modality == "Compound":
            keys = ["Metadata_Treatment", "Metadata_Pert_IName"]
            rule = "all-well mean by Treatment + Pert_IName"
        elif modality == "Crispr":
            keys = ["Metadata_Treatment", CRISPR_GUIDE]
            # Current JUMP data: each Broad sample has exactly one sgRNA sequence.
            pairs_check = work[["Metadata_Broad_Sample", CRISPR_GUIDE]].drop_duplicates()
            if (pairs_check.groupby("Metadata_Broad_Sample")[CRISPR_GUIDE].nunique() > 1).any():
                raise ValueError("A Crispr Broad sample contains multiple guide sequences")
            pairs_check.to_csv(RESULTS / "crispr_guide_sample_mapping.csv", index=False)
            rule = "all-well guide mean by Treatment + Target_Sequence, then Treatment median"
        else:
            keys = ["Metadata_Treatment"]
            rule = "all-well mean by Treatment"
        grouped = work.groupby(keys, sort=True, dropna=False, observed=True)
        intermediate = grouped[embedding_columns].mean().reset_index()
        counts = grouped.agg(n_input_wells=("Metadata_Treatment", "size"),
                             n_source_files=("Metadata_Source_Parquet", "nunique")).reset_index()
        intermediate = intermediate.merge(counts, on=keys, validate="one_to_one")
        intermediate["Metadata_All_Modality"] = modality
        intermediate_parts.append(intermediate)
        intermediate.to_parquet(RESULTS / f"{modality.lower()}_first_level_mean_profiles.parquet", index=False)
        if modality in ("Compound", "Orf"):
            final = intermediate.copy()
            final["n_first_level_units"] = 1
            if modality == "Orf":
                final["Metadata_Pert_IName"] = final.Metadata_Treatment
        else:
            final = intermediate.groupby("Metadata_Treatment", sort=True, observed=True)[embedding_columns].median().reset_index()
            final["Metadata_Pert_IName"] = final.Metadata_Treatment
            unit_counts = intermediate.groupby("Metadata_Treatment").size().rename("n_first_level_units").reset_index()
            final = final.merge(unit_counts, on="Metadata_Treatment", validate="one_to_one")
            # Count distinct files from raw wells, not by summing counts per guide/sample.
            final_counts = work.groupby("Metadata_Treatment").agg(
                n_input_wells=("Metadata_Treatment", "size"),
                n_source_files=("Metadata_Source_Parquet", "nunique")).reset_index()
            final = final.merge(final_counts, on="Metadata_Treatment", validate="one_to_one")
        final["Metadata_All_Modality"] = modality
        final["aggregation_rule"] = rule
        final[embedding_columns] = final[embedding_columns].astype("float32")
        profile_parts.append(final)
        assert int(final.n_input_wells.sum()) == len(work)
        aggregation_audit.append({"modality": modality, "input_wells": len(work),
                                  "first_level_units": len(intermediate),
                                  "final_profiles": len(final), "aggregation_rule": rule})
    profiles = pd.concat(profile_parts, ignore_index=True)[PROFILE_KEYS + embedding_columns +
        ["n_source_files", "n_input_wells", "n_first_level_units", "aggregation_rule"]]
    profiles = profiles.sort_values(PROFILE_KEYS).reset_index(drop=True)
    if profiles.duplicated(PROFILE_KEYS).any() or not np.isfinite(profiles[embedding_columns].to_numpy()).all():
        raise ValueError("Duplicate or nonfinite final profiles")
    profiles.to_parquet(RESULTS / "perturbation_profiles.parquet", index=False)
    pd.concat(intermediate_parts, ignore_index=True).to_parquet(RESULTS / "first_level_mean_profiles.parquet", index=False)
    aggregation_audit = pd.DataFrame(aggregation_audit)
    aggregation_audit.to_csv(RESULTS / "aggregation_audit.csv", index=False)
    display(aggregation_audit)
    print("Final profiles:", len(profiles), "Controls are retained.")

    return analyze_profiles(kind, entries, features, profiles, manifest, aggregation_audit, args)


def analyze_profiles(kind, entries, features, profiles, manifest, aggregation_audit, args):
    """Analyze aggregated profiles; also permits checking saved aggregation outputs."""
    from umap import UMAP
    sys.path.insert(0, str(args.qa_tools_dir))
    from audit_panel_alignment import require_matplotlib_panel_alignment
    ROOT = args.output_dir / kind
    RESULTS, FIGURES = ROOT / 'results', ROOT / 'figures'
    embedding_columns = features
    TOP_N, REPRESENTATION = args.top_n, args.representation
    parquet_files = entries
    print(f"[{kind}] Projecting {len(profiles)} profiles ({len(embedding_columns)} features)", flush=True)
    raw_matrix = profiles[embedding_columns].to_numpy(dtype=np.float32)
    if REPRESENTATION == "pca95":
        if min(raw_matrix.shape) < 2:
            raise ValueError(f"PCA requires at least two rows and features: {raw_matrix.shape}")
        pca = PCA(n_components=args.pca_variance, svd_solver="full")
        vectors = pca.fit_transform(raw_matrix).astype(np.float32)
        pd.DataFrame({
            "component": np.arange(1, pca.n_components_ + 1),
            "explained_variance_ratio": pca.explained_variance_ratio_,
            "cumulative_variance_ratio": np.cumsum(pca.explained_variance_ratio_),
        }).to_csv(RESULTS / "pca95_explained_variance.csv", index=False)
        print("PCA95 retained components:", pca.n_components_)
        print("PCA95 explained variance:", round(float(pca.explained_variance_ratio_.sum()), 4))
    else:
        vectors = raw_matrix
    if not np.isfinite(vectors).all():
        raise ValueError("Nonfinite analysis vectors")
    print("Analysis matrix:", vectors.shape)

    def cosine_distance(a, b):
        norm = np.linalg.norm(a) * np.linalg.norm(b)
        if not np.isfinite(norm) or norm <= 1e-12:
            raise ValueError("Zero or nonfinite vector norm")
        return float(1.0 - np.dot(a, b) / norm)

    records = []
    for gene, group in profiles.groupby("Metadata_Treatment", sort=True, observed=True):
        compounds = group[group["Metadata_All_Modality"].eq("Compound")]
        crispr = group[group["Metadata_All_Modality"].eq("Crispr")]
        orf = group[group["Metadata_All_Modality"].eq("Orf")]
        if len(compounds) == 0 or len(crispr) != 1 or len(orf) != 1:
            continue
        crispr_vector = vectors[crispr.index[0]]
        orf_vector = vectors[orf.index[0]]
        crispr_orf_distance = cosine_distance(crispr_vector, orf_vector)
        for idx in compounds.index:
            compound_vector = vectors[idx]
            compound_crispr = cosine_distance(compound_vector, crispr_vector)
            records.append({
                "Gene": gene,
                "Compound": profiles.at[idx, "Metadata_Pert_IName"],
                "Crispr": profiles.at[crispr.index[0], "Metadata_Pert_IName"],
                "Orf": profiles.at[orf.index[0], "Metadata_Pert_IName"],
                "Distance_to_CRISPR": compound_crispr,
                "Similarity_to_CRISPR": 1.0 - compound_crispr,
                "Distance_to_ORF": cosine_distance(compound_vector, orf_vector),
                "Distance_CRISPR_ORF": crispr_orf_distance,
            })

    pairs = pd.DataFrame.from_records(records)
    if pairs.empty:
        raise ValueError("No genes were matched across all three modalities")
    pairs = pairs.sort_values(["Distance_to_CRISPR", "Gene", "Compound"]).reset_index(drop=True)
    pairs["pair_rank"] = np.arange(1, len(pairs) + 1)
    pairs["is_control"] = pairs["Gene"].astype(str).str.lower().isin(CONTROL_NAMES)
    pairs.to_csv(RESULTS / f"compound_crispr_orf_pairs_{REPRESENTATION}.csv", index=False)

    # Include nontargeting controls in ranking; retain the control flag for plotting/annotation.
    gene_rank = pairs.drop_duplicates("Gene").copy()
    gene_rank["gene_rank"] = np.arange(1, len(gene_rank) + 1)
    top_genes = gene_rank["Gene"].head(TOP_N).tolist()
    gene_rank["is_highlighted"] = gene_rank["Gene"].isin(top_genes)
    gene_rank.to_csv(RESULTS / f"gene_rank_{REPRESENTATION}.csv", index=False)

    distance_long = pd.concat([
        pairs[["Gene", "Compound", "Distance_to_CRISPR"]].rename(
            columns={"Distance_to_CRISPR": "cosine_distance"}
        ).assign(modality_pair="Compound–Crispr"),
        pairs[["Gene", "Compound", "Distance_to_ORF"]].rename(
            columns={"Distance_to_ORF": "cosine_distance"}
        ).assign(modality_pair="Compound–Orf"),
        pairs.drop_duplicates("Gene")[["Gene", "Distance_CRISPR_ORF"]].rename(
            columns={"Distance_CRISPR_ORF": "cosine_distance"}
        ).assign(Compound=pd.NA, modality_pair="Crispr–Orf"),
    ], ignore_index=True)
    distance_long.to_csv(RESULTS / f"pairwise_cosine_distances_{REPRESENTATION}.csv", index=False)
    print("Matched genes:", pairs["Gene"].nunique(), "Compound pairs:", len(pairs))
    print(f"Top {TOP_N} unique genes:", ", ".join(top_genes))
    display(gene_rank.head(TOP_N)[["gene_rank", "Gene", "Compound", "Distance_to_CRISPR"]])

    if len(profiles) <= args.n_neighbors:
        raise ValueError(f"UMAP requires more profiles than n_neighbors: {len(profiles)} <= {args.n_neighbors}")
    if (np.linalg.norm(vectors, axis=1) <= 1e-12).any():
        raise ValueError("Zero-norm analysis profile; cosine distance is undefined")
    print(f"[{kind}] Fitting UMAP", flush=True)
    umap_xy = UMAP(
        n_neighbors=args.n_neighbors, min_dist=args.min_dist, metric=args.umap_metric, random_state=args.seed, n_jobs=1
    ).fit_transform(vectors)
    coordinates = profiles[PROFILE_KEYS + ["n_source_files", "n_input_wells"]].copy()
    coordinates["UMAP1"] = umap_xy[:, 0]
    coordinates["UMAP2"] = umap_xy[:, 1]
    coordinates["is_highlighted"] = coordinates["Metadata_Treatment"].isin(top_genes)
    coordinates["is_top10_cross_modality"] = coordinates["Metadata_Treatment"].isin(gene_rank["Gene"].head(10))
    coordinates.to_csv(RESULTS / f"umap_coordinates_{REPRESENTATION}.csv", index=False)
    print("UMAP coordinates:", coordinates.shape)

    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": [args.font_family, "DejaVu Sans"],
        "font.size": 8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "savefig.facecolor": "white",
    })

    def save_figure(fig, stem):
        save_umap_figure(fig, stem, args, kind)

    markers = {"Compound": "o", "Crispr": "^", "Orf": "D"}
    POINT_AREA = args.point_area  # 彩色点和灰色点使用相同的面积。
    CONTROL_COLOR = '#E69F00'
    fig = plot_top_targets_umap(coordinates, gene_rank['Gene'].head(10).tolist(), args, REPRESENTATION)
    save_figure(fig, FIGURES / f"cross_modality_top_targets_umap_{REPRESENTATION}")

    pair_order = ["Compound–Crispr", "Compound–Orf", "Crispr–Orf"]
    arrays = [
        distance_long.loc[distance_long["modality_pair"].eq(name), "cosine_distance"].to_numpy()
        for name in pair_order
    ]
    fig, ax = plt.subplots(figsize=(5.8, 4.6), dpi=args.dpi)
    boxes = ax.boxplot(arrays, positions=np.arange(3), widths=0.5,
                       patch_artist=True, showfliers=False,
                       medianprops={"color": "#20272B", "linewidth": 1.4},
                       whiskerprops={"color": "#5E6B72"},
                       capprops={"color": "#5E6B72"})
    fills = ["#84B3BF", "#B7A2C7", "#D4BD88"]
    for patch, fill in zip(boxes["boxes"], fills):
        patch.set(facecolor=fill, edgecolor="#3C4B51", alpha=0.8)
    rng = np.random.default_rng(args.seed)
    for i, values in enumerate(arrays):
        ax.scatter(rng.normal(i, 0.065, len(values)), values,
                   s=8, color="#253137", alpha=0.20, linewidths=0, rasterized=True)
    ax.set_xticks(np.arange(3), [f"{name}\nn={len(values)}" for name, values in zip(pair_order, arrays)])
    ax.set_ylabel("Cosine distance")
    ax.set_title(f"Matched target gene distances | {REPRESENTATION}", weight="bold")
    ax.yaxis.grid(True, color="#E8ECEE", linewidth=0.7)
    ax.set_axisbelow(True)
    fig.tight_layout()
    save_figure(fig, FIGURES / f"pairwise_cosine_distances_{REPRESENTATION}")

    # 每个基因在 Compound 配对中的均值；Crispr–Orf 每个基因只有一个配对。
    # Include controls in per-gene ranking; keep a noncontrol view for perturbation-only summaries.
    gene_similarity_all = pairs.groupby("Gene", observed=True).agg(
        mean_similarity_compound_crispr=("Distance_to_CRISPR", lambda x: (1.0 - x).mean()),
        mean_similarity_compound_orf=("Distance_to_ORF", lambda x: (1.0 - x).mean()),
        mean_similarity_crispr_orf=("Distance_CRISPR_ORF", lambda x: (1.0 - x.iloc[0])),
        n_compounds=("Compound", "nunique"),
    ).reset_index()
    gene_similarity_all["mean_similarity_all_pairs"] = gene_similarity_all[[
        "mean_similarity_compound_crispr",
        "mean_similarity_compound_orf",
        "mean_similarity_crispr_orf",
    ]].mean(axis=1)
    gene_similarity_all = gene_similarity_all.sort_values(
        ["mean_similarity_compound_orf", "Gene"], ascending=[False, True]
    ).reset_index(drop=True)
    gene_similarity_all["compound_orf_rank"] = np.arange(1, len(gene_similarity_all) + 1)
    mean_rank_order = gene_similarity_all.sort_values(
        ["mean_similarity_all_pairs", "Gene"], ascending=[False, True]
    ).index
    gene_similarity_all["mean_all_pairs_rank"] = 0
    gene_similarity_all.loc[mean_rank_order, "mean_all_pairs_rank"] = np.arange(1, len(mean_rank_order) + 1)
    gene_similarity_all.to_csv(RESULTS / f"gene_pairwise_similarity_{REPRESENTATION}.csv", index=False)
    gene_similarity = gene_similarity_all.loc[
        ~gene_similarity_all["Gene"].astype(str).str.strip().str.lower().isin(CONTROL_NAMES)
    ].copy()

    pair_metric_columns = {
        "Compound–Crispr": "mean_similarity_compound_crispr",
        "Compound–Orf": "mean_similarity_compound_orf",
        "Crispr–Orf": "mean_similarity_crispr_orf",
    }
    pair_means = (
        gene_similarity[list(pair_metric_columns.values())].mean()
        .rename(index={v: k for k, v in pair_metric_columns.items()})
        .rename("mean_cosine_similarity")
        .sort_values(ascending=False)
        .rename_axis("modality_pair")
        .reset_index()
    )
    pair_means["n_target_genes"] = len(gene_similarity)
    pair_means.to_csv(RESULTS / f"pairwise_similarity_means_{REPRESENTATION}.csv", index=False)
    display(gene_similarity_all.head(TOP_N)[["compound_orf_rank", "Gene", "mean_similarity_compound_orf", "n_compounds"]])
    display(pair_means)

    # 三种配对按跨目标基因平均 cosine similarity 排序；箱体展示基因间分布，菱形表示均值。
    ordered_pairs = pair_means["modality_pair"].tolist()
    metric_for_pair = {name: column for name, column in pair_metric_columns.items()}
    similarity_arrays = [gene_similarity[metric_for_pair[name]].to_numpy() for name in ordered_pairs]
    fig, ax = plt.subplots(figsize=(7.2, 4.8), dpi=args.dpi)
    bp = ax.boxplot(similarity_arrays, positions=np.arange(len(ordered_pairs)), widths=0.48,
                    patch_artist=True, showfliers=False,
                    medianprops={"color": "#20272B", "linewidth": 1.3},
                    whiskerprops={"color": "#5E6B72", "linewidth": 0.9},
                    capprops={"color": "#5E6B72", "linewidth": 0.9})
    pair_colors = ["#78A9B5", "#C77B61", "#9A8AB3"]
    for box, color in zip(bp["boxes"], pair_colors):
        box.set(facecolor=color, edgecolor="#3C4B51", alpha=0.78, linewidth=0.9)
    rng = np.random.default_rng(args.seed)
    for i, values in enumerate(similarity_arrays):
        ax.scatter(rng.normal(i, 0.055, len(values)), values, s=9, color="#28343A",
                   alpha=0.17, linewidths=0, rasterized=True, zorder=2)
        mean_value = values.mean()
        ax.scatter(i, mean_value, marker="D", s=42, color=pair_colors[i],
                   edgecolor="#20272B", linewidth=0.6, zorder=4)
        # Keep mean labels above the data rectangle, clear of grid/median lines.
        ax.annotate(f"Mean {mean_value:.3f}", (i, 1.015),
                    xycoords=("data", "axes fraction"), ha="center", va="bottom",
                    fontsize=8, color="#20272B")
    ax.set_xticks(np.arange(len(ordered_pairs)), [f"{name}\nn={len(gene_similarity)} genes" for name in ordered_pairs])
    ax.set_ylabel("Mean cosine similarity per target gene")
    ax.set_title(f"Average similarity across modality pairs | {REPRESENTATION}", weight="bold", pad=27)
    ax.yaxis.grid(True, color="#E8ECEE", linewidth=0.7)
    ax.set_axisbelow(True)
    fig.tight_layout()
    save_figure(fig, FIGURES / f"mean_pairwise_similarity_rank_{REPRESENTATION}")

    mean_top_genes = gene_similarity_all.sort_values("mean_all_pairs_rank")["Gene"].head(10).tolist()
    fig = plot_top_targets_umap(coordinates, mean_top_genes, args, REPRESENTATION)
    fig.axes[0].set_title(
        f"Top targets by mean three-pair similarity | {REPRESENTATION}",
        fontsize=fig.axes[0].title.get_fontsize(), weight="bold", pad=9
    )
    # Keep the existing output filename for downstream compatibility.
    save_figure(fig, FIGURES / f"umap_rank_compound_orf_similarity_{REPRESENTATION}")

    summary = {
        "input_dirs": {m: os.path.commonpath([str(p.parent) for modality, p, _ in entries if modality == m]) for m in MODALITIES},
        "input_kind": kind,

        "output_dir": str(ROOT),

        "umap_parameters": {"n_neighbors": args.n_neighbors, "min_dist": args.min_dist, "metric": args.umap_metric, "random_state": args.seed, "n_jobs": 1},
        "input_files": len(parquet_files),
        "input_wells": int(manifest["input_wells"].sum()),
        "aggregation": "all plates pooled; Crispr guide mean -> gene median; Compound Treatment+Pert_IName mean; Orf Treatment mean",
        "aggregation_by_modality": aggregation_audit.to_dict("records"),
        "crispr_guide_column": CRISPR_GUIDE,
        "feature_columns": len(embedding_columns),
        "representation": REPRESENTATION,
        "pca_variance_threshold": args.pca_variance,
        "pca_explained_variance": float(pca.explained_variance_ratio_.sum()) if REPRESENTATION == "pca95" else None,
        "analysis_dimensions": int(vectors.shape[1]),
        "final_profiles": len(profiles),
        "matched_genes": int(pairs["Gene"].nunique()),
        "compound_crispr_pairs": len(pairs),
        "highlighted_unique_genes": len(top_genes),
        "cross_modality_umap_top_genes": gene_rank["Gene"].head(10).tolist(),
        "mean_pairwise_umap_top_genes": mean_top_genes,
        "ranking": "minimum Compound-Crispr cosine distance across compounds for each gene",
        "controls_in_ranking": True,
        "top_genes": top_genes,
    }
    (RESULTS / f"summary_{REPRESENTATION}.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps({k: v for k, v in summary.items() if k != "top_genes"}, indent=2))



    return summary, profiles[PROFILE_KEYS], gene_similarity_all




def plot_top_targets_umap(coordinates, selected_genes, args, representation):
    """Display the top ten ranked targets, with legends below the plot."""
    selected_genes = list(selected_genes)[:10]
    selected = coordinates.Metadata_Treatment.isin(selected_genes)
    control = coordinates.Metadata_Treatment.astype(str).str.strip().str.lower().isin(CONTROL_NAMES)
    markers = {'Compound': 'o', 'Crispr': '^', 'Orf': 'D'}
    colors = dict(zip(selected_genes, mpl.colormaps['tab20'](np.linspace(0, 1, len(selected_genes)))))
    for gene in selected_genes:
        if gene.strip().lower() in CONTROL_NAMES:
            colors[gene] = '#E69F00'
    # scatter sizes are pt squared; increase the selected marker's linear size by 3 pt.
    selected_area = (np.sqrt(args.point_area) + 3.) ** 2
    fig, ax = plt.subplots(figsize=(7.2, 8.2), dpi=args.dpi)
    for modality in MODALITIES:
        member = coordinates.Metadata_All_Modality.eq(modality)
        background = coordinates.loc[member & ~selected & ~control]
        ax.scatter(background.UMAP1, background.UMAP2, marker=markers[modality],
                   s=args.point_area, c='#C9CDD0', alpha=.47, linewidths=0, rasterized=True, zorder=1)
        controls = coordinates.loc[member & control & ~selected]
        ax.scatter(controls.UMAP1, controls.UMAP2, marker=markers[modality],
                   s=args.point_area, c='#E69F00', edgecolors='#26343A', linewidths=.42, alpha=.98, zorder=4)
        foreground = coordinates.loc[member & selected]
        ax.scatter(foreground.UMAP1, foreground.UMAP2, marker=markers[modality],
                   s=selected_area, c=[colors[g] for g in foreground.Metadata_Treatment],
                   edgecolors='#26343A', linewidths=.42, alpha=.95, zorder=5)
    modality_handles = [Line2D([], [], marker=markers[m], linestyle='none', label=m,
                        markerfacecolor='#7F888D', markeredgecolor='none', markersize=6) for m in MODALITIES]
    fig.legend(handles=modality_handles, loc='upper center', bbox_to_anchor=(.53,.245),
               ncol=3, fontsize=8, frameon=False)
    gene_handles = [Line2D([], [], marker='o', linestyle='none', label=f'{i+1}  {gene}',
                    markerfacecolor=colors[gene], markeredgecolor='#26343A',
                    markeredgewidth=.3, markersize=6) for i,gene in enumerate(selected_genes)]
    fig.legend(handles=gene_handles, title='Top 10 pairs', loc='upper center',
               bbox_to_anchor=(.53,.19), ncol=5, fontsize=8, title_fontsize=9, frameon=False)
    ax.set(xlabel='UMAP1', ylabel='UMAP2')
    ax.set_title(f'Cross modality UMAP | {representation}', weight='bold', pad=9)
    # Keep the coordinate limits tied to the data, not to the legend layout.
    for column, setter in [('UMAP1', ax.set_xlim), ('UMAP2', ax.set_ylim)]:
        low, high = coordinates[column].min(), coordinates[column].max()
        padding = max(float(high-low), 1.) * .05
        setter(low-padding, high+padding)
    ax.set_aspect('equal', adjustable='box')
    fig.subplots_adjust(left=.12, right=.97, bottom=.31, top=.94)
    # Reduce all UMAP text by one point, including ticks and both legends.
    fig.canvas.draw()
    for text in fig.findobj(mpl.text.Text):
        text.set_fontsize(max(1., text.get_fontsize() - 1.))
    return fig


def save_umap_figure(fig, stem, args, kind):
    sys.path.insert(0, str(args.qa_tools_dir))
    from audit_panel_alignment import require_matplotlib_panel_alignment
    # Run checks in temporary storage; retain only the actual figure exports.
    import subprocess
    from tempfile import TemporaryDirectory
    print(f"[{kind}] Exporting and checking {stem.name}", flush=True)
    try:
        with TemporaryDirectory(prefix="descort_umap_qa_") as qa_directory:
            qa_stem = Path(qa_directory) / stem.name
            fig.canvas.draw()
            require_matplotlib_panel_alignment(
                fig,
                json_out=str(qa_stem.with_suffix(".alignment.json")),
                require_panel_labels=False,
                strict=True,
            )
            for extension in dict.fromkeys(args.figure_formats):
                options = {"bbox_inches": "tight"}
                if extension in {"png", "tiff"}:
                    options["dpi"] = args.dpi
                fig.savefig(stem.with_suffix(f".{extension}"), **options)
            # Check a temporary PDF when PDF is not among the requested exports.
            pdf_to_check = stem.with_suffix(".pdf") if "pdf" in args.figure_formats else qa_stem.with_suffix(".pdf")
            if "pdf" not in args.figure_formats:
                fig.savefig(pdf_to_check, bbox_inches="tight")
            for tool, extra in [
                ("audit_pdf_text.py", ["--min-pt", "5", "--json"]),
                ("audit_figure_collisions.py", ["--json-out", str(qa_stem.with_suffix(".collision.json"))]),
            ]:
                result = subprocess.run([sys.executable, str(args.qa_tools_dir / tool),
                                         str(pdf_to_check), *extra],
                                        capture_output=True, text=True)
                if result.returncode:
                    raise RuntimeError(f"Figure check failed: {tool}, {stem}\n"
                                       f"{result.stdout}\n{result.stderr}")
    finally:
        plt.close(fig)



def compare_results(completed, args):
    if len(completed) == 2:
        comparison_dir = args.output_dir / 'comparison'
        comparison_dir.mkdir(parents=True, exist_ok=True)
        model_summary, model_keys, model_metrics = completed['model']
        original_summary, original_keys, original_metrics = completed['original']
        membership = model_keys.merge(original_keys, on=PROFILE_KEYS, how='outer',
                                      indicator=True, validate='one_to_one')
        membership.to_csv(comparison_dir / 'profile_membership.csv', index=False)
        paired = model_metrics.merge(original_metrics, on='Gene', how='outer',
                                     suffixes=('_model', '_original'), indicator=True,
                                     validate='one_to_one')
        metric_names = ['mean_similarity_compound_crispr', 'mean_similarity_compound_orf',
                        'mean_similarity_crispr_orf', 'mean_similarity_all_pairs']
        for metric in metric_names:
            paired[f'{metric}_delta'] = paired[f'{metric}_model'] - paired[f'{metric}_original']
        paired.to_csv(comparison_dir / 'gene_similarity_model_vs_original.csv', index=False)
        # Full pair identities must also match for a like-for-like distance comparison.
        tables = {}
        for kind in ['model', 'original']:
            tables[kind] = pd.read_csv(args.output_dir / kind / 'results' /
                                      f'compound_crispr_orf_pairs_{args.representation}.csv')
        pair_keys = ['Gene', 'Compound', 'Crispr', 'Orf']
        pair_comparison = tables['model'].merge(tables['original'], on=pair_keys, how='outer',
                                               suffixes=('_model', '_original'), indicator=True,
                                               validate='one_to_one')
        pair_comparison.to_csv(comparison_dir / 'pair_distances_model_vs_original.csv', index=False)
        pd.DataFrame([dict(input_kind=k, **{key: value for key, value in completed[k][0].items()
                      if key != "input_kind" and isinstance(value, (str, int, float, bool))})
                      for k in completed]).to_csv(comparison_dir / 'run_summary.csv', index=False)
        (comparison_dir / 'comparison_audit.json').write_text(json.dumps({
            'profiles_match': bool(membership['_merge'].eq('both').all()),
            'matched_profile_counts': membership['_merge'].value_counts().to_dict(),
            'pair_identities_match': bool(pair_comparison['_merge'].eq('both').all()),
            'note': 'Each input has its own PCA and UMAP. Coordinates are not aligned. '
                    'All unmatched identities are retained in audit tables; missing metrics remain empty. '
                    'Gene deltas are descriptive; check pair membership before interpreting differences.'
        }, indent=2))
