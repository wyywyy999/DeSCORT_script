#!/usr/bin/env python3
"""Generic paired-profile PM and split-half replicate anchors for CRISPR or compounds."""
import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from datasets import load_from_disk

from .fast_cp_metrics import METRICS, deduplicate_controls, evaluate_files, fast_masks
from .finetune import infer_feature_order
from .replicate import evaluate_replicate, split_half_indices
from .utils import load_pickle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--gene2index', '--smiles2index', dest='label2index', required=True, type=Path)
    parser.add_argument('--context', required=True)
    parser.add_argument('--pert-col', default='Perturb_Gene')
    parser.add_argument('--control', default='nontargeting')
    parser.add_argument('--n-jobs', type=int, default=8)
    parser.add_argument('--blas-threads', type=int, default=4)
    parser.add_argument('--scipy-checks', type=int, default=100)
    parser.add_argument('--check-only', action='store_true', help='Inspect schema and one row; no PM, splits, DE or output files')
    parser.add_argument('--feature2index', required=True, type=Path)
    parser.add_argument('--base-seed', type=int, default=0)
    parser.add_argument('--splits', type=int, default=5)
    args = parser.parse_args()
    if min(args.n_jobs, args.blas_threads, args.scipy_checks) < 1:
        parser.error('Worker and validation counts must be positive')
    if args.splits < 2:
        parser.error('At least two splits are needed for the reported standard deviation')
    out = args.output_dir
    mapping = load_pickle(args.label2index)
    vocabulary = load_pickle(args.feature2index)
    labels_by_id = {int(i): str(s) for s, i in mapping.items()}
    names = {int(i): str(f) for f, i in vocabulary.items()}
    assert len(labels_by_id) == len(mapping)
    data = load_from_disk(str(args.dataset)).with_format('numpy')
    if args.check_only:
        required = {'Perturb_Gene', 'Perturb_Gene_Index', 'Perturb_expression',
                    'Control_expression', 'Feature_name_index'}
        if not len(data) or not required <= set(data.column_names):
            raise ValueError('Empty or unsupported paired-profile dataset')
        row = data[0]
        if labels_by_id[int(row['Perturb_Gene_Index'])] != str(row['Perturb_Gene']):
            raise ValueError('Gene-to-index mapping does not match the first row')
        ids = np.asarray(row['Feature_name_index'], dtype=np.int64)
        if not set(ids[ids > 0]) <= set(names):
            raise ValueError('Feature dictionary is missing input IDs')
        print(json.dumps(dict(dataset=str(args.dataset), rows=len(data),
            first_row_valid_features=int(np.count_nonzero(ids > 0)),
            context=args.context, control=args.control, pert_col=args.pert_col,
            output_dir=str(out), schema_and_first_row_mapping_valid=True,
            metrics_started=False), indent=2))
        return
    out.mkdir(parents=True, exist_ok=True)
    axis = infer_feature_order(data, max(names)+1)
    controls = np.empty((len(data), len(axis)), dtype=np.float32)
    truth = np.empty_like(controls)
    labels, selections = [], {}
    for start in range(0, len(data), 1024):
        stop = min(start+1024, len(data))
        batch = data[start:stop]
        ids = np.asarray(batch['Feature_name_index'])
        if ids.ndim == 2 and np.all(ids == ids[0]):
            key = np.asarray(ids[0], dtype=np.int64).tobytes()
            if key not in selections:
                lookup = {int(fid): j for j, fid in enumerate(ids[0])}
                selections[key] = [lookup[fid] for fid in axis]
                np.testing.assert_array_equal(ids[0, selections[key]], axis)
            select = selections[key]
            controls[start:stop] = np.asarray(batch['Control_expression'])[:, select]
            truth[start:stop] = np.asarray(batch['Perturb_expression'])[:, select]
        else:
            for j, row_ids in enumerate(batch['Feature_name_index']):
                row_ids = np.asarray(row_ids, dtype=np.int64)
                key = row_ids.tobytes()
                if key not in selections:
                    lookup = {int(fid): k for k, fid in enumerate(row_ids)}
                    selections[key] = [lookup[fid] for fid in axis]
                    np.testing.assert_array_equal(row_ids[selections[key]], axis)
                controls[start+j] = np.asarray(batch['Control_expression'][j])[selections[key]]
                truth[start+j] = np.asarray(batch['Perturb_expression'][j])[selections[key]]
        for index, given_label in zip(batch['Perturb_Gene_Index'], batch['Perturb_Gene']):
            label = labels_by_id[int(index)]
            if label != str(given_label) or label == args.control:
                raise ValueError('Incorrect perturbation/index mapping or control-labelled perturbation row')
            labels.append(label)
        if start == 0 or stop == len(data) or (start//1024) % 100 == 0:
            print(f'Prepared {stop:,}/{len(data):,} paired observations', flush=True)
    assert np.isfinite(truth).all() and np.isfinite(controls).all()
    obs = pd.DataFrame({args.pert_col: labels+[args.control]*len(labels)},
                       index=[f'perturb_{i}' for i in range(len(labels))]
                             +[f'paired_control_{i}' for i in range(len(labels))])
    var = pd.DataFrame({'feature_id': axis}, index=[names[i] for i in axis])
    full = ad.AnnData(np.concatenate([truth, controls]), obs=obs, var=var)
    # Split unique controls, so paired copies of the same control cannot leak across halves.
    full, control_counts = deduplicate_controls(full, args.pert_col, args.control)
    counts = full.obs[args.pert_col].value_counts()
    if counts.min() < 2:
        raise ValueError('Every perturbation and control must have at least two observations')
    full.write_h5ad(out/'ground_truth.h5ad', compression='lzf')
    perts = sorted(set(labels))
    labels_array = np.asarray(labels)
    groups = fast_masks(labels_array)
    mean = np.stack([truth[groups[label]].mean(0, dtype=np.float64)
                     for label in perts]).mean(0).astype(np.float32)
    pm = full.copy()
    pm.X[pm.obs[args.pert_col].ne(args.control).to_numpy()] = mean
    pm.uns.update(baseline=f'{args.context} full-context PM', target_perturbation_values_used=True,
                  weighting='Equal weight per perturbation after computing each perturbation mean')
    pm.write_h5ad(out/f'{args.context}_PM.h5ad', compression='lzf')
    perturbation_rows = len(truth)
    del pm, truth, controls, selections, groups
    full_counts = counts.rename_axis('perturbation').to_frame('n_full')
    full_counts.to_csv(out/'observation_counts.csv')
    pm_result = evaluate_files(out/f'{args.context}_PM.h5ad', out/'ground_truth.h5ad', out/'PM',
        cache_dir=out/'de_cache', pert_col=args.pert_col, control=args.control,
        dedup_controls=False, strict_alignment=True, copy_truth_controls=False,
        n_jobs=args.n_jobs, blas_threads=args.blas_threads, scipy_checks=args.scipy_checks, fdr_scope='per_pert',
        reuse_cache=True, show_progress=False, result_name='pm')
    full_de = pd.read_parquet(pm_result['manifest']['real_de']['cache_path'])
    seeds = [int(child.generate_state(1)[0])
             for child in np.random.SeedSequence(args.base_seed).spawn(args.splits)]
    runs, manifests = [], []
    full_labels = full.obs[args.pert_col].astype(str).to_numpy()
    for i, seed in enumerate(seeds, 1):
        directory = out/f'split_{i:02d}'
        directory.mkdir(exist_ok=True)
        left_ids, right_ids = split_half_indices(full_labels, seed)
        left, right = full[left_ids].copy(), full[right_ids].copy()
        assignment = full.obs.copy()
        assignment['half'] = ''
        assignment.loc[left.obs_names, 'half'] = 'reference'
        assignment.loc[right.obs_names, 'half'] = 'replicate'
        assignment.to_csv(directory/'split_assignments.csv', index_label='original_observation_id')
        left.write_h5ad(directory/'reference_half.h5ad', compression='lzf')
        right.write_h5ad(directory/'replicate_half.h5ad', compression='lzf')
        raw, manifest = evaluate_replicate(left, right, full_de, directory, out/'de_cache',
            pert_col=args.pert_col, control=args.control, n_jobs=args.n_jobs,
            blas_threads=args.blas_threads, scipy_checks=args.scipy_checks)
        manifest.update(split=i, seed=seed)
        (directory/'run_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
        runs.append(raw.rename(f'split_{i:02d}'))
        manifests.append(manifest)
        print(f'COMPLETED split {i}/{args.splits}', raw.to_dict(), flush=True)
    replicate = pd.DataFrame(runs).reindex(columns=METRICS)
    replicate.to_csv(out/'replicate_per_split.csv', index_label='split')
    summary = pd.DataFrame({args.context+'_PM': pm_result['raw'], 'replicate_mean': replicate.mean(),
                            'replicate_std': replicate.std(ddof=1),
                            'replicate_min': replicate.min(), 'replicate_max': replicate.max(),
                            'n_splits': replicate.count()})
    summary.to_csv(out/'PM_vs_replicate.csv', index_label='metric')
    protocol = dict(dataset=str(args.dataset.resolve()), context=args.context,
        pert_col=args.pert_col, control=args.control,
        base_seed=args.base_seed, split_seeds=seeds, seed_derivation=f'NumPy SeedSequence(base_seed).spawn({args.splits}); child.generate_state(1)[0]',
        perturbations=len(perts), perturbation_rows=perturbation_rows, features=len(axis),
        controls=control_counts, total_unique_control_rows=int(counts[args.control]),
        perturbation_counts_min=int(counts.drop(args.control).min()),
        perturbation_counts_max=int(counts.drop(args.control).max()),
        replicate=f'{args.splits} stratified disjoint half splits; each half uses its own controls',
        odd_counts='Floor(n/2) reference rows, remaining rows replicate; no observations discarded',
        pm='Full-context equal-perturbation-weight mean response, assigned to all perturbations',
        lfc='Full-reference significant gate (FDR<0.05, >=10 features), full denominator; half-effect absolute differences',
        data_space='Signed MAD-normalized Cell Painting features; mean differences replace RNA log fold changes',
        no_RNA_count_or_knockdown_filters=True, no_downsampling_to_400=True,
        min_gate_size=10, base_seed_derivation_matches_pdf_concept_but_not_verified_official_code=True,
        fidelity='Omit both-halves-no-significant-feature cases (undefined 0/0)',
        std_ddof=1, mse_is_additional_to_six_PDF_metrics=True, runs=manifests)
    (out/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    print(summary.to_string(), flush=True)


if __name__ == '__main__':
    main()
