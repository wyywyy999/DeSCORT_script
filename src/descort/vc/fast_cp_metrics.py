"""Fast, full-data Cell Painting metrics, with content-validated DE checkpoints.

No sampling or feature truncation. Metric formulas are delegated to the existing
src.descort.vc implementation. Runtime accelerations are restored on exit.
"""
from __future__ import annotations

import gc
import hashlib
import json
import logging
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.special import ndtr
from scipy.stats import mannwhitneyu
from threadpoolctl import threadpool_limits
from tqdm.auto import tqdm

from . import inference as vc

METRICS = [
    'pds_cosine', 'mse', 'expr_mse_unbiased_capped_norm',
    'de_wilcoxon_direction_fidelity_yield_raw',
    'de_wilcoxon_direction_reach_raw', 'de_wilcoxon_sig_jaccard',
    'de_wilcoxon_lfc_nmae',
]
_LOCK = threading.RLock()


def log(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def fast_masks(labels):
    names, codes = np.unique(labels, return_inverse=True)
    order = np.argsort(codes, kind='stable')
    edges = np.r_[0, np.bincount(codes).cumsum()]
    return {name: order[a:b] for name, a, b in zip(names, edges[:-1], edges[1:])}


def mwu_shared_reference(values, reference, codes, counts):
    """Same two-sided, asymptotic, tie-corrected MWU as SciPy; no continuity correction."""
    ref = np.sort(reference)
    nref = float(len(ref))
    ntarget = counts.astype(np.float64)
    left = np.searchsorted(ref, values, side='left')
    right = np.searchsorted(ref, values, side='right')
    u = np.bincount(codes, weights=(left + right) * 0.5, minlength=len(counts))
    _, reference_counts = np.unique(ref, return_counts=True)
    reference_counts = reference_counts.astype(np.float64)
    reference_ties = np.sum(reference_counts**3 - reference_counts)
    order = np.lexsort((values, codes))
    sorted_values, sorted_groups = values[order], codes[order]
    starts = np.r_[0, np.flatnonzero(
        (sorted_values[1:] != sorted_values[:-1]) |
        (sorted_groups[1:] != sorted_groups[:-1])
    ) + 1]
    t = np.diff(np.r_[starts, len(values)]).astype(np.float64)
    g, v = sorted_groups[starts], sorted_values[starts]
    r = (np.searchsorted(ref, v, 'right') - np.searchsorted(ref, v, 'left')).astype(np.float64)
    extra_ties = 3*r*r*t + 3*r*t*t + t*t*t - t
    ties = reference_ties + np.bincount(g, weights=extra_ties, minlength=len(counts))
    total = ntarget + nref
    variance = ntarget*nref/12 * ((total + 1) - ties/(total*(total - 1)))
    with np.errstate(divide='ignore', invalid='ignore'):
        z = np.abs(u - ntarget*nref/2) / np.sqrt(np.maximum(variance, 0))
        p = 2 * ndtr(-z)
    # The original evaluator also maps undefined p-values to 1 before BH.
    p = np.where(np.isfinite(p), p, 1.0)
    return np.clip(p, 0, 1), u


def validate_mwu():
    rng = np.random.default_rng(42)
    checks, maximum_error = 0, 0.0
    for mode in ['continuous', 'ties', 'constant', 'separated']:
        counts = np.array([1, 2, 5, 17, 41])
        codes = np.repeat(np.arange(len(counts)), counts)
        values, reference = rng.normal(size=len(codes)), rng.normal(size=83)
        if mode == 'ties':
            values, reference = np.round(values), np.round(reference)
        elif mode == 'constant':
            values[:], reference[:] = 1, 1
        elif mode == 'separated':
            values, reference = values + 30, reference - 30
        p, u = mwu_shared_reference(values, reference, codes, counts)
        for g in range(len(counts)):
            expected = mannwhitneyu(values[codes == g], reference,
                                   method='asymptotic', use_continuity=False)
            expected_p = expected.pvalue if np.isfinite(expected.pvalue) else 1.0
            np.testing.assert_allclose(u[g], expected.statistic, rtol=0, atol=0)
            np.testing.assert_allclose(p[g], expected_p, rtol=1e-10, atol=1e-13)
            maximum_error = max(maximum_error, float(abs(p[g] - expected_p)))
            checks += 1
    return {'checks': checks, 'max_p_absolute_error': maximum_error}


def load_input(path, pert_col, control, layer=None):
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    a = ad.read_h5ad(path)
    if layer is not None:
        if layer not in a.layers:
            raise ValueError(f'{path}: layer {layer!r} does not exist')
        a.X = a.layers[layer].copy()
    # Layers are not used by the metrics; avoid copying them during deduplication.
    a.layers.clear()
    a.raw = None
    if pert_col not in a.obs:
        raise ValueError(f'{path}: missing obs[{pert_col!r}]; columns={list(a.obs)}')
    if a.obs[pert_col].isna().any():
        raise ValueError(f'{path}: missing perturbation labels')
    if not a.var_names.is_unique or not a.n_vars:
        raise ValueError(f'{path}: feature names must be unique and nonempty')
    labels = a.obs[pert_col].astype(str).to_numpy()
    if control not in labels or len(np.unique(labels)) < 2:
        raise ValueError(f'{path}: need control={control!r} and at least one perturbation')
    if sparse.issparse(a.X):
        a.X = a.X.toarray()
    if a.X.dtype.kind not in 'fiu':
        raise TypeError(f'{path}: X must be a real numeric matrix')
    for start in range(0, a.n_obs, 8192):
        if not np.isfinite(a.X[start:start+8192]).all():
            raise ValueError(f'{path}: X contains NaN/Inf near row {start}')
    return a


def validate_pair(real, pred, pert_col, control, strict_alignment):
    if not real.var_names.equals(pred.var_names):
        raise ValueError('Real/pred feature names or order differ. Align them explicitly first.')
    rl, pl = (a.obs[pert_col].astype(str).to_numpy() for a in [real, pred])
    if set(rl) != set(pl):
        raise ValueError('Real/pred perturbation sets differ; no automatic intersection is performed.')
    if strict_alignment:
        if not real.obs_names.equals(pred.obs_names) or not np.array_equal(rl, pl):
            raise ValueError('Row IDs/order or labels differ. For intentionally unpaired samples, set STRICT_ALIGNMENT=False.')
    if control not in set(rl):
        raise ValueError(f'Missing control {control!r}')


def deduplicate_controls(a, pert_col, control):
    labels = a.obs[pert_col].astype(str).to_numpy()
    index = np.flatnonzero(labels == control)
    values = np.ascontiguousarray(a.X[index])
    if values.dtype.kind == 'f':
        values[values == 0] = 0  # canonicalize +0/-0 for numeric equality
    packed = values.view(np.dtype((np.void, values.dtype.itemsize * a.n_vars))).ravel()
    _, first = np.unique(packed, return_index=True)
    keep = np.sort(np.r_[np.flatnonzero(labels != control), index[first]])
    counts = {'control_before': int(len(index)), 'control_after': int(len(first)),
              'perturbation_rows': int(np.count_nonzero(labels != control))}
    if len(keep) == a.n_obs:
        return a, counts
    return a[keep].copy(), counts


def data_fingerprint(a, pert_col, control, fdr_scope):
    """Hash the actual processed matrix, labels and features, not just filenames."""
    h = hashlib.sha256()
    settings = {'shape': a.shape, 'dtype': a.X.dtype.str, 'pert_col': pert_col,
                'control': control, 'fdr_scope': fdr_scope,
                'algorithm_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'metric_source_sha256': hashlib.sha256(Path(vc.__file__).read_bytes()).hexdigest()}
    h.update(json.dumps(settings, sort_keys=True).encode())
    for strings in [a.var_names.astype(str).to_numpy(), a.obs[pert_col].astype(str).to_numpy()]:
        # Length-prefixed UTF-8 encoding avoids ambiguous concatenations.
        for s in strings:
            b = str(s).encode('utf-8')
            h.update(len(b).to_bytes(8, 'little'))
            h.update(b)
    for start in range(0, a.n_obs, 8192):
        h.update(memoryview(np.ascontiguousarray(a.X[start:start+8192])).cast('B'))
    return h.hexdigest(), settings


class _CompactDataframeLog(logging.Filter):
    def filter(self, record):
        if isinstance(record.args, tuple):
            record.args = tuple(
                f'<DataFrame: {len(value):,} rows, {len(value.columns)} columns>'
                if isinstance(value, pd.DataFrame) else value for value in record.args
            )
        return True


@contextmanager
def accelerated_runtime(show_progress=True):
    """Restore all runtime substitutions even if a notebook cell is interrupted."""
    with _LOCK:
        compact_log = _CompactDataframeLog()
        vc.evaluator_logger.addFilter(compact_log)
        old_masks = vc.PerturbationAnndataPair._make_masks
        old_de_metrics = vc._vcc2026_de_metrics
        def indexed_de(real_de, pred_de, perts, **kwargs):
            ri = real_de.groupby('target', sort=False).indices
            pi = pred_de.groupby('target', sort=False).indices
            result = {name: {} for name in vc.VCC2026_DE_METRICS}
            for gene in tqdm(perts, desc='DE metrics by gene', disable=not show_progress):
                one = old_de_metrics(real_de.iloc[ri[str(gene)]], pred_de.iloc[pi[str(gene)]],
                                     [gene], **{**kwargs, 'show_progress': False})
                for key, value in one.items():
                    result[key].update(value)
            return result
        vc.PerturbationAnndataPair._make_masks = staticmethod(fast_masks)
        vc._vcc2026_de_metrics = indexed_de
        try:
            yield
        finally:
            vc.PerturbationAnndataPair._make_masks = staticmethod(old_masks)
            vc._vcc2026_de_metrics = old_de_metrics
            vc.evaluator_logger.removeFilter(compact_log)


def compute_de(a, pair, which, n_jobs, check_count, fdr_scope, show_progress):
    masks = pair.pert_mask_real if which == 'real' else pair.pert_mask_pred
    targets = np.asarray(pair.perts)
    counts = np.array([len(masks[g]) for g in targets])
    indices = np.concatenate([masks[g] for g in targets])
    codes = np.repeat(np.arange(len(targets)), counts)
    log(f'{which}: arranging matrix for {len(targets)} genes × {a.n_vars} features')
    values = np.asfortranarray(a.X[indices])
    reference = np.asfortranarray(a.X[masks[pair.control_pert]])
    pvalues = np.empty((len(targets), a.n_vars))
    statistics = np.empty_like(pvalues)
    def work(j):
        return j, mwu_shared_reference(values[:, j], reference[:, j], codes, counts)
    with ThreadPoolExecutor(max_workers=n_jobs) as pool:
        for done, (j, (p, u)) in enumerate(tqdm(pool.map(work, range(a.n_vars)),
                                             total=a.n_vars, desc=f'{which} MWU',
                                             disable=not show_progress), 1):
            pvalues[:, j], statistics[:, j] = p, u
            if not show_progress and (done % 128 == 0 or done == a.n_vars):
                log(f'{which} MWU: {done}/{a.n_vars} features')
    rng, errors = np.random.default_rng(42), []
    for _ in range(check_count):
        g, j = int(rng.integers(len(targets))), int(rng.integers(a.n_vars))
        expected = mannwhitneyu(a.X[masks[targets[g]], j], reference[:, j],
                               method='asymptotic', use_continuity=False)
        expected_p = expected.pvalue if np.isfinite(expected.pvalue) else 1.0
        np.testing.assert_allclose(pvalues[g, j], expected_p, rtol=1e-9, atol=1e-13)
        np.testing.assert_allclose(statistics[g, j], expected.statistic, rtol=0, atol=0)
        errors.append(float(abs(pvalues[g, j] - expected_p)))
    bulk = pair._bulk_real if which == 'real' else pair._bulk_pred
    target_means = np.stack([bulk[g] for g in targets]).astype(float)
    reference_mean = bulk[pair.control_pert].astype(float)
    frame = pd.DataFrame({'target': np.repeat(targets, a.n_vars),
                          'feature': np.tile(pair.genes, len(targets)),
                          'p_value': pvalues.ravel(), 'statistic': statistics.ravel(),
                          'target_mean': target_means.ravel(),
                          'reference_mean': np.tile(reference_mean, len(targets)),
                          'signed_effect': (target_means-reference_mean).ravel()})
    frame = vc._apply_fdr(frame, fdr_scope)
    validation = {'scipy_checks': check_count, 'max_p_absolute_error': max(errors, default=0),
                  'n_comparisons': len(frame), 'passed': True}
    log(f'{which}: SciPy checks PASS; maximum p difference {validation["max_p_absolute_error"]}')
    return frame, validation


def cached_de(a, pair, which, cache_dir, reuse_cache, n_jobs, check_count, fdr_scope, show_progress):
    fingerprint, settings = data_fingerprint(a, pair.pert_col, pair.control_pert, fdr_scope)
    # Same real data can be reused for other predictions/baselines.
    stem = cache_dir / f'de_{fingerprint}'
    parquet, metadata = stem.with_suffix('.parquet'), stem.with_suffix('.json')
    if reuse_cache and parquet.is_file() and metadata.is_file():
        try:
            info = json.loads(metadata.read_text())
            if info.get('fingerprint') != fingerprint or not info['validation']['passed']:
                raise ValueError('cache metadata mismatch')
            frame = pd.read_parquet(parquet)
            required = {'target', 'feature', 'p_value', 'fdr', 'signed_effect'}
            if len(frame) != len(pair.perts)*a.n_vars or not required.issubset(frame):
                raise ValueError('cache schema/row count mismatch')
            log(f'{which}: cache HIT {parquet.name}')
            return frame, {**info, 'cache_hit': True, 'cache_path': str(parquet)}
        except Exception as exc:
            log(f'{which}: ignoring unusable cache: {exc}')
    log(f'{which}: cache MISS; computing DE from all samples')
    frame, validation = compute_de(a, pair, which, n_jobs, check_count, fdr_scope, show_progress)
    info = {'fingerprint': fingerprint, 'settings': settings, 'validation': validation}
    temporary = cache_dir / f'.{fingerprint}.{uuid.uuid4().hex}.parquet'
    frame.to_parquet(temporary, index=False)
    temporary.replace(parquet)
    temporary_json = cache_dir / f'.{fingerprint}.{uuid.uuid4().hex}.json'
    temporary_json.write_text(json.dumps(info, indent=2))
    temporary_json.replace(metadata)
    return frame, {**info, 'cache_hit': False, 'cache_path': str(parquet)}


def evaluate_files(prediction_path, ground_truth_path, outdir, *, cache_dir=None,
                   pert_col='Perturb_Gene', control='nontargeting',
                   prediction_layer=None, ground_truth_layer=None,
                   dedup_controls=True, strict_alignment=True, copy_truth_controls=False,
                   n_jobs=8, blas_threads=4, scipy_checks=100, fdr_scope='per_pert',
                   reuse_cache=True, show_progress=True, result_name='model'):
    """Evaluate every perturbation and feature and save seven metrics + provenance."""
    started = time.perf_counter()
    if n_jobs < 1 or blas_threads < 1 or scipy_checks < 1:
        raise ValueError('n_jobs, blas_threads and scipy_checks must be positive')
    if fdr_scope not in {'per_pert', 'global'}:
        raise ValueError('fdr_scope must be per_pert or global')
    if Path(result_name).name != result_name:
        raise ValueError('result_name must be a simple filename stem')
    outdir = Path(outdir).expanduser().resolve()
    cache_dir = Path(cache_dir).expanduser().resolve() if cache_dir else outdir/'de_cache'
    outdir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    validation = validate_mwu()
    log('Loading complete H5AD inputs (no gene sampling or feature truncation)')
    real = load_input(ground_truth_path, pert_col, control, ground_truth_layer)
    pred = load_input(prediction_path, pert_col, control, prediction_layer)
    validate_pair(real, pred, pert_col, control, strict_alignment)
    if copy_truth_controls:
        if not strict_alignment:
            raise ValueError('Copying controls requires strict row alignment')
        mask = real.obs[pert_col].astype(str).eq(control).to_numpy()
        pred.X[mask] = real.X[mask]
    counts = {}
    for which, a in [('real', real), ('pred', pred)]:
        labels = a.obs[pert_col].astype(str).to_numpy()
        counts[which] = {'control_before': int(np.count_nonzero(labels == control)),
                         'control_after': int(np.count_nonzero(labels == control)),
                         'perturbation_rows': int(np.count_nonzero(labels != control))}
        if dedup_controls:
            log(f'{which}: exact duplicate-control removal')
            a, counts[which] = deduplicate_controls(a, pert_col, control)
        if which == 'real':
            real = a
        else:
            pred = a
    validate_pair(real, pred, pert_col, control, strict_alignment)
    log(f'Data: {len(real.obs[pert_col].unique())-1} perturbations; {real.n_vars} features; '
        f'real={real.n_obs:,} rows, pred={pred.n_obs:,} rows')
    prep_seconds = time.perf_counter()-started
    with threadpool_limits(limits=blas_threads), accelerated_runtime(show_progress):
        pair = vc.PerturbationAnndataPair(real=real, pred=pred, pert_col=pert_col, control_pert=control)
        de_started = time.perf_counter()
        real_de, real_cache = cached_de(real, pair, 'real', cache_dir, reuse_cache, n_jobs,
                                       scipy_checks, fdr_scope, show_progress)
        pred_de, pred_cache = cached_de(pred, pair, 'pred', cache_dir, reuse_cache, n_jobs,
                                       scipy_checks, fdr_scope, show_progress)
        de_seconds = time.perf_counter()-de_started
        metric_started = time.perf_counter()
        results, aggregate, _ = vc.CellEvalMixin().compute_metrics(
            adata_pred=pred, adata_real=real, metrics_to_calculate=METRICS,
            de_pred=pred_de, de_real=real_de, control_pert=control, pert_col=pert_col,
            de_method='wilcoxon', de_backend='scipy', de_input_type='raw-continuous',
            num_threads=n_jobs, outdir=outdir/'cell_eval', fdr_scope=fdr_scope,
            pdex_kwargs={'fdr_scope': fdr_scope, 'show_progress': show_progress})
        raw = {}
        for metric in METRICS:
            if metric == 'expr_mse_unbiased_capped_norm':
                raw[metric] = float(aggregate.loc[aggregate.statistic == 'mean', metric].iloc[0])
            else:
                values = pd.to_numeric(results[metric], errors='coerce').to_numpy(float)
                finite = values[np.isfinite(values)]
                raw[metric] = float(finite.mean()) if len(finite) else np.nan
        metric_seconds = time.perf_counter()-metric_started
    report = {'prediction_path': str(Path(prediction_path).resolve()),
              'ground_truth_path': str(Path(ground_truth_path).resolve()),
              'prediction_layer': prediction_layer, 'ground_truth_layer': ground_truth_layer,
              'pert_col': pert_col, 'control': control, 'fdr_scope': fdr_scope,
              'dedup_controls': dedup_controls, 'strict_alignment': strict_alignment,
              'copy_truth_controls': copy_truth_controls,
              'n_perturbations': len(pair.perts), 'n_features': real.n_vars,
              'real_shape': list(real.shape), 'pred_shape': list(pred.shape),
              'counts': counts, 'real_de': real_cache, 'pred_de': pred_cache,
              'synthetic_validation': validation,
              'timing_seconds': {'preprocessing': prep_seconds, 'de_and_fingerprints': de_seconds,
                                 'metrics': metric_seconds, 'total': time.perf_counter()-started},
              'valid_metric_gene_counts': {m: int(np.isfinite(pd.to_numeric(results[m], errors='coerce')).sum()) for m in METRICS},
              'metric_source': str(Path(vc.__file__).resolve())}
    raw_series = pd.Series(raw, name='value').reindex(METRICS)
    raw_series.to_frame(name=0).to_parquet(outdir/f'{result_name}.parquet')
    raw_series.to_csv(outdir/f'{result_name}_metrics.csv')
    (outdir/'run_manifest.json').write_text(json.dumps(report, indent=2))
    log(f'Finished in {report["timing_seconds"]["total"]:.1f}s → {outdir/result_name}.parquet')
    del real, pred, real_de, pred_de, pair
    gc.collect()
    return {'raw': raw_series, 'results': results, 'aggregate': aggregate,
            'manifest': report, 'outdir': outdir}
