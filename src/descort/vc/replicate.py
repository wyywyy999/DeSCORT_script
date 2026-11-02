"""Split-half replicate anchors for signed Cell Painting metrics.

PDS uses independent controls on each side. The LFC-like replicate anchor uses
the full-reference significant-feature gate and denominator, as described in
vcc2026-metrics.pdf, Chapter 6. Signed mean effects replace RNA log fold changes.
"""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from . import inference as vc
from .fast_cp_metrics import METRICS, accelerated_runtime, cached_de, fast_masks

LFC = 'de_wilcoxon_lfc_nmae'


def split_half_indices(labels, seed):
    """Split each label independently; every row is assigned exactly once."""
    labels = np.asarray(labels).astype(str)
    rng = np.random.default_rng(seed)
    left, right = [], []
    groups = fast_masks(labels)
    for label in sorted(groups):
        rows = rng.permutation(groups[label])
        if len(rows) < 2:
            raise ValueError(f'{label!r} needs at least two observations')
        cut = len(rows)//2
        left.extend(rows[:cut]); right.extend(rows[cut:])
    left, right = np.sort(left), np.sort(right)
    np.testing.assert_array_equal(np.sort(np.r_[left, right]), np.arange(len(labels)))
    return left, right


def independent_control_pds(pair):
    """Reuse the existing PDS rank implementation, with each half's own origin."""
    offset = pair.bulk('real', pair.control_pert)-pair.bulk('pred', pair.control_pert)
    def bulk(which, label):
        values = pair.bulk(which, label)
        return values+offset if which == 'pred' else values
    proxy = SimpleNamespace(perts=pair.perts, effect=pair.effect, bulk=bulk,
                            control_pert=pair.control_pert)
    return vc._vcc2026_pds_cosine(proxy, show_progress=False)


def full_gate_replicate_lfc(full_de, left_de, right_de, perts, min_gate_size=10):
    """Use full-reference significance/denominator and independent half effects."""
    values, diagnostics = {}, []
    full_groups = dict(tuple(full_de.groupby('target', sort=False)))
    left_groups = dict(tuple(left_de.groupby('target', sort=False)))
    right_groups = dict(tuple(right_de.groupby('target', sort=False)))
    for pert in map(str, perts):
        frame = full_groups[pert]
        gate = frame.loc[(frame.fdr < .05) & np.isfinite(frame.signed_effect)
                         & frame.feature.astype(str).ne(pert)]
        denominator = float(np.abs(gate.signed_effect).sum())
        a = left_groups[pert].set_index('feature').signed_effect.reindex(gate.feature)
        b = right_groups[pert].set_index('feature').signed_effect.reindex(gate.feature)
        a = a.replace([np.inf, -np.inf], np.nan).fillna(0).to_numpy()
        b = b.replace([np.inf, -np.inf], np.nan).fillna(0).to_numpy()
        valid = len(gate) >= min_gate_size and np.isfinite(denominator) and denominator != 0
        values[pert] = float(np.abs(a-b).sum()/denominator) if valid else np.nan
        diagnostics.append(dict(perturbation=pert, full_gate_size=len(gate),
                                full_denominator=denominator, scored=bool(valid)))
    return values, pd.DataFrame(diagnostics)


def evaluate_replicate(left, right, full_de, outdir, cache_dir, *,
                       pert_col='Metadata_Smiles', control='CS(=O)C',
                       n_jobs=4, blas_threads=2, scipy_checks=100):
    """Evaluate right against left without sharing or copying control observations."""
    outdir, cache_dir = Path(outdir), Path(cache_dir)
    outdir.mkdir(parents=True, exist_ok=True); cache_dir.mkdir(parents=True, exist_ok=True)
    if set(left.obs_names) & set(right.obs_names):
        raise ValueError('Replicate halves must contain disjoint observation IDs')
    with threadpool_limits(limits=blas_threads), accelerated_runtime(False):
        pair = vc.PerturbationAnndataPair(real=left, pred=right, pert_col=pert_col, control_pert=control)
        real_de, real_cache = cached_de(left, pair, 'real', cache_dir, True, n_jobs,
                                        scipy_checks, 'per_pert', False)
        pred_de, pred_cache = cached_de(right, pair, 'pred', cache_dir, True, n_jobs,
                                        scipy_checks, 'per_pert', False)
        # Standard expression/DE calculations, before the two replicate-specific corrections.
        results, aggregate, _ = vc.CellEvalMixin().compute_metrics(
            adata_pred=right, adata_real=left,
            metrics_to_calculate=[m for m in METRICS if m != 'pds_cosine'],
            de_pred=pred_de, de_real=real_de, control_pert=control, pert_col=pert_col,
            de_method='wilcoxon', de_backend='scipy', de_input_type='raw-continuous',
            num_threads=n_jobs, outdir=outdir/'standard_half_diagnostics', fdr_scope='per_pert',
            pdex_kwargs={'fdr_scope': 'per_pert', 'show_progress': False})
        results['lfc_nmae_ordinary_half_gate_diagnostic'] = results[LFC]
        # PDF Chapter 3: when neither half calls a significant feature, Fidelity
        # is undefined (0/0) and is omitted from its mean, rather than scored 0.
        fidelity = 'de_wilcoxon_direction_fidelity_yield_raw'
        results['fidelity_ordinary_empty_zero_diagnostic'] = results[fidelity]
        called = set(real_de.loc[real_de.fdr.lt(.05), 'target'].astype(str))
        called.update(pred_de.loc[pred_de.fdr.lt(.05), 'target'].astype(str))
        both_empty = ~results.perturbation.astype(str).isin(called)
        results.loc[both_empty, fidelity] = np.nan
        results['pds_cosine'] = results.perturbation.map(independent_control_pds(pair))
        lfc, gates = full_gate_replicate_lfc(full_de, real_de, pred_de, pair.perts)
        results[LFC] = results.perturbation.map(lfc)
        raw = {}
        for metric in METRICS:
            raw[metric] = (float(aggregate.loc[aggregate.statistic.eq('mean'), metric].iloc[0])
                           if metric == 'expr_mse_unbiased_capped_norm'
                           else float(results[metric].mean()))
    raw = pd.Series(raw, name='value')
    results.to_csv(outdir/'per_perturbation_metrics.csv', index=False)
    gates.to_csv(outdir/'full_reference_lfc_gate.csv', index=False)
    raw.to_csv(outdir/'replicate_metrics.csv')
    manifest = dict(real_de=real_cache, pred_de=pred_cache, real_rows=left.n_obs,
                    pred_rows=right.n_obs, disjoint_observation_ids=True,
                    controls_shared=False, pds_control_origin='Each half uses its own controls',
                    lfc_estimator='Full-reference gate and denominator; half-reference effect vectors',
                    fidelity_empty_both_policy='Omit undefined 0/0, as PDF Chapter 3 specifies',
                    fidelity_both_empty_omitted=int(both_empty.sum()),
                    valid_metric_perturbations={m: int(results[m].notna().sum()) for m in METRICS})
    return raw, manifest
