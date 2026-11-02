"""Baseline-to-replicate scaling of the six VCC2026-compatible metrics."""
import numpy as np
import pandas as pd

SCORE_METRICS = [
    'pds_cosine', 'expr_mse_unbiased_capped_norm',
    'de_wilcoxon_direction_fidelity_yield_raw',
    'de_wilcoxon_direction_reach_raw', 'de_wilcoxon_sig_jaccard',
    'de_wilcoxon_lfc_nmae',
]


def baseline_to_replicate_scores(raw, baseline, replicate):
    """Scale with fixed anchors; only nMSE and LFC have PDF-defined clipping."""
    values = raw.loc[:, SCORE_METRICS].astype(float)
    baseline = pd.Series(baseline).reindex(SCORE_METRICS).astype(float)
    replicate = pd.Series(replicate).reindex(SCORE_METRICS).astype(float)
    if not (np.isfinite(values.to_numpy()).all()
            and np.isfinite(baseline.to_numpy()).all()
            and np.isfinite(replicate.to_numpy()).all()):
        raise ValueError('Raw metrics and anchors must be finite')
    span = replicate-baseline
    if span.eq(0).any():
        raise ValueError('Baseline and replicate anchors must differ')
    higher = [m for m in SCORE_METRICS if m not in {
        'expr_mse_unbiased_capped_norm', 'de_wilcoxon_lfc_nmae'}]
    if not (span.loc[higher].gt(0).all()
            and span.loc[['expr_mse_unbiased_capped_norm', 'de_wilcoxon_lfc_nmae']].lt(0).all()):
        raise ValueError('Replicate must be better than baseline in every scored metric')
    scores = (values-baseline)/span
    scores['expr_mse_unbiased_capped_norm'] = scores['expr_mse_unbiased_capped_norm'].clip(0, 1)
    scores['de_wilcoxon_lfc_nmae'] = scores['de_wilcoxon_lfc_nmae'].clip(lower=-6)
    scores['mean_score'] = scores.mean(axis=1, skipna=False)
    return scores
