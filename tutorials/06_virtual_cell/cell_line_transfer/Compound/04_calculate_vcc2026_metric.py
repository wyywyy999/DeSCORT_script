#!/usr/bin/env python3
"""CLI version of calculate_metric_vcc2026_unseen_gene_fast.ipynb."""

# Support both an installed package and direct execution from this checkout.
from pathlib import Path as _Path
import sys as _sys
_SOURCE = _Path(__file__).resolve().parents[4] / "src"
if str(_SOURCE) not in _sys.path:
    _sys.path.insert(0, str(_SOURCE))

import json
import os
from pathlib import Path
import sys
import warnings

for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(key, '4')

HERE = Path(__file__).resolve().parent

import numpy as np
import pandas as pd
from descort.vc.fast_cp_metrics import evaluate_files, validate_mwu

from descort.vc.metric_cli import inspect_input, parse_args


def main():
    args = parse_args()
    inputs = {'truth': (args.ground_truth_path, args.ground_truth_layer),
              'prediction': (args.prediction_path, args.prediction_layer)}
    if args.baseline_path is not None:
        inputs['baseline'] = (args.baseline_path, args.baseline_layer)
    checked = {name: inspect_input(path, layer, args.pert_col, args.control_name)
               for name, (path, layer) in inputs.items()}
    reference = checked['truth']
    for name, current in checked.items():
        if not reference[3].equals(current[3]):
            raise ValueError(f'{name}: feature names/order differ from ground truth; no automatic reordering or column removal')
        if reference[4] is not None and current[4] is not None and not np.array_equal(reference[4], current[4]):
            raise ValueError(f'{name}: feature IDs differ from ground truth')
        if set(reference[2]) != set(current[2]):
            raise ValueError(f'{name}: perturbation set differs from ground truth')
        if args.strict_alignment and (not reference[1].equals(current[1]) or not np.array_equal(reference[2], current[2])):
            raise ValueError(f'{name}: sample IDs/order/labels differ from ground truth')
    provenance = {name: entry[0] for name, entry in checked.items()}
    print(pd.DataFrame(provenance).T.to_string(), flush=True)
    if any(info['in_sample'] is True for info in provenance.values()):
        warnings.warn('Input has in_sample=True; check the actual model training/test split before interpreting unseen-compound performance.')
    args.outdir.mkdir(parents=True, exist_ok=True)
    (args.outdir/'input_provenance.json').write_text(json.dumps(provenance, indent=2, ensure_ascii=False)+'\n')
    (args.outdir/'cli_parameters.json').write_text(json.dumps(vars(args), indent=2, default=str)+'\n')
    print('Synthetic MWU validation:', validate_mwu(), flush=True)
    if args.check_only:
        print('Metadata/alignment checks passed; metrics were not calculated.', flush=True)
        return
    common = dict(cache_dir=args.cache_dir, pert_col=args.pert_col, control=args.control_name,
                  ground_truth_layer=args.ground_truth_layer, dedup_controls=args.dedup_controls,
                  strict_alignment=args.strict_alignment, copy_truth_controls=args.copy_truth_controls,
                  n_jobs=args.n_jobs, blas_threads=args.blas_threads, scipy_checks=args.scipy_checks,
                  fdr_scope=args.fdr_scope, reuse_cache=args.reuse_cache, show_progress=args.show_progress)
    model = evaluate_files(args.prediction_path, args.ground_truth_path, args.outdir,
                           prediction_layer=args.prediction_layer, result_name='model', **common)
    print(model['raw'].to_frame('model_raw').to_string(), flush=True)
    manifest = model['manifest']
    print('Perturbations:', manifest['n_perturbations'], 'features:', manifest['n_features'], flush=True)
    print('DE cache hits:', {k: manifest[k]['cache_hit'] for k in ('real_de', 'pred_de')}, flush=True)
    print('Timing (seconds):', manifest['timing_seconds'], flush=True)
    print('Valid perturbation counts:', manifest['valid_metric_gene_counts'], flush=True)
    if args.baseline_path is not None:
        baseline = evaluate_files(args.baseline_path, args.ground_truth_path, args.outdir/'baseline',
                                  prediction_layer=args.baseline_layer, result_name='baseline', **common)
        comparison = pd.DataFrame({'model': model['raw'], 'baseline': baseline['raw']})
        comparison.to_csv(args.outdir/'model_vs_baseline.csv')
        print(comparison.to_string(), flush=True)
    print('Results:', args.outdir.resolve(), flush=True)


if __name__ == '__main__':
    main()
