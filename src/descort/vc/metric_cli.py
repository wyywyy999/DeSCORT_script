"""CLI version of calculate_metric_vcc2026_unseen_gene_fast.ipynb."""
import argparse
from pathlib import Path
import anndata as ad


def boolean(value):
    if value.lower() in {'true', '1', 'yes'}:
        return True
    if value.lower() in {'false', '0', 'no'}:
        return False
    raise argparse.ArgumentTypeError('Expected true or false')


def inspect_input(path, layer, pert_col, control, count_name="compounds"):
    if not path.is_file():
        raise FileNotFoundError(path)
    data = ad.read_h5ad(path, backed='r')
    try:
        if pert_col not in data.obs:
            raise ValueError(f'{path}: missing label column {pert_col!r}')
        if data.obs[pert_col].isna().any():
            raise ValueError(f'{path}: missing perturbation labels')
        labels = data.obs[pert_col].astype(str)
        if not labels.eq(control).any() or labels.nunique() < 2:
            raise ValueError(f'{path}: need control {control!r} and at least one perturbation')
        if not data.var_names.is_unique or not data.n_vars:
            raise ValueError(f'{path}: feature names must be unique and nonempty')
        if layer is not None and layer not in data.layers:
            raise ValueError(f'{path}: missing layer {layer!r}')
        if 'control_key_word' in data.uns and str(data.uns['control_key_word']) != control:
            raise ValueError(f'{path}: control differs from exported control_key_word')
        info = dict(path=str(path.resolve()), layer=layer, samples=int(data.n_obs),
                    features=int(data.n_vars), **{count_name: int(labels.nunique()-1)},
                    controls=int(labels.eq(control).sum()))
        for key in ('model_path', 'train_dataset_path', 'inference_dataset_path',
                    'test_dataset_path', 'source_dataset', 'target_dataset', 'feature_alignment'):
            if key in data.uns:
                info[key] = str(data.uns[key])
        info['in_sample'] = bool(data.uns['in_sample']) if 'in_sample' in data.uns else None
        return info, data.obs_names.copy(), labels.to_numpy(), data.var_names.copy(), (
            data.var['feature_id'].to_numpy().copy() if 'feature_id' in data.var else None)
    finally:
        data.file.close()


def parse_args(pert_col='Metadata_Smiles', control_name='CS(=O)C'):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ground_truth_path', type=Path, required=True)
    parser.add_argument('--prediction_path', type=Path, required=True)
    parser.add_argument('--baseline_path', type=Path)
    parser.add_argument('--skip_baseline', action='store_true')
    parser.add_argument('--outdir', type=Path, required=True)
    parser.add_argument('--cache_dir', type=Path, required=True)
    parser.add_argument('--pert_col', default=pert_col)
    parser.add_argument('--control_name', default=control_name)
    for name in ('ground_truth_layer', 'prediction_layer', 'baseline_layer'):
        parser.add_argument('--'+name)
    for name, default in [('dedup_controls', True), ('strict_alignment', True),
                          ('copy_truth_controls', False), ('reuse_cache', True), ('show_progress', True)]:
        parser.add_argument('--'+name, type=boolean, default=default)
    parser.add_argument('--fdr_scope', choices=['per_pert', 'global'], default='per_pert')
    parser.add_argument('--n_jobs', type=int, default=8)
    parser.add_argument('--blas_threads', type=int, default=4)
    parser.add_argument('--scipy_checks', type=int, default=100)
    parser.add_argument('--check-only', action='store_true', help='Check H5AD metadata/alignment without calculating metrics')
    args = parser.parse_args()
    if args.skip_baseline:
        args.baseline_path = None
        args.baseline_layer = None
    if min(args.n_jobs, args.blas_threads, args.scipy_checks) < 1:
        parser.error('Worker, BLAS thread and SciPy check counts must be positive')
    if args.copy_truth_controls and not args.strict_alignment:
        parser.error('copy_truth_controls requires strict_alignment=true')
    if args.baseline_layer and not args.baseline_path:
        parser.error('baseline_layer requires baseline_path')
    return args


def parse_crispr_args():
    return parse_args(pert_col='Perturb_Gene', control_name='nontargeting')


def inspect_crispr_input(path, layer, pert_col, control):
    return inspect_input(path, layer, pert_col, control, count_name='perturbations')
