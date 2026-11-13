"""Generate perturbation profiles from normalized controls and a request CSV.

CSV columns: pert_col,number. Conditions must match the training vocabulary
(gene names for CRISPR; usually canonical SMILES for Compound).
"""
import argparse
import logging
import pickle

import numpy as np
import pandas as pd

def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pretrained_model_path', required=True)
    p.add_argument('--parquet_path', required=True, help='Normalized feature Parquet; not CLS embeddings')
    p.add_argument('--csv_path', required=True, help='Request CSV with pert_col and number columns')
    p.add_argument('--pert_col', required=True, help='Parquet perturbation column used to identify controls')
    p.add_argument('--control_label', default='nontargeting')
    p.add_argument('--csv_pert_col', default='pert_col')
    p.add_argument('--number_col', default='number')
    p.add_argument('--perturbation2index_pkl', required=True, help='Training gene2index or smiles2index pickle')
    p.add_argument('--index2condition_pkl', required=True, help='Training index2esm2 or index2morgan pickle')
    p.add_argument('--feature2index_pkl', required=True)
    p.add_argument('--need_feature_list_pkl', help='Ordered training feature list; required for checkpoints without a saved feature axis')
    p.add_argument('--output_path', required=True, help='Generated .parquet output')
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--replacement', choices=['auto', 'never', 'always'], default='auto')
    p.add_argument('--overwrite', action='store_true')
    return p.parse_args()

def load_pickle(path):
    with open(path, 'rb') as handle:
        return pickle.load(handle)

def prepare_inputs(args, config):
    """Validate all requests before running inference; preserve the feature axis."""
    vocab = load_pickle(args.perturbation2index_pkl)
    conditions = load_pickle(args.index2condition_pkl)
    feature_vocab = load_pickle(args.feature2index_pkl)
    df = pd.read_parquet(args.parquet_path)
    if args.pert_col not in df:
        raise ValueError(f'Parquet is missing perturbation column {args.pert_col}')
    positions = np.flatnonzero(df[args.pert_col].astype(str).eq(args.control_label).to_numpy())
    if not len(positions):
        raise ValueError(f'No controls match {args.pert_col}={args.control_label!r}')
    # Keep labels such as "NA" and leading zeros unchanged.
    requests = pd.read_csv(args.csv_path, dtype=str, keep_default_na=False)
    if not {args.csv_pert_col, args.number_col}.issubset(requests.columns) or requests.empty:
        raise ValueError('CSV must be nonempty and contain the requested condition and number columns')
    counts = pd.to_numeric(requests[args.number_col], errors='raise').to_numpy(dtype=np.float64)
    if not np.isfinite(counts).all() or (counts < 0).any() or (counts != np.floor(counts)).any() or (counts >= 2**63).any():
        raise ValueError('number must contain finite nonnegative integers smaller than 2**63')
    counts = counts.astype(np.int64)
    if not counts.any():
        raise ValueError('At least one request must have number > 0')
    ids = []
    for label in requests[args.csv_pert_col]:
        if not label or label not in vocab:
            raise ValueError(f'Condition {label!r} is absent from the training vocabulary')
        index = int(vocab[label])
        vector = np.asarray(conditions[index])
        if vector.shape != (config.esm2_input_dim,) or not np.isfinite(vector).all():
            raise ValueError(f'Condition {label!r} must have a finite vector of width {config.esm2_input_dim}')
        ids.append(index)
    if args.replacement == 'never' and (counts > len(positions)).any():
        raise ValueError('Requested number exceeds the control pool; use --replacement auto or always')
    reverse = {int(index): name for name, index in feature_vocab.items()}
    saved = getattr(config, 'population_feature_ids', None)
    if saved:
        feature_ids = [int(i) for i in saved]
        if len(feature_ids) != len(set(feature_ids)) or any(i not in reverse for i in feature_ids):
            raise ValueError('Saved feature axis is duplicated or missing from feature2index')
        features = [reverse[i] for i in feature_ids]
    else:
        if not args.need_feature_list_pkl:
            raise ValueError('Checkpoint has no population_feature_ids; provide the ordered training --need_feature_list_pkl')
        features = [name for name in load_pickle(args.need_feature_list_pkl) if name in feature_vocab]
        feature_ids = [int(feature_vocab[name]) for name in features]
    if not features or len(set(feature_ids)) != len(feature_ids):
        raise ValueError('Feature axis must be nonempty and unique')
    if any(i <= 0 or i >= config.feature_name_vocab_size for i in feature_ids):
        raise ValueError('Feature IDs must be positive and within the checkpoint vocabulary')
    length = int(config.max_seq_len)
    if length <= 0:
        raise ValueError('Checkpoint max_seq_len must be positive')
    if len(features) > length:
        logging.warning('Using the first %d/%d training features, matching inference without feature sampling', length, len(features))
        features, feature_ids = features[:length], feature_ids[:length]
    missing = [name for name in features if name not in df]
    if missing:
        raise ValueError(f'Control Parquet is missing training features: {missing[:10]}')
    values = df.iloc[positions][features].to_numpy(dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError('Selected control features contain NaN or infinite values; preprocess them first')
    metadata = df.iloc[positions][[c for c in df.columns if c.startswith('Metadata_')]].reset_index(drop=True)
    reserved = {'Metadata_Generation_ControlRow', 'Metadata_Generation_ControlLabel',
                'Metadata_Generation_RequestRow', 'Metadata_Generation_ConditionIndex'}
    if reserved.intersection(df.columns) or args.pert_col in reserved or args.pert_col in features:
        raise ValueError('Input column names conflict with generation provenance or feature columns')
    return requests, counts, ids, conditions, features, feature_ids, values, positions, metadata
