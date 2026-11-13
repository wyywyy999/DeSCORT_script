"""Generate perturbation profiles from normalized controls and a request CSV.

CSV columns: pert_col,number. Conditions must match the training vocabulary
(gene names for CRISPR; usually canonical SMILES for Compound).
"""
from pathlib import Path
import logging
import sys

_SOURCE = Path(__file__).resolve().parents[2] / 'src'
if str(_SOURCE) not in sys.path:
    sys.path.insert(0, str(_SOURCE))

import numpy as np
import pandas as pd

from descort.vc.generate_from_control import parse_args, prepare_inputs




def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
    if args.batch_size <= 0:
        raise ValueError('batch_size must be positive')
    output = Path(args.output_path)
    if output.suffix != '.parquet':
        raise ValueError('output_path must end with .parquet')
    if output.resolve() in {Path(args.parquet_path).resolve(), Path(args.csv_path).resolve()}:
        raise ValueError('Output must differ from both input files')
    if output.exists() and not args.overwrite:
        raise FileExistsError(f'{output} already exists; pass --overwrite to replace it')
    import torch
    import pyarrow as pa
    import pyarrow.parquet as pq
    from descort.model.configuration_descort import DeSCORTConfig
    from descort.model.modeling_descort import DeSCORTForPerturbPrediction

    config = DeSCORTConfig.from_pretrained(args.pretrained_model_path)
    requests, counts, indices, conditions, features, feature_ids, values, positions, metadata = prepare_inputs(args, config)
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable; use --device cpu')
    torch.manual_seed(args.seed)
    model, info = DeSCORTForPerturbPrediction.from_pretrained(
        args.pretrained_model_path, config=config, INDEX2ESM2_Dict=conditions, output_loading_info=True)
    if any(info.get(k) for k in ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs')):
        raise ValueError(f'Checkpoint is not a compatible trained perturbation model: {info}')
    model.to(device).eval()
    rng = np.random.default_rng(args.seed)
    n_features = len(features)
    length = int(config.max_seq_len)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Write to a temporary sibling; failed inference never leaves a complete-looking output.
    import tempfile
    with tempfile.NamedTemporaryFile(dir=output.parent, suffix='.parquet', delete=False) as handle:
        temporary = Path(handle.name)
    writer = None
    try:
        with torch.inference_mode():
            for request_row, (label, count, index) in enumerate(zip(requests[args.csv_pert_col], counts, indices)):
                count = int(count)
                replace = args.replacement == 'always' or (args.replacement == 'auto' and count > len(values))
                selected = rng.choice(len(values), size=count, replace=replace)
                logging.info('Condition=%s; number=%d; controls=%d; replacement=%s', label, count, len(values), replace)
                for start in range(0, count, args.batch_size):
                    selection = selected[start:start + args.batch_size]
                    size = len(selection)
                    control = torch.full((size, length), float(config.pad_token_value), device=device)
                    control[:, :n_features] = torch.as_tensor(values[selection], device=device)
                    names = torch.zeros((size, length), dtype=torch.long, device=device)
                    names[:, :n_features] = torch.as_tensor(feature_ids, dtype=torch.long, device=device)
                    pred = model.inference(Perturb_Gene_Index=torch.full((size,), index, dtype=torch.long, device=device),
                                           Feature_Name_Index=names, Control_Expr=control,
                                           Attention_mask=names.ne(0).long()).logits.float().cpu().numpy()
                    if pred.shape != (size, n_features) or not np.isfinite(pred).all():
                        raise ValueError(f'Invalid generated predictions: {pred.shape}')
                    meta = metadata.iloc[selection].reset_index(drop=True).copy()
                    meta[args.pert_col] = label
                    meta['Metadata_Generation_ControlRow'] = positions[selection]
                    meta['Metadata_Generation_ControlLabel'] = args.control_label
                    meta['Metadata_Generation_RequestRow'] = request_row
                    meta['Metadata_Generation_ConditionIndex'] = index
                    frame = pd.concat([meta, pd.DataFrame(pred, columns=features)], axis=1)
                    table = pa.Table.from_pandas(frame, preserve_index=False)
                    if writer is None:
                        writer = pq.ParquetWriter(temporary, table.schema)
                    writer.write_table(table)
        writer.close()
        writer = None
        temporary.replace(output)
    finally:
        if writer is not None:
            writer.close()
        temporary.unlink(missing_ok=True)
    logging.info('Saved %d generated profiles to %s', sum(map(int, counts)), output)


if __name__ == '__main__':
    main()
