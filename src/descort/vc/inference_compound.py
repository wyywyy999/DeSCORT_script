'Compound inference on the feature intersection saved by 02_finetune.py.'
import argparse
import logging
from pathlib import Path
import anndata as ad
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from ..data.collator import PerturbPredictionDataCollator
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('pretrained_model_path', 'smiles2index_pkl', 'feature2index_pkl',
                 'train_dataset_path', 'test_dataset_path', 'output_dir'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--index2morgan_pkl', '--index2esm2_pkl', '--index2unimol_pkl', dest='index2morgan_pkl', required=True,
                        help='Index-to-condition vector dictionary; width must match checkpoint esm2_input_dim.')
    parser.add_argument('--prediction_filename', default='prediction_data.h5ad')
    parser.add_argument('--ground_truth_filename', default='ground_truth.h5ad')
    parser.add_argument('--mean_baseline_filename', default='perturb_mean_data.h5ad')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--max_length', type=int, default=None,
                        help='Defaults to the checkpoint max_seq_len; must match training.')
    parser.add_argument('--pad_value', type=float, default=100)
    parser.add_argument('--use_sdpa', choices=('true', 'false'), default=None,
                        help='Optional attention implementation override; otherwise use checkpoint configuration.')
    parser.add_argument('--dataloader_num_workers', type=int, default=4)
    parser.add_argument('--dataloader_prefetch_factor', type=int, default=2)
    parser.add_argument('--gene_column', default='Perturb_Gene',
                        help='Output label column; values are SMILES, despite the compatibility name.')
    parser.add_argument('--control_key_word', '--control_label', dest='control_key_word', default='CS(=O)C')
    parser.add_argument('--log_file', default=None)
    parser.add_argument('--check-only', action='store_true', help='Validate checkpoint configuration, mappings and dataset alignment without model inference.')
    return parser.parse_args()


def invert_vocabulary(mapping, name):
    inverse = {int(index): label for label, index in mapping.items()}
    if len(inverse) != len(mapping):
        raise ValueError(f'{name} contains duplicate indices')
    return inverse


def build_collator(config, max_length=None, pad_value=100):
    """Never infer a new intersection from test data or reorder checkpoint features."""
    feature_ids = getattr(config, 'population_feature_ids', None)
    if getattr(config, 'population_feature_alignment', None) != 'intersection' or not feature_ids:
        raise ValueError('Checkpoint must contain population_feature_alignment="intersection" '
                         'and population_feature_ids from the current compound fine-tuning script. '
                         'Old union checkpoints and foundation checkpoints are not compatible.')
    length = config.max_seq_len if max_length is None else max_length
    if length <= 0 or length != config.max_seq_len:
        raise ValueError(f'max_length must match checkpoint max_seq_len={config.max_seq_len}')
    collator = PerturbPredictionDataCollator(
        max_length=length, sampling=False, pad_value=pad_value,
        feature_order=feature_ids, feature_alignment='intersection',
    )
    return collator, list(feature_ids[:length])


def build_dataloader(dataset, collator, args):
    kwargs = dict(dataset=dataset, batch_size=args.batch_size, shuffle=False,
                  collate_fn=collator, num_workers=args.dataloader_num_workers,
                  pin_memory=torch.device(args.device).type == 'cuda')
    if args.dataloader_num_workers:
        kwargs.update(prefetch_factor=args.dataloader_prefetch_factor, persistent_workers=True)
    return DataLoader(**kwargs)


def validate_batch(batch, feature_ids):
    """Check the output axis and matching values, including right padding."""
    ids = batch['Feature_Name_Index']
    expected = torch.zeros(ids.shape[1], dtype=ids.dtype, device=ids.device)
    expected[:len(feature_ids)] = torch.as_tensor(feature_ids, device=ids.device)
    if not torch.equal(ids, expected.expand_as(ids)):
        raise ValueError('Inference batch features do not match the saved training axis')
    if not torch.equal(batch['Attention_mask'].bool(), ids.ne(0)):
        raise ValueError('Feature mask does not match the saved training axis')
    for name in ('Control_Expr', 'Perturb_Expr'):
        if batch[name].shape != ids.shape or not torch.isfinite(batch[name]).all():
            raise ValueError(f'{name} has invalid shape or non-finite values')


def collect_test_outputs(model, dataloader, device, feature_ids):
    indices, truth, predictions, controls = [], [], [], []
    size = len(feature_ids)
    with torch.inference_mode():
        for batch in tqdm(dataloader, desc='Compound inference'):
            validate_batch(batch, feature_ids)
            inputs = {k: v.to(device) for k, v in batch.items()}
            # Perturb_Expr is only used below for ground truth, never for prediction.
            inputs.pop('Perturb_Expr')
            prediction = model.inference(**inputs).logits.float().cpu()
            if prediction.shape != batch['Perturb_Expr'][:, :size].shape:
                raise ValueError('Prediction shape does not match the saved feature axis')
            if not torch.isfinite(prediction).all():
                raise ValueError('Model produced non-finite predictions')
            indices.append(batch['Perturb_Gene_Index'].reshape(-1).numpy())
            truth.append(batch['Perturb_Expr'][:, :size].numpy())
            controls.append(batch['Control_Expr'][:, :size].numpy())
            predictions.append(prediction.numpy())
    if not predictions:
        raise ValueError('Inference dataset is empty')
    return tuple(np.concatenate(items) for items in (indices, truth, predictions, controls))


def calculate_train_mean(dataloader, feature_ids):
    """Equal weight per compound, independent of each compound's well count."""
    sums, counts = {}, {}
    for batch in tqdm(dataloader, desc='Train-mean baseline'):
        validate_batch(batch, feature_ids)
        values = batch['Perturb_Expr'][:, :len(feature_ids)].double()
        indices = batch['Perturb_Gene_Index'].reshape(-1)
        for index in torch.unique(indices):
            selected = indices == index
            key = int(index)
            sums[key] = sums.get(key, 0) + values[selected].sum(0)
            counts[key] = counts.get(key, 0) + int(selected.sum())
    if not counts:
        raise ValueError('Training dataset is empty')
    logger.info('Baseline averages %d compound means with equal weights', len(counts))
    return torch.stack([sums[k] / counts[k] for k in sorted(counts)]).mean(0).float().numpy()


def write_outputs(args, feature_ids, index2feature, index2smiles, indices,
                  truth, predictions, controls, train_mean, in_sample):
    labels = [index2smiles[int(index)] for index in indices]
    n = len(labels)
    labels += [args.control_key_word] * n
    obs = pd.DataFrame(index=[f'perturb_{i}' for i in range(n)] + [f'control_{i}' for i in range(n)])
    obs[args.gene_column] = labels
    obs['Metadata_Smiles'] = labels
    obs['is_control'] = [False] * n + [True] * n
    var = pd.DataFrame({'feature_id': feature_ids}, index=[index2feature[i] for i in feature_ids])
    metadata = dict(model_path=str(Path(args.pretrained_model_path).resolve()),
                    condition_embedding_path=str(Path(args.index2morgan_pkl).resolve()),
                    condition_embedding_dim=args.condition_dim,
                    smiles2index_path=str(Path(args.smiles2index_pkl).resolve()),
                    train_dataset_path=str(Path(args.train_dataset_path).resolve()),
                    inference_dataset_path=str(Path(args.test_dataset_path).resolve()),
                    in_sample=in_sample, feature_alignment='intersection',
                    control_key_word=args.control_key_word,
                    controls_description='Paired controls from the processed dataset; may repeat.')
    for values, filename in ((truth, args.ground_truth_filename),
                             (predictions, args.prediction_filename),
                             (np.broadcast_to(train_mean, truth.shape), args.mean_baseline_filename)):
        matrix = np.concatenate((values, controls)).astype(np.float32)
        output = ad.AnnData(X=matrix, obs=obs.copy(), var=var.copy(), uns=dict(metadata))
        path = Path(args.output_dir) / filename
        output.write_h5ad(path)
        logger.info('Saved %s: %s', path, output.shape)
