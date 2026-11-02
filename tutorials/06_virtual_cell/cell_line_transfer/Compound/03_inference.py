"""Compound inference on the feature intersection saved by 02_finetune.py."""

# Support both an installed package and direct execution from this checkout.
from pathlib import Path as _Path
import sys as _sys
_SOURCE = _Path(__file__).resolve().parents[4] / "src"
if str(_SOURCE) not in _sys.path:
    _sys.path.insert(0, str(_SOURCE))


import logging
import sys
from pathlib import Path

import numpy as np
import torch
from datasets import load_from_disk

from descort.model.configuration_descort import DeSCORTConfig
from descort.model.modeling_descort import DeSCORTForPerturbPrediction

logger = logging.getLogger(__name__)

from descort.vc.inference_compound import (
    parse_args,
    invert_vocabulary,
    build_collator,
    build_dataloader,
    validate_batch,
    collect_test_outputs,
    calculate_train_mean,
    write_outputs,
)
from descort.vc.utils import load_pickle


def main():
    args = parse_args()
    handlers = [logging.StreamHandler()]
    if args.log_file:
        Path(args.log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(args.log_file))
    logging.basicConfig(level=logging.INFO, handlers=handlers,
                        format='%(asctime)s | %(levelname)s | %(message)s')
    if args.batch_size <= 0 or args.dataloader_num_workers < 0 or args.dataloader_prefetch_factor <= 0:
        raise ValueError('Invalid batch size or DataLoader worker/prefetch settings')
    filenames = [args.prediction_filename, args.ground_truth_filename, args.mean_baseline_filename]
    if len(set(filenames)) != 3 or any(Path(name).name != name for name in filenames):
        raise ValueError('Output filenames must be three distinct filenames without directories')
    for name in ('pretrained_model_path', 'smiles2index_pkl', 'index2morgan_pkl',
                 'feature2index_pkl', 'train_dataset_path', 'test_dataset_path'):
        if not Path(getattr(args, name)).exists():
            raise FileNotFoundError(f'{name} does not exist: {getattr(args, name)}')
    config = DeSCORTConfig.from_pretrained(args.pretrained_model_path)
    collator, feature_ids = build_collator(config, args.max_length, args.pad_value)
    if args.use_sdpa is not None:
        config.use_sdpa = args.use_sdpa == 'true'
    index2smiles = invert_vocabulary(load_pickle(args.smiles2index_pkl), 'smiles2index')
    index2feature = invert_vocabulary(load_pickle(args.feature2index_pkl), 'feature2index')
    if any(index not in index2feature for index in feature_ids):
        raise ValueError('Feature vocabulary does not cover the saved feature axis')
    embeddings = load_pickle(args.index2morgan_pkl)
    args.condition_dim = config.esm2_input_dim
    train_dataset = load_from_disk(args.train_dataset_path).with_format('torch')
    in_sample = Path(args.train_dataset_path).resolve() == Path(args.test_dataset_path).resolve()
    test_dataset = train_dataset if in_sample else load_from_disk(args.test_dataset_path).with_format('torch')
    if not len(train_dataset) or not len(test_dataset):
        raise ValueError('Training and inference datasets must be nonempty')
    split_ids = []
    for name, dataset in [('train', train_dataset), ('test', test_dataset)]:
        indices = [int(i) for i in dataset['Perturb_Gene_Index']]
        if 'Perturb_Gene' in dataset.column_names:
            if any(index2smiles.get(i) != label for i, label in zip(indices, dataset['Perturb_Gene'])):
                raise ValueError(f'{name}: perturbation labels do not match the SMILES index mapping')
        split_ids.append(set(indices))
        validate_batch(collator([dataset[i] for i in range(min(2, len(dataset)))]), feature_ids)
    used_ids = split_ids[0] | split_ids[1]
    for index in used_ids:
        if index not in index2smiles or index not in embeddings:
            raise ValueError(f'Compound ID {index} is missing a SMILES label or condition embedding')
        if np.asarray(embeddings[index]).shape != (config.esm2_input_dim,):
            raise ValueError(f'Condition width for ID {index} does not match esm2_input_dim={config.esm2_input_dim}')
    logger.info('Training rows=%d; inference rows=%d; saved features=%d; output features=%d; length=%d',
                len(train_dataset), len(test_dataset), len(config.population_feature_ids),
                len(feature_ids), collator.max_length)
    if in_sample:
        logger.warning('Inference uses the training dataset: these are in-sample outputs, not held-out evaluation.')
    logger.info('Condition width=%d; train compounds=%d; test compounds=%d; overlap=%d',
                config.esm2_input_dim, len(split_ids[0]), len(split_ids[1]), len(split_ids[0] & split_ids[1]))
    if args.check_only:
        logger.info('Input checks passed; checkpoint length=%d; no model inference or H5AD output.', config.max_seq_len)
        return
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable; pass --device cpu for CPU inference')
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    model, loading_info = DeSCORTForPerturbPrediction.from_pretrained(
        args.pretrained_model_path, config=config, INDEX2ESM2_Dict=embeddings, output_loading_info=True)
    if any(loading_info.get(key) for key in ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs')):
        raise ValueError(f'Checkpoint weights do not match the inference model: {loading_info}')
    model.to(device).eval()
    logger.info('Device=%s; batch size=%d; SDPA=%s', device, args.batch_size, config.use_sdpa)
    test_loader = build_dataloader(test_dataset, collator, args)
    indices, truth, predictions, controls = collect_test_outputs(model, test_loader, device, feature_ids)
    del test_loader, model
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    train_loader = build_dataloader(train_dataset, collator, args)
    train_mean = calculate_train_mean(train_loader, feature_ids)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    write_outputs(args, feature_ids, index2feature, index2smiles, indices,
                  truth, predictions, controls, train_mean, in_sample)
    logger.info('Compound inference and baseline generation completed')


if __name__ == '__main__':
    main()
