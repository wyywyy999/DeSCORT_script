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



from descort.data.collator import PerturbPredictionDataCollator
from descort.model.modeling_descort import DeSCORTForPerturbPrediction


logger = logging.getLogger(__name__)

from descort.vc.inference_crispr import (
    parse_args,
    setup_logging,
    resolve_device,
    build_dataloader,
    collect_test_outputs,
    calculate_train_mean,
    index_to_gene,
    write_anndata,
)
from descort.vc.utils import load_pickle


def main():
    args = parse_args()
    setup_logging(args.log_file, args.log_level)

    for path_name in (
        "pretrained_model_path",
        "index2gene_pkl",
        "index2esm2_pkl",
        "index2feature_pkl",
        "train_dataset_path",
        "test_dataset_path",
    ):
        path = Path(getattr(args, path_name))
        if not path.exists():
            raise FileNotFoundError(f"{path_name} does not exist: {path}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading dictionaries")
    index2gene = load_pickle(args.index2gene_pkl)
    index2esm2_dict = load_pickle(args.index2esm2_pkl)
    index2feature = load_pickle(args.index2feature_pkl)

    logger.info("Loading model from %s", args.pretrained_model_path)
    model = DeSCORTForPerturbPrediction.from_pretrained(
        args.pretrained_model_path,
        INDEX2ESM2_Dict=index2esm2_dict,
    )
    prediction_head_type = getattr(model.config, "prediction_head_type", "mlm")
    predict_residual = bool(getattr(model.config, "predict_residual", False))
    prediction_semantics = (
        "control + predicted delta"
        if predict_residual
        else "direct absolute perturbed features"
    )
    logger.info(
        "Checkpoint prediction mode: head=%s, semantics=%s",
        prediction_head_type,
        prediction_semantics,
    )
    device = resolve_device(args.device)
    model.to(device)
    model.eval()
    logger.info("Using device: %s", device)

    logger.info("Loading train dataset from %s", args.train_dataset_path)
    train_dataset = load_from_disk(args.train_dataset_path)
    logger.info("Loading test dataset from %s", args.test_dataset_path)
    test_dataset = load_from_disk(args.test_dataset_path)
    logger.info(
        "Loaded %s training cells and %s test cells",
        f"{len(train_dataset):,d}",
        f"{len(test_dataset):,d}",
    )

    collator = PerturbPredictionDataCollator(
        max_length=args.max_length,
        sampling=args.sampling,
        pad_value=args.pad_value,
        keep_first_n_tokens=args.keep_first_n_tokens,
    )
    train_dataloader = build_dataloader(train_dataset, collator, args)
    test_dataloader = build_dataloader(test_dataset, collator, args)

    gene_indices, ground_truth, predictions, controls = collect_test_outputs(
        model,
        test_dataloader,
        device,
    )
    del test_dataloader
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    perturb_gene_labels = [
        index_to_gene(index2gene, gene_index) for gene_index in gene_indices
    ]
    feature_indices = test_dataset[0]["Feature_name_index"]
    if hasattr(feature_indices, "tolist"):
        feature_indices = feature_indices.tolist()
    feature_labels = [index2feature[int(index)] for index in feature_indices]
    if len(feature_labels) != ground_truth.shape[1]:
        raise ValueError(
            "Feature-name count does not match inference output: "
            f"{len(feature_labels)} != {ground_truth.shape[1]}"
        )
    gene_labels = perturb_gene_labels + [args.control_label] * len(controls)

    ground_truth_with_controls = np.concatenate((ground_truth, controls), axis=0)
    predictions_with_controls = np.concatenate((predictions, controls), axis=0)

    train_mean = calculate_train_mean(train_dataloader)
    if train_mean.shape[0] != ground_truth.shape[1]:
        raise ValueError(
            "Training-mean feature count does not match the test data: "
            f"{train_mean.shape[0]} != {ground_truth.shape[1]}"
        )
    mean_baseline = np.broadcast_to(train_mean, ground_truth.shape)
    mean_baseline_with_controls = np.concatenate((mean_baseline, controls), axis=0)

    outputs = (
        (
            ground_truth_with_controls,
            args.ground_truth_filename,
        ),
        (
            predictions_with_controls,
            args.prediction_filename,
        ),
        (
            mean_baseline_with_controls,
            args.mean_baseline_filename,
        ),
    )
    for data, filename in outputs:
        write_anndata(
            data=data,
            gene_labels=gene_labels,
            feature_labels=feature_labels,
            gene_column=args.gene_column,
            output_path=output_dir / filename,
        )

    logger.info("Inference and baseline generation completed")


if __name__ == "__main__":
    main()
