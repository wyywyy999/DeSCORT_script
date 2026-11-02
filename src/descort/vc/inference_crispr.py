'CRISPR inference data loading and export helpers.'
import argparse
import logging
from pathlib import Path
import anndata as ad
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
logger = logging.getLogger(__name__)


def add_boolean_argument(parser, name, default, help_text):
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        f"--{name}",
        dest=name,
        action="store_true",
        help=help_text,
    )
    group.add_argument(
        f"--no_{name}",
        dest=name,
        action="store_false",
        help=f"Disable {help_text.lower()}",
    )
    parser.set_defaults(**{name: default})


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run virtual-cell inference and save prediction, ground-truth, "
            "and train-mean-baseline AnnData files."
        )
    )

    # ---- Model and vocabulary files ----
    parser.add_argument(
        "--pretrained_model_path",
        required=True,
        help="Path to the fine-tuned virtual-cell model directory.",
    )
    parser.add_argument(
        "--index2gene_pkl",
        required=True,
        help="Path to the index-to-gene pickle file.",
    )
    parser.add_argument(
        "--index2esm2_pkl",
        required=True,
        help="Path to the index-to-ESM2 embedding pickle file.",
    )
    parser.add_argument(
        "--index2feature_pkl",
        required=True,
        help="Path to the index-to-feature-name pickle file.",
    )

    # ---- Dataset and output paths ----
    parser.add_argument(
        "--train_dataset_path",
        required=True,
        help="Path to the processed training dataset.",
    )
    parser.add_argument(
        "--test_dataset_path",
        required=True,
        help="Path to the processed test dataset.",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory in which all H5AD outputs are saved.",
    )
    parser.add_argument(
        "--prediction_filename",
        default="prediction_data.h5ad",
        help="Prediction H5AD filename. (default: %(default)s)",
    )
    parser.add_argument(
        "--ground_truth_filename",
        default="ground_truth.h5ad",
        help="Ground-truth H5AD filename. (default: %(default)s)",
    )
    parser.add_argument(
        "--mean_baseline_filename",
        default="perturb_mean_data.h5ad",
        help="Train-mean-baseline H5AD filename. (default: %(default)s)",
    )

    # ---- DataLoader / collator settings ----
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="Inference device. (default: %(default)s)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Inference batch size. (default: %(default)d)",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=1024,
        help="Maximum feature sequence length. (default: %(default)d)",
    )
    parser.add_argument(
        "--pad_value",
        type=float,
        default=100,
        help="Expression padding value. (default: %(default)s)",
    )
    parser.add_argument(
        "--keep_first_n_tokens",
        type=int,
        default=0,
        help="Number of leading features always retained. (default: %(default)d)",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=4,
        help="Number of DataLoader workers. (default: %(default)d)",
    )
    parser.add_argument(
        "--dataloader_prefetch_factor",
        type=int,
        default=2,
        help="Batches prefetched per worker. (default: %(default)d)",
    )
    add_boolean_argument(
        parser,
        "sampling",
        default=False,
        help_text="Sample features when a sequence exceeds max_length.",
    )
    add_boolean_argument(
        parser,
        "dataloader_pin_memory",
        default=True,
        help_text="Pin DataLoader memory.",
    )
    add_boolean_argument(
        parser,
        "dataloader_persistent_workers",
        default=True,
        help_text="Keep DataLoader workers alive between iterations.",
    )

    # ---- Output metadata ----
    parser.add_argument(
        "--gene_column",
        default="Perturb_Gene",
        help="Gene-label column written to AnnData.obs. (default: %(default)s)",
    )
    parser.add_argument(
        "--control_label",
        default="nontargeting",
        help="Label assigned to appended controls. (default: %(default)s)",
    )
    # ---- Logging ----
    parser.add_argument(
        "--log_file",
        default=None,
        help="Optional log-file path. Logs always go to stdout.",
    )
    parser.add_argument(
        "--log_level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"),
        help="Logging level. (default: %(default)s)",
    )

    return parser.parse_args()


def setup_logging(log_file, log_level):
    handlers = [logging.StreamHandler()]
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path))

    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(filename)s:%(lineno)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=getattr(logging, log_level),
        handlers=handlers,
    )


def resolve_device(requested_device):
    device = torch.device(requested_device)
    if device.type == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA is unavailable; falling back to CPU.")
        return torch.device("cpu")
    return device


def build_dataloader(dataset, collator, args):
    kwargs = {
        "dataset": dataset,
        "batch_size": args.batch_size,
        "shuffle": False,
        "num_workers": args.dataloader_num_workers,
        "collate_fn": collator,
        "pin_memory": args.dataloader_pin_memory,
    }
    if args.dataloader_num_workers > 0:
        kwargs["prefetch_factor"] = args.dataloader_prefetch_factor
        kwargs["persistent_workers"] = args.dataloader_persistent_workers
    return DataLoader(**kwargs)


def trim_padding(values, attention_mask):
    """Remove right padding, requiring a rectangular result for AnnData."""
    valid_lengths = attention_mask.sum(dim=1).to(dtype=torch.int64)
    if valid_lengths.numel() == 0:
        return values
    if not torch.all(valid_lengths == valid_lengths[0]):
        raise ValueError(
            "Samples in a batch have different valid feature lengths; "
            "cannot build a rectangular AnnData matrix."
        )
    return values[:, : int(valid_lengths[0].item())]


def collect_test_outputs(model, dataloader, device):
    gene_indices = []
    ground_truth = []
    predictions = []
    controls = []

    with torch.inference_mode():
        for batch in tqdm(dataloader, desc="Test inference", unit="batch"):
            batch = {
                key: value.to(device) if isinstance(value, torch.Tensor) else value
                for key, value in batch.items()
            }

            attention_mask = batch["Attention_mask"]
            control_expr = trim_padding(batch["Control_Expr"], attention_mask)
            perturb_expr = trim_padding(batch["Perturb_Expr"], attention_mask)
            output = model.inference(**batch)

            prediction = output.logits
            if prediction.shape != perturb_expr.shape:
                raise ValueError(
                    "Prediction shape does not match ground truth: "
                    f"{tuple(prediction.shape)} != {tuple(perturb_expr.shape)}"
                )

            controls.append(control_expr.float().cpu().numpy())
            ground_truth.append(perturb_expr.float().cpu().numpy())
            predictions.append(prediction.float().cpu().numpy())
            gene_indices.append(batch["Perturb_Gene_Index"].cpu().numpy())

    if not predictions:
        raise ValueError("The test dataset produced no inference batches.")

    return (
        np.concatenate(gene_indices, axis=0),
        np.concatenate(ground_truth, axis=0),
        np.concatenate(predictions, axis=0),
        np.concatenate(controls, axis=0),
    )


def calculate_train_mean(dataloader):
    """Calculate an equal-weighted mean over training perturbation means.

    Each perturbation is first averaged over all of its training wells, and the
    resulting perturbation means are then averaged with equal weight.  Thus a
    perturbation with more wells cannot dominate the train baseline.
    """
    feature_sums_by_gene = {}
    cell_counts_by_gene = {}

    for batch in tqdm(dataloader, desc="Train-mean baseline", unit="batch"):
        perturb_expr = batch["Perturb_Expr"]
        attention_mask = batch["Attention_mask"]
        perturb_expr = trim_padding(perturb_expr, attention_mask)
        gene_indices = batch["Perturb_Gene_Index"].reshape(-1)
        if gene_indices.shape[0] != perturb_expr.shape[0]:
            raise ValueError(
                "Perturb_Gene_Index count does not match Perturb_Expr rows: "
                f"{gene_indices.shape[0]} != {perturb_expr.shape[0]}"
            )

        perturb_expr = perturb_expr.to(dtype=torch.float64)
        for gene_index in torch.unique(gene_indices):
            gene_mask = gene_indices == gene_index
            gene_key = int(gene_index.item())
            gene_sum = perturb_expr[gene_mask].sum(dim=0)
            if gene_key in feature_sums_by_gene:
                feature_sums_by_gene[gene_key] += gene_sum
                cell_counts_by_gene[gene_key] += int(gene_mask.sum().item())
            else:
                feature_sums_by_gene[gene_key] = gene_sum
                cell_counts_by_gene[gene_key] = int(gene_mask.sum().item())

    if not feature_sums_by_gene:
        raise ValueError("The training dataset produced no batches.")

    perturbation_means = torch.stack(
        [
            feature_sums_by_gene[gene_index] / cell_counts_by_gene[gene_index]
            for gene_index in sorted(feature_sums_by_gene)
        ]
    )
    logger.info(
        "Train baseline equally averages %d perturbation means (well counts: min=%d, max=%d)",
        len(cell_counts_by_gene),
        min(cell_counts_by_gene.values()),
        max(cell_counts_by_gene.values()),
    )
    return perturbation_means.mean(dim=0).cpu().numpy()


def index_to_gene(index2gene, gene_index):
    try:
        return index2gene[int(gene_index)]
    except (KeyError, IndexError, TypeError) as error:
        raise KeyError(f"Gene index {int(gene_index)} is absent from index2gene.") from error


def write_anndata(data, gene_labels, feature_labels, gene_column, output_path):
    adata = ad.AnnData(X=data)
    adata.obs[gene_column] = gene_labels
    adata.var_names = feature_labels
    adata.write(output_path)
    logger.info("Saved %s with shape %s", output_path, adata.shape)
