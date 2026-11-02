"""Shared training arguments and feature-intersection helpers."""
import logging
from dataclasses import dataclass, field
import numpy as np
logger = logging.getLogger(__name__)


@dataclass
class VirtualCellArguments:
    """Arguments used only by the virtual-cell prediction task."""

    index2esm2_pkl: str = field(
        metadata={"help": "Path to the index-to-ESM2 embedding pickle file."}
    )
    gene_column: str = field(
        default="Perturb_Gene",
        metadata={"help": "Dataset column used for gene-grouped batches."},
    )
    genes_per_batch: int = field(default=1)
    wells_per_gene: int = field(default=8)
    gene_sampling_rounds: int = field(default=4)
    esm2_pca_path: str = field(
        default=None,
        metadata={"help": "Optional NPZ used to initialize the ESM2 input projection."},
    )
    freeze_esm2_pca_projection: bool = field(
        default=True,
        metadata={"help": "Freeze a PCA-initialized ESM2 input projection."},
    )


def infer_feature_order(dataset, vocab_size):
    """Find the sorted feature intersection across every training layout."""
    feature_dataset = dataset.select_columns(['Feature_name_index']).with_format('numpy')
    all_ids, layouts = set(), set()
    common_ids = None
    for start in range(0, len(feature_dataset), 256):
        for ids in feature_dataset[start:start + 256]['Feature_name_index']:
            ids = np.asarray(ids, dtype=np.int64)
            if ids.ndim != 1 or np.any(ids < 0) or np.any(ids >= vocab_size):
                raise ValueError('Feature_name_index contains invalid vocabulary indices')
            key = ids.tobytes()
            if key in layouts:
                continue
            valid_ids = ids[ids > 0]
            if len(np.unique(valid_ids)) != len(valid_ids):
                raise ValueError('A dataset row contains duplicate feature IDs')
            layouts.add(key)
            all_ids.update(valid_ids.tolist())
            current_ids = set(valid_ids.tolist())
            common_ids = current_ids if common_ids is None else common_ids & current_ids
    if not common_ids:
        raise ValueError('The training dataset has no common features (empty intersection)')
    logger.info('Aligning %d feature layouts to %d common feature IDs; dropping %d non-common IDs',
                len(layouts), len(common_ids), len(all_ids - common_ids))
    return sorted(common_ids)
