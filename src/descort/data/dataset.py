"""Data processors for DeSCORT training and evaluation."""

import os
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import tempfile
from time import perf_counter
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Union, Literal, Any, Tuple, Optional

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm
from datasets import Dataset, DatasetInfo, load_from_disk, concatenate_datasets, Features, Value, Sequence


PERTURBATION_FEATURES = Features({
    'Perturb_Gene': Value('string'),
    'Perturb_expression': Sequence(Value('float32')),
    'Control_expression': Sequence(Value('float32')),
    'Feature_name_index': Sequence(Value('int64')),
    'Perturb_Gene_Index': Value('int64'),
})


def _matrix_to_arrow(values, value_type):
    """Build a typed Arrow list column without per-value Python conversion."""
    values = np.ascontiguousarray(values)
    rows, width = values.shape
    offsets = np.arange(rows + 1, dtype=np.int64) * width
    if offsets[-1] > np.iinfo(np.int32).max:
        raise ValueError('One parquet file exceeds Arrow\'s list offset limit')
    return pa.ListArray.from_arrays(
        pa.array(offsets.astype(np.int32)), pa.array(values.reshape(-1), type=value_type),
    )


class DeSCORTForMaskedLMDataProcessor:
    """Data processor for Cell Painting parquet files for MLM training."""

    def __init__(self, nproc: int = 1, verbosity: Literal[0, 1] = 1):
        self.nproc = nproc
        self._setup_logging(verbosity)

    def _setup_logging(self, verbosity: int):
        if verbosity == 1:
            logging.basicConfig(
                format="%(asctime)s | %(levelname)s | %(filename)s:%(lineno)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO,
            )
        else:
            logging.basicConfig(level=logging.WARNING)

    @staticmethod
    def find_parquet_files(filepath: Union[Path, str]) -> List[Path]:
        if not isinstance(filepath, Path):
            filepath = Path(filepath).absolute()
        parquet_files = list(filepath.rglob("*.parquet"))
        logging.info("Found %d parquet files in %s", len(parquet_files), filepath)
        return parquet_files

    @staticmethod
    def filter_df_by_gene_list(df: pd.DataFrame, gene_list: List[str],
                                gene_column: str = 'Metadata_Foci_Barcode_MatchedTo_GeneCode') -> pd.DataFrame:
        if gene_column not in df.columns:
            raise ValueError(f"Column '{gene_column}' not found. Available: {list(df.columns)}")
        return df[df[gene_column].isin(gene_list)].copy()

    def extract_features(self, df: pd.DataFrame, Feature2index_dict: Dict,
                         need_feature_list: List = None) -> Tuple:
        feature_columns = [col for col in df.columns if not col.startswith('Metadata')]
        if need_feature_list is None:
            feature_columns_index = [Feature2index_dict[feature] for feature in feature_columns]
            df = df[feature_columns]
        else:
            intersected = set(need_feature_list).intersection(Feature2index_dict)
            need_feature_columns = [f for f in feature_columns if f in intersected]
            feature_columns_index = [Feature2index_dict[f] for f in need_feature_columns]
            df = df[need_feature_columns]
        return df.values, feature_columns_index

    def process_single_parquet(self, parquet_path, key_gene_column, need_genes_list,
                                FEATURE2INDEX_DICT, need_feature_list=None) -> Tuple[List[Dict], str, int]:
        try:
            df = pd.read_parquet(parquet_path)
            na_number = df.filter(regex='^(?!Meta)').isna().sum().sum()
            if na_number != 0:
                df = df.fillna(0)
            df = self.filter_df_by_gene_list(df, need_genes_list, key_gene_column)
            # Use parent dir name as prefix to avoid collisions when
            # the same plate name appears in different subdirectories.
            sample_name = f"{parquet_path.parent.name}_{os.path.basename(parquet_path).split('.')[0]}"

            data_dict = {
                'Perturb_Gene': df[key_gene_column].values,
                'Sample_name': sample_name,
            }
            data_dict['Feature_expression'], feature_name_index = self.extract_features(
                df, FEATURE2INDEX_DICT, need_feature_list,
            )
            data_dict['Feature_name_index'] = [feature_name_index] * df.shape[0]
            return [data_dict], sample_name, df.shape[0]
        except Exception as e:
            logging.error("Error processing %s: %s", parquet_path, e)
            return {}, None, None

    def mapping_dict(self, data_dict: Dict, GENE2INDEX_DICT: Dict) -> Dict:
        data_dict['Perturb_Gene_Index'] = [GENE2INDEX_DICT[gene] for gene in data_dict['Perturb_Gene']]
        return data_dict

    def save_to_disk(self, data_dict: Dict, output_dir: Path, index, max_shard_size: str = "5GB") -> None:
        data_name = data_dict['Sample_name']
        del data_dict['Sample_name']

        for key in ['Perturb_expression', 'Control_expression']:
            if key in data_dict:
                data_dict[key] = [[float(x) for x in seq] for seq in data_dict[key]]

        ds = Dataset.from_dict(data_dict)
        ds.set_format("torch")
        TARGET_FEATURES = Features({
            'Perturb_Gene': Value('string'),
            'Feature_expression': Sequence(Value('float32')),
            'Feature_name_index': Sequence(Value('int64')),
            'Perturb_Gene_Index': Value('int64'),
        })
        ds = ds.cast(TARGET_FEATURES)
        logging.info("Converted to Dataset: %d samples", len(ds))

        if not isinstance(output_dir, Path):
            output_dir = Path(output_dir)
        shard_output_dir = output_dir / f'{data_name}_{index}_index'
        ds.save_to_disk(str(shard_output_dir), num_proc=self.nproc, max_shard_size=max_shard_size)
        logging.info("Saved shard: %d samples to %s", len(ds), shard_output_dir)

    def __call__(self, parquet_dirs, output_dir, data_name, key_gene_column,
                 GENE2INDEX_DICT, FEATURE2INDEX_DICT, save_to_disk=True,
                 max_shard_size='5GB', need_genes_list=None, need_feature_list=None) -> Dict[str, Any]:
        if isinstance(parquet_dirs, str):
            parquet_dirs = [parquet_dirs]

        all_parquet_files = []
        for parquet_dir in parquet_dirs:
            all_parquet_files.extend(self.find_parquet_files(parquet_dir))
        if data_name is not None:
            all_parquet_files = [f for f in all_parquet_files if any(name in str(f) for name in data_name)]

        if not all_parquet_files:
            logging.warning("No parquet files found")
            return {'status': 'no_files_found'}

        logging.info("Processing %d parquet files", len(all_parquet_files))

        for parquet_path in tqdm(all_parquet_files, desc='Processing files'):
            try:
                data_dict_list, sample_name, cell_number = self.process_single_parquet(
                    parquet_path, key_gene_column, need_genes_list, FEATURE2INDEX_DICT, need_feature_list,
                )
                logging.info("Sample %s: %d cells", sample_name, cell_number)

                combined_data_dict = {}
                for index in range(len(data_dict_list)):
                    data_dict = self.mapping_dict(data_dict_list[index], GENE2INDEX_DICT=GENE2INDEX_DICT)
                    for key, value in data_dict.items():
                        if key == 'Sample_name':
                            combined_data_dict[key] = value
                        else:
                            combined_data_dict.setdefault(key, []).extend(value)

                if save_to_disk:
                    logging.info("Saving to disk...")
                    self.save_to_disk(combined_data_dict, output_dir, 0, max_shard_size=max_shard_size)
            except Exception as e:
                logging.error("Error processing %s: %s", parquet_path, e)
                continue

        return {'status': 'success'}

class DeSCORTForContrastiveLMDataProcessor(DeSCORTForMaskedLMDataProcessor):
    """Data processor for Cell Painting parquet files for Contrastive LM training."""
    pass

class DeSCORTPerturbPredictionProcessor(DeSCORTForMaskedLMDataProcessor):
    """Shared fast preprocessing for gene and compound perturbation datasets."""

    def __init__(self, nproc: int = 1, verbosity: Literal[0, 1] = 1,
                 control_key_word: str = "nontargeting", read_batch_size: int = 8192):
        if nproc < 1 or read_batch_size < 1:
            raise ValueError('nproc and read_batch_size must be positive')
        super().__init__(nproc=nproc, verbosity=verbosity)
        self.control_key_word = control_key_word
        self.read_batch_size = read_batch_size

    def _prepare_file(self, parquet_path, key_gene_column, need_genes_list,
                      FEATURE2INDEX_DICT, need_feature_list, GENE2INDEX_DICT=None):
        parquet = pq.ParquetFile(parquet_path)
        columns = parquet.schema_arrow.names
        if key_gene_column not in columns:
            raise ValueError(f"{parquet_path}: missing column {key_gene_column!r}")
        labels = parquet.read(columns=[key_gene_column], use_threads=False).column(0).to_pandas()
        is_control = labels.eq(self.control_key_word).fillna(False).to_numpy(dtype=bool)
        allowed = labels.notna() if need_genes_list is None else labels.isin(need_genes_list)
        if GENE2INDEX_DICT is not None:
            is_mapped = labels.isin(GENE2INDEX_DICT)
            unknown = ((~is_mapped if need_genes_list is None else allowed & ~is_mapped)
                       & ~is_control).sum()
            if unknown:
                logging.warning('%s: excluding %d rows with missing/unmapped labels', parquet_path, unknown)
            allowed &= is_mapped
        is_target = ~is_control & allowed.to_numpy(dtype=bool)
        if not is_target.any():
            logging.warning('%s: no selected perturbation rows; skipping', parquet_path)
            return None
        if not is_control.any():
            raise ValueError(f"{parquet_path}: no controls with {key_gene_column}={self.control_key_word!r}")

        selected_features = set(FEATURE2INDEX_DICT)
        if need_feature_list is not None:
            selected_features.intersection_update(need_feature_list)
        feature_columns = [name for name in columns
                           if not name.startswith('Metadata') and name in selected_features]
        if not feature_columns:
            raise ValueError(f"{parquet_path}: no features shared with the selected vocabulary")
        feature_indices = np.asarray([FEATURE2INDEX_DICT[name] for name in feature_columns], dtype=np.int64)

        # Retain controls independently of gene/embedding lists. Read feature batches
        # to avoid materializing the full single-cell parquet as a pandas DataFrame.
        keep = is_target | is_control
        values = np.empty((int(keep.sum()), len(feature_columns)), dtype=np.float32)
        source_offset = output_offset = 0
        for batch in parquet.iter_batches(batch_size=self.read_batch_size,
                                           columns=feature_columns, use_threads=False):
            mask = keep[source_offset:source_offset + batch.num_rows]
            source_offset += batch.num_rows
            count = int(mask.sum())
            if count:
                selected = batch.filter(pa.array(mask)).to_pandas().fillna(0)
                values[output_offset:output_offset + count] = selected.to_numpy(dtype=np.float32)
                output_offset += count
        labels = labels.loc[keep].reset_index(drop=True)
        target_rows = np.flatnonzero(is_target[keep])
        control_rows = np.flatnonzero(is_control[keep])
        target_labels = labels.iloc[target_rows].reset_index(drop=True)
        groups = target_labels.groupby(target_labels, sort=False).indices
        # Explicit gene lists retain their historical order; without a list,
        # use first appearance, matching the compound preprocessing entry point.
        group_order = list(groups) if need_genes_list is None else [g for g in need_genes_list if g in groups]
        rng = np.random.default_rng(42)
        ordered_rows, paired_rows, group_sizes = [], [], []
        for label in group_order:
            rows = target_rows[groups[label]]
            ordered_rows.append(rows)
            paired_rows.append(rng.choice(control_rows, size=len(rows), replace=len(rows) > len(control_rows)))
            group_sizes.append(len(rows))
        return (labels, values, feature_indices, np.concatenate(ordered_rows),
                np.concatenate(paired_rows), group_sizes)

    def process_dataset(self, parquet_path, key_gene_column, GENE2INDEX_DICT,
                        FEATURE2INDEX_DICT, need_genes_list=None, need_feature_list=None):
        """Return a typed Dataset using the same pairing logic for both tasks."""
        plan = self._prepare_file(parquet_path, key_gene_column, need_genes_list,
                                  FEATURE2INDEX_DICT, need_feature_list, GENE2INDEX_DICT)
        if plan is None:
            return None
        labels, values, feature_indices, ordered_rows, paired_rows, _ = plan
        ordered_labels = labels.iloc[ordered_rows].tolist()
        output = pa.Table.from_arrays([
            pa.array(ordered_labels, type=pa.string()),
            _matrix_to_arrow(values[ordered_rows], pa.float32()),
            _matrix_to_arrow(values[paired_rows], pa.float32()),
            _matrix_to_arrow(np.broadcast_to(feature_indices, (len(ordered_rows), len(feature_indices))), pa.int64()),
            pa.array([GENE2INDEX_DICT[label] for label in ordered_labels], type=pa.int64()),
        ], schema=PERTURBATION_FEATURES.arrow_schema)
        dataset = Dataset(output, info=DatasetInfo(features=PERTURBATION_FEATURES))
        dataset.set_format('torch')
        return dataset

    def process_single_parquet(self, parquet_path, key_gene_column, need_genes_list,
                                FEATURE2INDEX_DICT, need_feature_list=None):
        """Keep the legacy per-group dictionary interface on the shared fast path."""
        plan = self._prepare_file(parquet_path, key_gene_column, need_genes_list,
                                  FEATURE2INDEX_DICT, need_feature_list)
        path = Path(parquet_path)
        sample_name = f'{path.parent.name}_{path.stem}'
        if plan is None:
            return [], sample_name, 0
        labels, values, feature_indices, ordered_rows, paired_rows, group_sizes = plan
        result, start = [], 0
        for size in group_sizes:
            end = start + size
            result.append({
                'Perturb_Gene': labels.iloc[ordered_rows[start:end]].tolist(),
                'Sample_name': sample_name,
                'Perturb_expression': values[ordered_rows[start:end]],
                'Control_expression': values[paired_rows[start:end]],
                'Feature_name_index': [feature_indices.tolist()] * size,
            })
            start = end
        return result, sample_name, len(labels)

    def save_to_disk(self, data_dict, output_dir, index, max_shard_size='5GB'):
        """Compatibility path for callers supplying an already combined dictionary."""
        data_name = data_dict['Sample_name']
        dataset = Dataset.from_dict({k: v for k, v in data_dict.items() if k != 'Sample_name'},
                                    features=PERTURBATION_FEATURES)
        dataset.set_format('torch')
        dataset.save_to_disk(str(Path(output_dir) / f'{data_name}_{index}_index'),
                             num_proc=1, max_shard_size=max_shard_size)

    def _process_and_save(self, item, *, output_dir, key_gene_column, GENE2INDEX_DICT,
                          FEATURE2INDEX_DICT, need_genes_list, need_feature_list,
                          save_to_disk, max_shard_size):
        index, path = item
        start = perf_counter()
        dataset = self.process_dataset(path, key_gene_column, GENE2INDEX_DICT,
                                       FEATURE2INDEX_DICT, need_genes_list, need_feature_list)
        if dataset is None:
            return None
        shard = Path(output_dir) / f'{index:06d}_{path.parent.name}_{path.stem}_0_index'
        if save_to_disk:
            dataset.save_to_disk(str(shard), num_proc=1, max_shard_size=max_shard_size)
        logging.info('%s: %d rows, %.2f s including save', path.name, len(dataset), perf_counter() - start)
        return str(shard), len(dataset), None if save_to_disk else dataset

    def __call__(self, parquet_dirs, output_dir, data_name, key_gene_column,
                 GENE2INDEX_DICT, FEATURE2INDEX_DICT, save_to_disk=True,
                 max_shard_size='5GB', need_genes_list=None, need_feature_list=None):
        if isinstance(parquet_dirs, (str, Path)):
            parquet_dirs = [parquet_dirs]
        files = set()
        for directory in parquet_dirs:
            path = Path(directory)
            if not path.is_dir():
                raise NotADirectoryError(path)
            files.update(file.resolve() for file in self.find_parquet_files(path))
        names = [data_name] if isinstance(data_name, str) else (data_name or [])
        names = [name for name in names if name.strip()]
        files = sorted(file for file in files if not names or any(name in str(file) for name in names))
        if not files:
            return {'status': 'no_files_found', 'num_samples': 0, 'dataset_paths': []}
        if save_to_disk:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
        worker = partial(self._process_and_save, output_dir=output_dir,
                         key_gene_column=key_gene_column, GENE2INDEX_DICT=GENE2INDEX_DICT,
                         FEATURE2INDEX_DICT=FEATURE2INDEX_DICT, need_genes_list=need_genes_list,
                         need_feature_list=need_feature_list, save_to_disk=save_to_disk,
                         max_shard_size=max_shard_size)
        with ThreadPoolExecutor(max_workers=min(self.nproc, len(files))) as executor:
            results = [result for result in executor.map(worker, enumerate(files)) if result is not None]
        return {'status': 'success' if results else 'no_samples',
                'num_samples': sum(count for _, count, _ in results),
                'dataset_paths': [path for path, _, _ in results] if save_to_disk else [],
                'datasets': [dataset for _, _, dataset in results] if not save_to_disk else []}

    def process_and_merge(self, *, parquet_dirs, output_dir, save_dir, data_name,
                          key_gene_column, GENE2INDEX_DICT, FEATURE2INDEX_DICT,
                          need_genes_list=None, need_feature_list=None, merge_nproc=2):
        """Run both tutorial tasks through one isolated processing/merge pipeline."""
        start = perf_counter()
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        run_dir = Path(tempfile.mkdtemp(prefix='perturb_', dir=output_dir))
        result = self(parquet_dirs, run_dir, data_name, key_gene_column,
                      GENE2INDEX_DICT, FEATURE2INDEX_DICT,
                      need_genes_list=need_genes_list, need_feature_list=need_feature_list)
        if not result['num_samples']:
            raise ValueError(f"No training/evaluation samples produced: {result['status']}")
        processed = perf_counter()
        Path(save_dir).parent.mkdir(parents=True, exist_ok=True)
        merged = merge_datasets_direct(run_dir, save_dir, nproc=merge_nproc, skip_errors=False)
        end = perf_counter()
        logging.info('Processed %d rows from %d files; process/save=%.2fs, merge/save=%.2fs, total=%.2fs',
                     len(merged), len(result['dataset_paths']), processed - start, end - processed, end - start)
        return merged


def merge_datasets_direct(data_dir: str, save_dir: str, nproc: int = 2, skip_errors: bool = True):
    """Directly merge all dataset shards into one."""
    data_dir = Path(data_dir)
    dataset_paths = sorted(d for d in data_dir.glob("*") if d.is_dir() and any(d.glob("*.arrow")))
    if not dataset_paths:
        raise ValueError(f"No valid datasets found in {data_dir}")

    print(f"Found {len(dataset_paths)} dataset shards")

    def safe_load(path):
        try:
            ds = load_from_disk(str(path))
            print(f"Loaded: {path.name} ({len(ds):,d} samples)")
            return ds
        except Exception as e:
            if skip_errors:
                print(f"Skipped {path.name}: {e}")
                return None
            raise

    print("Loading all datasets...")
    if nproc < 1:
        raise ValueError("nproc must be positive")
    with ThreadPoolExecutor(max_workers=min(nproc, len(dataset_paths))) as executor:
        all_datasets = list(executor.map(safe_load, dataset_paths))
    valid_datasets = [ds for ds in all_datasets if ds is not None]
    if not valid_datasets:
        raise ValueError("No valid data available for merging")

    total_samples = sum(len(ds) for ds in valid_datasets)
    print(f"Merging {len(valid_datasets)} datasets ({total_samples:,d} total samples)...")
    merged = concatenate_datasets(valid_datasets)

    print(f"Saving to {save_dir}")
    merged.save_to_disk(save_dir)
    print(f"Done! Total: {len(merged):,d} samples")
    return merged
