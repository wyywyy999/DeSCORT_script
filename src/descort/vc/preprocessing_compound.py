'Build compound training and optional unseen-compound test datasets.'
import argparse
from pathlib import Path
from .utils import load_pickle


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_data_paths", nargs="+", required=True,
                        help="Source directories; parquet files are found recursively.")
    parser.add_argument("--train_output_dir", required=True,
                        help="Parent directory for this run's training shards.")
    parser.add_argument("--train_save_dir", required=True,
                        help="Directory for the merged training dataset.")
    parser.add_argument("--train_gene_list_pkl", help="Optional training SMILES list.")
    parser.add_argument("--test_gene_list_pkl", help="Optional test SMILES list.")
    parser.add_argument("--test_output_dir", help="Parent directory for test shards.")
    parser.add_argument("--test_save_dir", help="Directory for the merged test dataset.")
    parser.add_argument("--gene2index_pkl", required=True,
                        help="SMILES-to-index pickle (compatible argument name).")
    parser.add_argument("--feature2index_pkl", required=True)
    parser.add_argument("--need_feature_list_pkl", required=True)
    parser.add_argument("--key_gene_column", default="Metadata_Smiles")
    parser.add_argument("--train_data_names", nargs="*", default=None,
                        help="Optional path substrings; omitted or empty means all parquet files.")
    parser.add_argument("--test_data_names", nargs="*", default=None,
                        help="Optional test path substrings; omitted or empty means all parquet files.")
    parser.add_argument("--control_key_word", default="CS(=O)C",
                        help="Control value in key_gene_column (default: DMSO).")
    parser.add_argument("--nproc", type=int, default=4,
                        help="Number of parquet files processed concurrently (default: 4).")
    parser.add_argument("--merge_nproc", type=int, default=2,
                        help="Number of concurrent shard loaders (default: 2).")
    parser.add_argument("--verbosity", type=int, choices=(0, 1), default=1)
    args = parser.parse_args(argv)
    args.train_data_names = [name for name in (args.train_data_names or []) if name.strip()] or None
    args.test_data_names = [name for name in (args.test_data_names or []) if name.strip()] or None
    test_options = [args.test_gene_list_pkl, args.test_output_dir, args.test_save_dir]
    if any(test_options) and not (all(test_options) and args.train_gene_list_pkl):
        parser.error("Test processing requires train_gene_list_pkl, test_gene_list_pkl, test_output_dir and test_save_dir")
    if args.test_data_names and not all(test_options):
        parser.error("test_data_names requires test processing options")
    output_paths = [Path(p).resolve() for p in [args.train_output_dir, args.train_save_dir,
                    args.test_output_dir, args.test_save_dir] if p]
    if len(set(output_paths)) != len(output_paths):
        parser.error("Training/test shard and merged output directories must be distinct")
    if args.nproc < 1 or args.merge_nproc < 1:
        parser.error("nproc and merge_nproc must be positive")
    return args


def load_smiles_list(path, gene2index_dict, control_key_word):
    if not path:
        return None
    values = list(load_pickle(path))
    if not values or not all(isinstance(value, str) for value in values):
        raise ValueError(f"Expected a nonempty SMILES string list: {path}")
    if len(set(values)) != len(values):
        raise ValueError(f"Duplicate SMILES in {path}")
    missing = set(values) - set(gene2index_dict) - {control_key_word}
    if missing:
        raise ValueError(f"SMILES absent from the index dictionary in {path}: {sorted(missing)}")
    if not (set(values) - {control_key_word}):
        raise ValueError(f"No perturbation SMILES in {path}")
    return values
