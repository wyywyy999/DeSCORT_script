# Support both an installed package and direct execution from this checkout.
from pathlib import Path as _Path
import sys as _sys
_SOURCE = _Path(__file__).resolve().parents[2] / "src"
if str(_SOURCE) not in _sys.path:
    _sys.path.insert(0, str(_SOURCE))

import os
import sys
import argparse
import pickle

from descort.data.dataset import DeSCORTForContrastiveLMDataProcessor
from descort.data.dataset import merge_datasets_direct


def parse_args():
    parser = argparse.ArgumentParser(
        description="Process CellPainting data for DeSCORT masked LM pretraining."
    )

    # ---- Data source paths ----
    parser.add_argument(
        "--source_data_paths", nargs="+", required=True,
        help="One or more source parquet directories."
    )

    # ---- Output directories ----
    parser.add_argument(
        "--train_output_dir", required=True,
        help="Directory for processed train data output."
    )
    parser.add_argument(
        "--train_save_dir", required=True,
        help="Directory for merged train data."
    )

    # ---- Pickle / vocabulary files ----
    parser.add_argument(
        "--gene2index_pkl", required=True,
        help="Path to label2index pickle file."
    )
    parser.add_argument(
        "--feature2index_pkl", required=True,
        help="Path to feature2index vocabulary pickle file."
    )
    parser.add_argument(
        "--need_feature_list_pkl", required=True,
        help="Path to the need_feature_list pickle file (e.g. union feature list)."
    )

    # ---- Column / gene settings ----
    parser.add_argument(
        "--key_gene_column", default="Metadata_Foci_Barcode_MatchedTo_GeneCode",
        help="Column name used as the gene key. (default: %(default)s)"
    )
    parser.add_argument(
        "--train_data_names", nargs="+", default=None,
        help="List of data names to process for training. (default: None)"
    )
    # ---- Processing options ----
    parser.add_argument(
        "--nproc", type=int, default=1,
        help="Number of processes for the data processor. (default: %(default)d)"
    )
    parser.add_argument(
        "--merge_nproc", type=int, default=2,
        help="Number of processes for merging datasets. (default: %(default)d)"
    )
    parser.add_argument(
        "--verbosity", type=int, default=1,
        help="Verbosity level. (default: %(default)d)"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # ---- Load dictionaries ----
    with open(args.gene2index_pkl, "rb") as f:
        gene2index_dict = pickle.load(f)

    with open(args.feature2index_pkl, "rb") as f:
        feature2index_dict = pickle.load(f)

    with open(args.need_feature_list_pkl, "rb") as f:
        need_feature_list = pickle.load(f)

    ems2_genes_list = list(gene2index_dict.keys())

    # ---- Processor ----
    processor = DeSCORTForContrastiveLMDataProcessor(
        nproc=args.nproc, verbosity=args.verbosity
    )

    # ==================== Train ====================

    os.makedirs(args.train_output_dir, exist_ok=True)
    os.makedirs(args.train_save_dir, exist_ok=True)

    processor(
        parquet_dirs=args.source_data_paths,
        output_dir=args.train_output_dir,
        data_name=args.train_data_names,
        key_gene_column=args.key_gene_column,
        GENE2INDEX_DICT=gene2index_dict,
        FEATURE2INDEX_DICT=feature2index_dict,
        need_genes_list=ems2_genes_list,
        save_to_disk=True,
        need_feature_list=need_feature_list,
    )

    merge_datasets_direct(
        data_dir=args.train_output_dir,
        save_dir=args.train_save_dir,
        nproc=args.merge_nproc,
        skip_errors=True,
    )
    print(f"[INFO] Train processing done. Merged data saved to: {args.train_save_dir}")

if __name__ == "__main__":
    main()