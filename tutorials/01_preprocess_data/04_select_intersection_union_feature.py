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
import logging
import pandas as pd


from tqdm import tqdm
from descort.preprocess.preprocess import find_files, setup_logging


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute intersection and union of feature columns "
                    "across parquet files, optionally filtered by a vocabulary."
    )

    # ---- Source data ----
    parser.add_argument(
        "--source_data_paths", nargs="+", required=True,
        help="One or more directories containing parquet files."
    )

    # ---- Vocabulary ----
    parser.add_argument(
        "--vocab_file", default=None,
        help="Optional pickle file containing a feature2index dict. "
             "If provided, only features present in this vocab are retained."
    )

    # ---- Output ----
    parser.add_argument(
        "--output_dir", required=True,
        help="Directory to save output pickle files."
    )
    parser.add_argument(
        "--output_prefix", default="features",
        help="Prefix for output filenames. "
             "Output: {prefix}_union_feature_{N}.pkl and "
             "{prefix}_intersect_feature_{N}.pkl. (default: %(default)s)"
    )

    # ---- Logging ----
    parser.add_argument(
        "--log_file", default=None,
        help="Path to log file. If not provided, logs go to stdout only."
    )
    parser.add_argument(
        "--log_level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level. (default: %(default)s)"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # ---- Setup logging ----
    logger = setup_logging(
        log_file=args.log_file,
        level=getattr(logging, args.log_level),
        logger_name="select_features",
    )

    logger.info("=" * 60)
    logger.info("Select Intersection / Union Features")
    logger.info("=" * 60)

    # ---- Load vocabulary ----
    vocab_set = None
    if args.vocab_file:
        logger.info("Loading vocabulary from: %s", args.vocab_file)
        with open(args.vocab_file, "rb") as f:
            vocab = pickle.load(f)
        vocab_set = set(vocab.keys())
        logger.info("Vocabulary size: %d", len(vocab_set))

    # ---- Collect all parquet files ----
    all_files = []
    for parquet_dir in args.source_data_paths:
        if not os.path.exists(parquet_dir):
            logger.error("Directory does not exist: %s", parquet_dir)
            continue
        logger.info("Searching directory: %s", parquet_dir)
        all_files.extend(find_files(parquet_dir))

    logger.info("Found total %d parquet files", len(all_files))
    if not all_files:
        logger.error("No parquet files found!")
        return

    # ---- Collect column sets ----
    all_columns_sets = []
    for parquet_file in tqdm(all_files, desc="Reading parquet files"):
        try:
            df = pd.read_parquet(parquet_file)
            all_columns_sets.append(set(df.columns))
        except Exception as e:
            logger.error("Failed to read %s: %s", parquet_file.name, e)
            continue

    if not all_columns_sets:
        logger.error("No valid data found!")
        return

    # ---- Compute union and intersection ----
    logger.info("Computing union of all columns...")
    union_columns = set.union(*all_columns_sets)
    logger.info("Union size: %d", len(union_columns))

    logger.info("Computing intersection of all columns...")
    intersect_columns = set.intersection(*all_columns_sets)
    logger.info("Intersection size: %d", len(intersect_columns))

    # ---- Identify metadata columns ----
    meta_columns = [col for col in union_columns if "Meta" in col]
    logger.info("Metadata columns in union: %d", len(meta_columns))

    # ---- Filter by vocabulary ----
    if vocab_set is not None:
        intersect_with_vocab = intersect_columns.intersection(vocab_set)
        intersect_with_vocab.update(meta_columns)
        union_with_vocab = union_columns.intersection(vocab_set)
        union_with_vocab.update(meta_columns)
    else:
        intersect_with_vocab = intersect_columns
        union_with_vocab = union_columns

    logger.info("Intersection (with vocab): %d features", len(intersect_with_vocab))
    logger.info("Union (with vocab):        %d features", len(union_with_vocab))

    # ---- Save ----
    os.makedirs(args.output_dir, exist_ok=True)

    union_path = os.path.join(
        args.output_dir,
        f"{args.output_prefix}_union_feature_{len(union_with_vocab)}.pkl",
    )
    with open(union_path, "wb") as f:
        pickle.dump(union_with_vocab, f)
    logger.info("Saved: %s", union_path)

    intersect_path = os.path.join(
        args.output_dir,
        f"{args.output_prefix}_intersect_feature_{len(intersect_with_vocab)}.pkl",
    )
    with open(intersect_path, "wb") as f:
        pickle.dump(intersect_with_vocab, f)
    logger.info("Saved: %s", intersect_path)

    logger.info("Done.")


if __name__ == "__main__":
    main()