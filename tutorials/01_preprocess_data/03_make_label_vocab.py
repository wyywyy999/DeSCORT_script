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
        description="Build label vocabulary (feature2index / index2feature) "
                    "from the union of unique values in a label column across "
                    "all parquet files."
    )

    # ---- Source data ----
    parser.add_argument(
        "--source_data_paths", nargs="+", required=True,
        help="One or more directories containing parquet files."
    )

    # ---- Label column ----
    parser.add_argument(
        "--label_column", default="Metadata_Broad_Sample",
        help="Column name used as the label key. (default: %(default)s)"
    )

    # ---- Output ----
    parser.add_argument(
        "--output_dir", required=True,
        help="Directory to save the vocabulary pickle files."
    )
    parser.add_argument(
        "--output_prefix", default="vocab",
        help="Prefix for output file names. "
             "Output: {prefix}_feature2index_{N}.pkl and "
             "{prefix}_index2feature_{N}.pkl. (default: %(default)s)"
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
        logger_name="make_label_vocab",
    )

    logger.info("=" * 60)
    logger.info("Starting label vocabulary building pipeline")
    logger.info("=" * 60)

    # ---- Collect data files ----
    all_parquet_files = []
    for parquet_dir in args.source_data_paths:
        if not os.path.exists(parquet_dir):
            logger.error("Directory does not exist: %s", parquet_dir)
            continue
        logger.info("Searching directory: %s", parquet_dir)
        files_in_dir = find_files(parquet_dir)
        all_parquet_files.extend(files_in_dir)

    logger.info("Found total %d parquet files", len(all_parquet_files))
    if len(all_parquet_files) == 0:
        logger.error("No parquet files found, exiting")
        return

    # ---- Collect unique labels from each file ----
    logger.info("Collecting unique labels from column: %s", args.label_column)
    label_sets = []
    for parquet_file in tqdm(all_parquet_files, desc="Processing files"):
        sample_name = os.path.basename(parquet_file).split(".")[0]
        logger.debug("Processing: %s", sample_name)

        try:
            df = pd.read_parquet(parquet_file, columns=[args.label_column])
        except Exception as e:
            logger.error("Failed to read %s: %s", parquet_file, e)
            continue

        if args.label_column not in df.columns:
            logger.warning(
                "Column '%s' not found in %s, skipping", args.label_column, sample_name
            )
            continue

        label_sets.append(set(df[args.label_column].dropna().unique()))

    if not label_sets:
        logger.error("No valid labels collected, exiting")
        return

    # ---- Build union vocabulary ----
    label_union = list(set.union(*label_sets))
    logger.info("Total unique labels: %d", len(label_union))

    feature2index = {label: idx for idx, label in enumerate(label_union)}
    index2feature = {idx: label for idx, label in enumerate(label_union)}

    # ---- Save ----
    os.makedirs(args.output_dir, exist_ok=True)

    feat2idx_path = os.path.join(
        args.output_dir,
        f"{args.output_prefix}_label2index_{len(feature2index)}.pkl",
    )
    with open(feat2idx_path, "wb") as f:
        pickle.dump(feature2index, f)
    logger.info("Saved label2index: %s", feat2idx_path)

    idx2feat_path = os.path.join(
        args.output_dir,
        f"{args.output_prefix}_index2label_{len(index2feature)}.pkl",
    )
    with open(idx2feat_path, "wb") as f:
        pickle.dump(index2feature, f)
    logger.info("Saved index2label: %s", idx2feat_path)

    logger.info("Vocabulary building completed!")


if __name__ == "__main__":
    main()