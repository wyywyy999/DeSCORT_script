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
        description="Filter parquet files to keep only features present in "
                    "a given union feature list, then save with metadata."
    )

    # ---- Source data ----
    parser.add_argument(
        "--source_data_paths", nargs="+", required=True,
        help="One or more directories containing parquet files."
    )

    # ---- Feature list ----
    parser.add_argument(
        "--feature_list_pkl", required=True,
        help="Path to a pickle file containing the union/intersection feature "
             "set (e.g. from 04_select_intersection_union_feature.py)."
    )

    # ---- Output ----
    parser.add_argument(
        "--output_path", required=True,
        help="Directory to save the filtered parquet files. "
             "Batch subdirectory structure is preserved."
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
        logger_name="get_union_feature_data",
    )

    logger.info("=" * 60)
    logger.info("Starting union feature data filtering pipeline")
    logger.info("=" * 60)

    # ---- Load feature list ----
    logger.info("Loading feature list from: %s", args.feature_list_pkl)
    with open(args.feature_list_pkl, "rb") as f:
        common_feature = pickle.load(f)
    logger.info("Feature list size: %d", len(common_feature))

    # ---- Collect data files ----
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

    # ---- Ensure output directory exists ----
    os.makedirs(args.output_path, exist_ok=True)

    # ---- Process each file ----
    total = len(all_files)
    logger.info("Starting to process %d files", total)

    for idx, parquet_file in enumerate(all_files, 1):
        sample_name = os.path.basename(parquet_file).split(".")[0]
        batch_name = parquet_file.parent.name
        logger.info(
            "Processing file %d/%d: %s (batch: %s)", idx, total, sample_name, batch_name
        )

        try:
            df = pd.read_parquet(parquet_file)
        except Exception as e:
            logger.error("Failed to read %s: %s", parquet_file, e)
            continue

        # Separate metadata and features
        meta_df = df.filter(regex="^Meta")
        feature_df = df.filter(regex="^(?!Meta)")

        # Select only features in the common feature list
        select_features = [
            feat for feat in feature_df.columns if feat in common_feature
        ]
        select_feature_df = feature_df[select_features]

        # Merge and save
        select_df = pd.concat([meta_df, select_feature_df], axis=1)

        batch_output_dir = os.path.join(args.output_path, batch_name)
        os.makedirs(batch_output_dir, exist_ok=True)
        output_file = os.path.join(batch_output_dir, sample_name + ".parquet")
        select_df.to_parquet(output_file)

        logger.info(
            "Saved: %s (features: %d/%d)",
            output_file, len(select_features), len(feature_df.columns),
        )

    logger.info("All %d files processed!", total)


if __name__ == "__main__":
    main()