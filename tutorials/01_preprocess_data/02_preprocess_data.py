# Support both an installed package and direct execution from this checkout.
from pathlib import Path as _Path
import sys as _sys
_SOURCE = _Path("/fse/home/wangyongyan/Project/CellPainting/DeSCORT/other_model/DeSCORT/src")
if str(_SOURCE) not in _sys.path:
    _sys.path.insert(0, str(_SOURCE))

import os
import sys
import argparse
import logging
import tempfile
import pandas as pd
from pathlib import Path


from tqdm import tqdm
from descort.preprocess.preprocess import (
    find_files,
    setup_logging,
    normalization_func,
    map_channel_names,
)


def _absolute_path(value):
    """Expand user paths and resolve CLI paths to absolute paths."""
    return str(_Path(value).expanduser().resolve())


def parse_args():
    parser = argparse.ArgumentParser(
        description="Preprocess CellPainting parquet files: aggregation, "
                    "normalization, clip, and feature selection."
    )

    # ---- Source data ----
    parser.add_argument(
        "--source_data_paths", type=_absolute_path, nargs="+", required=True,
        help="One or more directories containing raw parquet files."
    )
    parser.add_argument(
        "--need_plate_list", nargs="*", default=None,
        help="List of plate names to filter. If not provided, all plates are used."
    )

    # ---- Output paths ----
    parser.add_argument(
        "--median_output_path", type=_absolute_path, default=None,
        help="Directory to save median-aggregated parquet files. "
             "Required when --do_save_median is set."
    )
    parser.add_argument(
        "--normalization_output_path", type=_absolute_path, default=None,
        help="Directory to save normalization output parquet files. "
             "Required when --do_save_normalization is set."
    )
    parser.add_argument(
        "--clip_output_path", type=_absolute_path, required=True,
        help="Directory to save clipped output parquet files."
    )
    parser.add_argument(
        "--normalization_select_feature_output_path", type=_absolute_path, required=True,
        help="Directory to save feature-selected parquet files."
    )

    # ---- Aggregation / processing ----
    parser.add_argument(
        "--need_columns", nargs="+", required=True,
        help="Columns used for aggregation strata."
    )
    parser.add_argument(
        "--data_level",
        type=str,
        choices=["well", "sc"],
        default="well",
        help="Data aggregation level: 'well' for well-level, 'sc' for single-cell. (default: well)"
    )
    parser.add_argument(
        "--do_map_channel_names", action="store_true", default=False,
        help="Map old channel names (DAPI_Painting, Phalloidin, ConA, Mito) "
             "to new names (DNA, AGP, ER, Mito). (default: %(default)s)"
    )
    parser.add_argument(
        "--do_save_median", action="store_true", default=False,
        help="Save intermediate median-aggregated parquet files. (default: %(default)s)"
    )
    parser.add_argument(
        "--do_save_normalization", action="store_true", default=False,
        help="Save intermediate normalization parquet files. (default: %(default)s)"
    )

    # ---- Normalization ----
    parser.add_argument(
        "--normalize_method", default="mad_robustize",
        choices=["standardize", "robustize", "mad_robustize", "spherize"],
        help="Normalization method. (default: %(default)s)"
    )
    parser.add_argument(
        "--normalize_samples_query", default=None,
        help="Pandas query string to select control samples for normalization. "
             "Example: \"Metadata_Foci_Barcode_MatchedTo_GeneCode == 'nontargeting'\". "
             "If not provided, uses all samples."
    )

    # ---- Clip ----
    parser.add_argument(
        "--clip_lower_percentile", type=float, default=1,
        help="Lower percentile for clipping. (default: %(default)s)"
    )
    parser.add_argument(
        "--clip_upper_percentile", type=float, default=99,
        help="Upper percentile for clipping. (default: %(default)s)"
    )

    # ---- Feature selection ----
    parser.add_argument(
        "--feature_select_operations", nargs="+",
        default=["variance_threshold", "correlation_threshold",
                 "drop_na_columns", "blocklist"],
        help="Feature selection operations. (default: %(default)s)"
    )
    parser.add_argument(
        "--na_cutoff", type=float, default=0.05,
        help="NA cutoff for feature selection. (default: %(default).2f)"
    )
    parser.add_argument(
        "--corr_threshold", type=float, default=0.9,
        help="Correlation threshold for feature selection. (default: %(default).1f)"
    )

    # ---- Logging ----
    parser.add_argument(
        "--log_file", type=_absolute_path, default=None,
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
        logger_name="preprocess_data",
    )

    # ---- Collect data files ----
    data_list = []
    for parquet_dir in args.source_data_paths:
        if not os.path.exists(parquet_dir):
            logger.error("Directory does not exist: %s", parquet_dir)
            continue
        logger.info("Searching directory: %s", parquet_dir)
        files_in_dir = find_files(parquet_dir)
        data_list.extend(files_in_dir)

    logger.info("Found %d files total", len(data_list))

    # ---- Filter by plate list ----
    if args.need_plate_list:
        data_list = [
            file for file in data_list
            if any(plate in str(file) for plate in args.need_plate_list)
        ]
        logger.info("After plate filtering: %d files", len(data_list))

    if not data_list:
        logger.warning("No files to process!")
        return

    # ---- Resolve intermediate output paths ----
    # Use temp dirs for median/normalization unless explicitly requested to save.
    _temp_dirs = []
    if args.do_save_median:
        if not args.median_output_path:
            logger.error("--do_save_median requires --median_output_path")
            return
        median_output_path = Path(args.median_output_path)
    else:
        _temp_median = tempfile.mkdtemp(prefix="median_")
        _temp_dirs.append(_temp_median)
        median_output_path = Path(_temp_median)
        logger.info("Median outputs will not be saved (use --do_save_median to keep)")

    if args.do_save_normalization:
        if not args.normalization_output_path:
            logger.error("--do_save_normalization requires --normalization_output_path")
            return
        normalization_output_path = Path(args.normalization_output_path)
    else:
        _temp_norm = tempfile.mkdtemp(prefix="normalization_")
        _temp_dirs.append(_temp_norm)
        normalization_output_path = Path(_temp_norm)
        logger.info("Normalization outputs will not be saved (use --do_save_normalization to keep)")

    clip_output_path = Path(args.clip_output_path)
    normalization_select_feature_output_path = Path(args.normalization_select_feature_output_path)

    # ---- Process each plate ----
    total_plates = len(data_list)
    logger.info("Starting to process %d data files", total_plates)

    for plate_idx, plate_path in enumerate(data_list, 1):
        sample_name = os.path.basename(plate_path).split(".")[0]
        logger.info(
            "Processing file %d/%d: %s", plate_idx, total_plates, sample_name
        )

        try:
            df = pd.read_parquet(plate_path)
            logger.info("Data loaded: %d rows, %d columns", df.shape[0], df.shape[1])
        except Exception as e:
            logger.error("Failed to load %s: %s", plate_path, e)
            continue

        # Map channel names if requested
        if args.do_map_channel_names:
            logger.info("Mapping channel names...")
            df.columns = map_channel_names(df.columns)

        # Run normalization pipeline
        try:
            result = normalization_func(
                df=df,
                sample_name=sample_name,
                need_columns=args.need_columns,
                median_output_path=median_output_path,
                normalization_output_path=normalization_output_path,
                clip_output_path=clip_output_path,
                normalization_select_feature_output_path=normalization_select_feature_output_path,
                normalize_method=args.normalize_method,
                normalize_samples_query=args.normalize_samples_query,
                data_level=args.data_level,
                clip_lower_percentile=args.clip_lower_percentile,
                clip_upper_percentile=args.clip_upper_percentile,
                feature_select_operations=args.feature_select_operations,
                na_cutoff=args.na_cutoff,
                corr_threshold=args.corr_threshold,
            )
            logger.info("Output files: %s", result)
        except Exception as e:
            logger.error("Failed to process %s: %s", sample_name, e)
            continue

        logger.info("File %s processing completed%s", sample_name, "=" * 40)

    logger.info("All %d files processed!", total_plates)


if __name__ == "__main__":
    main()