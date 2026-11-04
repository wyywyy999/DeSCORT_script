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
from pathlib import Path


from tqdm import tqdm
from descort.preprocess.preprocess import find_files, process_csv_to_parquet, setup_logging


def _absolute_path(value):
    """Expand user paths and resolve CLI paths to absolute paths."""
    return str(_Path(value).expanduser().resolve())


def parse_args():
    parser = argparse.ArgumentParser(description="Batch-convert CSV files to Parquet format.")
    parser.add_argument("--csv_dir", type=_absolute_path, required=True, help="Input directory containing CSV files.")
    parser.add_argument("--out_dir", type=_absolute_path, required=True, help="Output directory for the converted Parquet files.")
    parser.add_argument("--pattern", default="*.csv", help="File matching pattern (default: %(default)s).")
    parser.add_argument("--log_file", type=_absolute_path, default=None, help="Log file path. If omitted, log to the console only.")

    return parser.parse_args()

def main():
    args = parse_args()

    logger = setup_logging(
        log_file=args.log_file,
        level=logging.INFO,
        logger_name="CSV_to_Parquet",
        log_to_console=True,
    )

    logger.info("=" * 60)
    logger.info("Starting CSV to Parquet conversion")
    logger.info("=" * 60)

    csv_dir_path = Path(args.csv_dir)
    out_dir_path = Path(args.out_dir)

    if not csv_dir_path.exists():
        logger.error(f"Input directory does not exist: {csv_dir_path}")
        return

    out_dir_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {out_dir_path}")

    all_csv_files = find_files(csv_dir_path, args.pattern)
    if not all_csv_files:
        logger.warning(f"No files matching '{args.pattern}' found!")
        return

    logger.info(f"Found {len(all_csv_files)} files to process")

    success_count = 0
    failed_count = 0

    for csv_file in tqdm(all_csv_files, desc="Processing files"):
        if process_csv_to_parquet(csv_file, out_dir_path):
            success_count += 1
        else:
            failed_count += 1

    logger.info("=" * 60)
    logger.info("Processing completed!")
    logger.info(f"  Successfully processed: {success_count} files")
    logger.info(f"  Failed: {failed_count} files")
    logger.info(f"  Total: {len(all_csv_files)} files")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
