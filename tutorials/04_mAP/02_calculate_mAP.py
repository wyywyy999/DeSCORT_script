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
import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed


from descort.preprocess.preprocess import setup_logging, find_files
from descort.mAP.utils import (
    calculate_repilcates_activity_mAP_parallel,
    calculate_sister_activity_mAP_parallel,
    calculate_sister_consistency_mAP_parallel,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Calculate mAP metrics for CellPainting embeddings."
    )

    # ---- mAP type ----
    parser.add_argument(
        "--mAP_type", required=True,
        choices=["replicates_activity", "sister_activity",
                 "sister_consistency"],
        help="Type of mAP calculation to run."
    )

    # ---- Data source paths ----
    parser.add_argument(
        "--source_data_paths", nargs="+", required=True,
        help="One or more source parquet/embedding directories."
    )

    # ---- Pickle / vocabulary files ----
    parser.add_argument(
        "--need_feature_list_pkl", required=True,
        help="Path to the need_feature_list pickle file (e.g. union feature list)."
    )
    parser.add_argument(
        "--feature2index_pkl", required=True,
        help="Path to feature2index vocabulary pickle file."
    )

    # ---- Output ----
    parser.add_argument(
        "--save_path", required=True,
        help="Directory to save the mAP results."
    )

    # ---- Column / gene settings ----
    parser.add_argument(
        "--perturbation_gene_key", default="Metadata_Foci_Barcode_MatchedTo_GeneCode",
        help="Column name used as the perturbation gene key. (default: %(default)s)"
    )
    parser.add_argument(
        "--sgRNA_barcode_key", default="Metadata_Foci_Barcode_MatchedTo_Barcode",
        help="Column name used as the sgRNA barcode key (only for replicates_activity). "
             "(default: %(default)s)"
    )
    parser.add_argument(
        "--negative_control_list", nargs="+", default=["nontargeting"],
        help="List of negative control values. (default: %(default)s)"
    )

    # ---- Aggregation / processing ----
    parser.add_argument(
        "--choose_type", default="well-level",
        help="Aggregation level. (default: %(default)s)"
    )
    parser.add_argument(
        "--do_intersect_features", action="store_true", default=False,
        help="Whether to intersect features across plates (only keeping common columns). "
             "(default: %(default)s)"
    )
    parser.add_argument(
        "--do_groupby_mean", action="store_true", default=False,
        help="Whether to group by need_columns and take mean before merging. "
             "(default: %(default)s)"
    )
    parser.add_argument(
        "--need_columns", nargs="+", required=True,
        help="Columns to use for metadata extraction."
    )
    parser.add_argument(
        "--group_by_columns", nargs="+", required=True,
        help="Columns to use for groupby mean."
    )

    # ---- Data filtering ----
    parser.add_argument(
        "--need_plate_list", nargs="+", default=None,
        help="List of plate names to filter (e.g. CP186A CP186B). If not provided, all plates are used."
    )

    # ---- mAP calculation settings ----
    parser.add_argument(
        "--random_seeds", nargs="+", type=int, default=[42, 2026, 2025, 2024, 2023],
        help="List of random seeds for reproducibility. (default: %(default)s)"
    )
    parser.add_argument(
        "--n_jobs", type=int, default=20,
        help="Number of parallel jobs. (default: %(default)d)"
    )
    parser.add_argument(
        "--sample_method", default="random",
        choices=["random", "all"],
        help="Control sampling method. (default: %(default)s)"
    )
    parser.add_argument(
        "--n_number", type=int, default=2,
        help="Multiplier for number of control samples per query sample. (default: %(default)d)"
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
        logger_name="calculate_mAP",
    )

    # ---- Load dictionaries ----
    logger.info("Loading feature list and dictionaries...")
    with open(args.need_feature_list_pkl, "rb") as f:
        need_feature_list = pickle.load(f)
    logger.info(f"Loaded feature list with {len(need_feature_list)} features")

    with open(args.feature2index_pkl, "rb") as f:
        FEATURE2INDEX_DICT = pickle.load(f)
    logger.info(f"Loaded feature dictionary with {len(FEATURE2INDEX_DICT)} features")

    Meta_feature = [feature for feature in need_feature_list if "Meta" in feature]
    intersected_feature_list = [feat for feat in need_feature_list if feat in FEATURE2INDEX_DICT]
    intersected_feature_list.extend(Meta_feature)
    logger.info(f"Intersected feature list size: {len(intersected_feature_list)}")

    # ---- Use columns from args ----
    need_columns = args.need_columns
    group_by_columns = args.group_by_columns
    logger.info(f"Using need_columns: {need_columns}")
    logger.info(f"Using group_by_columns: {group_by_columns}")

    # ---- Select mAP function ----
    mAP_type = args.mAP_type
    if mAP_type == "replicates_activity":
        mAP_func = calculate_repilcates_activity_mAP_parallel
        use_barcode = True
        use_control = True
    elif mAP_type == "sister_activity":
        mAP_func = calculate_sister_activity_mAP_parallel
        use_barcode = False
        use_control = True
    elif mAP_type == "sister_consistency":
        mAP_func = calculate_sister_consistency_mAP_parallel
        use_barcode = False
        use_control = False
    else:
        raise ValueError(f"Unknown mAP_type: {mAP_type}")

    logger.info(f"mAP type: {mAP_type} | use_barcode: {use_barcode} | use_control: {use_control}")

    need_plate_list = args.need_plate_list

    # ---- Collect data files ----
    data_list = []
    for parquet_dir in args.source_data_paths:
        if not os.path.exists(parquet_dir):
            logger.error(f"Directory does not exist: {parquet_dir}")
            continue
        logger.info(f"Searching directory: {parquet_dir}")
        files_in_dir = find_files(parquet_dir)
        data_list.extend(files_in_dir)

    logger.info(f"Found {len(data_list)} files total")

    # ---- Filter by plate list ----
    if need_plate_list is not None:
        data_list = list({
            file for file in data_list
            for need_plate in need_plate_list
            if need_plate in str(file)
        })
    logger.info(f"After plate filtering: {len(data_list)} files")

    # ---- Load and aggregate data ----
    data_mean_list = []
    for plate_path in tqdm(data_list, desc="Loading plates"):
        single_plate_data = pd.read_parquet(plate_path)

        # # Handle Broad_Sample replacement for compound datasets
        # if "Metadata_Broad_Sample" in single_plate_data.columns:
        #     single_plate_data["Metadata_Broad_Sample"] = (
        #         single_plate_data["Metadata_Broad_Sample"]
        #         .replace(["None", "none", ""], "nontargeting")
        #         .fillna("nontargeting")
        #     )

        meta_data = single_plate_data[need_columns]
        feature_data = single_plate_data.filter(regex="^(?!Meta)")
        feature_data = feature_data.fillna(0)
        new_feature_data = pd.concat([meta_data, feature_data], axis=1)

        if args.do_groupby_mean:
            new_feature_data = new_feature_data.groupby(group_by_columns, as_index=False).mean()

        data_mean_list.append(new_feature_data)

    # ---- Merge data ----
    if args.do_intersect_features:
        data_common_list = [
            df[list(set(df.columns) & set(intersected_feature_list))]
            for df in data_mean_list
        ]
        common_columns = set.intersection(*[set(df.columns) for df in data_common_list])
        data_common_list = [df[list(common_columns)] for df in data_common_list]
        all_data = pd.concat(data_common_list, ignore_index=True)
    else:
        all_data = pd.concat(data_mean_list, ignore_index=True)

    logger.info(f"Merged data shape: {all_data.shape}")

    # ---- Get perturbation gene list ----
    Perturbation_Gene = [
        gene for gene in all_data[args.perturbation_gene_key].unique()
        if gene not in args.negative_control_list
    ]
    logger.info(f"Number of perturbation genes: {len(Perturbation_Gene)}")

    # ---- Filter out negative controls if needed ----
    if use_control:
        all_samples = all_data
    else:
        all_samples = all_data[
            ~all_data[args.perturbation_gene_key].isin(args.negative_control_list)
        ]

    # ---- Process each random seed ----
    for random_seed in args.random_seeds:
        seed_save_path = os.path.join(args.save_path, str(random_seed))
        os.makedirs(seed_save_path, exist_ok=True)
        logger.info(f"{'=' * 60}")
        logger.info(f"Processing random_seed={random_seed}")
        logger.info(f"{'=' * 60}")

        # ---- Run mAP calculation in parallel ----
        logger.info(f"Starting mAP calculation with {args.n_jobs} jobs...")
        func_kwargs = dict(
            all_data=all_samples,
            Perturbation_Gene_key=args.perturbation_gene_key,
            sample_method=args.sample_method,
            n_number=args.n_number,
            random_seed=random_seed,
        )
        if use_barcode:
            func_kwargs["sgRNA_barcode_key"] = args.sgRNA_barcode_key
        if use_control:
            func_kwargs["negative_control_list"] = args.negative_control_list

        results = Parallel(n_jobs=args.n_jobs, verbose=10, timeout=99999, batch_size="auto")(
            delayed(mAP_func)(gene=gene, **func_kwargs)
            for gene in tqdm(Perturbation_Gene, desc="Processing genes")
        )

        # ---- Filter None results and save ----
        results = [r for r in tqdm(results, desc="Filtering None results") if r is not None]
        logger.info(f"Valid results: {len(results)}")

        if len(results) > 0:
            results_df = pd.concat(results, ignore_index=True)
            output_file = os.path.join(seed_save_path, "ap_scores_dict.pickle")
            with open(output_file, "wb") as f:
                pickle.dump(results_df, f)
            logger.info(f"Results saved to: {output_file}")
        else:
            logger.warning("No valid results to save!")

    logger.info("All seeds processed!")


if __name__ == "__main__":
    main()