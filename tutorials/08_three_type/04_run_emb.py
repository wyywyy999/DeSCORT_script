"""Extract plate embeddings with a pretrained DeSCORT model."""
# Support both an installed package and direct execution from this checkout.
from pathlib import Path as _Path
import sys as _sys
_SOURCE = _Path("/fse/home/wangyongyan/Project/CellPainting/DeSCORT/other_model/DeSCORT/src")
if str(_SOURCE) not in _sys.path:
    _sys.path.insert(0, str(_SOURCE))

import os
import sys
import torch
import argparse
import pickle
import logging
import numpy as np
import pandas as pd


from tqdm import tqdm
from datasets import Dataset


from descort.preprocess.preprocess import setup_logging, find_files
# ---- Import custom modules ----
from descort.data.dataset import DeSCORTForContrastiveLMDataProcessor
from descort.data.collator import (
    DeSCORTForContrastiveLMDataCollator,
)
from descort.model.modeling_descort import DeSCORTForContrastiveLM


def parse_args():
    parser = argparse.ArgumentParser(
        description="Calculate all embeddings for CellPainting plates (no split) using a pretrained DeSCORT model."
    )

    # ---- Data source paths ----
    parser.add_argument(
        "--source_data_paths", nargs="+", required=True,
        help="One or more source parquet directories."
    )

    # ---- Pickle / vocabulary files ----
    parser.add_argument(
        "--gene2index_pkl", required=True,
        help="Path to gene2index pickle file."
    )
    parser.add_argument(
        "--feature2index_pkl", required=True,
        help="Path to feature2index vocabulary pickle file."
    )
    parser.add_argument(
        "--need_feature_list_pkl", required=True,
        help="Path to the need_feature_list pickle file (e.g. union feature list)."
    )

    # ---- Model settings ----
    parser.add_argument(
        "--pretrained_model_path", required=True,
        help="Path to the pretrained model directory."
    )
    parser.add_argument(
        "--save_path", required=True,
        help="Directory to save the output embeddings."
    )

    # ---- Data filtering ----
    parser.add_argument(
        "--key_gene_column", default="Metadata_Foci_Barcode_MatchedTo_GeneCode",
        help="Column name used as the gene key. (default: %(default)s)"
    )
    parser.add_argument(
        "--need_plate_list", nargs="*", default=None,
        help="List of plate names to filter (e.g. CP186A CP186B). If not provided, all plates are used."
    )
    parser.add_argument(
        "--metadata_parts", nargs="+", type=int, default=[-2, -5],
        help="Path part indices for extracting metadata (Batch, Type). (default: %(default)s)"
    )

    # ---- Model / device settings ----
    parser.add_argument(
        "--device", default="cuda:0",
        help="Device to run the model on. (default: %(default)s)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=64,
        help="Batch size for the dataloader. (default: %(default)d)"
    )
    parser.add_argument(
        "--max_length", type=int, default=1024,
        help="Maximum sequence length. (default: %(default)d)"
    )

    # ---- Collator / processing settings ----
    parser.add_argument(
        "--do_mlm", action="store_true", default=False,
        help="Whether to do masked language modeling. (default: %(default)s)"
    )
    parser.add_argument(
        "--mlm_probability", type=float, default=0.0,
        help="MLM probability. (default: %(default).1f)"
    )
    parser.add_argument(
        "--nproc", type=int, default=1,
        help="Number of processes for the data processor. (default: %(default)d)"
    )
    parser.add_argument(
        "--verbosity", type=int, default=1,
        help="Verbosity level for the data processor. (default: %(default)d)"
    )
    parser.add_argument(
        "--sampling", action="store_true", default=True,
        help="Enable random feature sampling when a profile exceeds max_length. (default: %(default)s)"
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
        logger_name="calculate_emb",
    )

    # ---- Load dictionaries ----
    logger.info("Loading gene and feature dictionaries...")
    with open(args.gene2index_pkl, "rb") as f:
        GENE2INDEX_DICT = pickle.load(f)
    logger.info(f"Successfully loaded gene dictionary with {len(GENE2INDEX_DICT)} genes")

    with open(args.feature2index_pkl, "rb") as f:
        FEATURE2INDEX_DICT = pickle.load(f)
    logger.info(f"Successfully loaded feature dictionary with {len(FEATURE2INDEX_DICT)} features")

    with open(args.need_feature_list_pkl, "rb") as f:
        need_feature_list = pickle.load(f)
    logger.info(f"Successfully loaded feature list with {len(need_feature_list)} features")

    key_gene_column = args.key_gene_column
    need_genes_list = list(GENE2INDEX_DICT.keys())

    # ---- Collect data files ----
    data_list = []
    for parquet_dir in args.source_data_paths:
        if not os.path.exists(parquet_dir):
            logger.error(f"Directory does not exist: {parquet_dir}")
            continue
        logger.info(f"Searching directory: {parquet_dir}")
        files_in_dir = find_files(parquet_dir)
        data_list.extend(files_in_dir)

    # ---- Filter data by test plate list ----
    if args.need_plate_list:
        data_list = [
            plate for plate in data_list
            if any(test in str(plate) for test in args.need_plate_list)
        ]

    logger.info(f"Filtered data list: {data_list}, count: {len(data_list)}")

    # ---- Initialize data processor ----
    logger.info("Initializing DeSCORTForContrastiveLMDataProcessor...")
    Processor = DeSCORTForContrastiveLMDataProcessor(
        nproc=args.nproc, verbosity=args.verbosity
    )
    logger.info("Data processor initialized successfully")

    # ---- Initialize data collator ----
    logger.info("Initializing DeSCORTForContrastiveLMDataCollator...")
    DeSCORT_Collator = DeSCORTForContrastiveLMDataCollator(
        do_mlm=args.do_mlm,
        mlm_probability=args.mlm_probability,
        mask_value=-100,
        pad_value=100,
        max_length=args.max_length,
        sampling=args.sampling,
    )
    logger.info("Data collator initialized successfully")

    # ---- Load pretrained model ----
    logger.info(f"Loading pretrained model: {args.pretrained_model_path}")
    model = DeSCORTForContrastiveLM.from_pretrained(
        args.pretrained_model_path,
    )
    logger.info("Model loaded successfully")

    # ---- Set device ----
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    if torch.cuda.is_available():
        logger.info(f"GPU info: {torch.cuda.get_device_name(0)}")

    model.eval()
    model = model.to(device)
    logger.info("Model moved to target device and set to eval mode")

    # ---- Make save directory ----
    os.makedirs(args.save_path, exist_ok=True)
    logger.info(f"Save path: {args.save_path}")

    # ---- Process each plate ----
    total_plates = len(data_list)
    logger.info(f"Starting to process {total_plates} data files")

    for plate_idx, plate in enumerate(data_list, 1):
        logger.info(f"Processing file {plate_idx}/{total_plates}: {plate}")
        Metadata_Batch = plate.parts[args.metadata_parts[0]]
        Metadata_type = plate.parts[args.metadata_parts[1]]

        single_data_path = plate
        logger.info(f"Data file path: {single_data_path}")

        # ---- Load parquet ----
        try:
            logger.info(f"Loading Parquet file: {plate}")
            single_gene_data = pd.read_parquet(single_data_path)
            logger.info(
                f"Data loaded: {single_gene_data.shape[0]} rows, "
                f"{single_gene_data.shape[1]} columns"
            )
        except Exception as e:
            logger.error(f"Failed to load data file: {e}")
            continue

        df = single_gene_data.copy()

        # ---- Filter by gene list ----
        logger.info(f"Filtering genes, using gene list: {len(need_genes_list)} genes")
        df = Processor.filter_df_by_gene_list(df, need_genes_list, key_gene_column)
        logger.info(f"Data shape after gene filtering: {df.shape}")

        # ---- Extract metadata ----
        logger.info("Extracting metadata...")
        df_metadata = df.filter(regex="^Metadata").reset_index(drop=True)
        logger.info(f"Metadata shape: {df_metadata.shape}")

        assert df_metadata.shape[0] == df.shape[0], (
            f"Metadata row count ({df_metadata.shape[0]}) does not match "
            f"dataframe row count ({df.shape[0]})!"
        )

        # ---- Extract sample name ----
        sample_name = os.path.basename(single_data_path).split(".")[0]
        logger.info(f"Sample name: {sample_name}")

        # ---- Build data dict ----
        logger.info("Building data dictionary...")
        data_dict = {
            "Perturb_Gene": df[key_gene_column].values,
        }
        logger.info(f"Perturb gene count: {len(data_dict['Perturb_Gene'])}")

        # ---- Extract features ----
        logger.info("Extracting feature expression...")
        data_dict["Feature_expression"], feature_name_index = (
            Processor.extract_features(
                df, FEATURE2INDEX_DICT, need_feature_list=need_feature_list
            )
        )
        logger.info(
            f"Feature expression shape: {data_dict['Feature_expression'].shape}"
        )

        # ---- Add feature name index ----
        data_dict["Feature_name_index"] = [feature_name_index] * df.shape[0]
        logger.info(f"Feature name index: {feature_name_index}")

        logger.info("Custom info added")

        # ---- Map dict ----
        logger.info("Executing dictionary mapping...")
        data_dict = Processor.mapping_dict(
            data_dict,
            GENE2INDEX_DICT=GENE2INDEX_DICT,
        )
        logger.info("Dictionary mapping completed")

        # ---- Create dataset ----
        logger.info("Creating HuggingFace Dataset...")
        try:
            ds = Dataset.from_dict(data_dict)
            ds.set_format("torch")
            logger.info(f"Dataset created successfully: {len(ds)} samples")
        except Exception as e:
            logger.error(f"Failed to create dataset: {e}")
            continue

        # ---- Create dataloader ----
        logger.info("Creating dataloader...")
        try:
            eval_dataloader = torch.utils.data.DataLoader(
                ds,
                batch_size=args.batch_size,
                collate_fn=DeSCORT_Collator,
                shuffle=False,
            )
            logger.info(
                f"Dataloader created successfully, batch size: {args.batch_size}, "
                f"total batches: {len(eval_dataloader)}"
            )
        except Exception as e:
            logger.error(f"Failed to create dataloader: {e}")
            continue

        # ---- Clear GPU cache ----
        logger.info("Clearing GPU cache...")
        torch.cuda.empty_cache()

        # ---- Extract embeddings ----
        logger.info("Starting to extract embeddings...")
        all_embeddings = []
        total_batches = len(eval_dataloader)

        try:
            for batch_idx, batch in enumerate(
                tqdm(eval_dataloader, desc=f"Processing {sample_name}", unit="batch")
            ):
                logger.debug(f"Processing batch {batch_idx + 1}/{total_batches}")

                # Move data to GPU
                batch_gpu = {}
                for key, value in batch.items():
                    if isinstance(value, torch.Tensor):
                        batch_gpu[key] = value.to(device)
                    else:
                        batch_gpu[key] = value

                # Get latent embedding
                with torch.no_grad():
                    emb_out = model.get_latent_embedding(**batch_gpu)

                hidden_states = emb_out["logits"]
                all_embeddings.append(hidden_states.detach().cpu().numpy())

                # Log progress every 10 batches
                if (batch_idx + 1) % 100 == 0:
                    logger.info(
                        f"Progress: {batch_idx + 1}/{total_batches} batches completed"
                    )

            logger.info(
                f"Embedding extraction completed, {len(all_embeddings)} batches processed"
            )

        except Exception as e:
            logger.error(f"Error extracting embeddings: {e}")
            continue

        # ---- Stack embeddings ----
        logger.info("Merging all embeddings...")
        try:
            all_embeddings = np.vstack(all_embeddings)
            logger.info(
                f"Embeddings merged, final shape: {all_embeddings.shape}"
            )
        except Exception as e:
            logger.error(f"Failed to merge embeddings: {e}")
            continue

        # ---- Create embedding DataFrame ----
        logger.info("Creating embedding DataFrame...")
        all_embeddings_df = pd.DataFrame(
            all_embeddings,
            columns=[f"emb_{i}" for i in range(all_embeddings.shape[1])],
        )
        logger.info(f"Embedding DataFrame shape: {all_embeddings_df.shape}")

        # ---- Merge metadata and embeddings ----
        logger.info("Merging metadata and embeddings...")
        try:
            assert df_metadata.shape[0] == all_embeddings_df.shape[0], (
                f"Metadata row count ({df_metadata.shape[0]}) does not match "
                f"all_embedding row count ({all_embeddings_df.shape[0]})!"
            )
            emb_df = pd.concat([df_metadata, all_embeddings_df], axis=1)
            logger.info(f"Final merged DataFrame shape: {emb_df.shape}")
            logger.info(f"Column names: {list(emb_df.columns)[:10]}...")
        except Exception as e:
            logger.error(f"Failed to merge metadata and embeddings: {e}")
            continue

        # ---- Save ----
        try:
            plate_save_dir = os.path.join(
                args.save_path, Metadata_type, Metadata_Batch
            )
            os.makedirs(plate_save_dir, exist_ok=True)
            plate_save_path = os.path.join(
                plate_save_dir, sample_name + ".parquet"
            )
            if "Metadata_Broad_Sample" in emb_df.columns:
                emb_df["Metadata_Broad_Sample"] = (
                    emb_df["Metadata_Broad_Sample"].astype(str)
                )
            emb_df.to_parquet(plate_save_path)
            logger.info(f"Final save path: {plate_save_path}")
        except Exception as e:
            logger.error(f"Failed to save: {e}")
            continue

        logger.info(f"File {plate} processing completed\n{'=' * 50}")

    logger.info(f"All {total_plates} files processed!")


if __name__ == "__main__":
    main()