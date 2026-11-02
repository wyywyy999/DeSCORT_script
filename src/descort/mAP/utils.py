import os
import sys
import pickle
import logging
import numpy as np
import scanpy as sc
import pandas as pd
import anndata as ad
from tqdm import tqdm
# from scib import metrics
from sklearn.metrics.pairwise import cosine_similarity

from sklearn.metrics import f1_score
from copairs import map
from copairs.matching import assign_reference_index
from datetime import datetime
from joblib import Parallel, delayed
from typing import Union, List
from pathlib import Path
from typing import Union, List, Optional, Dict



# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(
    log_file: Optional[str] = None,
    level: int = logging.INFO,
    logger_name: Optional[str] = None,
    log_to_console: bool = True,
) -> logging.Logger:
    """
    Configure and return a logger with console and optional file output.

    Parameters
    ----------
    log_file : str, optional
        Path to the log file. If None, only console logging is used.
    level : int
        Logging level (default: logging.INFO).
    logger_name : str, optional
        Logger name. If None, uses the calling module's __name__.
    log_to_console : bool
        Whether to add a console handler (default: True).

    Returns
    -------
    logging.Logger
    """
    logger = logging.getLogger(logger_name or __name__)
    logger.setLevel(level)

    # Clear existing handlers to avoid duplicates
    logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    if log_file:
        os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

def find_files(
    directory: Union[Path, str],
    pattern: str = "*.parquet",
) -> List[Path]:
    """
    Recursively find all files matching a glob pattern in a directory.

    Parameters
    ----------
    directory : Path or str
        Root directory to search.
    pattern : str
        Glob pattern to match (default: "*.parquet").

    Returns
    -------
    List[Path]
        Sorted list of matching file paths.
    """
    if isinstance(directory, str):
        directory = Path(directory).absolute()
    elif not isinstance(directory, Path):
        directory = Path(str(directory)).absolute()

    files = sorted(directory.rglob(pattern))
    logging.info(
        "Found %d files matching '%s' in %s", len(files), pattern, directory
    )
    return files



def calculate_repilcates_activity_mAP_parallel(gene, 
                                        all_data,
                                        Perturbation_Gene_key,
                                        sgRNA_barcode_key,
                                        negative_control_list,
                                        sample_method = 'random',
                                        n_number= 2,
                                        random_seed = 42):
    np.random.seed(random_seed)
    query_samples_df = all_data[all_data[Perturbation_Gene_key] == gene].reset_index(drop=True)
    control_samples_df = all_data[all_data[Perturbation_Gene_key].isin(negative_control_list)].reset_index(drop=True)
    barcode_list = list(query_samples_df[sgRNA_barcode_key].unique())
    barcode_result_list = []
    for barcode in barcode_list:

        barcode_sample_df = query_samples_df[query_samples_df[sgRNA_barcode_key] == barcode]
        if barcode_sample_df.shape[0] > 1:
        # 随机采样control
            if sample_method == 'random':
                n_select = min(barcode_sample_df.shape[0]*n_number, control_samples_df.shape[0])
                # n_select =  len(control_samples)
                
                control_indices = np.random.choice(range(len(control_samples_df)), size=n_select, replace=False)
                sample_control_df = control_samples_df.loc[control_indices]
            else:
                sample_control_df = control_samples_df
            
            new_all_data_df = pd.concat([barcode_sample_df, sample_control_df]).reset_index(drop=True)

            similarities = cosine_similarity(barcode_sample_df.filter(regex='^(?!Meta)'), new_all_data_df.filter(regex='^(?!Meta)'))

        
            ap_scores = pd.DataFrame()
            for i, query_similarity in enumerate(similarities):
                # 1. 排除自身，获取排序索引
                sorted_indices = np.argsort(query_similarity)[::-1]
                sorted_indices = sorted_indices[sorted_indices != i]  # 关键：去掉自己
                
                # 2. 获取排序后的 compound 标签
                sorted_compounds = new_all_data_df[Perturbation_Gene_key].iloc[sorted_indices]
                
                # 3. 找到所有 relevant 样本的位置（0-indexed），转 1-indexed rank
                relevant_positions = np.where(sorted_compounds == gene)[0]
                relevant_ranks = relevant_positions + 1  # 转换为 1-indexed ranks
                
                # print(f"Query {i+1}: Found {len(relevant_ranks)} relevant samples")
                if len(relevant_ranks) == 0:
                    print(f"Warning: No relevant samples found for query {i+1} of Gene {gene}")
                    continue
                precisions = np.arange(1, len(relevant_ranks) + 1) / relevant_ranks
                ap = np.mean(precisions)
                ap_scores.loc[i, 'gene'] = gene
                ap_scores.loc[i, 'barcode'] = barcode
                ap_scores.loc[i, 'n_pos_pairs'] = barcode_sample_df.shape[0]
                ap_scores.loc[i, 'n_total_pairs'] = new_all_data_df.shape[0]
                ap_scores.loc[i, 'average_precision'] = ap
                ap_scores['n_pos_pairs'] = ap_scores['n_pos_pairs'].astype(int)
                ap_scores['n_total_pairs'] = ap_scores['n_total_pairs'].astype(int)

            if ap_scores.empty:
                continue
            pos_sameby = ['gene','barcode']
            activity_ap = map.mean_average_precision(
            ap_scores, pos_sameby, null_size=10000, threshold=0.05, seed= 0
            )
            barcode_result_list.append(activity_ap)
        else:
            continue
    if len(barcode_result_list) == 0:
        result_df = None
    else:
        result_df = pd.concat(barcode_result_list)
    
    return result_df


def calculate_sister_activity_mAP_parallel(gene, 
                                        all_data,
                                        Perturbation_Gene_key,
                                        negative_control_list,
                                        sample_method = 'random',
                                        n_number=2,
                                        random_seed = 42):
    np.random.seed(random_seed)
    query_samples_df = all_data[all_data[Perturbation_Gene_key] == gene].reset_index(drop = True)
    control_samples_df = all_data[all_data[Perturbation_Gene_key].isin(negative_control_list)].reset_index(drop=True)
    if query_samples_df.shape[0] <= 1:
        return None
    
    if query_samples_df.shape[0] > 1:
    # 随机采样control
        if sample_method == 'random':
            n_select = min(query_samples_df.shape[0]*n_number, control_samples_df.shape[0])
            # n_select =  len(control_samples)
            control_indices = np.random.choice(range(len(control_samples_df)), size=n_select, replace=False)
            sample_control_df = control_samples_df.loc[control_indices]
        else:
            sample_control_df = control_samples_df
        
        new_all_data_df = pd.concat([query_samples_df, sample_control_df]).reset_index(drop=True)
        similarities = cosine_similarity(query_samples_df.filter(regex='^(?!Meta)'), new_all_data_df.filter(regex='^(?!Meta)'))

    
        ap_scores = pd.DataFrame()
        for i, query_similarity in enumerate(similarities):
            # 1. 排除自身，获取排序索引
            sorted_indices = np.argsort(query_similarity)[::-1]
            sorted_indices = sorted_indices[sorted_indices != i]  # 关键：去掉自己
            
            # 2. 获取排序后的 compound 标签
            sorted_compounds = new_all_data_df[Perturbation_Gene_key].iloc[sorted_indices]
            
            # 3. 找到所有 relevant 样本的位置（0-indexed），转 1-indexed rank
            relevant_positions = np.where(sorted_compounds == gene)[0]
            relevant_ranks = relevant_positions + 1  # 转换为 1-indexed ranks
            
            # print(f"Query {i+1}: Found {len(relevant_ranks)} relevant samples")
            if len(relevant_ranks) == 0:
                print(f"Warning: No relevant samples found for query {i+1} of Gene {gene}")
                continue
            precisions = np.arange(1, len(relevant_ranks) + 1) / relevant_ranks
            ap = np.mean(precisions)
            ap_scores.loc[i, 'gene'] = gene
            # ap_scores.loc[i, 'barcode'] = barcode
            ap_scores.loc[i, 'n_pos_pairs'] = query_samples_df.shape[0]
            ap_scores.loc[i, 'n_total_pairs'] = new_all_data_df.shape[0]
            ap_scores.loc[i, 'average_precision'] = ap
            ap_scores['n_pos_pairs'] = ap_scores['n_pos_pairs'].astype(int)
            ap_scores['n_total_pairs'] = ap_scores['n_total_pairs'].astype(int)

        if ap_scores.empty:
            return None
        pos_sameby = ['gene']
        activity_ap = map.mean_average_precision(
        ap_scores, pos_sameby, null_size=10000, threshold=0.05, seed= 0
        )
        return activity_ap


def calculate_sister_consistency_mAP_parallel(gene, 
                                        all_data,
                                        Perturbation_Gene_key,
                                        sample_method = 'random',
                                        n_number = 2,
                                        random_seed = 42):
    np.random.seed(random_seed)
    query_samples_df = all_data[all_data[Perturbation_Gene_key] == gene].reset_index(drop=True)
    if query_samples_df.shape[0] <= 1:
            print(f"跳过 {gene}: 无匹配样本")
            return None
    other_samples_df = all_data[all_data[Perturbation_Gene_key] != gene].reset_index(drop=True)
        
        # 随机采样control
    if sample_method == 'random':
        n_select = min(query_samples_df.shape[0]*n_number, other_samples_df.shape[0])
        # n_select =  len(control_samples)
        other_indices = np.random.choice(range(len(other_samples_df)), size=n_select, replace=False)
        sample_control_df = other_samples_df.loc[other_indices]
    elif sample_method == 'all' :
        sample_control_df = other_samples_df
    
    new_all_data_df = pd.concat([query_samples_df, sample_control_df]).reset_index(drop=True)
    similarities = cosine_similarity(query_samples_df.filter(regex='^(?!Meta)'), new_all_data_df.filter(regex='^(?!Meta)'))

    
    ap_scores = pd.DataFrame()
    for i, query_similarity in enumerate(similarities):
        # 1. 排除自身，获取排序索引
        sorted_indices = np.argsort(query_similarity)[::-1]
        sorted_indices = sorted_indices[sorted_indices != i]  # 关键：去掉自己
        
        # 2. 获取排序后的 compound 标签
        sorted_compounds = new_all_data_df[Perturbation_Gene_key].iloc[sorted_indices]
        
        # 3. 找到所有 relevant 样本的位置（0-indexed），转 1-indexed rank
        relevant_positions = np.where(sorted_compounds == gene)[0]
        relevant_ranks = relevant_positions + 1  # 转换为 1-indexed ranks
        
        # print(f"Query {i+1}: Found {len(relevant_ranks)} relevant samples")
        if len(relevant_ranks) == 0:
            print(f"Warning: No relevant samples found for query {i+1} of Gene {gene}")
            continue
        precisions = np.arange(1, len(relevant_ranks) + 1) / relevant_ranks
        ap = np.mean(precisions)
        ap_scores.loc[i, 'gene'] = gene
        # ap_scores.loc[i, 'barcode'] = barcode
        ap_scores.loc[i, 'n_pos_pairs'] = query_samples_df.shape[0]
        ap_scores.loc[i, 'n_total_pairs'] = new_all_data_df.shape[0]
        ap_scores.loc[i, 'average_precision'] = ap
        ap_scores['n_pos_pairs'] = ap_scores['n_pos_pairs'].astype(int)
        ap_scores['n_total_pairs'] = ap_scores['n_total_pairs'].astype(int)

    if ap_scores.empty:
        return None
    pos_sameby = ['gene']
    activity_ap = map.mean_average_precision(
    ap_scores, pos_sameby, null_size=10000, threshold=0.05, seed= 0
    )

    return activity_ap


