import logging
import sys
from datetime import datetime
from typing import Union, List
from pathlib import Path
import anndata as ad
import os
import pickle
import scanpy as sc
from joblib import Parallel, delayed
from tqdm import tqdm
from ..mAP.utils import calculate_sister_activity_mAP_parallel, calculate_sister_consistency_mAP_parallel
import pandas as pd
from .lisi_pure_python import clisi, ilisi
import numpy as np
import pandas as pd
from scib.metrics import metrics

# ========== 配置日志 ==========
def setup_logging(output_dir, log_name=None):
    """设置日志配置，同时输出到控制台和文件"""
    # 创建日志目录
    log_dir = os.path.join(output_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)

    # 生成日志文件名（带时间戳）
    if log_name is not None:
        log_file = os.path.join(log_dir, f'processing_{log_name}.log')
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = os.path.join(log_dir, f'processing_{timestamp}.log')

    # 创建logger
    logger = logging.getLogger('CellPainting')
    logger.setLevel(logging.DEBUG)

    # 清除之前的handlers（避免重复）
    if logger.hasHandlers():
        logger.handlers.clear()

    # 文件处理器（记录所有日志）
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)

    # 控制台处理器（只显示INFO及以上）
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(message)s')
    console_handler.setFormatter(console_formatter)

    # 添加处理器
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger, log_file

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def calculate_batch_bio_metrics(adata,
                                batch_key,
                                label_key,
                                cluster_key,
                                embed ='X_embed' ):
    logger.info("Calculating batch bio metrics with scib...")
    mertric_dict = {}

    scib_results = metrics(
    adata,
    adata,
    batch_key=batch_key,
    label_key=label_key,
    cluster_key=cluster_key,
    embed=embed,
    isolated_labels_asw_=False,
    silhouette_=True,
    hvg_score_=False,
    graph_conn_=True,
    pcr_=False,
    kBET_=True,
    nmi_=True,
    ari_=True,
    ilisi_=False,
    clisi_=False,
)
    
    
    for index, row in scib_results.iterrows():
        mertric_dict[index] = row[0]
    
    clisi_score_fast = clisi(
    adata,
    label_key=label_key,
    mode='direct',
    use_rep = embed,
    subsample=None,
    verbose=True,
)
    
    ilisi_score_fast = ilisi(
    adata,
    batch_key=batch_key,
    mode='direct',
    use_rep = embed,
    subsample=None,
    verbose=True,
)
    mertric_dict['clisi_score_fast'] = clisi_score_fast
    mertric_dict['ilisi_score_fast'] = ilisi_score_fast

    logger.info("Batch bio metrics calculation completed.")
    return mertric_dict

def find_h5ad_files(filepath: Union[Path, str]) -> List[Path]:
    """
    Recursively find all h5ad files in the specified directory
    
    **Parameters:**
    filepath : Union[Path, str]
        | Directory path to search
        
    **Returns:**
    List[Path] - List of found parquet file paths
    """
    if not isinstance(filepath, Path):
        filepath = Path(filepath).absolute()

    h5ad_files = list(filepath.rglob("*.h5ad"))

    logger.info(f"Found {len(h5ad_files)} h5ad files in {filepath}")
    return h5ad_files


def h5ad_2_df(adata,
              embs ='X',):
    if embs =='X':
        X_data = adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X
        
        df = pd.DataFrame(
        X_data,
        index=adata.obs_names,
        columns=adata.var_names
    )
    else:
        X_data = adata.obsm[embs]
        df = pd.DataFrame(
        X_data,
        index=adata.obs_names,
        # columns=adata.var_names
    )

    # 创建表达矩阵的 DataFrame
    

    # 合并 obs 元数据
    for col in adata.obs.columns:
        if 'Meta' in col:
            df[col] = adata.obs[col].values
        else:
            new_col = 'Metadata_' + col
            df[new_col] = adata.obs[col].values

    return df
    
def calculate_mAP(df, 
                  Perturbation_Gene_key,
                  negative_control,
                  sample_method = 'random',
                  n_number = 2,
                  random_seed = 42,
                  n_jobs = 10
                  ):
        # 提取表达矩阵
    
    Perturbation_Gene = list(df[Perturbation_Gene_key].unique())
    Perturbation_Gene.remove(negative_control)

    logger.info(f"Calculating mAP for {len(Perturbation_Gene)} genes (negative control: {negative_control})")
    logger.info("Running with_control mAP calculation in parallel...")

    with_control_mAP_results = Parallel(n_jobs=n_jobs, verbose=10)(
    delayed(calculate_sister_activity_mAP_parallel)(
        gene = gene, 
        all_data = df,
        Perturbation_Gene_key =Perturbation_Gene_key,
        negative_control_list = [negative_control],
        sample_method = sample_method,
        n_number = n_number,
        random_seed = random_seed
            )
                for gene in tqdm(Perturbation_Gene, desc='Processing genes')
            )

    with_control_mAP_results = [r for r in tqdm(with_control_mAP_results,desc = 'with_control_mAP_results') if r is not None]


    with_control_mAP_results_df = pd.concat(with_control_mAP_results, ignore_index=True)
    
    none_control_df = df[df[Perturbation_Gene_key] != negative_control]
            
    with_other_mAP_results = Parallel(n_jobs=n_jobs, verbose=10)(
    delayed(calculate_sister_consistency_mAP_parallel)(
        gene = gene, 
        all_data = none_control_df,
        Perturbation_Gene_key =Perturbation_Gene_key,
        sample_method = sample_method,
        n_number = n_number,
        random_seed = random_seed
            )
                for gene in tqdm(Perturbation_Gene, desc='Processing genes')
            )

    with_other_mAP_results = [r for r in tqdm(with_other_mAP_results,desc = 'with_other_mAP_results') if r is not None]


    with_other_mAP_results_df = pd.concat(with_other_mAP_results, ignore_index=True)

    logger.info(f"mAP calculation completed. "
                f"with_control: {len(with_control_mAP_results_df)} results, "
                f"with_other: {len(with_other_mAP_results_df)} results")


    return with_control_mAP_results_df, with_other_mAP_results_df


