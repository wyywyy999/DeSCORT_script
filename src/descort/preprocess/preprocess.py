import os
import sys
import time
import logging
import pycytominer
import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path
from typing import List, Union,Optional,Literal



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


def process_csv_to_parquet(csv_path: Path, output_dir: Path) -> bool:
    """处理单个CSV文件，转换为Parquet格式"""
    try:
        # 构建输出路径
        output_filename = f"{csv_path.stem}.parquet"
        output_path = output_dir / output_filename
        
        # 检查是否已处理
        if output_path.exists():
            logging.debug(f"Skipping {csv_path.name} - output already exists")
            return True
        
        df = pd.read_csv(csv_path)
        
        output_path.parent.mkdir(parents=True, exist_ok=True)  # 确保目录存在
        df.to_parquet(output_path, index=False)
        logging.info(f"Saved to {output_path}")
        return True
        
    except Exception as e:
        logging.error(f"Failed to process {csv_path}: {e}")
        return False
    
    

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

def normalization_func(
    df: pd.DataFrame,
    sample_name: str,
    need_columns: List[str],
    median_output_path: Path,
    normalization_output_path: Path,
    clip_output_path: Path,
    normalization_select_feature_output_path: Path,
    normalize_method: Literal["standardize", "robustize", "mad_robustize", "spherize"] = 'mad_robustize',
    normalize_samples_query: Optional[str] = None,
    data_level: str = 'well',
    clip_lower_percentile: float = 1,
    clip_upper_percentile: float = 99,
    feature_select_operations: List[str] = ['variance_threshold', 'correlation_threshold', 'drop_na_columns', 'blocklist'],
    na_cutoff: float = 0.05,
    corr_threshold: float = 0.9,
) -> dict:
    """
    完整的数据归一化处理函数
    
    Parameters
    ----------
    df : pd.DataFrame
        输入的原始数据框
    sample_name : str
        样本名称
    need_columns : List[str]
        聚合需要的列名
    median_output_path : Path
        中位数聚合结果输出路径
    normalization_output_path : Path
        归一化结果输出路径
    clip_output_path : Path
        clip处理结果输出路径
    normalization_select_feature_output_path : Path
        特征选择结果输出路径
    normalize_method : str

    normalize_samples_query : str, optional
        归一化使用的样本查询条件
        对于 'mad_robustize' 通常需要指定对照样本
        对于 'zscore' 可以为 None (使用所有样本)
    data_level : str
        数据级别，'well' 或其他
    clip_lower_percentile : float
        clip下分位数，默认 1
    clip_upper_percentile : float
        clip上分位数，默认 99
    feature_select_operations : List[str]
        特征选择操作列表
    na_cutoff : float
        NA值截断阈值
    corr_threshold : float
        相关性阈值
        
    Returns
    -------
    dict
        包含处理结果路径的字典
    """
    logger = logging.getLogger(__name__)
    
    # 确保输出目录存在
    os.makedirs(median_output_path, exist_ok=True)
    os.makedirs(normalization_output_path, exist_ok=True)
    os.makedirs(clip_output_path, exist_ok=True)
    os.makedirs(normalization_select_feature_output_path, exist_ok=True)
    
    # 构建输出文件路径
    median_output_file = os.path.join(median_output_path, f'{sample_name}_median.parquet')
    normalization_output_file = os.path.join(normalization_output_path, f'{sample_name}_normalization.parquet')
    clip_output_file = os.path.join(clip_output_path, f'{sample_name}_normalization_clip_data.parquet')
    feature_select_output_file = os.path.join(
        normalization_select_feature_output_path, 
        f'{sample_name}_normalization_feature_select.parquet'
    )
    
    result = {
        'median_file': median_output_file,
        'normalization_file': normalization_output_file,
        'clip_file': clip_output_file,
        'feature_select_file': feature_select_output_file,
        'normalize_method': normalize_method
    }
    
    try:
        # 1. 聚合 (如果不是well级别)
        if data_level != 'well':
            logger.info(f"Starting aggregation for {sample_name}")
            start_time = time.time()
            
            median_df = pycytominer.aggregate(
                population_df=df,
                strata=need_columns,
                operation="median",
                output_file=median_output_file,
                output_type="parquet",
                float_format="%.3f",
            )
            
            logger.info(f"Aggregation completed, time: {time.time() - start_time:.2f}s")
            current_df = median_df
        else:
            current_df = df
            logger.info(f"Skipping aggregation for {sample_name} (well level data)")
        
        # 2. 归一化
        logger.info(f"Starting normalization for {sample_name} using method: {normalize_method}")
        start_time = time.time()
        

        if normalize_samples_query is None:
            logger.warning("MAD robustize normalization recommended to specify control samples. Using 'all'.")
            samples_param = 'all'
        else:
            samples_param = normalize_samples_query
        
        normalizer_df = pycytominer.normalize(
            profiles=current_df,
            method=normalize_method,
            samples=samples_param,
            output_file=normalization_output_file,
            output_type="parquet",
            float_format="%.3f",
            )
        
        logger.info(f"Normalization completed, time: {time.time() - start_time:.2f}s")
        
        # 3. Clip (1, 99) 和列过滤
        logger.info(f"Starting clip processing for {sample_name}")
        start_time = time.time()
        
        normalization_select_feature_df = pd.read_parquet(normalization_output_file)
        
        # 计算每列的均值（只对数值列）
        numeric_cols = normalization_select_feature_df.select_dtypes(include=[np.number]).columns
        column_means = normalization_select_feature_df[numeric_cols].mean()
        
        logger.info(f"Original data shape: {normalization_select_feature_df.shape}")
        logger.info(f"Numeric columns count: {len(column_means)}")
        
        # 找出需要删除的列
        # 条件1: 均值 > 1000
        cols_mean_gt_1000 = column_means[column_means > 1000].index
        # 条件2: 均值 == 0
        cols_mean_eq_0 = column_means[column_means == 0].index
        # 合并需要删除的列
        cols_to_drop = cols_mean_gt_1000.union(cols_mean_eq_0)
        
        logger.info(f"Columns with mean > 1000: {len(cols_mean_gt_1000)}")
        logger.info(f"Columns with mean == 0: {len(cols_mean_eq_0)}")
        logger.info(f"Total columns to drop: {len(cols_to_drop)}")
        
        if len(cols_to_drop) > 0:
            logger.debug(f"First 10 columns to drop: {cols_to_drop[:10].tolist()}")
        
        # 删除这些列
        data_cleaned = normalization_select_feature_df.drop(columns=cols_to_drop)
        
        # 分离特征和元数据
        feature_data = data_cleaned.filter(regex="^(?!Meta)")
        feature_data = feature_data.fillna(0)
        meta_data = data_cleaned.filter(regex='^(Meta)')
        
        # 执行clip
        lower = np.percentile(feature_data.values, clip_lower_percentile)
        upper = np.percentile(feature_data.values, clip_upper_percentile)
        feature_clipped = np.clip(feature_data, lower, upper)
        
        # 合并并保存
        final_df = pd.concat([meta_data, feature_clipped], axis=1)
        final_df.to_parquet(clip_output_file)
        
        logger.info(f"Clip processing completed, time: {time.time() - start_time:.2f}s")
        logger.info(f"Lower: {lower:.4f}, Upper: {upper:.4f}")
        
        # 4. 特征选择
        logger.info(f"Starting feature selection for {sample_name}")
        start_time = time.time()
        
        normalization_df = pd.read_parquet(clip_output_file)
        
        pycytominer.feature_select(
            profiles=normalization_df,
            features='infer',
            samples='all',
            operation=feature_select_operations,
            na_cutoff=na_cutoff,
            corr_threshold=corr_threshold,
            output_file=feature_select_output_file,
            output_type='parquet',
        )
        
        logger.info(f"Feature selection completed, time: {time.time() - start_time:.2f}s")
        logger.info(f"All processing completed for {sample_name}")
        
        return result
        
    except Exception as e:
        logger.error(f"Error processing {sample_name}: {e}")
        raise


def map_channel_names(column_list):
    """
    将旧版通道名映射到新版通道名
    """
    mapping = {
        'DAPI_Painting': 'DNA',
        'Phalloidin': 'AGP',
        'ConA': 'ER',
        'Mito': 'Mito'
    }
    
    new_column_list = []
    for col in column_list:
        new_col = col
        for old, new in mapping.items():
            # 替换通道名部分，例如 "Intensity_MeanIntensity_Phalloidin" -> "Intensity_MeanIntensity_AGP"
            if f'_{old}_' in new_col or new_col.endswith(f'_{old}'):
                new_col = new_col.replace(old, new)
        new_column_list.append(new_col)
    
    return new_column_list