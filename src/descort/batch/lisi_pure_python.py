"""
纯 Python 实现的 cLISI 和 iLISI 指标
对齐 scib.metrics.lisi_graph 的实现逻辑

与 scib 的对齐点:
  - 使用 kNN 图上的最短路径距离找邻居（而非 PCA 空间欧氏距离）
  - 图由 scanpy 的 k=15 邻居图构建
  - LISI core (Simpson's Index + perplexity beta 搜索) 与 scib 完全一致

两种模式:
  - graph: 在 kNN 图上 BFS 找 k0 个最短路径邻居（对齐 scib，推荐）
           ≈ 93k 细胞 / subsample=10000 / 约 15-40 秒
  - direct: 直接在 PCA 空间算 kNN（快速近似，不严格对齐 scib）

用法:
    from lisi_pure_python import clisi, ilisi

    score_c = clisi(adata, label_key='celltype', mode='graph')
    score_i = ilisi(adata, batch_key='batch', mode='graph')
"""

from typing import Optional
import numpy as np
import pandas as pd
import scanpy as sc
from numba import njit, prange
from scipy.sparse import csr_matrix, issparse


# ============================================================================
# Core LISI — 与 scib 完全一致
# ============================================================================

@njit
def _Hbeta(D_row: np.ndarray, beta: float):
    """
    计算 H 和 P（perplexity 匹配的辅助函数）。
    与 scib.metrics.lisi.Hbeta 完全一致。
    """
    D_row = np.copy(D_row)
    D_row[np.isnan(D_row)] = 0.0
    P = np.exp(-D_row * beta)
    sumP = np.sum(P)
    if sumP == 0.0:
        return 0.0, np.zeros_like(P)
    H = np.log(sumP) + beta * np.sum(D_row * P) / sumP
    P /= sumP
    return H, P


@njit
def _one_hot(vector: np.ndarray, num_classes: int) -> np.ndarray:
    """Numba 兼容的 one-hot 编码。"""
    result = np.zeros((len(vector), num_classes), dtype=np.float64)
    for i in range(len(vector)):
        result[i, int(vector[i])] = 1.0
    return result


@njit
def _compute_lisi_for_cell(
    knn_idx: np.ndarray,
    D_act: np.ndarray,
    batch_labels: np.ndarray,
    n_batches: int,
    perplexity: float,
    tol: float = 1e-5,
) -> float:
    """
    为单个细胞计算 LISI (1/Simpson's Index)。
    与 scib.metrics.lisi.compute_simpson_index_graph 逻辑一致。

    knn_idx:      k0 个邻居的索引 (0-based)
    D_act:        到 k0 个邻居的最短路径距离
    batch_labels: 所有细胞的类别编码
    n_batches:    唯一类别数
    perplexity:   困惑度参数
    """
    if np.any(np.isnan(D_act)) or len(D_act) == 0:
        return 1.0

    logU = np.log(perplexity)
    beta = 1.0
    betamin = -np.inf
    betamax = np.inf

    H, P = _Hbeta(D_act, beta)
    Hdiff = H - logU
    tries = 0

    while np.abs(Hdiff) > tol and tries < 50:
        if Hdiff > 0.0:
            betamin = beta
            if betamax == np.inf:
                beta *= 2.0
            else:
                beta = (beta + betamax) / 2.0
        else:
            betamax = beta
            if betamin == -np.inf:
                beta /= 2.0
            else:
                beta = (beta + betamin) / 2.0

        H, P = _Hbeta(D_act, beta)
        Hdiff = H - logU
        tries += 1

    if H == 0.0:
        return -1.0

    batch = batch_labels[knn_idx]
    B = _one_hot(batch, n_batches)

    sumP = np.zeros(n_batches, dtype=np.float64)
    for j in range(len(P)):
        for k in range(n_batches):
            sumP[k] += P[j] * B[j, k]

    simpson = 0.0
    for k in range(n_batches):
        simpson += sumP[k] * sumP[k]

    if simpson < 1e-10:
        return float(n_batches)
    return 1.0 / simpson


@njit(parallel=True)
def _compute_lisi_all(
    knn_indices: np.ndarray,
    knn_distances: np.ndarray,
    batch_labels: np.ndarray,
    n_batches: int,
    perplexity: float,
    tol: float = 1e-5,
) -> np.ndarray:
    """为所有细胞并行计算 LISI。"""
    n_cells = knn_indices.shape[0]
    lisi = np.zeros(n_cells, dtype=np.float64)

    for i in prange(n_cells):
        knn_idx = knn_indices[i].astype(np.int64)
        D = knn_distances[i]
        lisi[i] = _compute_lisi_for_cell(
            knn_idx, D, batch_labels, n_batches, perplexity, tol
        )

    return lisi


# ============================================================================
# 图最短路径邻居搜索 — 对齐 scib 的 C++ knn_graph.o
# ============================================================================

@njit
def _bfs_knn_all_cells(
    adj_indptr: np.ndarray,
    adj_indices: np.ndarray,
    k0: int,
    max_depth: int = 6,
) -> tuple:
    """
    对每个细胞在 kNN 图上做 BFS，找到 k0 个最短路径邻居。
    距离 = BFS 跳数（hop count）。

    使用时间戳 visited 数组避免每细胞重置 O(n)。

    参数:
        adj_indptr:  CSR 行指针 (n+1,)
        adj_indices: CSR 列索引 (nnz,)
        k0:          要找的邻居数
        max_depth:   BFS 最大深度

    返回:
        knn_idx:  (n_cells, k0) 邻居索引
        knn_dist: (n_cells, k0) 邻居距离（跳数）
    """
    n = len(adj_indptr) - 1

    knn_idx = np.zeros((n, k0), dtype=np.int32)
    knn_dist = np.full((n, k0), np.nan, dtype=np.float64)

    visited = np.zeros(n, dtype=np.int32)
    frontier = np.zeros(n, dtype=np.int32)
    next_frontier = np.zeros(n, dtype=np.int32)
    timestamp = 0

    for cell in range(n):
        timestamp += 1
        visited[cell] = timestamp

        frontier[0] = cell
        frontier_size = 1

        found_count = 0
        depth = 0

        while found_count < k0 and frontier_size > 0 and depth < max_depth:
            depth += 1
            next_size = 0

            for fi in range(frontier_size):
                node = frontier[fi]
                for nb_ii in range(adj_indptr[node], adj_indptr[node + 1]):
                    nb = adj_indices[nb_ii]
                    if visited[nb] != timestamp:
                        visited[nb] = timestamp
                        next_frontier[next_size] = nb
                        next_size += 1
                        knn_idx[cell, found_count] = nb
                        knn_dist[cell, found_count] = depth
                        found_count += 1
                        if found_count >= k0:
                            break
                if found_count >= k0:
                    break

            frontier, next_frontier = next_frontier, frontier
            frontier_size = next_size

    return knn_idx, knn_dist


def _neighbors_on_graph(
    adata,
    k0: int,
    subsample: Optional[int] = None,
    random_state: int = 0,
    verbose: bool = False,
):
    """
    在 kNN 图上用 BFS 找 k0 个最短路径邻居。
    对齐 scib 的 C++ knn_graph.o 行为。

    如果 subsample 设值，对采样子集计算邻居，返回子集在原始坐标中的索引。
    """
    if "neighbors" not in adata.uns:
        raise AttributeError(
            "adata 没有 kNN 图。请先运行 sc.pp.neighbors(adata, n_neighbors=15)"
        )

    n_cells = adata.n_obs
    rng = np.random.RandomState(random_state)

    if subsample is not None and subsample < n_cells:
        cell_idx = np.sort(rng.choice(n_cells, subsample, replace=False))
        if verbose:
            print(f"采样 {subsample}/{n_cells} 个细胞进行 LISI 计算")
    else:
        cell_idx = np.arange(n_cells)

    # 获取 kNN 图（CSR）
    connectivities = adata.obsp["connectivities"].tocsr()

    # 构建对称无权重图
    # 每个细胞的邻居为 connectivities[i] 中非零的列
    adj_indptr = connectivities.indptr.astype(np.int32)
    adj_indices = connectivities.indices.astype(np.int32)

    # 如果采样子集，需要从全图中提取子图的邻居关系
    # 但我们这里只对特定细胞做 BFS，所以 BFS 在全图上跑，只采集子集的结果
    if subsample is not None and subsample < n_cells:
        # 先全量 BFS
        if verbose:
            print("BFS 搜索图最短路径邻居 ...")

        knn_idx_all, knn_dist_all = _bfs_knn_all_cells(
            adj_indptr, adj_indices, k0=k0, max_depth=6,
        )

        # 提取子集
        knn_idx = knn_idx_all[cell_idx]
        knn_dist = knn_dist_all[cell_idx]
    else:
        if verbose:
            print("BFS 搜索图最短路径邻居 ...")

        knn_idx, knn_dist = _bfs_knn_all_cells(
            adj_indptr, adj_indices, k0=k0, max_depth=6,
        )

    # 检查未找到足够邻居的细胞
    n_missing = np.sum(np.isnan(knn_dist[:, -1]))
    if n_missing > 0:
        if verbose:
            print(f"警告: {n_missing}/{len(cell_idx)} 个细胞未找到 {k0} 个邻居")

    return knn_idx, knn_dist, cell_idx


# ============================================================================
# 直接 kNN 模式（快速近似）
# ============================================================================

def _neighbors_direct(
    adata,
    k0: int,
    use_rep: str = "X_pca",
    subsample: Optional[int] = None,
    random_state: int = 0,
    verbose: bool = False,
):
    """
    直接在 PCA/embedding 空间计算 k0 个最近邻（含自环）。
    不对齐 scib，作为快速近似使用。

    ** 与 scib 的区别 **:
      scib: 在 k=15 的 kNN 图上跑最短路径，取 k0=90
      此处: 直接算 PCA 空间中的 k0=90 个欧氏近邻
    """
    rng = np.random.RandomState(random_state)
    n_cells = adata.n_obs

    if subsample is not None and subsample < n_cells:
        idx = np.sort(rng.choice(n_cells, subsample, replace=False))
        adata_sub = adata[idx].copy()
    else:
        idx = np.arange(n_cells)
        adata_sub = adata.copy()

    sc.pp.neighbors(adata_sub, n_neighbors=k0, use_rep=use_rep, copy=False)

    distances = adata_sub.obsp["distances"]
    n = adata_sub.n_obs
    knn_idx = np.zeros((n, k0), dtype=np.int32)
    knn_dist = np.zeros((n, k0), dtype=np.float64)

    for i in range(n):
        row_start = distances.indptr[i]
        row_end = distances.indptr[i + 1]
        neighbors = distances.indices[row_start:row_end]
        dists = distances.data[row_start:row_end]
        order = np.argsort(dists)
        n_actual = min(len(order), k0)
        knn_idx[i, :n_actual] = neighbors[order[:n_actual]]
        knn_dist[i, :n_actual] = dists[order[:n_actual]]
        if n_actual < k0:
            for j in range(n_actual, k0):
                knn_idx[i, j] = i
                knn_dist[i, j] = 0.0

    return knn_idx, knn_dist, idx


# ============================================================================
# 主 LISI 计算 — 对齐 scib 的 lisi_graph_py
# ============================================================================

def lisi_graph_py(
    adata,
    obs_key: str,
    mode: str = "graph",
    use_rep: str = "X_pca",
    k0: int = 90,
    perplexity: Optional[float] = None,
    subsample: Optional[int] = None,
    verbose: bool = False,
    random_state: int = 0,
):
    """
    LISI 主计算。对齐 scib.metrics.lisi.lisi_graph_py。

    步骤:
      1. 在 kNN 图上找 k0 个最短路径邻居（mode='graph'）
         或 PCA 空间找 k0 个欧氏近邻（mode='direct'）
      2. 对每个细胞: beta 搜索匹配 perplexity → 计算 Simpson's Index → 取倒数
      3. 返回每个细胞的 LISI 值

    参数:
        mode: 'graph'（对齐 scib）/ 'direct'（快速近似）
    """
    # Step 1: 找邻居
    if mode == "graph":
        knn_idx, knn_dist, cell_idx = _neighbors_on_graph(
            adata, k0=k0, subsample=subsample,
            random_state=random_state, verbose=verbose,
        )
    elif mode == "direct":
        knn_idx, knn_dist, cell_idx = _neighbors_direct(
            adata, k0=k0, use_rep=use_rep,
            subsample=subsample, random_state=random_state, verbose=verbose,
        )
    else:
        raise ValueError(f"未知 mode: {mode}，请使用 'graph' 或 'direct'")

    # Step 2: 准备标签编码
    all_categories = adata.obs[obs_key].astype("category")
    obs_codes = all_categories.cat.codes.values[cell_idx]
    n_categories = adata.obs[obs_key].nunique()

    # Step 3: 去除自环
    # 在 BFS/graph 模式下，第一个邻居不是自环（因为 BFS 不包含自身）
    # 在 direct 模式下，第一个是自环
    # 统一处理: 跳过 index == cell_idx 的行
    n_actual = len(cell_idx)
    # knn_idx 的每一行第一个邻居就是第一个找到的邻居（不是自环）
    # 我们使用所有 k0 个邻居（scib 的行为也是使用所有找到的邻居，没有单独去自环）
    # 看看 scib：C++ 的 indices 不包含自环，BFS 也不包含
    # direct 模式包含自环，需要跳过第一个

    if mode == "direct":
        # direct 模式第一个是自环
        used_knn_idx = knn_idx[:, 1:k0]
        used_knn_dist = knn_dist[:, 1:k0]
        effective_k = k0 - 1
    else:
        # graph/BFS 模式不自包含自身
        used_knn_idx = knn_idx
        used_knn_dist = knn_dist
        effective_k = k0

    # Step 4: 计算 LISI
    if perplexity is None or perplexity >= effective_k:
        perplexity = np.floor(effective_k / 3)

    lisi_values = _compute_lisi_all(
        knn_indices=used_knn_idx,
        knn_distances=used_knn_dist,
        batch_labels=obs_codes,
        n_batches=n_categories,
        perplexity=perplexity,
    )

    return lisi_values, cell_idx


# ============================================================================
# 公开 API — cLISI 和 iLISI
# ============================================================================

def clisi(
    adata,
    label_key: str,
    mode: str = "graph",
    use_rep: str = "X_pca",
    k0: int = 90,
    perplexity: Optional[float] = None,
    subsample: Optional[int] = None,
    scale: bool = True,
    verbose: bool = False,
    random_state: int = 0,
) -> float:
    """
    Cell-type LISI (cLISI) 分数。

    对齐 scib.metrics.clisi_graph。

    解释:
        cLISI = 每个细胞邻居中细胞类型的有效混合度
        - 原始值: 1（纯）~ nlabs（完全混合）
        - 缩放值: 1（好, 纯）~ 0（差, 混合）

    参数:
        mode: 'graph'（默认，对齐 scib）/ 'direct'（快速近似）
    """
    lisi_values, _ = lisi_graph_py(
        adata=adata, obs_key=label_key,
        mode=mode, use_rep=use_rep,
        k0=k0, perplexity=perplexity,
        subsample=subsample,
        verbose=verbose, random_state=random_state,
    )

    median_score = np.nanmedian(lisi_values)

    if scale:
        nlabs = adata.obs[label_key].nunique()
        if nlabs <= 1:
            return 1.0
        # cLISI: 1 -> 好, nlabs -> 差
        scaled = (nlabs - median_score) / (nlabs - 1)
        return max(0.0, min(1.0, scaled))
    else:
        return median_score


def ilisi(
    adata,
    batch_key: str,
    mode: str = "graph",
    use_rep: str = "X_pca",
    k0: int = 90,
    perplexity: Optional[float] = None,
    subsample: Optional[int] = None,
    scale: bool = True,
    verbose: bool = False,
    random_state: int = 0,
) -> float:
    """
    Integration LISI (iLISI) 分数。

    对齐 scib.metrics.ilisi_graph。

    解释:
        iLISI = 每个细胞邻居中 batch 的有效混合度
        - 原始值: 1（差, 单一 batch）~ nbatches（好, 均匀混合）
        - 缩放值: 1（好, 混合）~ 0（差, 分离）
    """
    lisi_values, _ = lisi_graph_py(
        adata=adata, obs_key=batch_key,
        mode=mode, use_rep=use_rep,
        k0=k0, perplexity=perplexity,
        subsample=subsample,
        verbose=verbose, random_state=random_state,
    )

    median_score = np.nanmedian(lisi_values)

    if scale:
        nbatches = adata.obs[batch_key].nunique()
        if nbatches <= 1:
            return 1.0
        # iLISI: 1 -> 差, nbatches -> 好
        scaled = (median_score - 1) / (nbatches - 1)
        return max(0.0, min(1.0, scaled))
    else:
        return median_score
