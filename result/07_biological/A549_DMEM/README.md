# A549 DMEM 生物学分析结果预览

这些图片来自原项目 `tutorials/07_biological_new/result/` 中已有的 A549 分析，原样复制用于 [生物学教程](../../../tutorials/07_biological/README.md) 展示。本次没有重新计算完整分析，也没有修改图像。

| 图片 | 内容 | 原分析目录，相对于旧项目的 `tutorials/07_biological_new/result/` |
|---|---|---|
| [pathway_umap.png](pathway_umap.png) | DeSCORT 通路叠加 UMAP | `01_umap_analysis/cpg0021_A549_DMEM/model/figures/` |
| [corum_similarity.png](corum_similarity.png) | CORUM 背景校正相似性分布 | `02_CORUM_STRING/cpg0021_A549_DMEM/figures/` |
| [embedding_clusters.png](embedding_clusters.png) | DeSCORT 10 簇 UMAP | `03_de_cluster/hit_10_cluster/figures/` |
| [cluster3_enrichment.png](cluster3_enrichment.png) | 历史运行中簇 3 的富集气泡图 | `04_cluster_enrichment/figures/` |

原始文件名和 SHA-256 校验值见 [manifest.json](manifest.json)。聚类编号、基因数及通路条目属于各图片对应的历史运行；重新运行教程时，应读取本次生成的结果表进行解释。
