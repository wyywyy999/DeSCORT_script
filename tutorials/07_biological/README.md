# 生物学解释

输入输出：embedding、原形态、CORUM/STRING/基因集 → 聚类、通路与富集图表。

按任务选择 `01_umap_analysis.py`、`02_CORUM_STRING.py`、`03_de_cluster.py`、`04_cluster_enrichment.py`。前两者比较表征，第三个负责聚类/差异形态/富集，第四个读取已有富集表选择簇绘图。查询基因参数为空时保留跳过近邻图的行为。先用 `01_umap_analysis.py --stage check` 检查输入；指定 `--output-dir`、`--plates`、`--original-features-pkl` 和离线基因集资源。

```bash
bash tutorials/07_biological/run.sh --help
```

从任意工作目录执行时，可使用本入口的绝对路径。参数应以当前脚本 `--help` 为准；历史实验 Bash 中的服务器路径需要按本机配置调整。
