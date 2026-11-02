# 预处理与词表

输入输出：原始 CSV / parquet → 聚合、归一化、裁剪特征与词表。

依次参考 `01_csv2parquet.py` → `02_preprocess_data.py` → `03_make_label_vocab.py` → `04_select_intersection_union_feature.py` → `05_get_union_feature_data.py`。先完成特征预处理，再开展下游任务。词表与特征集的生成辅助脚本在 resources 的来源索引中记录。

```bash
bash tutorials/01_preprocess_data/run.sh --help
```

从任意工作目录执行时，可使用本入口的绝对路径。参数应以当前脚本 `--help` 为准；历史实验 Bash 中的服务器路径需要按本机配置调整。
