# 形态检索 mAP

输入输出：embedding parquet、元数据 → mAP、显著性与命中结果。

`01_calculate_emb.py` 调用统一的 embedding 提取实现，修复旧版不存在的 `preprocess_22` 导入；然后运行 `02_calculate_mAP.py`。使用 copairs 的原有计算方式，保持控制标签、扰动列和 sample_method 一致。

```bash
bash tutorials/04_mAP/run.sh --help
```

从任意工作目录执行时，可使用本入口的绝对路径。参数应以当前脚本 `--help` 为准；历史实验 Bash 中的服务器路径需要按本机配置调整。
