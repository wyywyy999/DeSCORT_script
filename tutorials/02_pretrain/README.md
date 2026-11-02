# 表征预训练

输入输出：归一化 parquet、标签词表、特征词表 → Arrow → 基础模型。

运行 `01_process_data.py`，需要时用 `02_combined_dataset.py` 合并，再执行 `02_pretrain_model.py`。训练使用双视图重建与对比学习；输出模型通常位于 `output_dir/models`。可以从 `configs/pretrain/example.json` 开始。

```bash
bash tutorials/02_pretrain/run.sh --help
```

从任意工作目录执行时，可使用本入口的绝对路径。参数应以当前脚本 `--help` 为准；历史实验 Bash 中的服务器路径需要按本机配置调整。
