# 表征微调与 embedding

输入输出：基础模型、任务 parquet → 微调模型、embedding parquet。

依次运行 `01_process_data.py` → `02_finetune.py` → `03_calculate_emb.py`。提取入口复用 `descort.embedding.extract`，支持 CPU。使用真实检查点时显式给出所有词表与所选特征，避免把扰动预测模型误作表征模型。

```bash
bash tutorials/03_finetune/run.sh --help
```

从任意工作目录执行时，可使用本入口的绝对路径。参数应以当前脚本 `--help` 为准；历史实验 Bash 中的服务器路径需要按本机配置调整。
