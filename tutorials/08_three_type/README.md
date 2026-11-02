# 跨扰动类型共享表征

输入输出：CRISPR / Compound / ORF 归一化数据 → 共享模型与 embedding。

依次运行 `01_process_data.py`、按需 `02_combined_dataset.py`、`02_finetune.py`、`03_calculate_emb.py`。不同扰动类型必须共用相同 feature ID 定义；标签映射也应记录来源。该任务复用公共模型、回调与提取函数。已有跨类型 mAP/生物学比较结果保留在原项目，其入口通过论文图与基线索引定位。

```bash
bash tutorials/08_three_type/run.sh --help
```

从任意工作目录执行时，可使用本入口的绝对路径。参数应以当前脚本 `--help` 为准；历史实验 Bash 中的服务器路径需要按本机配置调整。
