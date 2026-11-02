# cell_line_transfer / Crispr

本组输入为归一化形态数据、扰动/特征映射及条件向量；输出为处理后的 dataset、训练模型、H5AD 预测与指标表。

依次执行 `01_process_data.py` → `02_finetune.py` → `03_inference.py` → `04_calculate_vcc2026_metric.py`。

```bash
bash tutorials/06_virtual_cell/cell_line_transfer/Crispr/run.sh 01_process_data.py --help
```

各阶段参数、控制规则与对齐约定见 [虚拟细胞总览](../../README.md)。历史参数 `.sh` 仅作为实验配置来源，执行前核对所有输入与输出路径。
