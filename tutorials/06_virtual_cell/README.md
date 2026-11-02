# 虚拟细胞

采用原项目 `06_vc_new` 中已提取公共模块的四组任务入口，统一到本目录；不能仅因目录名含 new 就认为与所有历史实验等价。旧 `06_vc`、`06_vc_compound` 的探索、消融和结果仍在原项目，版本来源见整理清单。

| 任务 | CRISPR | Compound |
|---|---|---|
| 未见扰动生成 | `unseen_gene_generation/Crispr` | `unseen_gene_generation/Compound` |
| 跨细胞系迁移 | `cell_line_transfer/Crispr` | `cell_line_transfer/Compound` |

每组运行顺序：`01_process_data.py` → `02_finetune.py` → `03_inference.py` → `04_calculate_vcc2026_metric.py`。各组同时提供历史参数 Bash 和无默认参数的 `run.sh`。

```bash
bash tutorials/06_virtual_cell/unseen_gene_generation/Crispr/run.sh 01_process_data.py --help
bash tutorials/06_virtual_cell/unseen_gene_generation/Compound/run.sh 03_inference.py --help
```

CRISPR 条件向量通常来自 ESM2；Compound 可使用 Morgan / UniMol2，通过文件内容及 `esm2_input_dim` 对齐，保留历史参数名称。控制标签、训练测试扰动列表、输入 feature axis、均值基线权重和指标族要分别记录。

跨细胞系任务需对目标细胞系单独准备数据，明确 few-shot 训练与测试的细胞系/plate 划分；目录名本身不自动实施数据隔离。未见 Compound 任务目录保留 historical `unseen_gene_generation` 命名，语义为未见化合物。

Compound 检查点必须保存 intersection feature axis；在测试集上重新推导交集会改变模型输入与评估含义。推理输出带来源元信息，指标入口默认检查特征轴与扰动集合。
