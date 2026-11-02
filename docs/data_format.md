# 数据格式与对齐

## Parquet

元数据列以 `Metadata` 开头；其他列按原处理器逻辑视为形态特征。请避免把无关数值/字符串列混入非元数据区域。扰动列由 `--key_gene_column` 显式指定，例如 CRISPR 的 `Metadata_Foci_Barcode_MatchedTo_GeneCode`、Compound 的 `Metadata_Smiles`。不同任务的控制标签可能是 `nontargeting`、SMILES 或其他实验标记，需要与词表一致。

特征名称通过 `feature2index.pkl` 映射到整数 ID，0 为 padding ID；所选特征列表是名称列表。当前处理器取输入 parquet 中出现、且同时在词表和所选特征集合中的特征，保持输入列的顺序，不简单照列表重排。

## Hugging Face / Arrow dataset

`Dataset.save_to_disk` 输出目录包含 Arrow 与元信息文件，应整体提供给训练脚本。

| 用途 | 保存字段 |
|---|---|
| 对比/重建 | `Perturb_Gene`、`Feature_expression`、`Feature_name_index`、`Perturb_Gene_Index` |
| 扰动预测 | `Perturb_Gene`、`Perturb_expression`、`Control_expression`、`Feature_name_index`、`Perturb_Gene_Index` |

字段大小写有意义。Collator 输出的 `Masked_Expr`、`Feature_Name_Index_1`、`Attention_mask` 等是模型参数名，不应与磁盘字段混用。

形态值 padding 为 100，mask 值为 -100；重建标签中 -100 表示不计算该位置损失。实际值来自各入口参数和检查点，整理不更改这些含义。

## 条件表示

扰动标签 → index、index → 条件 embedding、名称 → embedding 三类映射不可互换。虚拟细胞训练中参数仍保留历史名称 `--index2esm2_pkl`，但文件可以是 ESM2、Morgan 或 UniMol2 的向量映射；`esm2_input_dim` 必须与真实宽度一致。Compound 推理入口还保留 `--index2morgan_pkl` / `--smiles2index_pkl` 等参数名，具体以 `--help` 为准。

## Embedding / AnnData

表征提取输出为保留 `Metadata*` 的 parquet，表征列为 `emb_0 ... emb_(hidden_size-1)`。目录层级受 `--metadata_parts` 影响；扁平示例应显式设置该参数。

虚拟细胞输出 H5AD 包含 `.X`、扰动与控制标签、`.var.feature_id` 和 `.uns` 的模型/输入来源信息。Compound 推理使用微调检查点保存的 `population_feature_ids` 与 intersection 对齐；不能在测试集上重新选择轴。

评估前检查：特征名称、ID 与顺序，扰动集合，控制标签，输入 layer，`in_sample`，是否存在重复配对对照。`--strict-alignment` 进一步要求样本 ID、顺序与标签一致。指标输入是连续形态特征，不能误当 RNA count 执行 log1p。
