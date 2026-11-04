# 工作流

统一任务顺序与输入输出见 [教程总览](../../tutorials/README.md)。

下载与预处理入口：

- [数据下载](../../tutorials/00_download/README.md)：从 Cell Painting Gallery 获取 cpg0021、cpg0000 和 cpg0016 原始特征。
- [数据预处理与词表](../../tutorials/01_preprocess_data/README.md)：CSV 转换 → 聚合、归一化、裁剪与特征筛选 → 标签词表 → 特征交并集 → 提取所选特征数据。

主链路：数据下载 → 预处理 → parquet / 词表 → Arrow 数据集 → 预训练 → 表征微调 → embedding → mAP / 批次效应 / 生物学解释。

虚拟细胞支线：归一化形态 + 扰动条件向量 + 训练/测试划分 → 扰动预测微调 → 对照驱动推理 → 连续特征评估。CRISPR 与 Compound、未见扰动与跨细胞系分别有明确入口。

跨扰动类型分析保留独立任务，复用相同模型、数据和提取函数；论文作图读取已完成实验的结果表。
