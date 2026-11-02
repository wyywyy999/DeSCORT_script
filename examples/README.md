# 最小示例

运行 `python examples/minimal_embedding.py --output-dir outputs/minimal_embedding`。

示例数据与随机小模型由代码生成，无需下载真实数据或预训练权重，使用 CPU。检查 3 个样本 × 16 维 embedding、数值有限和两个元数据列完整。它覆盖 parquet → 正式模型加载 → 公共提取入口 → parquet 输出。

真实任务先换成训练后的模型、真实特征/标签映射及相同的预处理方案。示例模型不能用于科学结论。
