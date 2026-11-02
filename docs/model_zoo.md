# 模型与检查点

当前材料没有可确认的公开下载链接，不提供虚构模型地址。沿用原实验机器上的检查点时，请显式传入路径。

| 模型 | 主要用途 | 需要的附属信息 |
|---|---|---|
| 基础 `DeSCORTForContrastiveLM` | 预训练后的通用表征 | config、权重、特征词表、归一化方案 |
| 表征微调模型 | mAP、批次、生物学、跨扰动类型分析 | 微调数据、特征集、表征提取参数 |
| `DeSCORTForPerturbPrediction` | CRISPR / Compound 虚拟细胞 | 条件向量映射、维度、特征轴、训练/测试划分与损失设置 |

一个可用的 Transformers 模型目录至少包含 `config.json` 及模型权重文件，原脚本通常在输出目录下保存 `models/`，中间训练状态可能位于 `checkpoint-*`。不能把基础表征模型目录直接当成已完成 Compound intersection 微调的扰动检查点。

记录检查点路径与 SHA-256、训练 commit/快照、`config.json`、词表校验值和数据划分。当前 `resources/manifest.json` 只登记原项目本地资源，不宣称提供训练权重。
