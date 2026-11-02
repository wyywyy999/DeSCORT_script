# 架构图与实际实现

架构图使用用户提供的 `Figure_1_ModelFramework.png`，文档副本为 `assets/workflow.png`。

| 图中部分 | 实际实现 |
|---|---|
| CellProfiler / Pycytominer 输入处理 | `preprocess/preprocess.py`、`tutorials/01_preprocess_data`；本包接收已有形态特征，不直接处理显微图像 |
| 视图采样、掩码 | `data/collator.py` 的 `DeSCORTForContrastiveLMDataCollator` |
| 形态值编码 | `model/modeling_descort.py` 的 `AutoDiscretizationEmbedding` |
| 特征名称编码 | `DescortConditionEmbeddings` |
| `[CLS]` 与编码器 | `DeSCORTModel`、`DeSCORTEncoder` |
| RMSNorm / Self-Attention / MoE | `DeepseekRMSNorm`、`MultiHeadAttention`、`DeepseekMoE`、`DeSCORTLayer` |
| 重建、投影及对比学习 | `MLMHead`、`DeSCORTForContrastiveLM`、`SupConLoss` |
| 条件扰动生成 | `DeSCORTPerturbModel`、`DeSCORTForPerturbPrediction`、`vc/*` |
| 下游检索、整合与解释 | `mAP`、`batch`、`biological` |

源码默认编码器为 8 层、隐藏维度 256、8 个注意力头；MoE 默认 8 个路由专家、每 token 选 4 个专家、1 个共享专家。实际运行以检查点 `config.json` 为准。形态值与特征名称的 embedding 在相加前分别执行 LayerNorm；这与编码器内部的 RMSNorm 是不同步骤。

图中展示的实验数据规模与下游效果不由本次代码整理重新验证。模型前向、损失函数、归一化和指标公式均沿用原实现；架构图不作为擅自更改实现的依据。
