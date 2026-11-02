# 公共 API

统一导入名为 `descort`，安装目录为 `src/descort`。

```python
from descort import DeSCORTConfig, DeSCORTForContrastiveLM, DeSCORTForPerturbPrediction
from descort.data import DeSCORTForContrastiveLMDataProcessor
from descort.data.collator import DeSCORTForMaskedLMDataCollator, PerturbPredictionDataCollator
from descort.vc import MetricsEvaluator
```

`DeSCORTForContrastiveLM(config)` 用于表征训练；`from_pretrained(path)` 加载对应模型。`model.get_latent_embedding(**batch).logits` 返回 `[batch, hidden_size]` 的 CLS embedding。该接口使用单视图 `Masked_Expr`、`Feature_Name_Index_1`、`Attention_mask`；可由 `DeSCORTForMaskedLMDataCollator` 生成后，将 `Feature_Name_Index` 显式改名为 `Feature_Name_Index_1`。

`DeSCORTForPerturbPrediction(config, INDEX2ESM2_Dict)` 用于虚拟细胞任务；条件映射、维度和 feature axis 都需要与训练保持一致。`DeSCORTForMaskedLM` 是保留的旧接口，带额外构造参数；当前推荐的预训练与 embedding 工作流使用 `DeSCORTForContrastiveLM`，本次不改写旧任务类。

| 模块 | 作用 |
|---|---|
| `model` | 模型、配置、输出数据结构、扰动损失 |
| `data` | parquet 处理、collator、分组 sampler |
| `preprocess` | CSV 转换、特征归一化、通道名映射 |
| `training` | 共用 `LossLoggingCallback` |
| `embedding.extract` | 共用 `parse_args()` / `main()`；支持 `python -m descort.embedding.extract` |
| `mAP` | 活性、姐妹扰动一致性/活性 mAP |
| `batch` | 校正后端与整合指标 |
| `vc` | 条件扰动、推理、连续特征指标、replicate 与 scaling |
| `biological` | PCA/UMAP、CORUM/STRING、聚类和富集 |

`descort.modeling_descort` 仅为正式 `descort.model.modeling_descort` 的兼容导入。历史模型副本位于 `archive/legacy_models`，不进入安装包。
