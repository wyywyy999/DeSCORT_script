# 基线与历史实验索引

基线代码与大型训练结果保留在原项目 `../..`。本副本通过 `manifest.json` 定位，不混入核心安装包。

| 原项目目录 | 方法/内容 |
|---|---|
| `tutorials/06_vc/MLP` | CRISPR 线性/MLP |
| `tutorials/06_vc/Rigde_model` | 历史 Ridge 对照与审计，保留原目录拼写 |
| `tutorials/06_vc_compound/MLP` | Compound 线性/MLP |
| `tutorials/06_vc_compound/Ridge` | Compound Ridge |
| `tutorials/06_vc_compound/GEARS` | 本地 GEARS 数据/训练/评估适配 |
| `tutorials/06_vc_compound/STATE` | 本地 STATE 适配 |
| `tutorials/06_vc_compound/delta_transfer` | delta transfer 对照 |

这些是原实验的本地适配，不自动代表原方法论文的完整官方实现或相同调参预算。每个方法需要记录条件表示、特征集合、训练划分、控制规则、指标输入和独立环境，保证比较口径一致。
