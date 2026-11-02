# 资源与来源

本整理副本没有复制原始数据、大型 embedding、词表 pickle 和模型权重。`manifest.json` 登记原项目 `../../common_files/` 的本地资源路径、大小，以及可负担的小文件校验值；未确认的来源/许可证不推测填写。

| 资源 | 原项目位置 | 用法 |
|---|---|---|
| 特征名称词表 | `common_files/feature_vocab` | 名称 → ID，0 为 padding |
| 扰动标签词表 | `common_files/label_vocab` | 基因/化合物 → label ID |
| 特征集合 | `common_files/feature_intersection_union` | 选定名称集合 |
| ESM2 | `common_files/ESM2_files`、`06_vc` | 基因条件表示 |
| Morgan / UniMol2 | `common_files/06_vc_compound`、`06_vc/Morgan` | 化合物条件表示与生成脚本 |
| 生物学资源 | `common_files/biological`，工作区 `LLLM` 等 | CORUM、STRING、GO/KEGG 和手工注释 |

请显式传入绝对路径或在 `configs/local/paths.sh` 配置。`resources` 目前是索引，不伪装成已打包的数据目录。记录清单的 `sha256: null` 表示大文件未读取校验，不代表文件损坏。

数据与模型公开下载位置尚未确认，见 `docs/model_zoo.md`。
