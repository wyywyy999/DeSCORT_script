# DeSCORT 项目整理框架草案

> 状态：供审阅的结构方案，尚未执行项目整理。  
> 日期：2026-11-02。  
> 本次仅新增此 Markdown；不移动、删除或重命名现有代码、模型、数据及结果。此文件所在目录目前只是方案存放位置。

## 1. 整理目标与参考

将 DeSCORT 整理成同时支持**安装使用、教程运行和论文实验复现**的研究代码仓库。

参考的是本地克隆版本的实际目录与文档：

- Cellist 的仓库根目录实际位于 `other_model/Cellist/Cellist/`。
- DeSCOPE 的仓库根目录位于 `other_model/DeSCOPE/`。

| 参考内容 | Cellist / DeSCOPE 的做法 | DeSCORT 建议采用的方式 |
|---|---|---|
| 项目介绍 | 根目录 README，包含方法、安装和使用说明 | 根目录统一 `README.md`，链接所有主要任务 |
| 安装配置 | `pyproject.toml`，源码位于 `src/` | 用 `pyproject.toml` 维护统一的包信息与依赖 |
| 公共代码 | `src/Cellist`、`src/descope` | 继续使用 `src/descort`，逐步统一导入方式 |
| 文档 | `docs/`，安装、教程和 API 分开 | 增加安装、输入格式、工作流和 API 文档 |
| 教程 | Cellist 提供命令行说明；DeSCOPE 按任务组织教程 | 保留编号教程，每项任务提供 Bash + Python + README |
| 软件入口 | Cellist 提供 `cellist` 命令 | 第一阶段保留 Bash 入口；统一 CLI 可后续增加 |

重点是借鉴仓库组织方式。DeSCORT 的输入格式、模型结构和实验流程按现有实现保留。

## 2. 当前项目中需要处理的主要问题

1. 根目录目前主要是 `Readme.md`、`src`、`tutorials`、`common_files` 等，安装与项目介绍尚不完整。
2. 已有公共模块包括 `model`、`data`、`preprocess`、`mAP`、`batch`、`vc`、`biological`，无需重新设计所有算法。
3. `06_vc`、`06_vc_compound`、`06_vc_new` 同时存在；`07_biological` 和 `07_biological_new` 也存在新旧版本。
4. 部分源码包含旧版文件及 `copy` 副本，需要核对当前入口后再确定正式版本。
5. 教程目录混有脚本、日志、缓存、模型检查点、AnnData 和作图结果，代码入口较难辨认。
6. 部分脚本依赖服务器绝对路径，以及 `LLLM`、`Final_well_level_new_LLM` 等其他项目中的资源。
7. 已有测试可以保留并利用；部分包目录的 `__init__.py` 和 `src.descort` / `descort` 导入方式需要统一。

## 3. 建议的目标目录

以下全部是**拟议结构**，不表示这些文件现在已经创建或已经可运行。

```text
DeSCORT/
├── README.md                         # 方法、安装、快速使用、任务导航
├── LICENSE                           # 许可证类型待确认
├── CITATION.cff                      # 作者及引用信息，按真实信息填写
├── pyproject.toml                    # 包元信息、依赖、构建配置
├── .gitignore                        # 忽略数据、模型、输出、缓存等
│
├── assets/
│   └── workflow.png                  # 已确认可用于发布的方法总览图
│
├── docs/
│   ├── index.md                      # 文档导航
│   ├── installation.md
│   ├── data_format.md                # parquet / Arrow / AnnData 格式
│   ├── resources.md                  # 词表、ESM2、Morgan、UniMol2 等
│   ├── model_zoo.md                  # 模型、适用任务、下载及配置说明
│   ├── workflows/                    # 各任务的输入、步骤和输出
│   ├── api/                          # 公共模块说明
│   └── reproducibility.md            # 参数、随机种子、指标口径
│
├── configs/
│   ├── paths.example.yaml            # 可共享的路径配置示例
│   ├── local/                        # 本机配置，不提交 Git
│   ├── pretrain/
│   ├── finetune/
│   ├── virtual_cell/
│   ├── biological/
│   └── deepspeed/                    # 汇总训练配置，减少重复副本
│
├── environments/
│   ├── environment.yml               # 主要环境
│   └── README.md                     # CUDA、DeepSpeed、可选依赖说明
│
├── src/descort/
│   ├── __init__.py
│   ├── arguments.py
│   ├── model/                        # 配置、模型、输出结构和损失
│   ├── data/                         # 数据处理器、collator、sampler
│   ├── preprocess/                   # 原始特征处理、MAD 等
│   ├── training/                     # 可复用训练辅助函数，逐步抽取
│   ├── embedding/                    # 公共 embedding 提取函数，逐步抽取
│   ├── mAP/                          # 先保留现有名字，避免无必要的改动
│   ├── batch/
│   ├── vc/                           # 虚拟细胞处理、训练辅助、推理和指标
│   └── biological/                   # UMAP、CORUM/STRING、聚类与富集绘图
│
├── tutorials/
│   ├── README.md                     # 选择任务与运行顺序
│   ├── 00_download/
│   ├── 01_preprocess_data/
│   ├── 02_pretrain/
│   ├── 03_finetune/                   # 包含 embedding 提取示例
│   ├── 04_mAP/
│   ├── 05_batch_effect/
│   ├── 06_virtual_cell/
│   │   ├── unseen_gene_generation/
│   │   │   ├── Crispr/               # 处理 → 训练 → 推理 → 指标
│   │   │   └── Compound/
│   │   ├── cell_line_transfer/
│   │   │   ├── Crispr/
│   │   │   └── Compound/
│   │   └── baselines/                # 基线入口、条件表示和环境说明
│   ├── 07_biological/
│   │   ├── 01_umap_analysis.py / .sh
│   │   ├── 02_CORUM_STRING.py / .sh
│   │   ├── 03_de_cluster.py / .sh
│   │   └── 04_cluster_enrichment.py / .sh
│   ├── 08_three_type/
│   └── 09_paper_plot_code/            # 论文图的复现入口
│
├── resources/
│   ├── README.md
│   ├── manifest.json                 # 资源来源、版本、形状、校验值
│   ├── feature_vocab/
│   ├── label_vocab/
│   ├── feature_sets/
│   └── perturbation_embeddings/      # 小文件或获取说明；大文件外部存放
│
├── tests/                            # 现有测试 + 关键流程的小规模检查
├── examples/                         # 最小示例数据、配置、期望输出说明
├── benchmarks/                       # 对照模型的适配代码与复现实验配置
└── outputs/                          # 本地可选输出根目录，不提交 Git
```

补充说明：

- `outputs/` 只是建议的本地默认位置；你现有的 `/fse/DC/...` 数据、检查点和结果路径可以继续使用。
- `configs/` 的具体读取方式在实施阶段确定；首阶段可以先保留 Bash 顶部的显式路径变量。
- `training/`、`embedding/` 只收纳实际共用函数，不为了目录完整而制造空层级。
- Sphinx / Read the Docs 可以像两个参考项目一样后续接入；第一阶段先让 Markdown 文档完整。

## 4. 现有内容如何对应到新框架

| 当前位置 | 建议归属 | 整理原则 |
|---|---|---|
| `src/descort/model/` | 同名公共模块 | 核对正式版本，保持模型权重和配置兼容 |
| `src/descort/modeling_descort.py`、模型 `copy` 文件 | 历史版本清单或归档 | 先查导入关系和行为差异，再决定是否退出主线 |
| `tutorials/00_download`、`01_preprocess_data` | 保留相应编号 | 补 README、输入格式和资源获取方式 |
| `tutorials/02_pretrain`、`03_finetune` | 保留相应编号 | 明确处理数据、训练和 embedding 提取三个阶段 |
| `tutorials/04_mAP`、`05_batch_effect` | 保留任务模块 | 统一输入、输出与结果解释 |
| `06_vc`、`06_vc_compound`、`06_vc_new` | `06_virtual_cell` | 先逐项核对流程、默认参数和数值，再确定统一主线 |
| `07_biological`、`07_biological_new` | `07_biological` | 以当前维护的脚本为候选入口，旧 notebook 留作追溯 |
| `08_three_type` | 保留独立任务目录 | 共用模型与数据函数，保留该任务的实验配置 |
| `09_paper_plot_code` | 论文作图教程 | 明确每张图读取哪个结果文件 |
| `common_files` | `resources` 或外部资源清单 | 区分词表、小型资源、大型 embedding 和生成脚本 |
| 各任务中的 GEARS / MLP / Ridge / STATE 等 | `benchmarks` + 教程入口 | 保留本地适配和环境；注明与原方法的差别 |
| 各目录下的 `result`、日志和检查点 | 本机输出路径 | 先建立索引，不直接移动现有结果 |
| `other_model/Cellist`、`DeSCOPE`、`Unimol2` | 本地参考或外部工具 | 发布仓库通过说明或获取脚本引用，不复制整份克隆仓库 |
| `src/tutorials` 等额外脚本副本 | 待核对清单 | 检查引用后再归类，不根据目录名直接删除 |

**新版目录不自动等于正确版本。** 特别是虚拟细胞流程，必须核对任务方向、条件表示、特征集合、训练阶段和评估设置。

## 5. Python、Bash 与公共函数的分工

继续遵守目前已经采用的方式：

- **教程 Python 保留 `main()`**：解析参数、组织步骤、保存结果。
- **公共函数放在 `src/descort`**：模型、数据处理、计算和绘图实现。
- **Bash 保留可编辑配置区**：绝对路径、显卡、batch size、epochs、任务参数。
- Notebook 用于交互探索和展示；正式流程提供可独立执行的 Python / Bash 入口。
- 后续安装目标是 `import descort`，逐步替换 `sys.path` 拼接及 `src.descort` 导入；过渡期间提供必要的旧入口兼容。

生物学分析的职责划分建议保持当前约定：

| 入口 | 负责内容 |
|---|---|
| `01_umap_analysis` | PCA/UMAP、通路叠加和距离比较；显式设置查询基因时，独立计算近邻并生成 figure 7 |
| `02_CORUM_STRING` | CORUM / STRING 相似性及背景校正分析 |
| `03_de_cluster` | 聚类、差异形态特征、富集检验、结果表及主要聚类分析图 |
| `04_cluster_enrichment` | 从富集表选择指定簇，筛选和绘制富集气泡图 |

## 6. 路径、资源与环境

### 路径

- 本机数据路径仍允许使用绝对路径，放在 Bash 顶部或本机路径配置中。
- 共享示例使用 `/absolute/path/to/...` 等明确占位符，文档说明每个路径对应什么。
- 先列出对其他项目目录的依赖，再决定迁入小型资源还是提供下载/生成步骤。
- 命令行覆盖 Bash 默认值的规则保持一致，路径中有空格时也应正确传参。

### 资源

每项资源至少记录：用途、来源、版本、对应词表、维度或数量、生成方法、校验值。

重点包括：

- 特征词表及各数据集特征集合。
- 基因/SMILES 与编号的双向映射。
- ESM2、Morgan、UniMol2 embedding 及对应条件维度。
- hit gene 列表、训练/测试扰动划分。
- CORUM、STRING、GO、KEGG 注释来源和快照版本。

已有映射编号、特征顺序和条件向量对应关系在整理时保持不变，不重新编号。

### 环境

- 先从当前能运行的环境整理精简依赖，再验证可安装性。
- 当前已有环境文件记录 Python 3.9、PyTorch 2.7.0、Transformers 4.51.3 等；这些是现有环境记录，不直接视为已验证的全部兼容版本范围。
- 不直接照搬 Cellist / DeSCOPE 的 Python 最低版本要求。
- PyTorch/CUDA 安装单独说明；DeepSpeed、R 批次校正、UniMol2 和基线模型的环境分别说明。
- 已有环境快照保留用于复现；正式依赖清单避免绑定整台服务器所有无关软件包。

## 7. 根目录 README 建议内容

1. DeSCORT 是什么、解决什么问题、方法总览图。
2. 支持的任务：预训练、表征微调、mAP、批次效应、虚拟细胞、生物学解释、跨扰动类型分析。
3. 安装及安装检查。
4. 一个小型端到端示例：处理输入 → 加载模型 → 提取 embedding。
5. 数据与预训练模型获取方式、词表和格式约定。
6. 教程导航及每个任务的运行顺序。
7. 输入/输出示例、常见参数、CPU/GPU 需求。
8. 论文实验复现配置与结果索引。
9. 引用、作者、许可证、致谢和问题反馈方式。

下载链接、论文信息、许可证和软件包发布状态按真实资料填写；尚未发布的内容标记为待补充。

## 8. 后续实施顺序

| 阶段 | 工作 | 验证内容 |
|---|---|---|
| 1. 建立清单 | 记录正式入口、旧副本、外部资源、重要结果和环境 | 明确现有流程到底调用哪些文件 |
| 2. 补项目外壳 | README、安装配置、文档导航、忽略规则 | 在隔离环境进行导入和安装检查 |
| 3. 统一教程 | 逐个任务合并重复入口，配置路径与参数，保留兼容入口 | 用同一小批数据比较处理结果和数值输出 |
| 4. 核对复现 | 对齐词表、条件向量、特征、检查点、指标定义和默认行为 | 沿用现有测试，补必要的端到端小样本检查 |
| 5. 准备发布 | 整理资源获取方式、模型说明、论文图入口 | 检查新用户能否按文档完成最小工作流 |

第一次整理不要求重跑全部训练，也不同时更改算法、损失函数和默认超参数。

## 9. 整理时保持的边界

- 不改变模型前向、训练损失、归一化和指标公式。
- 保持默认值和空参数含义可追溯，例如空文件列表处理所有 parquet、未设置查询基因则跳过近邻图。
- 原有模型、结果、随机划分和映射文件先保留，记录来源后再考虑归档。
- 不把整个第三方仓库、服务器原始数据或训练检查点直接加入发布仓库。
- `legacy`、`new`、`copy` 等命名最终由版本清单解释；不根据名字直接认定哪个文件可删除。
- 是否在原项目逐步整理，还是另外建立干净的发布副本，在审阅本框架后确定。

## 10. 本次交付与下一步

本次交付只有这份框架草案。

建议优先审阅三点：**教程任务的划分是否合适、是否采用独立发布副本、首批需要完整保留哪些任务**。框架确认后，再编写具体文件迁移清单并执行整理。
