# DeSCORT

面向 Cell Painting 形态特征的表征学习与扰动预测代码。模型通过形态值与特征名称编码、带 `[CLS]` 的 Transformer/MoE 编码器，以及重建和对比学习任务学习形态表征，支持后续检索、批次效应分析、虚拟细胞生成和生物学解释。

![DeSCORT 模型架构](assets/workflow.png)

这是按照 [整理框架草稿](DeSCORT_repository_framework_draft.md) 建立的**独立整理副本**。原项目位于 `../..`；原始代码、数据、模型与结果保持原位。整理与验证范围见 [整理说明](docs/reorganization/README.md)。

## 安装

在本目录执行，使用项目已有的 DeSCORT 环境或 Python ≥ 3.9 环境：

```bash
python -m pip install -e '.[train,preprocess,vc,analysis]'
python -c "import descort; print(descort.__version__)"
```

GPU 训练先准备匹配本机 CUDA 的 PyTorch；DeepSpeed 单独安装 `.[distributed]`。批次校正的 R/TensorFlow 等后端单独管理，见 [安装文档](docs/installation.md)。本次实际验证使用已有 Python 3.9.15 / PyTorch 2.7.0 / Transformers 4.51.3 环境，版本记录见 [environments](environments/README.md)。

## 最小可运行示例

```bash
python examples/minimal_embedding.py --output-dir outputs/minimal_embedding
```

该示例生成 3 行合成 parquet、一个随机初始化的小模型，并通过正式 embedding 入口保存形态表征；在 CPU 上检查输出形状、有限值和元数据。**它验证软件链路，不代表训练后的模型效果。** 再次运行请选择新的空输出目录。

真实模型的基本用法：

```python
from descort import DeSCORTForContrastiveLM
model = DeSCORTForContrastiveLM.from_pretrained('/absolute/path/to/models')
model.eval()
# 输入字段与提取入口见 docs/data_format.md 和 docs/api/index.md。
```

## 任务导航

| 任务 | 入口 | 输入 → 输出 |
|---|---|---|
| 数据下载 | [00_download](tutorials/00_download/README.md) | 数据源 → 原始文件 |
| 预处理与词表 | [01_preprocess_data](tutorials/01_preprocess_data/README.md) | 原始特征 → 归一化 parquet、映射与特征集合 |
| 预训练 | [02_pretrain](tutorials/02_pretrain/README.md) | parquet → Arrow dataset → 基础模型 |
| 表征微调与 embedding | [03_finetune](tutorials/03_finetune/README.md) | 基础模型、标注数据 → 表征模型与 embedding |
| mAP | [04_mAP](tutorials/04_mAP/README.md) | embedding → 检索指标 |
| 批次效应 | [05_batch_effect](tutorials/05_batch_effect/README.md) | AnnData → 校正结果和指标 |
| 虚拟细胞 | [06_virtual_cell](tutorials/06_virtual_cell/README.md) | 对照形态、扰动向量 → 扰动形态预测 |
| 生物学分析 | [07_biological](tutorials/07_biological/README.md) | 形态表征、注释 → 通路、聚类、富集分析 |
| 跨扰动类型 | [08_three_type](tutorials/08_three_type/README.md) | CRISPR / Compound / ORF → 共享表征分析 |
| 论文作图 | [09_paper_plot_code](tutorials/09_paper_plot_code/README.md) | 已有结果表 → 论文图 |

每个主任务提供 `run.sh SCRIPT.py [参数...]`；数据路径和训练参数仍由 Bash/命令行显式指定。详见 [教程总览](tutorials/README.md)。

## 配置、资源与复现

- [路径与运行配置](configs/README.md)：可复制到 `configs/local/`，先 `--dry-run` 查看完整命令。
- [资源清单](resources/README.md)：原项目资源只读复用；不复制大型 embedding、数据与权重。
- [输入格式](docs/data_format.md)、[模型与代码对应](docs/model_architecture.md)、[复现约定](docs/reproducibility.md)。
- [基线索引](benchmarks/README.md) 保留各方法原适配位置与环境说明，不把第三方仓库一起打包。
- 测试：`python -m unittest discover -s tests -v`。

包版本 `0.1.0` 是本地整理版本，尚不表示已向 PyPI 发布。论文 DOI、作者引用信息、公开权重下载地址和许可证尚未在提供的项目材料中确认，因此不生成虚构的 `CITATION.cff` 或许可证文件。参考组织方式来自本地 DeSCOPE 与 Cellist；DeSCORT 模型实现保持自身代码。
