# DeSCORT

面向 Cell Painting 形态特征的表征学习与扰动预测代码。模型通过形态值与特征名称编码、带 `[CLS]` 的 Transformer/MoE 编码器，以及重建和对比学习任务学习形态表征，支持后续检索、批次效应分析、虚拟细胞生成和生物学解释。

![DeSCORT 模型架构](assets/workflow.png)

源代码与任务教程见 [GitHub 仓库](https://github.com/wyywyy999/DeSCORT_script)，当前测试包为 [TestPyPI 上的 descort 0.1.3](https://test.pypi.org/project/descort/0.1.3/)。

## 安装

下面按“创建环境 → 安装 PyTorch → 按需安装 DeepSpeed → 安装 DeSCORT → 检查安装”的顺序操作。完成前置步骤后，在 TestPyPI 和 GitHub 两种安装方式中选择一种即可。

当前代码在 Linux 上验证，要求 **Python ≥ 3.9**。已使用的核心版本为 Python **3.9.15**、PyTorch **2.7.0**、Transformers **4.51.3**；使用 GPU 训练时，还需要兼容的 NVIDIA 驱动。

### Step 1：创建 Python 环境

建议使用独立的 Conda 环境。以下以 Python 3.9 为例：

```bash
conda create -n DeSCORT python=3.9 pip -y
conda activate DeSCORT
python -m pip install --upgrade pip setuptools wheel
```

### Step 2：安装 PyTorch

根据机器的 NVIDIA 驱动支持情况选择 PyTorch 的 CUDA 构建。以下是 PyTorch **2.7.0** 的安装示例，**选择其中一种执行**；完整平台支持与命令见 [PyTorch 官方安装说明](https://pytorch.org/get-started/previous-versions/)。

CUDA 11.8：

```bash
python -m pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cu118
```

CUDA 12.6：

```bash
python -m pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cu126
```

CUDA 12.8：

```bash
python -m pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cu128
```

仅使用 CPU：

```bash
python -m pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cpu
```

安装后检查版本及 GPU 是否可用：

```bash
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA build:', torch.version.cuda); print('GPU available:', torch.cuda.is_available())"
```

CPU 版本的 `GPU available` 为 `False` 是正常的；GPU 训练环境需要能检测到可用 GPU。

### Step 3：安装 DeepSpeed（分布式训练时使用）

使用教程中以 `deepspeed` 启动的预训练或微调脚本时，需要先安装 DeepSpeed。以下使用本项目已使用的 **0.18.0** 版本：

```bash
python -m pip install deepspeed==0.18.0
ds_report
```

DeepSpeed 的部分 C++/CUDA 算子会在使用时编译，需要对应的编译工具链。若使用 CUDA 算子，应核对本机 CUDA Toolkit 与 PyTorch 的 CUDA 构建版本；详情见 [DeepSpeed 官方安装说明](https://www.deepspeed.ai/tutorials/advanced-install/)。运行单卡常规推理或 CPU 分析可以跳过本步骤。仓库中的分布式训练配置位于 [configs/deepspeed/zero2.json](configs/deepspeed/zero2.json)。

### Step 4：安装 DeSCORT

#### 方式一：通过包索引安装（当前使用 TestPyPI）

目前 `descort==0.1.3` 已发布到 **TestPyPI 测试站**，尚未发布到正式 PyPI。先只从 TestPyPI 下载 DeSCORT 的 wheel，再从正式 PyPI 安装依赖。pip 同时使用多个索引时不会按主、备用顺序选择依赖，而会在所有索引中选择满足约束的候选版本；此前的双索引命令可能选到测试站上的同名依赖，例如 `python-dateutil==3.9.0`。参见 [pip 官方说明](https://pip.pypa.io/en/stable/cli/pip_install/#finding-packages)。

```bash
python -m pip download --no-deps --only-binary=:all: \
  --index-url https://test.pypi.org/simple/ \
  --dest ./descort_download \
  descort==0.1.3

python -m pip install \
  --index-url https://pypi.org/simple/ \
  ./descort_download/descort-0.1.3-py3-none-any.whl \
  'python-dateutil==2.9.0.post0'
```

从 0.1.1 起，默认一起安装训练、预处理、虚拟细胞和下游分析依赖，无需填写方括号中的依赖组。DeepSpeed 已在步骤 3 单独处理。0.1.3 增加 `python-dateutil>=2.9.0.post0,<3` 约束，避免选中与 botocore 冲突的 3.x 版本。

如果已按旧命令安装并出现 `botocore` 与 `python-dateutil` 冲突，先在同一个 DeSCORT 环境中执行：

```bash
python -m pip install --index-url https://pypi.org/simple/ 'python-dateutil==2.9.0.post0'
python -m pip check
```

若 `pip check` 仍列出冲突，需要根据完整输出进一步处理。NumPy 从 2.x 降为 1.26.4 符合本项目 `numpy>=1.23,<2` 的约束，不是这里的报错原因。环境快照与发布依赖的区别和已核实冲突见 [依赖核查报告](environments/dependency-audit-0.1.2.md)。

如果使用的是依赖已经准备好的环境，只安装测试包本身，可执行：

```bash
python -m pip install \
  --index-url https://test.pypi.org/simple/ \
  --no-deps descort==0.1.3
```

`--no-deps` 会跳过依赖安装，仅适用于已经具备所需依赖的环境。上述包索引安装提供 Python 包；若要运行仓库中的完整教程和 Bash 脚本，使用下面的 GitHub 方式获取源码。模型权重、词表及实验数据仍需按任务准备。

#### 方式二：从 GitHub 克隆并安装

先准备 Git，再克隆仓库并在仓库根目录安装：

```bash
git clone https://github.com/wyywyy999/DeSCORT_script.git
cd DeSCORT_script
python -m pip install .
```

源码 0.1.3 默认一起安装训练、预处理、虚拟细胞和下游分析依赖，无需填写依赖组。如果仓库为私有仓库，克隆时需要使用具有访问权限的 GitHub 账号完成 Git 认证。

需要修改源码并持续开发时，将上面的安装命令换成可编辑安装：

```bash
python -m pip install -e .
```

`.` 表示当前仓库；`-e` 让环境关联当前源码目录。修改源码后重新启动 Python 程序即可使用修改后的代码，不必每次重新安装。如果当前环境的依赖已经准备好，只需关联源码，可以执行 `python -m pip install -e . --no-deps`。

### Step 5：检查安装

两种安装方式都可以使用同一组检查命令：

```bash
python -c "import descort; from descort import DeSCORTConfig, DeSCORTForContrastiveLM; print('DeSCORT:', descort.__version__); print('Package path:', descort.__file__)"
python -m pip show descort
```

需要确认 DeepSpeed 环境时，另行运行 `ds_report`。批次校正中的 R、scIB、DESC/TensorFlow 等后端按对应任务准备，见 [安装文档](docs/installation.md)。

`environments/environment.yml` 和 `environments/requirement.txt` 记录了原服务器的完整环境，其中包含额外工具及已知版本约束冲突，并不是上面两种安装方式的必经步骤；本节通过项目声明的依赖安装 DeSCORT。

## 加载已有模型

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

## 文档、结果图与复现

- [输入格式](docs/data_format.md)、[模型与代码对应](docs/model_architecture.md)、[复现约定](docs/reproducibility.md)。
- [环境记录](environments/README.md) 与 [完整环境依赖清单](environments/requirement.txt)。
- [教程结果图](result/README.md)：按任务与示例数据集集中保存预览图和来源说明。

当前测试发布版本为 `0.1.3`，发布地址为 TestPyPI。论文 DOI、作者引用信息、公开权重下载地址和许可证尚未在提供的项目材料中确认，因此不生成虚构的 `CITATION.cff` 或许可证文件。参考组织方式来自本地 DeSCOPE 与 Cellist；DeSCORT 模型实现保持自身代码。
