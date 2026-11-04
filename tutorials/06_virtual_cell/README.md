# 虚拟细胞：未见基因扰动生成

采用原项目 `06_vc_new` 中已提取公共模块的四组任务入口，统一到本目录；不能仅因目录名含 new 就认为与所有历史实验等价。旧 `06_vc`、`06_vc_compound` 的探索、消融和结果仍在原项目，版本来源见整理清单。

| 任务 | CRISPR | Compound |
|---|---|---|
| 未见扰动生成 | `unseen_gene_generation/Crispr` | `unseen_gene_generation/Compound` |
| 跨细胞系迁移 | `cell_line_transfer/Crispr` | `cell_line_transfer/Compound` |

每组运行顺序：`01_process_data.py` → `02_finetune.py` → `03_inference.py` → `04_calculate_vcc2026_metric.py`。各组同时提供历史参数 Bash 和无默认参数的 `run.sh`。下文给出 CRISPR 未见基因生成的完整示例。

```bash
bash tutorials/06_virtual_cell/unseen_gene_generation/Crispr/run.sh 01_process_data.py --help
bash tutorials/06_virtual_cell/unseen_gene_generation/Compound/run.sh 03_inference.py --help
```

CRISPR 条件向量通常来自 ESM2；Compound 可使用 Morgan / UniMol2，通过文件内容及 `esm2_input_dim` 对齐，保留历史参数名称。控制标签、训练测试扰动列表、输入 feature axis、均值基线权重和指标族要分别记录。

跨细胞系任务需对目标细胞系单独准备数据，明确 few-shot 训练与测试的细胞系/plate 划分；目录名本身不自动实施数据隔离。未见 Compound 任务目录保留 historical `unseen_gene_generation` 命名，语义为未见化合物。

Compound 检查点必须保存 intersection feature axis；在测试集上重新推导交集会改变模型输入与评估含义。推理输出带来源元信息，指标入口默认检查特征轴与扰动集合。

## HeLa DMEM：unseen gene generation tutorial

本例参照原项目 `tutorials/06_vc_new/unseen_gene_generation/Crispr` 中的实际配置，使用 **20210422_6W_CP257 的 HeLa DMEM 数据**，在 50 个训练基因上微调，然后预测训练列表以外的基因扰动形态。按照“数据准备 → 微调 → 推理 → 指标评估 → 结果查看”的顺序运行；命令调用的是本仓库整理后的入口。

输入是归一化的 Cell Painting 形态特征和基因 ESM2 向量，输出是预测的扰动后形态特征，而不是基因表达矩阵。虽然原始数据目录包含 `well_level_new`，本例选中的文件名为 `single_cell_profiles`，下文统一称为“行/样本”，不把每行误认为一个独立 well。

| 步骤 | 入口 | 主要用途 |
|---|---|---|
| 1 | [01_process_data.py](unseen_gene_generation/Crispr/01_process_data.py) | 按训练/测试基因列表筛选，并配对同一文件中的对照，保存 Hugging Face Dataset |
| 2 | [02_finetune.sh](unseen_gene_generation/Crispr/02_finetune.sh) | 从已有检查点微调 CRISPR 扰动预测模型 |
| 3 | [03_inference.py](unseen_gene_generation/Crispr/03_inference.py) | 输出真实形态、预测形态和训练均值基线的 H5AD |
| 4 | [04_calculate_vcc2026_metric.py](unseen_gene_generation/Crispr/04_calculate_vcc2026_metric.py) | 检查对齐，计算模型和基线的七项指标 |

本例的训练与测试使用相同的五个 plate：`CP257A CP257B CP257D CP257F CP257H`，通过**非对照基因集合不重叠**定义 unseen；它不是跨 plate 或跨细胞系泛化。对照可以在两侧共享。这里的 unseen 仅说明本次微调的基因划分，不能据此断言初始化检查点及其全部上游训练从未见过这些基因。

> 以下命令用于重新运行完整实验。本次文档核对读取了现有资源、parquet 元数据和基因列，未重新执行完整训练、全量推理或指标计算，也没有可直接引用的该配置结果图；下文提供运行后的结果检查与绘图命令。

## Step 0　准备环境和输入

### 0.1 安装环境

在仓库根目录执行。PyTorch 的 CUDA 构建与本机驱动、DeepSpeed 编译环境的配置见[根目录安装说明](../../README.md#安装)。本例训练使用 DeepSpeed，数据准备和指标计算使用 CPU，推理默认使用一张 GPU。

```bash
cd /fse/home/wangyongyan/Project/CellPainting/DeSCORT/other_model/DeSCORT
conda activate DeSCORT
python -m pip install -e .
python -m pip install deepspeed==0.18.0
ds_report
```

这是完整数据示例：当前所选五个文件共约 197 万行，测试筛选后约 125 万行。数据处理、H5AD 导出和评估会占用较多内存与磁盘；推理脚本会将测试结果汇总到内存，调整 batch size 只减少每批的开销，不会消除最终矩阵的内存需求。

### 0.2 输入文件与格式

| 输入 | 本例要求 |
|---|---|
| 归一化形态 parquet | 包含 `Metadata_Foci_Barcode_MatchedTo_GeneCode`、形态数值列和 `nontargeting` 对照；对照必须在各个参与处理的文件内存在 |
| 训练基因列表 | `few_shot/gene_50_train.pkl`，当前含 50 个非对照基因和 1 个对照条目 |
| 测试基因列表 | `few_shot/gene_50_test.pkl`，当前含 13,803 个非对照候选基因和 1 个对照条目；实际出现的基因由所选 parquet 决定 |
| 基因编码 | `gene2index_21611.pkl` 和逆映射 `index2gene_21611.pkl` |
| ESM2 向量 | 基因名到向量的字典用于筛选；基因 ID 到向量的字典用于模型，两者必须使用相同编码、相同向量；本例宽度为 5,120 |
| 特征编码 | `feature2index_vocabulary_7366.pkl` 和逆映射 `index2feature_vocabulary_7366.pkl` |
| 所选形态特征 | `intersection_columns_1107.pkl`，与词表、各文件的实际列取交集，顺序沿用 parquet 列顺序 |
| 初始化检查点 | 原脚本指定的 A549 扰动预测 `models` 目录，包含 `config.json` 和权重文件；不是本例训练产生的 HeLa 检查点 |

基因列表、映射、ESM2、数据和模型权重需要单独准备，没有随 Python 安装包一起下载。下文使用当前服务器已有路径；在其他机器上请替换。

### 0.3 统一配置路径

在**同一个 Bash 会话**中执行本教程，后续命令沿用这些变量。重新运行实验时更换 `EXPERIMENT_NAME`，以便保存独立输出。

```bash
export REPO_ROOT="$PWD"
export CP_WORKSPACE="/fse/home/wangyongyan/Project/CellPainting"
export ORIGINAL_PROJECT="$CP_WORKSPACE/DeSCORT"
export DESCORT_RESOURCE_ROOT="$ORIGINAL_PROJECT/common_files"
export TUTORIAL_DIR="$REPO_ROOT/tutorials/06_virtual_cell/unseen_gene_generation/Crispr"
export PYTHON_BIN="python"
export DEEPSPEED_BIN="$(command -v deepspeed)"

export SOURCE_DATA="/fse/DC/CP_data/feature_data/20210422_6W_CP257/raw_data/well_level_new/normalization_clip_data"
export TRAIN_GENES="$DESCORT_RESOURCE_ROOT/06_vc/random_gene_list/few_shot/gene_50_train.pkl"
export TEST_GENES="$DESCORT_RESOURCE_ROOT/06_vc/random_gene_list/few_shot/gene_50_test.pkl"
export GENE2INDEX="$DESCORT_RESOURCE_ROOT/ESM2_files/gene2index_21611.pkl"
export INDEX2GENE="$DESCORT_RESOURCE_ROOT/label_vocab/cpg0021/index2gene_21611.pkl"
export GENE2ESM2="$DESCORT_RESOURCE_ROOT/ESM2_files/ESM2_pert_features_with_nontargeting.pkl"
export INDEX2ESM2="$DESCORT_RESOURCE_ROOT/ESM2_files/Index2ESM2_pert_features_with_nontargeting_19791.pkl"
export FEATURE2INDEX="$DESCORT_RESOURCE_ROOT/feature_vocab/feature2index_vocabulary_7366.pkl"
export INDEX2FEATURE="$DESCORT_RESOURCE_ROOT/feature_vocab/index2feature_vocabulary_7366.pkl"
export SELECTED_FEATURES="$CP_WORKSPACE/Final_well_level_new_LLM/common_files/04_new_data_generation/intersection_columns_1107.pkl"
export PRETRAINED_MODEL_PATH="/fse/DC/CP_data/feature_data/benchmark_MAD_output/All_MAD_model/paper_model/06_vc/unseen_gene/DeSCORT/A549_13k_5k_direct_mlm_mean1_dir1_marg01_3ep/models"

export EXPERIMENT_NAME="HeLa_DMEM_50genes_unseen_gene_generation_50epochs"
export EXAMPLE_OUTPUT="$REPO_ROOT/outputs/06_virtual_cell/unseen_gene_generation/Crispr/$EXPERIMENT_NAME"
export TRAIN_DATASET_PATH="$EXAMPLE_OUTPUT/data/train"
export TEST_DATASET_PATH="$EXAMPLE_OUTPUT/data/test"
export FINETUNE_OUTPUT="$EXAMPLE_OUTPUT/finetune"
export MODEL_PATH="$FINETUNE_OUTPUT/models"
export INFERENCE_OUTPUT_DIR="$EXAMPLE_OUTPUT/inference"
export METRIC_OUTPUT="$EXAMPLE_OUTPUT/metrics"
export GENE_COLUMN="Metadata_Foci_Barcode_MatchedTo_GeneCode"
export CONTROL_LABEL="nontargeting"
export ESM2_INPUT_DIM=5120
PLATES=(CP257A CP257B CP257D CP257F CP257H)
mkdir -p "$EXAMPLE_OUTPUT/logs"
```

`PRETRAINED_MODEL_PATH` 是步骤 2 的初始化模型，`MODEL_PATH` 是步骤 2 输出、步骤 3 加载的模型。不要混用。`DESCORT_RESOURCE_ROOT` 显式指向原项目的共享资源，所有新结果写入本仓库 `outputs`，不覆盖原实验目录。

### 0.4 检查基因划分、映射和实际特征轴

下面只读取 pickle、parquet schema 和基因列，不运行训练或加载完整形态矩阵。输出清单保存到 `input_check.json`，便于记录本次实际输入。它也补充了入口 `--check-only` 不会完成的基因不重叠及跨文件特征顺序检查。

```bash
"$PYTHON_BIN" - <<'PY'
import json
import os
import pickle
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq


def load(name):
    path = Path(os.environ[name])
    if not path.is_file():
        raise FileNotFoundError(f'{name}: {path}')
    with path.open('rb') as handle:
        return pickle.load(handle)


control = os.environ['CONTROL_LABEL']
gene_column = os.environ['GENE_COLUMN']
gene2index, index2gene = load('GENE2INDEX'), load('INDEX2GENE')
gene2esm2, index2esm2 = load('GENE2ESM2'), load('INDEX2ESM2')
feature2index, index2feature = load('FEATURE2INDEX'), load('INDEX2FEATURE')
selected = set(load('SELECTED_FEATURES'))
splits = {}
for split, name in [('train', 'TRAIN_GENES'), ('test', 'TEST_GENES')]:
    genes = list(load(name))
    if len(genes) != len(set(genes)):
        raise ValueError(f'{split}: 基因列表有重复条目')
    effective = [g for g in genes if g in gene2esm2 and g != control]
    if not effective:
        raise ValueError(f'{split}: 没有具备 ESM2 的非对照基因')
    for gene in effective:
        idx = gene2index[gene]
        if index2gene[idx] != gene:
            raise ValueError(f'基因映射不互逆: {gene}')
        a, b = np.asarray(gene2esm2[gene]), np.asarray(index2esm2[idx])
        if a.shape != (int(os.environ['ESM2_INPUT_DIM']),) or not np.array_equal(a, b):
            raise ValueError(f'ESM2 维度或两套映射不一致: {gene}')
    splits[split] = set(effective)
overlap = splits['train'] & splits['test']
if overlap:
    raise ValueError(f'训练/测试非对照基因重叠: {sorted(overlap)[:10]}')

source = Path(os.environ['SOURCE_DATA'])
if not source.is_dir():
    raise NotADirectoryError(source)
files = sorted(source.rglob('*.parquet'))
plates = ['CP257A', 'CP257B', 'CP257D', 'CP257F', 'CP257H']
records, reference_axis = [], None
seen = {'train': set(), 'test': set()}
for plate in plates:
    matched = [p for p in files if plate in str(p)]
    if len(matched) != 1:
        raise ValueError(f'{plate}: 预期 1 个 parquet，实际 {len(matched)} 个')
    path = matched[0]
    parquet = pq.ParquetFile(path)
    axis = [n for n in parquet.schema_arrow.names
            if not n.startswith('Metadata') and n in selected and n in feature2index]
    if not axis or (reference_axis is not None and axis != reference_axis):
        raise ValueError(f'{plate}: 特征轴为空或顺序与其他文件不同')
    reference_axis = axis
    if any(index2feature[feature2index[n]] != n for n in axis):
        raise ValueError(f'{plate}: 特征映射不互逆')
    labels = parquet.read(columns=[gene_column]).column(0).to_pandas()
    counts = {split: int(labels.isin(genes).sum()) for split, genes in splits.items()}
    counts['controls'] = int(labels.eq(control).sum())
    if counts['controls'] == 0:
        raise ValueError(f'{plate}: 缺少 {control} 对照')
    for split in splits:
        seen[split].update(set(labels.dropna()) & splits[split])
    records.append(dict(plate=plate, path=str(path), features=len(axis), **counts))

model = Path(os.environ['PRETRAINED_MODEL_PATH'])
if not (model / 'config.json').is_file():
    raise FileNotFoundError(f'初始化模型缺少 config.json: {model}')
if not (list(model.glob('*.safetensors')) or list(model.glob('pytorch_model*.bin'))):
    raise FileNotFoundError(f'初始化模型缺少权重: {model}')
report = dict(plates=records,
              gene_list_counts={k: len(v) for k, v in splits.items()},
              observed_gene_counts={k: len(v) for k, v in seen.items()},
              selected_feature_entries=len(selected), actual_features=len(reference_axis),
              feature_names=reference_axis)
output = Path(os.environ['EXAMPLE_OUTPUT']) / 'input_check.json'
output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
print(json.dumps({k: v for k, v in report.items() if k != 'feature_names'}, indent=2))
print('Input check saved:', output)
PY
```

当前服务器数据的只读核对结果如下；它不是模型性能结果，也不代表其他版本输入的固定样本数：

| 项目 | 当前值 |
|---|---:|
| 训练非对照基因列表 / 实际出现基因 | 50 / 50 |
| 测试非对照基因列表 / 实际出现基因 | 13,803 / 12,965 |
| 所选文件中的训练扰动行 | 8,885 |
| 所选文件中的测试扰动行 | 1,251,580 |
| 五个文件中原始对照行总数 | 13,620 |
| 特征列表条目 / 实际共同形态特征 | 1,107 / 1,106 |

文件名中的 `1107` 是特征列表的名称，**不保证最终有 1,107 个输入特征**。本例五个文件实际有相同顺序的 1,106 个有效特征；训练与推理仍沿用原配置的 `max_length=1107`，多出的位置用 padding 和 attention mask 处理。不能在测试阶段重新选择或重排特征。

## Step 1　构建训练与测试 Dataset

### 1.1 先检查入口参数与资源

用 Bash 数组保存整组参数，后续检查和正式处理共用同一配置：

```bash
PROCESS_ARGS=(
  --source_data_paths "$SOURCE_DATA"
  --train_output_dir "$EXAMPLE_OUTPUT/tmp_train"
  --train_save_dir "$TRAIN_DATASET_PATH"
  --test_output_dir "$EXAMPLE_OUTPUT/tmp_test"
  --test_save_dir "$TEST_DATASET_PATH"
  --gene2index_pkl "$GENE2INDEX"
  --esm2_dict_pkl "$GENE2ESM2"
  --feature2index_pkl "$FEATURE2INDEX"
  --train_gene_list_pkl "$TRAIN_GENES"
  --test_gene_list_pkl "$TEST_GENES"
  --need_feature_list_pkl "$SELECTED_FEATURES"
  --key_gene_column "$GENE_COLUMN"
  --control_key_word "$CONTROL_LABEL"
  --train_data_names "${PLATES[@]}"
  --test_data_names "${PLATES[@]}"
  --nproc 1 --merge_nproc 2 --verbosity 1
)
bash "$TUTORIAL_DIR/run.sh" 01_process_data.py "${PROCESS_ARGS[@]}" --check-only
```

`--check-only` 会报告 ESM2 筛选后的列表条目数、所选特征列表长度及输出路径；列表条目数可能包含 `nontargeting`。它不扫描全部数据、不自动证明训练测试基因不重叠，也不代表实际输出样本数，需结合步骤 0.4。

### 1.2 正式处理

```bash
bash "$TUTORIAL_DIR/run.sh" 01_process_data.py "${PROCESS_ARGS[@]}"
```

处理器按列表筛选非对照扰动行，对每个基因从**同一个 parquet** 的对照行中随机配对；每个文件使用固定随机种子 42，对照不足时允许有放回抽样。对照用于提供模型输入，不作为 `nontargeting` 训练目标行另行输出。各文件的 Dataset 分片保存在独立临时子目录，再分别合并到训练、测试目录；形态缺失值按当前实现填 0。

主要输出：

| 路径 | 含义 |
|---|---|
| `data/train/` | 训练 Dataset，可用 `datasets.load_from_disk` 读取 |
| `data/test/` | 测试 Dataset，基因集合与训练侧分离 |
| `tmp_train/perturb_*/`、`tmp_test/perturb_*/` | 本次处理的临时分片，正式实验保留处理记录后可另行管理 |

路径相对于 `$EXAMPLE_OUTPUT`。每行字段为 `Perturb_Gene`、`Perturb_expression`、`Control_expression`、`Feature_name_index`、`Perturb_Gene_Index`；前者是基因名称，最后一个是模型使用的基因 ID。

### 1.3 检查实际输出

```bash
"$PYTHON_BIN" - <<'PY'
import os
from datasets import load_from_disk

splits = {}
for name, env in [('train', 'TRAIN_DATASET_PATH'), ('test', 'TEST_DATASET_PATH')]:
    ds = load_from_disk(os.environ[env])
    labels = set(ds.with_format(None)['Perturb_Gene'])
    if os.environ['CONTROL_LABEL'] in labels:
        raise ValueError(f'{name}: 对照不应成为独立预测目标')
    first = ds.with_format(None)[0]
    print(name, 'rows=', len(ds), 'genes=', len(labels),
          'features=', len(first['Feature_name_index']))
    assert len(first['Perturb_expression']) == len(first['Control_expression']) == len(first['Feature_name_index'])
    splits[name] = labels
assert not (splits['train'] & splits['test']), '实际 Dataset 的训练/测试基因重叠'
PY
```

## Step 2　在 50 个训练基因上微调

使用整理后的 `02_finetune.sh`，显式覆盖原脚本的输入、输出和 GPU 设置。下面的 `0,1` 是物理 GPU 编号，应改成当前分配给本任务的 GPU；原脚本默认 `0,7`，不要直接沿用。

```bash
GPU_IDS=0,1 \
EPOCHS=50 \
PER_DEVICE_TRAIN_BATCH_SIZE=64 \
LEARNING_RATE=2e-4 \
OUTPUT_DIR="$FINETUNE_OUTPUT" \
bash "$TUTORIAL_DIR/02_finetune.sh" --overwrite_output_dir False
```

脚本从前面导出的 `TRAIN_DATASET_PATH`、`PRETRAINED_MODEL_PATH`、`DESCORT_RESOURCE_ROOT` 和 `DEEPSPEED_BIN` 读取配置，使用本仓库的 [ZeRO-2 配置](../../configs/deepspeed/zero2.json)。这里追加 `--overwrite_output_dir False`，防止复用非空训练目录；需要重跑时使用新的实验名。该入口从头开始本次微调，不会自动恢复中断的 checkpoint。

本例沿用的主要配置：

| 参数 | 值 / 含义 |
|---|---|
| 形态编码器 | hidden size 256，8 层，8 个 attention heads |
| MoE | 8 个 routed experts，每 token 选 4 个，1 个 shared expert |
| ESM2 输入 | 5,120 维，不启用额外降维和输入归一化 |
| 特征序列 | 最大长度 1,107；`sampling=False`、padding 值 100 |
| 输出头 | `prediction_head_type=mlm`，`predict_residual=False` |
| 预测语义 | 直接预测扰动后绝对形态，不把预测值再加到对照上 |
| 损失权重 | mean 1.0、direction 1.0、marginal quantile 0.1；不启用 global delta prior |
| 优化 | seed 42、50 epochs、每设备 batch 64、学习率 `2e-4`、cosine、warmup ratio 0.1 |
| 精度与存储 | `bf16=True`；每 epoch 保存 checkpoint，最多保留 5 个 |

默认按同基因组织训练 batch；测试 Dataset 不传入训练，也没有在每轮训练中用于挑选最优 checkpoint。最终模型保存在 `$FINETUNE_OUTPUT/models/`，Trainer 状态和 TensorBoard 日志等位于同一训练输出目录。GPU 不支持 bf16 时，可在命令末尾追加 `--bf16 False`，但精度、显存和速度将与本例配置不同。

## Step 3　对未见基因生成扰动形态

### 3.1 加载本例模型并推理

```bash
CUDA_VISIBLE_DEVICES=0 \
bash "$TUTORIAL_DIR/run.sh" 03_inference.py \
  --pretrained_model_path "$MODEL_PATH" \
  --index2gene_pkl "$INDEX2GENE" \
  --index2esm2_pkl "$INDEX2ESM2" \
  --index2feature_pkl "$INDEX2FEATURE" \
  --train_dataset_path "$TRAIN_DATASET_PATH" \
  --test_dataset_path "$TEST_DATASET_PATH" \
  --output_dir "$INFERENCE_OUTPUT_DIR" \
  --prediction_filename prediction_data.h5ad \
  --ground_truth_filename ground_truth.h5ad \
  --mean_baseline_filename perturb_mean_data.h5ad \
  --device cuda:0 --batch_size 64 --max_length 1107 \
  --pad_value 100 --keep_first_n_tokens 0 --no_sampling \
  --dataloader_num_workers 4 --dataloader_prefetch_factor 2 \
  --dataloader_pin_memory --dataloader_persistent_workers \
  --gene_column Perturb_Gene --control_label "$CONTROL_LABEL" \
  --log_file "$EXAMPLE_OUTPUT/logs/inference.log" --log_level INFO
```

`CUDA_VISIBLE_DEVICES=0` 选择物理 GPU，`--device cuda:0` 指可见 GPU 中的逻辑编号。CPU 调试时可将后者改为 `cpu`，不建议用 CPU 完成本例的全量推理。

模型根据对照形态和测试基因对应的 ESM2 条件向量生成预测；测试扰动真实形态用于输出 ground truth，不作为模型生成结果的输入。推理时根据 attention mask 去除 padding，并沿用测试 Dataset 的实际特征名称。

### 3.2 三份 H5AD 的含义

| 文件 | `X` 的内容 |
|---|---|
| `ground_truth.h5ad` | 测试扰动真实形态 + 逐行配对的对照形态 |
| `prediction_data.h5ad` | 测试扰动模型预测 + 相同的配对对照形态 |
| `perturb_mean_data.h5ad` | 重复的训练扰动均值向量 + 相同的配对对照形态 |

均位于 `$INFERENCE_OUTPUT_DIR`。标签保存在 `obs['Perturb_Gene']`，形态特征名称保存在 `var_names`；数据写在 `X`，本例不指定 layer。每份输出先排列所有测试扰动行，再追加同样数量的配对对照行；因对照会重复采样，追加后的行数不能当作独立对照数量。

训练均值基线先对每个训练基因的全部扰动行求均值，再对这些基因均值**等权平均**；不是对所有训练行直接求均值，也不使用测试扰动真实值计算。三份文件中的对照来自相同测试配对输入，并非模型生成的对照预测。

## Step 4　计算七项 Cell Painting 指标

本例调用仓库中的 VCC2026 风格指标实现，适配的是带正负值的连续形态特征。DE 使用 Wilcoxon/Mann–Whitney 检验和 BH 校正；方向/效应量采用扰动相对对照的均值差，不能把名称中的 `lfc` 解释成对形态值取 log2 fold change。

### 4.1 先检查 H5AD 对齐

```bash
METRIC_ARGS=(
  --ground_truth_path "$INFERENCE_OUTPUT_DIR/ground_truth.h5ad"
  --prediction_path "$INFERENCE_OUTPUT_DIR/prediction_data.h5ad"
  --baseline_path "$INFERENCE_OUTPUT_DIR/perturb_mean_data.h5ad"
  --outdir "$METRIC_OUTPUT"
  --cache_dir "$EXAMPLE_OUTPUT/de_cache"
  --pert_col Perturb_Gene --control_name "$CONTROL_LABEL"
  --dedup_controls true --strict_alignment true --copy_truth_controls false
  --fdr_scope per_pert --n_jobs 8 --blas_threads 4 --scipy_checks 100
  --reuse_cache true --show_progress true
)
bash "$TUTORIAL_DIR/run.sh" 04_calculate_vcc2026_metric.py "${METRIC_ARGS[@]}" --check-only
```

检查特征名称及顺序、样本 ID/顺序、逐行扰动标签与扰动集合；不自动重排或删列。CRISPR 导出文件本身不含完整训练溯源信息，因此仍需保存步骤 0 的输入记录和训练日志。`--check-only` 会写入检查记录并执行小型合成 MWU 验证，但不会计算实际预测指标。

### 4.2 正式评估模型和基线

```bash
bash "$TUTORIAL_DIR/run.sh" 04_calculate_vcc2026_metric.py "${METRIC_ARGS[@]}"
```

默认对完全相同的对照向量去重，在全部测试扰动和全部实际特征上计算；`fdr_scope=per_pert` 表示每个扰动内跨特征进行多重检验校正。DE 缓存按输入内容与设置校验后复用；改变数据或校正方式会生成不同缓存。

| 指标 | 含义 / 解读 |
|---|---|
| `pds_cosine` | 用形态扰动效应的余弦距离检索真实扰动，分数越高越好 |
| `mse` | 扰动均值形态的均方误差，越低越好 |
| `expr_mse_unbiased_capped_norm` | 校正采样噪声并归一化、截断后的误差，越低越好 |
| `de_wilcoxon_direction_fidelity_yield_raw` | 对显著形态效应的方向一致性与产出衡量，越高越好 |
| `de_wilcoxon_direction_reach_raw` | 在方向一致性要求下的效应覆盖范围，越高越好 |
| `de_wilcoxon_sig_jaccard` | 真实/预测显著形态特征集合的 Jaccard，越高越好 |
| `de_wilcoxon_lfc_nmae` | 对显著形态效应的归一化绝对误差，越低越好 |

这些是原始指标，不是经过 replicate 上界和基线缩放后的统一分数。除归一化 MSE 使用实现中的聚合比值外，汇总主要对各指标的有效扰动值取均值；需同时看 `valid_metric_gene_counts`，缺少有效显著特征时部分指标可能为 NaN，不能一律当成 0。

主要输出相对于 `$METRIC_OUTPUT`：

| 文件 | 含义 |
|---|---|
| `input_provenance.json`、`cli_parameters.json` | 输入检查与本次评估参数 |
| `model_metrics.csv`、`model.parquet` | 模型七项原始汇总指标 |
| `cell_eval/results.csv` | 模型逐扰动指标 |
| `cell_eval/agg_results.csv` | 模型聚合统计 |
| `run_manifest.json` | 实际基因/特征数、对照去重数量、缓存及耗时、有效指标基因数 |
| `baseline/baseline_metrics.csv`、`baseline/baseline.parquet` | 训练均值基线指标 |
| `baseline/run_manifest.json` | 基线评估记录 |
| `model_vs_baseline.csv` | 模型与基线的七项汇总指标对照 |

## Step 5　查看结果与生成对照图

完成步骤 4 后，查看汇总表与实际评估范围：

```bash
"$PYTHON_BIN" - <<'PY'
import json
import os
from pathlib import Path
import pandas as pd

folder = Path(os.environ['METRIC_OUTPUT'])
print(pd.read_csv(folder / 'model_vs_baseline.csv', index_col=0).to_string())
manifest = json.loads((folder / 'run_manifest.json').read_text())
for key in ['n_perturbations', 'n_features', 'counts', 'valid_metric_gene_counts']:
    print(key, ':', manifest[key])
PY
```

下面为每个指标分别绘制模型与训练均值基线，纵轴采用各指标自己的原始量纲。图片由本次实际结果生成，不预填示例性能数值。

```bash
MPLBACKEND=Agg "$PYTHON_BIN" - <<'PY'
import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

root = Path(os.environ['EXAMPLE_OUTPUT'])
frame = pd.read_csv(Path(os.environ['METRIC_OUTPUT']) / 'model_vs_baseline.csv', index_col=0)
fig, axes = plt.subplots(2, 4, figsize=(18, 8))
lower_better = {'mse', 'expr_mse_unbiased_capped_norm', 'de_wilcoxon_lfc_nmae'}
for ax, (metric, row) in zip(axes.flat, frame.iterrows()):
    values = row[['model', 'baseline']].to_numpy(dtype=float)
    ax.bar(['DeSCORT', 'Train mean'], values, color=['#3575A5', '#A7A7A7'])
    direction = 'lower is better' if metric in lower_better else 'higher is better'
    ax.set_title(metric.replace('_', ' ') + '\n' + direction, fontsize=9, wrap=True)
    ax.set_ylabel('Raw metric')
    for i, value in enumerate(values):
        if np.isfinite(value):
            ax.annotate(f'{value:.4g}', (i, value), xytext=(0, 5),
                        textcoords='offset points', ha='center', fontsize=9)
        else:
            ax.text(i, 0, 'NaN', ha='center')
for ax in axes.flat[len(frame):]:
    ax.set_visible(False)
fig.tight_layout()
output = root / 'plots'
output.mkdir(parents=True, exist_ok=True)
fig.savefig(output / 'model_vs_train_mean.png', dpi=200, bbox_inches='tight')
fig.savefig(output / 'model_vs_train_mean.pdf', bbox_inches='tight')
plt.close(fig)
print('Plots saved:', output)
PY
```

整套实验的新输出结构为：

```text
outputs/06_virtual_cell/unseen_gene_generation/Crispr/<EXPERIMENT_NAME>/
├── input_check.json
├── data/{train,test}/
├── tmp_train/、tmp_test/
├── finetune/models/
├── inference/{ground_truth,prediction_data,perturb_mean_data}.h5ad
├── logs/inference.log
├── de_cache/
├── metrics/model_vs_baseline.csv
└── plots/model_vs_train_mean.{png,pdf}
```

如需切换到其他基因划分，需同步更换 `TRAIN_GENES`、`TEST_GENES` 和实验名，从步骤 0 重新检查并构建 Dataset；不能只修改输出目录名称。Compound 和跨细胞系迁移仍使用各自任务入口，其条件向量、对照标签与训练/测试定义不同，不直接套用本例的基因映射。
