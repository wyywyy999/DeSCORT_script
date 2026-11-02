# 安装与环境

在新仓库根目录运行命令；原项目不需要改动。

```bash
python -m pip install -e .                         # 模型、数据处理基础依赖
python -m pip install -e '.[train,preprocess,vc,analysis]'
python -c 'from descort import DeSCORTConfig, DeSCORTForContrastiveLM; print(DeSCORTConfig().model_type)'
```

可选依赖组：`train` 用于 Transformers Trainer；`preprocess` 用于 pycytominer；`vc` 用于 AnnData 和虚拟细胞指标；`analysis` 用于 mAP、Scanpy、生物学分析；`distributed` 用于 DeepSpeed。保持 Transformers 4.51 系列，以配合当前模型接口。

批次校正需要按方法准备 `bbknn`、`harmonypy`、`scanorama`、`mnnpy`、`desc`；scIB 评估需要 `scib`，kBET 还依赖 R / rpy2 等原实验环境。它们不作为基础安装的强制依赖。各适配器只在运行对应方法时加载相应后端。

本机可复用已有环境：

```bash
conda activate DeSCORT
python -m unittest discover -s tests -v
```

隔离安装可使用 `python -m venv --system-site-packages /tmp/descort-check`，再用该环境的 pip 安装本仓库。无网络且构建依赖已满足时，可用 `--no-build-isolation --no-deps` 检查打包和导入。该方式仅验证现有依赖下的安装，不验证从网络全新解决依赖。

正式多 GPU 训练使用 `configs/deepspeed/zero2.json`，它来自原项目已有的 ZeRO-2 配置。GPU 数量、CUDA、bf16 支持需要由实际运行机器决定；CPU 最小示例不启用 DeepSpeed。

受限或只读环境中若 Scanpy/Numba 报 `no locator available`，设置可写缓存目录后运行：

```bash
export NUMBA_CACHE_DIR="${TMPDIR:-/tmp}/descort-numba-cache"
```
