# DeSCORT 0.1.2 依赖核查

后续修正：0.1.3 增加 `python-dateutil>=2.9.0.post0,<3`，并采用分开下载测试包、从正式 PyPI 安装依赖的命令。以下保留 0.1.2 原始核查结论，环境快照仍是历史记录。

核查日期：2026-11-04。范围为 TestPyPI 已发布的 0.1.2 元数据、对应本地 wheel/sdist、`pyproject.toml`、本目录环境快照，以及用户提供的安装日志。本次不修改已有 Conda 环境，不重新发布版本。

## 1. 发布包使用什么依赖

TestPyPI 0.1.2 两个发布文件的 SHA256 均与本地 `dist/testpypi/0.1.2` 相同。wheel 中的 `Requires-Dist` 与 `pyproject.toml` 一致：24 项默认依赖，另有可选 DeepSpeed。

安装没有读取 `environment.yml` 或 `requirement.txt`，wheel 也没有包含 `environments/`。这里声明的是版本范围，不是原实验环境的精确复刻。用户日志中的版本与快照不同，是安装策略的直接结果。

| 包 | 环境快照 | 0.1.2 声明 | 用户安装日志 | 判断 |
|---|---|---|---|---|
| NumPy | 1.23.5 | >=1.23,<2 | 1.26.4 | 满足发布约束；替换原有 2.0.2 是预期行为 |
| pandas | 2.3.3 | >=2.0,<3 | 2.3.3 | 满足发布约束 |
| Transformers | 4.51.3 | >=4.51.3,<4.52 | 4.51.3 | 满足发布约束 |
| datasets | 4.0.0 | >=4,<5 | 4.5.0 | 满足发布约束，但不是快照中的同一版本 |
| AnnData | 0.10.8 | >=0.10,<0.12 | 0.10.9 | 满足发布约束 |
| scikit-learn | 1.1.3 | >=1.6 | 1.6.1 | 日志满足约束；快照不满足 |
| Scanpy | 1.9.3 | >=1.10,<1.12 | 1.10.3 | 日志满足约束；快照不满足 |
| python-dateutil | 2.9.0.post0 | 无直接限制，属于传递依赖 | 3.9.0 | 与已安装 botocore 要求冲突 |
| PyTorch | 2.7.0 | >=2.7,<3 | 本次日志未列出 | 不能仅凭这段日志确认已安装版本及 CUDA 构建 |

来源：[TestPyPI 0.1.2 元数据](https://test.pypi.org/pypi/descort/0.1.2/json)、本地 `pyproject.toml`、本目录两个环境文件。

## 2. 此次安装的明确冲突及索引来源

日志中的 `botocore==1.42.84` 要求 `python-dateutil>=2.1,<3.0.0`，安装结果为 `python-dateutil==3.9.0`，因此不满足约束。

核查时，正式 PyPI 的 `/pypi/python-dateutil/3.9.0/json` 返回 404，TestPyPI 的同版本接口存在。结合用户使用的双索引命令，这表明该候选版本来自测试索引。没有安装报告或原始下载 URL，不能据此逐个确认日志里所有其他包的来源。

`--extra-index-url` 并不是“第一个源找不到才去备用源”。pip 会综合全部索引的候选版本；当某个测试版满足依赖范围、版本号又更高时，它可能被选中。这次问题的直接修复是换回正式 PyPI 的 `python-dateutil==2.9.0.post0`：

```bash
conda activate DeSCORT
python -m pip install --index-url https://pypi.org/simple/ 'python-dateutil==2.9.0.post0'
python -m pip check
python -c "import descort, torch; from descort import DeSCORTConfig, DeSCORTForPerturbPrediction; print(descort.__version__, torch.__version__, torch.cuda.is_available())"
```

日志显示 DeSCORT 已安装，但不代表整个环境已经通过依赖检查。`pip check` 可以检查已安装包的声明约束；模型导入检查可验证当前环境下的基本导入，它们均不能代替完整任务运行验证。

来源：[botocore 1.42.84 元数据](https://pypi.org/pypi/botocore/1.42.84/json)、[TestPyPI dateutil 3.9.0 元数据](https://test.pypi.org/pypi/python-dateutil/3.9.0/json)、[正式 PyPI dateutil 2.9.0.post0](https://pypi.org/project/python-dateutil/2.9.0.post0/)、[pip 多索引候选选择](https://pip.pypa.io/en/stable/cli/pip_install/#finding-packages)、[pip check](https://pip.pypa.io/en/stable/cli/pip_check/)。

## 3. 原环境快照中已核实的不一致

以下按 Python 3.9 的适用条件核对官方包元数据。只列出本次核实的条目，不宣称已经穷尽完整环境快照的所有冲突。

| 冲突 | 证据 |
|---|---|
| `chex==0.1.90` 与 `numpy==1.23.5` | Chex 要求 `numpy>=1.24.1` |
| `decoupler==1.8.0` 与 `numba==0.56.3` | decoupler 要求 `numba>=0.60.0,<0.61.0` |
| 快照的 `scikit-learn==1.1.3` 与 DeSCORT 0.1.2 | 发布包要求 `scikit-learn>=1.6` |
| 快照的 `scanpy==1.9.3` 与 DeSCORT 0.1.2 | 发布包要求 `scanpy>=1.10,<1.12` |

前两项是快照自身的不一致；后两项是快照与当前发布声明的不一致。用户当前日志中的 NumPy 1.26.4、Numba 0.60.0、scikit-learn 1.6.1、Scanpy 1.10.3 不存在表中这几组版本组合，但仍应通过实际环境的 `pip check` 检查其他包。

来源：[Chex 0.1.90 元数据](https://pypi.org/pypi/chex/0.1.90/json)、[decoupler 1.8.0 元数据](https://pypi.org/pypi/decoupler/1.8.0/json)、DeSCORT 发布元数据及本地环境文件。因此不建议在当前环境直接运行 `pip install -r environments/requirement.txt` 来“修复”此次问题，这可能重新引入已知冲突。

## 4. 安装与验证边界

根目录 README 已改为先使用 `pip download --no-deps` 从 TestPyPI 下载指定 DeSCORT wheel，再以正式 PyPI 为索引安装该本地 wheel 及其依赖，避免同时搜索两个索引。本次未更改发布包的依赖范围，也未重发 TestPyPI。

之前发布验证覆盖包构建、Twine 检查、远端哈希、临时目录无依赖安装，以及已有 DeSCORT 环境中的模型导入。它没有在全新环境中安装和运行全部依赖；本次用户日志暴露的混合索引问题不在此前验证覆盖范围内。

本次核查也未更改服务器已有环境。用户安装可能位于另一台机器或另一 Conda 环境，因此当前环境的完整兼容状态还需以用户在对应环境运行的 `python -m pip check` 和实际任务检查为准。
