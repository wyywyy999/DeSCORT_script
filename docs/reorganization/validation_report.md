# 整理验证报告

验证日期：2026-11-02。新副本：`DeSCORT/other_model/DeSCORT`。

| 项目 | 结果 | 证据 |
|---|---|---|
| 原项目保护 | 2,880 个原文件 SHA-256 全部一致，原时间戳已恢复 | `original_restoration_check.json`、`original_files.json` |
| 原有回归测试在新副本运行 | 46 / 46 通过 | `checks/unit_tests.log` |
| 模型与数值核心保持 | 9 个核心模块与原文件逐字节相同 | `numerical_preservation.json` |
| embedding 公共函数 | 函数 AST 与原提取实现一致 | `numerical_preservation.json` |
| 三个 embedding 教程入口 | 在相同小模型与输入下输出逐值相等 | `checks/embedding_parity.log` |
| CPU 最小链路 | 3 行合成 parquet → 随机模型保存/加载 → 3×16 embedding；数值有限、元数据相同 | `checks/minimal_embedding.log` |
| 命令行跨工作目录 | 44 个 Python 入口从 `/tmp` 执行 `--help` 全部通过 | `cli_checks.json` |
| 安装包 | wheel 构建成功；在 `/tmp` 隔离 venv 安装，离开源码目录成功导入模型/config/评估 API | `checks/wheel_build.log`、`checks/wheel_install.log` |
| 打包内容 | 仅含 descort 包与元信息，无教程数据、历史模型副本或检查点 | `structural_checks.json` |
| 语法、文档链接、配置 | Python/Bash 检查通过，无失效的整理文档本地链接；argv 中空格保留、未定义路径变量拒绝运行 | `structural_checks.json` |
| 资源索引 | 127 个本地资源；小文件 SHA-256，大文件不读载荷 | `../../resources/manifest.json` |
| 架构图 | assets/workflow.png 与用户原图逐字节相同 | `structural_checks.json` |

测试使用现有 Python 3.9.15、PyTorch 2.7.0、Transformers 4.51.3 等依赖。隔离 venv 共享已有依赖，安装新构建的 wheel；未从网络全新安装全部依赖。现有 venv 引导的旧 setuptools 不支持项目元信息，因此 wheel 在已有 setuptools 69.5.1 环境构建后安装到隔离环境。

首次 mAP `--help` 检查因受限环境中的 Numba 缓存目录不可写失败；指定 `/tmp/descort-numba-cache` 后通过。安装文档记录该环境设置。未为此改变指标公式。

原项目测试中原本有 1 个 provenance 样例错误；仅在新副本的测试样例补齐字段，原文件已恢复。没有通过放宽模型/指标断言消除错误。

未验证的范围：真实 GPU 训练、完整论文数据、完整第三方基线训练、全部论文图重绘与科学性能。当前结果证明整理后的导入、包装入口、核心回归和小型推理链路可用，不能据此声称论文结果已完整复现。

可重新核对原文件：

```bash
python scripts/check_original_integrity.py
```
