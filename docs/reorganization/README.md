# 整理范围与原项目保护

整理日期：2026-11-02。所有最终新代码位于 `DeSCORT/other_model/DeSCORT`。

用户补充要求前曾在原目录实施修改，随后已从整理前快照恢复原文件，并逐文件核对 SHA-256。`original_restoration_check.json` 记录 2,880 个文件、0 个差异；`original_files.json` 是原文件清单。原代码、数据、模型、结果保留原位。

本副本完成：

- src 布局打包、可选依赖、`import descort` 与必要的 `__init__.py`。
- 以正式 `model/modeling_descort.py` 为模型实现；其他历史版本仅归档供追溯。
- 三处相同 embedding 提取逻辑统一为 `embedding.extract`，修复 mAP 的失效导入；相同训练日志回调统一为 `training`。
- 四组虚拟细胞流程采用原 `06_vc_new` 入口；生物学 Python/Bash 采用原 `07_biological_new`，保留现有算法与任务职责。
- 各主线脚本使用文件相对路径定位源码；通用 `run.sh` 原样传参；JSON profile 提供可检查的命令。
- 批次效应脚本补 `main()` 和路径参数，后端按需加载；模型前向、损失和指标公式未重写。
- 架构图、安装/格式/模型/复现文档、资源与基线来源清单、CPU 合成示例。

`migration_map.json` 记录**新副本的来源对应**，不是原项目文件已经迁移或删除的声明。历史 notebook、消融、基线、检查点和结果不根据名称自动删除，也不全部复制到新仓库。

整理前原测试运行 46 项，其中 1 项失败：输出测试样例缺少新增 provenance 字段。只在新副本测试中补齐该样例，并验证元信息保存；原项目测试文件已恢复原样。最终新副本检查见 `validation_report.md`。

尚需真实项目材料/算力完成的部分：公开数据和权重下载地址、作者引用与许可证、完整 GPU 训练及论文指标复现、全部第三方环境和每张论文图的重跑。它们不由目录整理自动解决。

可运行 `python scripts/check_original_integrity.py` 只读核对原文件。下载入口在新副本中补充了 `main()`、显式输出目录和 `--dry-run`，保留原数据源与默认选择，不执行真实下载。
