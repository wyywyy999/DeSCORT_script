# 教程与正式入口

## Documentation

完整教程可构建为带导航和搜索的在线文档站。Read the Docs 构建配置已包含在仓库中；部署和本地预览步骤见 [Documentation hosting](../docs/hosting.md)。网站首次构建成功后，可从 Read the Docs 的 **View docs** 获取实际访问地址。

所有入口均位于这个独立副本内；只读复用原项目的资源与模型时，请显式指定绝对路径。

通用调用：`bash tutorials/任务/run.sh SCRIPT.py [参数...]`。`run.sh --help` 列出该目录可用脚本；`SCRIPT.py --help` 显示脚本参数。`PYTHON_BIN` 可指定解释器。该包装器原样转发参数，不加载历史实验 Bash 中的默认数据路径。

各任务原有 Bash 保留可编辑实验参数；其中仍可能含服务器数据/模型路径。新运行建议先从 `configs` 配置或显式参数开始。大型数据、检查点、实验探索目录和历史结果不复制到此副本。

| 顺序 | 工作流文档 |
|---|---|
| 00 | [下载](00_download/README.md) |
| 01 | [预处理](01_preprocess_data/README.md) |
| 02 | [预训练](02_pretrain/README.md) |
| 03 | [表征微调与提取](03_finetune/README.md) |
| 04 | [mAP](04_mAP/README.md) |
| 05 | [批次效应](05_batch_effect/README.md) |
| 06 | [虚拟细胞](06_virtual_cell/README.md) |
| 07 | [生物学解释](07_biological/README.md) |
| 08 | [跨扰动类型](08_three_type/README.md) |
| 09 | [论文图](09_paper_plot_code/README.md) |
