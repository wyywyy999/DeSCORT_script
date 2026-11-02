# 外部资源

资源入口见 [resources/README.md](../resources/README.md) 和机器可读 [manifest](../resources/manifest.json)。原 `common_files` 保持原位，不复制大型 embedding。

新目录内 `PROJECT` 指向新仓库；生物学公共模块允许通过 `DESCORT_RESOURCE_ROOT` 与 `DESCORT_WORKSPACE_ROOT` 显式覆盖资源/工作区根路径。在当前嵌套布局下，默认可只读定位旧项目的 `common_files` 与 CellPainting 工作区。

其他处理和训练入口仍通过命令行传入资源路径；不要假定新目录含有原项目所有数据。CORUM/STRING、GO/KEGG、手工通路注释、ESM2/Morgan/UniMol2 的来源版本应随实验记录。生物学分析若未给离线基因集 JSON，原实现可能获取并缓存 Enrichr 库；离线复现时需提供固定库快照。
