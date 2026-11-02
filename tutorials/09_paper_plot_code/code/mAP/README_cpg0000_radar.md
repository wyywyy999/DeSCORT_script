> 当前 `plot_cpg0000_radar.py` 已改为分块热图入口。运行及输出说明见 [README_cpg0000_heatmap.md](README_cpg0000_heatmap.md)。以下为此前雷达图的历史说明。

# cpg0000 三类 mAP 雷达图

运行（从项目根目录）：

```bash
MPLCONFIGDIR=/tmp/mplconfig_cpg0000 /data/home/wangyongyan/miniconda3/envs/DeSCORT/bin/python \
  DeSCORT/tutorials/09_paper_plot_code/code/mAP/plot_cpg0000_radar.py
```

脚本自动定位项目输入，并复用 `paper_plot_style.py`。默认输出到 `result/mAP/cpg0000_radar/`。可用 `--input-root` 指定其他结果根目录；`--output-dir` 必须位于共享 `result/` 内，以兼容统一的审计文件归档。

生成三张独立雷达图，每张导出 PDF、可编辑 SVG 和 600 dpi PNG：

| 文件前缀 | 标题 | 输入分类 |
| --- | --- | --- |
| single_sgrna_replicate | Single sgRNA replicate | replicates_level/perturbation_activity |
| sister_sgrna_replicate | Sister sgRNA replicate | sister_level/perturbation_activity |
| sister_perturbation_consistency | Sister perturbation consistency | sister_level/perturbation_consistency |

每张图从正上方开始顺时针排列 12 个角：先 A549，再 U2OS；每种细胞均按 Compound_Long、Compound_Short、Crispr_Long、Crispr_Short、Orf_Long、Orf_Short 排列。

标签去掉细胞系前缀，例如 `Crispr_Long`；A549 使用棕色字体，U2OS 使用紫色字体。顶部统一图例包含模型线条和细胞系颜色方块。完整细胞系名称和实际角顺序保存在 CSV 和元数据中。

三条曲线分别为 Original、Foundation、Finetune，对应 original、foundation、finetune 目录。修改 `METHOD_LABELS` 可改变图例显示名称。

每个文件使用全部 `mean_average_precision` 行的算术平均，再对 42、2026、2025、2024、2023 五个 seed 的均值求平均。淡色带表示五个 seed 均值的 ±1 样本标准差（ddof=1）。五个 seed 是随机采样重复，不代表生物学独立重复；没有进行显著性检验或显著结果筛选。每张图的径向上限取该图全部曲线的最大均值加 0.1，最高为 1，即 `min(1, max_mean + 0.1)`。五个圆环等距，具体上限和圆环间隔记录在 `plot_metadata.json` 中；不同图的径向上限可能不同，不对每个角单独缩放。Original 为蓝色，Foundation 为灰色，Finetune 为红色；三个方法的均值多边形内部使用同色透明填充。超出合法 mAP 范围的 SD 带会截到 0–1，并记录截断数量。

不做对象交集筛选，每种方法展示自身文件的全部结果。不同方法对象数可能不同；`five_seed_values.csv` 保存每个文件的原始/使用行数，不应把差值直接解释为严格配对对象上的提升。代码遇到缺失文件、空表、非有限值或超出 [0,1] 的值会报错。

配套输出：

- `five_seed_values.csv`：540 个 seed 级汇总及输入文件定位。
- `five_seed_summary.csv`：108 个 task/axis/method 的均值、SD、seed 数和对象数范围。
- `radar_mean_values.csv`：按 12 个角排列的均值矩阵。
- `plot_metadata.json`：角顺序、统计口径和数据保留计数。

渲染审计结果位于 `result/_qa/mAP/cpg0000_radar/`。脚本导出时记录对齐审计；三张图均为单面板，因此多面板对齐不适用。此次三份 PDF 均通过字体和重叠审计，三张图片均已目视检查。静态检查需同时包含共享样式模块；单独扫描入口脚本会漏识别共享模块中的字体和导出设置。

顶部统一图例包含三个模型的线条符号及 A549/U2OS 的颜色方块；图下方不再显示均值/SD 说明文字，统计定义仍保留在本说明和元数据中。

图中不显示 Radial scale 文字；径向范围与圆环间隔仍保存在元数据中。

字体与同目录参考 notebook 统一：共享 `paper_plot_style.apply_paper_style()`，标题 15 pt 加粗，角标签和图例 12.5 pt 常规字重，sans-serif 字体优先 Arial、Helvetica、DejaVu Sans。字体参数写入元数据，实际 PDF 字体核对结果保存在审计目录 `typography_comparison.json`。

只显示 Foundation 和 Finetune 的数值：Foundation 数字位于灰色多边形内部，Finetune 数字位于红色多边形外侧，均为灰色 16 pt、三位小数。Original 蓝色曲线保留，不标数值。不使用引导线。每张图共有 24 个数值标注，CSV 仍保留全部三种方法的完整精度数据。为容纳图内标注，画布为 20.5 × 19.8 英寸。

最外圈径向上限为 `min(1, 当前图最大均值 + 0.1)`，具体范围记录在 `plot_metadata.json`。

外围条件标签锚点设为最外圈半径的 1.04 倍，使 Orf_Short 等标签靠近外圈。

当前三张雷达图的标题、图例和条件标签为 20 pt，数值标注为 16 pt；标题加粗。仅在雷达脚本中覆盖字号，其他图的共享字体配置不变。为保留标注间距，画布随字号增大。
