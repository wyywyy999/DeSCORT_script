# cpg0000 分块 mAP 热图

原 `plot_cpg0000_radar.py` 已改为热图入口，保留文件名便于继续从原位置运行：

```bash
MPLCONFIGDIR=/tmp/mplconfig_cpg0000 /data/home/wangyongyan/miniconda3/envs/DeSCORT/bin/python \
  DeSCORT/tutorials/09_paper_plot_code/code/mAP/plot_cpg0000_radar.py
```

默认输出到 `result/mAP/cpg0000_heatmap/`。支持 `--input-root` 和 `--output-dir`；输出路径需在共享 `result/` 内。

- 横轴：12 个配置，先 A549，再 U2OS；每种细胞按 Compound、Crispr、Orf，各自 Long、Short 排列。图中使用细胞系、扰动类型和时长三层共享标题显示数据集含义；CSV 保存完整标识。
- 纵向三个分块：Single sgRNA replicate、Sister sgRNA replicate、Sister perturbation consistency。
- 每块行顺序：DeSCORT、Original、Foundation。DeSCORT 对应输入的 `finetune` 目录，其余对应同名目录。
- 每个格子为五个 seed 的 mAP 均值，保留三位小数。每个 seed 先对其文件中全部 `mean_average_precision` 行求均值，再对五个均值求算术平均。
- 三个分块共用固定的 0.4–1.0 绝对 mAP 色标，没有逐行/逐列标准化；低值为蓝色、高值为红色，中间使用近白色过渡。同一任务、同一配置中，完整精度的最大值加粗；并列最大值都加粗。
- 标签/标题为 20 pt，数值为 16 pt。A549 分组标题和细线为棕色、U2OS 为紫色。单元格和 PDF/SVG 文字均可编辑；PNG 为 600 dpi。

全部 540 个输入文件用于 108 个单元格，不进行显著性筛选、对象交集筛选或缺失值填补。不同方法的对象数可能不同，差异不等价于严格配对对象的效果。五个 seed 是随机采样重复，不代表生物学独立重复，不推断显著性。

本图只显示均值；五个 seed 的样本标准差（ddof=1）保存在配套 CSV 中。所有分块使用同一统计定义。源文件格式错误、空表、缺失文件或无效数值会直接报错。

输出文件：

- `cpg0000_mAP_block_heatmap.pdf/.svg/.png`：三块合并的热图。
- `heatmap_mean_values.csv`：有序的 9 × 12 均值表。
- `heatmap_seed_sd.csv`：对应的 seed 样本标准差。
- `heatmap_source_data.csv`：含完整任务名、数据集名和方法显示名的长表。
- `five_seed_values.csv`：每个 seed 的均值、保留行数和源文件路径。
- `five_seed_summary.csv`：任务/数据集/方法的均值、SD、seed 数和对象数范围。
- `plot_metadata.json`：列/行顺序、聚合规则、数据保留计数和字体配置。

渲染对齐、PDF 字体及重叠检查保存在 `result/_qa/mAP/cpg0000_heatmap/`。此前已生成的雷达图文件保留在原输出目录。

当前视觉布局：红蓝渐变，固定绝对 mAP 0.4–1.0 色标；六组 Long/Short 列对之间留白，两种细胞系之间使用更宽间隔。任务标题以 a/b/c 标记，置于各分块上方；DeSCORT 行标签加粗，同组最佳数值仍以粗体标记。保留标签 20 pt、数值 16 pt，画布为 22 × 10.2 英寸。

红蓝色标表示 mAP 的绝对数值：蓝色低、红色高，0.7 仅为颜色过渡中点，不代表显著性阈值、基线或零效应。色标范围覆盖全部当前数据；代码会检查是否有数值超出范围，防止静默截断。

最新排版将任务标题与热图区左边缘对齐，压缩列对间隙和任务分块间距；细胞系/扰动/时长表头使用固定物理间距，色条居中放在整张热图下方。红蓝色标和 20/16 pt 字号不变。
