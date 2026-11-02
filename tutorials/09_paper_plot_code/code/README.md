# 论文绘图代码

三个 notebook 共用 `paper_plot_style.py` 中的 `FONT` 字号表和 `save_paper_figure()` 导出函数。
使用 DeSCORT Python 环境，从上到下运行 notebook，图和统计表会写入上一级的 `result/`。

| 元素 | 字号（pt） |
| --- | ---: |
| 默认字体、坐标轴标签 | 13 |
| 图标题、分组标题 | 15 |
| 表格 Method 表头 | 15.5 |
| 子图标题、方法名称、面板字母 | 13.5 |
| 刻度、图例、数值、说明文字 | 12.5 |
| 色条标题 | 14.5 |

每张图同时保存 PNG（300 dpi）、SVG、PDF；SVG 和 PDF 保留文字。

```text
result/
├── mAP/
│   ├── single_sgRNA_activity/
│   ├── sister_sgRNA_consistency/
│   └── sister_sgRNA_activity/
├── batch_correction/
│   ├── metrics/
│   └── umap/
│       └── individual/
└── _qa/                 # 字体、布局和遮挡检查
```

输入数据路径、方法筛选、五次重复的统计方式和显著性检验均沿用原 notebook。
批次校正结果继续排除 DESC，保留 DeSCORT。
