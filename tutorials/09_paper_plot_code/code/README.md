# Paper plotting code

The three notebooks share the `FONT` size dictionary and `save_paper_figure()` export function from `paper_plot_style.py`.
Use the DeSCORT Python environment and execute the notebooks from top to bottom. Figures and statistical tables are written to the parent `result/` directory.

| Element | Font size (pt) |
| --- | ---: |
| Default font and axis labels | 13 |
| Figure titles and group headings | 15 |
| Method header in tables | 15.5 |
| Subplot titles, method names, and panel letters | 13.5 |
| Ticks, legends, values, and explanatory text | 12.5 |
| Colorbar title | 14.5 |

Each figure is exported as PNG (300 dpi), SVG, and PDF. SVG and PDF retain text.

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
└── _qa/                 # Font, layout, and overlap checks
```

Input paths, method selection, statistics across five repetitions, and significance tests follow the original notebooks.
Batch-correction results continue to exclude DESC and retain DeSCORT.
