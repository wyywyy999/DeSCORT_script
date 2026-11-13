> `plot_cpg0000_radar.py` now generates block heatmaps. See [README_cpg0000_heatmap.md](README_cpg0000_heatmap.md) for current execution and output instructions. The following documents the earlier radar plots.

# cpg0000 radar plots for three mAP tasks

Run from the project root:

```bash
MPLCONFIGDIR=/tmp/mplconfig_cpg0000 /data/home/wangyongyan/miniconda3/envs/DeSCORT/bin/python \
  DeSCORT/tutorials/09_paper_plot_code/code/mAP/plot_cpg0000_radar.py
```

The script locates project inputs automatically and reuses `paper_plot_style.py`. Outputs default to `result/mAP/cpg0000_radar/`. Use `--input-root` for another result root; `--output-dir` must remain within the shared `result/` directory to preserve the common audit layout.

Three separate radar plots are exported, each as PDF, editable SVG, and 600 dpi PNG:

| Filename prefix | Title | Input category |
| --- | --- | --- |
| single_sgrna_replicate | Single sgRNA replicate | replicates_level/perturbation_activity |
| sister_sgrna_replicate | Sister sgRNA replicate | sister_level/perturbation_activity |
| sister_perturbation_consistency | Sister perturbation consistency | sister_level/perturbation_consistency |

Each plot has 12 axes, clockwise from the top: A549 followed by U2OS, each ordered Compound_Long, Compound_Short, Crispr_Long, Crispr_Short, Orf_Long, Orf_Short.

Labels omit the cell-line prefix, for example `Crispr_Long`. A549 labels are brown and U2OS labels purple. A shared top legend contains method lines and cell-line color patches. Full cell-line names and actual axis order are saved in CSV files and metadata.

The curves are Original, Foundation, and Finetune, reading the original, foundation, and finetune directories. Change `METHOD_LABELS` to update legend names.

Each file is summarized as the arithmetic mean of all `mean_average_precision` rows, followed by the mean across seeds 42, 2026, 2025, 2024, and 2023. Shading represents ±1 sample standard deviation (ddof=1) of seed means. Seeds are random-sampling repetitions, not independent biological replicates; no significance testing or significant-result filtering is applied. Each plot's radial limit is the largest mean plus 0.1, capped at 1: `min(1, max_mean + 0.1)`. Five rings are equally spaced. Limits and ring spacing are recorded in `plot_metadata.json`; radial limits can differ between plots, without per-axis rescaling. Original is blue, Foundation gray, and Finetune red, with matching translucent polygon fills. SD bands outside the valid mAP range are clipped to 0–1 and clipping counts are recorded.

No shared-object filter is applied; each method displays all rows in its own files. Object counts may differ between methods. `five_seed_values.csv` records original and retained row counts; do not interpret differences directly as improvements on strictly paired objects. Missing files, empty tables, nonfinite values, and values outside [0,1] cause errors.

Accompanying outputs:

- `five_seed_values.csv`: 540 seed-level summaries and input file locations.
- `five_seed_summary.csv`: mean, SD, seed count, and object-count range for 108 task/axis/method combinations.
- `radar_mean_values.csv`: mean-value matrix ordered by the 12 axes.
- `plot_metadata.json`: axis order, statistical definitions, and retention counts.

Rendering audit results are in `result/_qa/mAP/cpg0000_radar/`. Alignment audits are recorded during export. The three plots are single panels, so multipanel alignment is not applicable. All three PDFs passed font and overlap checks, and all images were visually inspected. Static checks must include the shared style module: scanning only the entry script misses shared font and export settings.

The shared top legend contains line symbols for the three models and color patches for A549/U2OS. The mean/SD explanation is no longer displayed below the plots; statistical definitions remain documented here and in metadata.

Plots do not display Radial scale text; radial limits and ring spacing remain in metadata.

Fonts match the reference notebooks through `paper_plot_style.apply_paper_style()`: titles are bold 15 pt, axis labels and legends regular 12.5 pt, with Arial, Helvetica, and DejaVu Sans preferred as sans-serif fonts. Settings are recorded in metadata, and PDF font checks are saved as `typography_comparison.json` in the audit directory.

Only Foundation and Finetune have value labels: Foundation values sit inside the gray polygon and Finetune values outside the red polygon, all gray, 16 pt, and formatted to three decimals. The blue Original curve remains without value labels. No leader lines are used. Each plot has 24 value labels; CSV files retain full-precision values for all three methods. The canvas is 20.5 × 19.8 inches to accommodate labels.

The outer radial limit is `min(1, the largest mean in the current plot + 0.1)`; the actual range is recorded in `plot_metadata.json`.

Outer condition labels are anchored at 1.04 times the outer ring radius, bringing labels such as Orf_Short closer to the ring.

The three current radar plots use 20 pt for titles, legends, and condition labels, and 16 pt for values; titles are bold. Font sizes are overridden only in the radar script, leaving the shared font configuration of other figures unchanged. The canvas expands with font size to preserve annotation spacing.
