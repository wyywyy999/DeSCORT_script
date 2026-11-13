# cpg0000 mAP block heatmap

The original `plot_cpg0000_radar.py` now generates heatmaps; its filename is retained so it can still be run from the same location:

```bash
MPLCONFIGDIR=/tmp/mplconfig_cpg0000 /data/home/wangyongyan/miniconda3/envs/DeSCORT/bin/python \
  DeSCORT/tutorials/09_paper_plot_code/code/mAP/plot_cpg0000_radar.py
```

The default output is `result/mAP/cpg0000_heatmap/`. `--input-root` and `--output-dir` are supported; outputs must remain within the shared `result/` directory.

- Columns: 12 configurations, A549 followed by U2OS. Within each cell line, Compound, Crispr, and Orf each have Long and Short conditions. Three shared header levels show cell line, perturbation type, and duration; CSV files preserve full identifiers.
- Three row blocks: Single sgRNA replicate, Sister sgRNA replicate, and Sister perturbation consistency.
- Rows within each block: DeSCORT, Original, Foundation. DeSCORT reads the `finetune` input directory; the other methods read their respective directories.
- Each cell shows the five-seed mean mAP to three decimal places. Each seed is first averaged over all `mean_average_precision` rows in its file, then the five means are averaged arithmetically.
- All blocks share a fixed absolute mAP color scale of 0.4–1.0, without row or column normalization. Lower values are blue, higher values red, with a near-white transition. The full-precision maximum for each task/configuration is bold; ties are all bold.
- Labels and headings use 20 pt; values use 16 pt. A549 group headings and thin rules are brown, U2OS purple. Cells and PDF/SVG text remain editable; PNG uses 600 dpi.

All 540 input files contribute to 108 cells, without significance filtering, shared-object filtering, or missing-value imputation. Methods may have different object counts, so differences are not equivalent to effects on strictly paired objects. The five seeds are random-sampling repetitions, not independent biological replicates; no significance is inferred.

The figure displays means only. Five-seed sample standard deviations (ddof=1) are saved in the accompanying CSV. All blocks use the same statistical definition. Malformed input files, empty tables, missing files, or invalid numbers cause errors.

Output files:

- `cpg0000_mAP_block_heatmap.pdf/.svg/.png`: combined three-block heatmap.
- `heatmap_mean_values.csv`: ordered 9 × 12 matrix of means.
- `heatmap_seed_sd.csv`: corresponding seed sample standard deviations.
- `heatmap_source_data.csv`: long-form table with full task, dataset, and method display names.
- `five_seed_values.csv`: per-seed means, retained row counts, and input paths.
- `five_seed_summary.csv`: mean, SD, seed count, and object-count range per task/dataset/method.
- `plot_metadata.json`: column/row order, aggregation rules, retention counts, and font settings.

Rendering alignment, PDF font, and overlap checks are saved in `result/_qa/mAP/cpg0000_heatmap/`. Previously generated radar files remain in the original output directory.

Current layout: a red–blue gradient with a fixed absolute mAP scale of 0.4–1.0. The six Long/Short column pairs are separated by gaps, with a larger gap between cell lines. Task headings labeled a/b/c sit above each block. DeSCORT row labels and within-group best values are bold. Labels remain 20 pt and values 16 pt on a 22 × 10.2 inch canvas.

The red–blue scale represents absolute mAP: blue is low, red high. The 0.7 midpoint is only a color transition, not a significance threshold, baseline, or zero effect. The scale covers all current data; the code checks for out-of-range values to prevent silent clipping.

The latest layout aligns task headings with the heatmap's left edge and reduces gaps between column pairs and task blocks. Cell-line/perturbation/duration headers use fixed physical spacing, and the colorbar is centered below the full heatmap. The red–blue scale and 20/16 pt font sizes are unchanged.
