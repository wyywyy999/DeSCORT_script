#!/usr/bin/env python3
"""Blocked cpg0000 mAP heatmap (historical radar-script filename).

Columns are the 12 A549/U2OS dataset configurations. Three task blocks each
contain DeSCORT, Original and Foundation rows. Every cell is the mean of the
five seed-level all-row mAP means; no significance or item-intersection filter.
Seed SDs and individual run values are exported alongside the mean heatmap.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))
from paper_plot_style import (  # noqa: E402
    RESULT_ROOT, apply_paper_style, save_paper_figure,
)

WORKSPACE = CODE_ROOT.parents[3]
DEFAULT_INPUT = (WORKSPACE / 'Paper_LLm/tutorials/01_mAP/'
                 'well_level_attention_mask_result')
DEFAULT_OUTPUT = RESULT_ROOT / 'mAP/cpg0000_heatmap'
SEEDS = ('42', '2026', '2025', '2024', '2023')
METHODS = ('finetune', 'original', 'foundation')
METHOD_LABELS = {'finetune': 'DeSCORT', 'original': 'Original',
                 'foundation': 'Foundation'}
FONT = {'title': 20, 'label': 20, 'tick': 20, 'annotation': 16}
CELL_COLORS = {'A549': '#8C510A', 'U2OS': '#5E3C99'}
COLOR_RANGE = (0.4, 1.0)
HEATMAP_COLORS = ['#79B2DC', '#C5DDED', '#FAFAF8', '#F1C2B8', '#EC9589']
TASKS = {
    'single_sgrna_replicate': (
        'Single sgRNA replicate', 'replicates_level/perturbation_activity'),
    'sister_sgrna_replicate': (
        'Sister sgRNA replicate', 'sister_level/perturbation_activity'),
    'sister_perturbation_consistency': (
        'Sister perturbation consistency', 'sister_level/perturbation_consistency'),
}
TASK_DISPLAY = {
    'single_sgrna_replicate': 'Single sgRNA\nreplicate',
    'sister_sgrna_replicate': 'Sister sgRNA\nreplicate',
    'sister_perturbation_consistency': 'Sister perturbation\nconsistency',
}
CONDITIONS = tuple((cell, perturbation, duration)
                   for cell in ('A549', 'U2OS')
                   for perturbation in ('Compound', 'Crispr', 'Orf')
                   for duration in ('long', 'short'))
AXIS_ORDER = tuple(f'{cell}_{pert.lower()}_{duration}'
                   for cell, pert, duration in CONDITIONS)
DATASET_NAMES = tuple(f'{cell}_{pert}_{duration.capitalize()}'
                      for cell, pert, duration in CONDITIONS)


def read_results(input_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Keep all observations; record row counts and file provenance per run."""
    records = []
    for task, (_, relative) in TASKS.items():
        root = input_root / relative / 'well-level/cpg0000'
        for (cell, pert, duration), axis in zip(CONDITIONS, AXIS_ORDER):
            for method in METHODS:
                for seed in SEEDS:
                    path = (root / method / pert / f'{cell}_{duration}' /
                            seed / 'ap_scores_dict.pickle')
                    with path.open('rb') as handle:
                        frame = pickle.load(handle)  # Trusted project-generated files.
                    if (not isinstance(frame, pd.DataFrame) or frame.empty or
                            'mean_average_precision' not in frame):
                        raise ValueError(f'Expected nonempty mAP DataFrame: {path}')
                    values = pd.to_numeric(frame['mean_average_precision'],
                                           errors='raise').to_numpy(dtype=float)
                    if not np.isfinite(values).all() or not ((values >= 0) & (values <= 1)).all():
                        raise ValueError(f'Invalid mAP values: {path}')
                    records.append({
                        'task': task, 'axis': axis, 'cell_line': cell,
                        'perturbation': pert, 'duration': duration,
                        'method': method, 'seed': seed,
                        'n_source_rows': len(values), 'n_used_rows': len(values),
                        'map_run_mean': float(values.mean()), 'source_file': str(path),
                    })
    runs = pd.DataFrame(records)
    summary = runs.groupby(['task', 'axis', 'method'], sort=False).agg(
        map_mean=('map_run_mean', 'mean'),
        map_sd=('map_run_mean', 'std'),
        n_seeds=('seed', 'size'),
        n_rows_min=('n_source_rows', 'min'),
        n_rows_max=('n_source_rows', 'max'),
    ).reset_index()
    if len(runs) != 540 or not summary.n_seeds.eq(5).all():
        raise ValueError('Expected 3 tasks x 12 axes x 3 methods x 5 seeds')
    return runs, summary


def heatmap_tables(summary: pd.DataFrame):
    """Fix row and column order explicitly; never cluster or normalize per row."""
    row_order = pd.MultiIndex.from_product([TASKS, METHODS], names=['task', 'method'])
    mean = summary.pivot(index=['task', 'method'], columns='axis', values='map_mean')
    sd = summary.pivot(index=['task', 'method'], columns='axis', values='map_sd')
    mean = mean.reindex(index=row_order, columns=AXIS_ORDER)
    sd = sd.reindex(index=row_order, columns=AXIS_ORDER)
    if mean.shape != (9, 12) or mean.isna().any().any() or sd.isna().any().any():
        raise ValueError('Expected all 108 task/method/dataset combinations')
    return mean, sd


def draw_heatmap(mean: pd.DataFrame):
    """Hierarchical headers and three aligned task blocks; absolute mAP scale."""
    apply_paper_style()
    cmap = LinearSegmentedColormap.from_list(
        'map_absolute', HEATMAP_COLORS)
    if mean.to_numpy().min() < COLOR_RANGE[0] or mean.to_numpy().max() > COLOR_RANGE[1]:
        raise ValueError('Heatmap values exceed the shared color range; update COLOR_RANGE')
    norm = Normalize(vmin=COLOR_RANGE[0], vmax=COLOR_RANGE[1])
    # Gaps explicitly separate perturbation pairs and the two cell-line groups.
    pair_starts = np.array([0.0, 2.18, 4.36, 7.04, 9.22, 11.4])
    column_x = np.array([start + offset for start in pair_starts for offset in (0, 1)])
    xmin, xmax = -.5, 12.9
    fig, axes = plt.subplots(3, 1, figsize=(22, 10.2), sharex=True,
                             gridspec_kw={'hspace': .42}, dpi=300)
    fig.subplots_adjust(left=.12, right=.98, bottom=.18, top=.715)
    images = []
    for ax, task, letter in zip(axes, TASKS, 'abc'):
        values = mean.xs(task, level='task').loc[list(METHODS)].to_numpy(dtype=float)
        for group, start in enumerate(pair_starts):
            image = ax.pcolormesh(start + np.arange(3) - .5,
                                  np.arange(len(METHODS) + 1) - .5,
                                  values[:, 2 * group:2 * group + 2],
                                  cmap=cmap, norm=norm, shading='flat',
                                  edgecolors='white', linewidth=.8)
            images.append(image)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(len(METHODS) - .5, -.5)
        ax.set_yticks(np.arange(len(METHODS)), [METHOD_LABELS[m] for m in METHODS],
                      fontsize=FONT['tick'])
        ax.tick_params(axis='y', length=0, pad=13)
        for label, method in zip(ax.get_yticklabels(), METHODS):
            label.set_color('#293D49' if method == 'finetune' else '#59636B')
            label.set_fontweight('bold' if method == 'finetune' else 'normal')
        ax.set_xticks([])
        ax.tick_params(axis='both', which='both', length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        for row in range(len(METHODS)):
            for col, x in enumerate(column_x):
                value = values[row, col]
                ax.text(x, row, f'{value:.3f}', ha='center', va='center',
                        fontsize=FONT['annotation'], color='#3F3F3F',
                        fontweight='bold' if value == values[:, col].max() else 'normal')
        # Fixed physical offsets also provide auditable panel-label anchors.
        ax.annotate(letter, xy=(0, 1), xycoords='axes fraction',
                    xytext=(-28, 12), textcoords='offset points',
                    ha='left', va='bottom', fontsize=FONT['label'],
                    fontweight='bold', color='#25343D', annotation_clip=False)
        ax.annotate(TASKS[task][0], xy=(0, 1), xycoords='axes fraction',
                    xytext=(0, 12), textcoords='offset points',
                    ha='left', va='bottom', fontsize=FONT['label'],
                    fontweight='bold', color='#25343D', annotation_clip=False)

    position = axes[0].get_position()
    def figure_x(value):
        return position.x0 + (value - xmin) / (xmax - xmin) * position.width

    # Physical spacing keeps the hierarchy consistent at export resolution.
    points_per_figure = fig.get_figheight() * 72
    duration_y = position.y1 + 54 / points_per_figure
    perturbation_y = duration_y + 37 / points_per_figure
    perturbation_line_y = perturbation_y - 17 / points_per_figure
    cell_y = perturbation_y + 42 / points_per_figure
    cell_line_y = cell_y - 17 / points_per_figure
    # Shared three-level header: cell line, perturbation type, exposure duration.
    for cell, first, last in [('A549', 0, 2), ('U2OS', 3, 5)]:
        left = pair_starts[first] - .5
        right = pair_starts[last] + 1.5
        fig.text(figure_x((left + right) / 2), cell_y, cell,
                 fontsize=FONT['label'], color=CELL_COLORS[cell],
                 fontweight='bold', ha='center', va='center')
        fig.add_artist(plt.Line2D([figure_x(left), figure_x(right)], [cell_line_y, cell_line_y],
                                  transform=fig.transFigure, color=CELL_COLORS[cell],
                                  linewidth=1.4, solid_capstyle='butt'))
    for group, start in enumerate(pair_starts):
        perturbation = ('Compound', 'Crispr', 'Orf')[group % 3]
        fig.text(figure_x(start + .5), perturbation_y, perturbation,
                 fontsize=FONT['label'], ha='center', va='center', color='#344550')
        fig.add_artist(plt.Line2D([figure_x(start - .43), figure_x(start + 1.43)],
                                  [perturbation_line_y, perturbation_line_y], transform=fig.transFigure,
                                  color='#CFD9E0', linewidth=.7, solid_capstyle='butt'))
        for offset, duration in [(0, 'Long'), (1, 'Short')]:
            fig.text(figure_x(start + offset), duration_y, duration,
                     fontsize=FONT['tick'], ha='center', va='center', color='#59636B')

    colorbar_width = .30
    colorbar_left = position.x0 + (position.width - colorbar_width) / 2
    cax = fig.add_axes([colorbar_left, .105, colorbar_width, .014])
    colorbar = fig.colorbar(images[0], cax=cax, orientation='horizontal',
                           ticks=np.linspace(COLOR_RANGE[0], COLOR_RANGE[1], 7))
    colorbar.ax.tick_params(labelsize=FONT['tick'], length=3, width=.6, colors='#59636B')
    colorbar.outline.set_visible(False)
    colorbar.set_label('Mean mAP', fontsize=FONT['label'], labelpad=10, color='#344550')
    return fig, cax


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-root', type=Path, default=DEFAULT_INPUT)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT,
                        help='Output below the shared paper result directory')
    args = parser.parse_args()
    args.output_dir.resolve().relative_to(RESULT_ROOT.resolve())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    runs, summary = read_results(args.input_root)
    mean, sd = heatmap_tables(summary)
    runs.to_csv(args.output_dir / 'five_seed_values.csv', index=False)
    summary.to_csv(args.output_dir / 'five_seed_summary.csv', index=False)
    for table, filename in [(mean, 'heatmap_mean_values.csv'), (sd, 'heatmap_seed_sd.csv')]:
        table.rename(columns=dict(zip(AXIS_ORDER, DATASET_NAMES))).to_csv(args.output_dir / filename)
    # Source data retain machine method names and expose the display mapping explicitly.
    summary.assign(method_display=summary.method.map(METHOD_LABELS),
                   task_display=summary.task.map({k: v[0] for k, v in TASKS.items()}),
                   dataset=summary.axis.map(dict(zip(AXIS_ORDER, DATASET_NAMES)))).to_csv(
                       args.output_dir / 'heatmap_source_data.csv', index=False)
    fig, colorbar_ax = draw_heatmap(mean)
    stem = args.output_dir / 'cpg0000_mAP_block_heatmap'
    save_paper_figure(fig, stem, exclude_axes=[colorbar_ax], dpi=600)
    plt.close(fig)
    metadata = {
        'plot_type': 'Three stacked task blocks, each with three method rows',
        'dataset_order': DATASET_NAMES, 'task_order': list(TASKS),
        'method_order': list(METHODS), 'method_labels': METHOD_LABELS,
        'seeds': SEEDS, 'n_input_files': len(runs), 'n_heatmap_cells': int(mean.size),
        'n_source_rows': int(runs.n_source_rows.sum()),
        'n_used_rows': int(runs.n_used_rows.sum()),
        'aggregation': 'All-row mean per seed, then mean of the five seed means',
        'uncertainty': 'Sample SD (ddof=1) retained in CSV; color and cell labels show means only',
        'uncertainty_unit': 'Resampling seed; not biological replication',
        'significance_filter': None, 'item_intersection_filter': None,
        'color_scale': list(COLOR_RANGE), 'per_row_normalization': False,
        'cell_format': '.3f', 'bold': 'Best full-precision value within each task/dataset',
        'typography': {'family': 'sans-serif', 'labels_pt': 20, 'values_pt': 16},
        'layout': 'Three task panels; shared hierarchical cell-line/perturbation/duration headers',
        'column_pair_starts': [0, 2.18, 4.36, 7.04, 9.22, 11.4],
        'palette': HEATMAP_COLORS,
        'numeric_text_color': '#3F3F3F', 'figure_inches': [22, 10.2],
    }
    (args.output_dir / 'plot_metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(f'Read {len(runs)} inputs; plotted {mean.shape[0]} rows x {mean.shape[1]} columns')
    print(f'Saved PDF/SVG/600-dpi PNG: {stem}')


if __name__ == '__main__':
    main()
