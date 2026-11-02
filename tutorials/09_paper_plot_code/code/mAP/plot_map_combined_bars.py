"""Assemble the three existing mAP bar plots with shared labels and Arial text.

Quantitative grid: compare DeSCORT and Original across single-reagent activity,
sister-reagent activity and sister-perturbation consistency on three plates.
All original five-seed values and existing statistical tests are retained.
"""
from pathlib import Path
import hashlib
import json
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Patch
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import pandas as pd

PAPER_PLOT_ROOT = Path('/fse/home/wangyongyan/Project/CellPainting/DeSCORT/tutorials/09_paper_plot_code')
sys.path.insert(0, str(PAPER_PLOT_ROOT/'code'))
from paper_plot_style import FONT, RESULT_ROOT, apply_paper_style, save_paper_figure, figure_qa_stem
from audit_panel_alignment import require_matplotlib_panel_alignment


def draw_combined_map_bars():
    """Replot the existing seed summaries; do not rasterize or crop source PDFs."""
    for font in sorted((PAPER_PLOT_ROOT/'code/fonts/arial').glob('*.TTF')):
        font_manager.fontManager.addfont(str(font))
    # Require genuine Arial rather than silently accepting a fallback family.
    arial_paths = {
        weight: font_manager.findfont(
            font_manager.FontProperties(family='Arial', weight=weight),
            fallback_to_default=False)
        for weight in ['normal', 'bold']
    }
    apply_paper_style()
    mpl.rcParams.update({'font.family': 'Arial', 'font.sans-serif': ['Arial'], 'font.size': 13,
                         'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    root = RESULT_ROOT/'mAP'
    plates = ('CP186H', 'CP257N', 'CP257H')
    methods = ('DeSCORT', 'Original')
    seeds = ('42', '2023', '2024', '2025', '2026')
    colors = {'DeSCORT': '#BC3C28', 'Original': '#0072B5'}
    offsets = {'DeSCORT': -.18, 'Original': .18}
    jitter = np.linspace(-.105, .105, len(seeds))
    panels = [
        dict(title='Single SgRNA Activity', folder='single_sgRNA_activity',
             values='five_run_values.csv', value_column='map_all',
             tests='paired_delta_map_all_tests.csv', ylim=1.27,
             ticks=[0, .25, .5, .75, 1.], tick_format='%.2f',
             bracket=(1.055, 1.075), stars_y=1.10),
        dict(title='Sister SgRNA Activity', folder='sister_sgRNA_activity',
             values='sister_sgRNA_activity_five_run_values.csv',
             value_column='map_mean_used_genes',
             tests='sister_sgRNA_activity_paired_tests.csv', ylim=1.13,
             ticks=[0, .2, .4, .6, .8, 1.], tick_format='%.1f',
             bracket=(1.020, 1.045), stars_y=1.052),
        dict(title='Sister Perturbation Consistency', folder='sister_sgRNA_consistency',
             values='five_run_values_shared_genes.csv', value_column='map_all_shared',
             tests='paired_all_genes_exploratory_tests.csv', ylim=.56,
             ticks=[0, .1, .2, .3, .4, .5], tick_format='%.1f',
             bracket=(.475, .490), stars_y=.515),
    ]
    stem = root/'combined_mAP_barplots'
    qa = figure_qa_stem(stem)
    # Equal plot areas; retain the original, explicitly labelled y-axis scales.
    fig, axes = plt.subplots(1, 3, figsize=(17.0, 5.1), dpi=300, facecolor='white')
    fig.subplots_adjust(left=.06, right=.99, top=.77, bottom=.16, wspace=.15)
    source_tables, bar_tables, provenance, label_offsets_pt = [], [], {}, []
    for ax, panel in zip(axes, panels):
        values_path = root/panel['folder']/panel['values']
        tests_path = root/panel['folder']/panel['tests']
        values = pd.read_csv(values_path, dtype={'seed': str})
        tests = pd.read_csv(tests_path).set_index('plate')
        expected = {(p, m, s) for p in plates for m in methods for s in seeds}
        actual = set(values[['plate', 'method', 'seed']].itertuples(index=False, name=None))
        if len(values) != 30 or actual != expected or values.duplicated(['plate', 'method', 'seed']).any():
            raise ValueError(f"Unexpected seed/plate/method coverage: {panel['title']}")
        if set(tests.index) != set(plates) or not tests.index.is_unique:
            raise ValueError('Statistical tests must cover exactly the same three plates')
        raw = values[panel['value_column']].to_numpy(float)
        if not np.isfinite(raw).all() or not np.logical_and(raw >= 0, raw <= 1).all():
            raise ValueError('Invalid mAP input')
        source_tables.append(values[['plate', 'method', 'seed', panel['value_column']]]
                             .rename(columns={panel['value_column']: 'mAP'})
                             .assign(panel=panel['title']))
        provenance[panel['title']] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                                     for p in [values_path, tests_path]}
        ax.set_title(panel['title'], fontsize=FONT['title'], fontweight='bold',
                     color='#1B272D', pad=14)
        ax.set_xlim(-.55, len(plates)-.45)
        ax.set_ylim(0, panel['ylim'])
        ax.set_xticks(range(len(plates)), plates)
        ax.set_yticks(panel['ticks'])
        ax.yaxis.set_major_formatter(FormatStrFormatter(panel['tick_format']))
        ax.spines[['top', 'right']].set_visible(False)
        ax.spines[['left', 'bottom']].set_color('#78838A')
        ax.tick_params(axis='both', labelsize=FONT['tick'], colors='#34434A')
        # The same axes-height fraction gives the same physical inset in all panels.
        # Allow clearance for the lowest sister-activity seed point as well as SD.
        label_inset_fraction = .15
        for i, plate in enumerate(plates):
            for method in methods:
                sample = values[values.plate.eq(plate) & values.method.eq(method)]
                y = sample.set_index('seed').loc[list(seeds), panel['value_column']].to_numpy(float)
                mean, sd = float(y.mean()), float(y.std(ddof=1))
                x = i+offsets[method]
                ax.bar(x, mean, width=.31, color=colors[method], alpha=.85,
                       edgecolor='none', zorder=2)
                ax.errorbar(x, mean, yerr=sd, fmt='none', ecolor='#20262A',
                            elinewidth=1.5, capsize=4.5, capthick=1.5, zorder=4)
                ax.scatter(x+jitter, y, s=50, facecolor='white', edgecolor='#20262A',
                           linewidth=.9, alpha=.25, zorder=5)
                label_y = mean-label_inset_fraction*panel['ylim']
                assert label_y > 0
                ax.text(x, label_y, f'{mean:.3f}', ha='center', va='center',
                        fontsize=FONT['annotation'], fontweight='bold', color='white', zorder=6)
                bar_tables.append(dict(panel=panel['title'], plate=plate, method=method,
                                       n_seeds=len(y), mean=mean, sd=sd, label_y=label_y,
                                       label_inset_axes_fraction=label_inset_fraction))
            lo, hi = panel['bracket']
            ax.plot([i+offsets['DeSCORT'], i+offsets['DeSCORT'],
                     i+offsets['Original'], i+offsets['Original']],
                    [lo, hi, hi, lo], color='#59656B', linewidth=.8, zorder=6)
            ax.text(i, panel['stars_y'], tests.loc[plate, 'stars'], ha='center', va='bottom',
                    fontsize=FONT['annotation'], fontweight='bold', color='#34434A')
        label_offsets_pt.append(label_inset_fraction*ax.get_position().height*fig.get_figheight()*72)
    np.testing.assert_allclose(label_offsets_pt, label_offsets_pt[0], atol=.01)
    fig.supylabel('Mean average precision', x=.015, fontsize=FONT['label'])
    handles = [Patch(facecolor=colors[m], edgecolor='none', alpha=.85, label=m) for m in methods]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.525, .965),
               ncol=2, frameon=False, fontsize=FONT['legend'])
    fig.canvas.draw()
    require_matplotlib_panel_alignment(fig, json_out=qa.with_suffix('.alignment.json'),
                                       tolerance_pt=1.5, gutter_tolerance_pt=1.5, strict=True)
    save_paper_figure(fig, stem, check_alignment=False, dpi=300)
    pd.concat(source_tables, ignore_index=True).to_csv(stem.with_suffix('.source_data.csv'), index=False)
    pd.DataFrame(bar_tables).to_csv(stem.with_suffix('.bar_summary.csv'), index=False)
    metadata = dict(font='Arial', font_files=arial_paths, source_sha256=provenance,
                    observations=90, exclusions=0, center='Mean of five seed means',
                    error_bars='Sample standard deviation across five seeds',
                    points='All five seed means; horizontal jitter is deterministic',
                    significance='Existing paired five-seed t-tests, Holm-adjusted across three plates within each task',
                    note='Seed repeats are computational repeats, not independent biological replicates.',
                    y_scales='Original y ranges retained and explicitly tick-labelled per panel',
                    numeric_label_inset_pt=label_offsets_pt,
                    changes='One shared y label and legend; requested titles; equal physical numeric-label inset; Arial',
                    original_single_panel_outputs_preserved=True)
    qa.with_suffix('.provenance.json').write_text(json.dumps(metadata, indent=2)+'\n')
    print(f'Saved {stem}.pdf / .svg / .png')
    return fig, stem


if __name__ == '__main__':
    combined_fig, combined_stem = draw_combined_map_bars()
    try:
        from IPython.display import display
        display(combined_fig)
    finally:
        plt.close(combined_fig)
