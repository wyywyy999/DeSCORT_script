"""Standalone cluster enrichment selection and plotting from saved cluster results."""
from pathlib import Path
import argparse
import logging
import re

import matplotlib as mpl
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .utils import TUTORIAL_DIR, export
from .audit_panel_alignment import require_matplotlib_panel_alignment


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--enrichment-file', type=Path, default=Path('/fse/home/wangyongyan/Project/CellPainting/DeSCORT/tutorials/07_biological_new/result/03_de_cluster/hit_10_cluster/results/cluster_pathway_complex_enrichment.csv'))
    p.add_argument('--output-dir', type=Path, default=Path('/fse/home/wangyongyan/Project/CellPainting/DeSCORT/tutorials/07_biological_new/result/04_cluster_enrichment'))
    p.add_argument('--cluster', '--enrichment-cluster', dest='cluster', type=int, nargs='+', default=[3],
                   help='One or more cluster IDs separated by spaces')
    p.add_argument('--fdr', '--enrichment-fdr', dest='fdr', type=float, default=.05)
    p.add_argument('--figure-formats', nargs='+', type=str.lower, choices=['png','pdf','svg'], default=['pdf','svg','png'])
    p.add_argument('--png-dpi', type=int, default=400)
    p.add_argument('--top-n', type=int, default=10, help='Maximum terms to display, ranked by ascending raw P value')
    return p


def load_enrichment(path, cluster):
    enrichment = pd.read_csv(path, low_memory=False)
    required = {'representation','cluster','database','term','overlap_n','p_adj_BH_within_database'}
    if not required <= set(enrichment.columns):
        raise ValueError(f'Missing enrichment columns: {sorted(required-set(enrichment.columns))}')
    available = sorted(enrichment.loc[enrichment.representation == 'DeSCORT', 'cluster'].unique())
    requested = [cluster] if np.isscalar(cluster) else cluster
    missing = [c for c in requested if c not in available]
    if missing:
        raise ValueError(f'Clusters {missing} not found for DeSCORT; available IDs: {available}')
    return enrichment


def select_cluster_enrichment(enrichment: pd.DataFrame, cluster: int, fdr: float, results: Path) -> pd.DataFrame:
    """Export significant DeSCORT terms using database-wide BH correction."""
    selected = enrichment.loc[
        (enrichment["representation"] == "DeSCORT")
        & (enrichment["cluster"] == cluster)
        & (enrichment["p_adj_BH_within_database"] < fdr)
    ].copy()
    selected["overlap_n"] = selected["overlap_n"].astype(int)
    selected["neg_log10_fdr"] = -np.log10(
        np.maximum(selected["p_adj_BH_within_database"].astype(float), 1e-300)
    )
    selected = selected.sort_values(
        ["neg_log10_fdr", "overlap_n", "database", "term"],
        ascending=[False, False, True, True],
    ).reset_index(drop=True)
    selected["label"] = selected["database"].astype(str) + "  " + selected["term"].astype(str)
    selected.to_csv(results / f"cluster{cluster}_DeSCORT_enrichment_significant.csv", index=False)
    return selected


def wrap_term_label(term, width=36):
    """Remove annotation IDs and wrap a long pathway name into two lines."""
    name = re.sub(r'\s*\(GO:\d+\)', '', str(term)).replace('_', ' ')
    name = re.sub(r'^(?:GO[ _](?:BP|CC|MF)|CORUM|KEGG|ANNOTATED)\s*[:|—-]?\s+', '', name)
    if len(name) <= width:
        return name
    spaces = [m.start() for m in re.finditer(r'\s+', name)]
    if not spaces:
        middle = len(name)//2
        return name[:middle]+'\n'+name[middle:]
    middle = min(spaces, key=lambda i: abs(i-len(name)/2))
    return name[:middle].rstrip()+'\n'+name[middle:].lstrip()


def prepare_enrichment_plot_data(tested, top_n):
    """Keep raw-P-passing rows and mark the raw-P-ranked plotting subset."""
    if top_n < 1:
        raise ValueError('top_n must be positive')
    if 'cluster_size' not in tested or (tested.cluster_size <= 0).any():
        raise ValueError('A positive cluster_size is required to calculate GeneRatio')
    data = tested.copy()
    data['GeneRatio'] = data.overlap_n/data.cluster_size
    data['Count'] = data.overlap_n.astype(int)
    data['p.adjust'] = data.p_adj_BH_within_database.astype(float)
    data['neg_log10_p_adjust'] = -np.log10(np.maximum(data['p.adjust'],1e-300))
    data['neg_log10_p_value'] = -np.log10(np.maximum(data.p_value.astype(float),1e-300))
    data = data.sort_values(['p_value','GeneRatio','database','term'],
                            ascending=[True,False,True,True],kind='stable').reset_index(drop=True)
    data['passes_raw_p_005'] = data.p_value < .05
    data['plotted'] = data.index.isin(data.index[data.passes_raw_p_005][:top_n])
    data['display_term'] = data.term.map(wrap_term_label)
    return data.loc[data.passes_raw_p_005].copy()


def plot_cluster_enrichment_bubble(selected, cluster, fdr, figures, qa, formats, dpi, top_n=10):
    """Filter raw P<.05, rank by ascending raw P; color=−log10(raw p_value); area=Count."""
    eligible = prepare_enrichment_plot_data(selected,top_n)
    figures.mkdir(parents=True, exist_ok=True); qa.mkdir(parents=True, exist_ok=True)
    eligible.to_csv(figures/f'figure_cluster{cluster}_enrichment_bubble_data.csv',index=False)
    selected = eligible.loc[eligible.plotted].copy()
    logging.info('Cluster %s: %d terms pass raw P < 0.05; displaying %d by ascending raw P',cluster,len(eligible),len(selected))
    if selected.empty:
        logging.info('No terms pass raw P < 0.05; bubble plot skipped')
        return
    with mpl.rc_context({'font.family':'sans-serif', 'font.sans-serif':['Arial','DejaVu Sans'],
                         'font.size':9, 'pdf.fonttype':42, 'svg.fonttype':'none',
                         'axes.spines.top':False, 'axes.spines.right':False}):
        fig, ax = plt.subplots(figsize=(8.4,max(4.8,.60*len(selected)+1.6)), dpi=dpi)
        fig.subplots_adjust(left=.47,right=.76,bottom=.14,top=.88)
        y = np.arange(len(selected))[::-1]
        size_scale = 32.0
        points = ax.scatter(selected.GeneRatio, y, s=size_scale*selected.Count,
                            c=selected['neg_log10_p_value'], cmap=mpl.colors.LinearSegmentedColormap.from_list('blue_to_red', ['#2166AC', '#B2182B']),
                            edgecolor='white', linewidth=.7, zorder=3)
        ax.set_yticks(y, selected.display_term)
        ax.set(xlabel='GeneRatio', ylabel='')
        fig.text(.60,.035,'Selection: raw P < 0.05; ranked by P value (ascending)',
                 ha='center', va='center', fontsize=8)
        low, high = selected.GeneRatio.min(), selected.GeneRatio.max()
        pad = max((high-low)*.22,high*.10,.005)
        ax.set_xlim(max(0,low-pad), high+pad)
        ax.set_ylim(-.7,len(selected)-.3)
        ax.grid(axis='x',color='#E7EBEE',linewidth=.7);ax.set_axisbelow(True)
        cax = fig.add_axes([.82,.59,.025,.25])
        cb=fig.colorbar(points,cax=cax)
        cb.set_label('−log10(P value)', fontsize=9)
        cb.ax.tick_params(labelsize=8)
        counts = sorted(set(int(v) for v in np.quantile(selected.Count,[0,.5,1])))
        handles = [ax.scatter([],[],s=size_scale*c,color='#778899',edgecolor='white',label=str(c))
                   for c in counts]
        ax.legend(handles=handles,title='Count',loc='upper left',bbox_to_anchor=(1.13,.42),
                  frameon=False,labelspacing=1.4,handletextpad=1.5,borderaxespad=0)
        stem = figures/f'figure_cluster{cluster}_enrichment_bubble'
        fig.canvas.draw()
        require_matplotlib_panel_alignment(fig,json_out=str(qa/(stem.name+'.alignment.json')),
            overlay_svg=str(qa/(stem.name+'.alignment.svg')),exclude_axes=[cax],
            require_panel_labels=False,strict=True)
        export(fig,stem,dpi,formats=formats)

