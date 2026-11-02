#!/usr/bin/env python3
"""Run the umap_analysis workflow using functions from descort.biological."""

# Support both an installed package and direct execution from this checkout.
from pathlib import Path as _Path
import sys as _sys
_SOURCE = _Path(__file__).resolve().parents[2] / "src"
if str(_SOURCE) not in _sys.path:
    _sys.path.insert(0, str(_SOURCE))

from pathlib import Path
import sys


import logging
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from descort.biological.utils import (
    get_files,
    inspect_input,
    resolve_hit_file,
    save_json,
)
from descort.biological.umap_analysis import (
    LOG,
    analyze,
    compare,
    gene_sets,
    parser,
    redraw_combined_pathways,
    run_neighbor_plots,
)


def main():
    args=parser().parse_args()
    if not 0<args.pca_variance<1:raise ValueError('pca-variance must be in (0,1)')
    if not 0<args.top_fraction<=1 or not 0<=args.compact_percentile<=100:raise ValueError('Invalid fraction/percentile')
    if not 0<args.pathway_plot_fraction<=1:raise ValueError('pathway-plot-fraction must be in (0,1]')
    if args.min_go_kegg_genes<2 or args.min_annotated_genes<2 or args.threads<1 or args.max_pathway_plots<0:raise ValueError('Invalid size/thread parameter')
    args.output_dir=args.output_dir.resolve();args.output_dir.mkdir(parents=True,exist_ok=True)
    logging.getLogger('fontTools').setLevel(logging.WARNING)
    logging.basicConfig(level=logging.INFO,format='%(asctime)s | %(levelname)s | %(message)s',handlers=[logging.StreamHandler(),logging.FileHandler(args.output_dir/f'{args.stage}.log')])
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['DejaVu Sans'],'font.size':10,'axes.linewidth':.8,'pdf.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
    save_json(args.output_dir/f'{args.stage}_parameters.json',vars(args))
    model=args.model_results or args.output_dir/'model/results'
    original=args.original_results or args.output_dir/'original/results'
    if args.query_top_n < 1:raise ValueError('query-top-n must be positive')
    if args.stage=='query':
        if args.query_gene:
            run_neighbor_plots(args)
        else:
            LOG.info('No query genes specified; skipping figure 7')
        return
    if args.stage=='combined-plot':
        redraw_combined_pathways(model,args,'model')
        redraw_combined_pathways(original,args,'original')
        LOG.info('Combined UMAP figures redrawn from cached results');return
    if args.stage=='check':
        for rep in ['model','original']:
            inspect_input(get_files(getattr(args,rep+'_input'),args.plates),rep,args)
        resolve_hit_file(args.hit_file)
        if not args.annotated_gene_sets.is_file():raise FileNotFoundError('Missing annotated gene sets')
        LOG.info('Input checks passed');return
    if args.stage in ['all','model','original']:
        sets=gene_sets(args)
        if args.stage in ['all','model']:model=analyze('model',args,sets)
        if args.stage in ['all','original']:original=analyze('original',args,sets)
    if args.stage in ['all','compare']:compare(model,original,args)
    if args.stage=='all' and args.query_gene:
        run_neighbor_plots(args,model,original)
    LOG.info('Completed stage %s; results: %s',args.stage,args.output_dir)


if __name__ == '__main__':
    main()
