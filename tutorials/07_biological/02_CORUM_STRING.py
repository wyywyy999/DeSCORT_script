#!/usr/bin/env python3
"""Run the corum_string workflow using functions from descort.biological."""

# Support both an installed package and direct execution from this checkout.
from pathlib import Path as _Path
import sys as _sys
_SOURCE = _Path(__file__).resolve().parents[2] / "src"
if str(_SOURCE) not in _sys.path:
    _sys.path.insert(0, str(_SOURCE))

from pathlib import Path
import sys


import logging
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from descort.biological import utils as io
from descort.biological.corum_string import (
    LOG,
    build_pairs,
    collect_profiles,
    compute,
    paired_changes,
    parser,
    plot_corum_distribution,
    plot_string_similarity_line,
    plot_results,
)


def main():
    args=parser().parse_args()
    if not 0<args.pca_variance<1 or not 0<=args.corum_min_coverage<=1:raise ValueError('Invalid PCA/coverage fraction')
    if not 0<=args.string_low_score<args.string_high_score<=1000:raise ValueError('Invalid STRING score thresholds')
    if args.threads<1 or args.string_chunk_size<1 or args.pair_chunk_size<1:raise ValueError('Threads and chunk sizes must be positive')
    if args.stage not in ['corum-plot','string-plot']:
        for path in [args.corum_file,args.string_file]:
            if not path.is_file():raise FileNotFoundError(f'Required input missing: {path}')
    args.output_dir=args.output_dir.resolve();args.output_dir.mkdir(parents=True,exist_ok=True)
    logging.basicConfig(level=logging.INFO,format='%(asctime)s | %(levelname)s | %(message)s',handlers=[logging.StreamHandler(),logging.FileHandler(args.output_dir/f'{args.stage}.log')])
    logging.getLogger('fontTools').setLevel(logging.WARNING)
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['DejaVu Sans'],'font.size':8,'axes.labelsize':9,'axes.linewidth':.7,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none'})
    io.save_json(args.output_dir/f'{args.stage}_parameters.json',vars(args))
    if args.stage=='corum-plot':
        table=args.output_dir/'results/CORUM_background_corrected.csv'
        if not table.is_file():raise FileNotFoundError(f'Saved CORUM results missing: {table}')
        folder=args.output_dir/'figures';folder.mkdir(exist_ok=True)
        plot_corum_distribution(pd.read_csv(table),args,folder)
        LOG.info('Redrew CORUM distribution from %s',table)
        return
    if args.stage=='string-plot':
        table=args.output_dir/'results/STRING_background_corrected.csv'
        if not table.is_file():raise FileNotFoundError(f'Saved STRING results missing: {table}')
        folder=args.output_dir/'figures';folder.mkdir(exist_ok=True)
        plot_string_similarity_line(pd.read_csv(table),args,folder)
        LOG.info('Redrew STRING line plot with fixed similarity intervals from %s',table)
        return
    if args.stage=='check':
        io.resolve_hit_file(args.hit_file)
        for rep in ['model','original']:io.inspect_input(io.get_files(getattr(args,rep+'_input'),args.plates),rep,args)
        LOG.info('Inputs checked; strategy=select_genes_then_PCA, background subtraction enabled');return
    result=args.output_dir/'results';folder=args.output_dir/'figures'
    result.mkdir(exist_ok=True);folder.mkdir(exist_ok=True)
    profiles,genes=collect_profiles(args,result)
    pairs=build_pairs(genes,args,result)
    metrics,corum,string,bins,evrs=compute(profiles,genes,pairs,args,result)
    plot_results(corum,string,bins,evrs,args,folder)
    paired_changes(metrics,corum,string,args,result,folder)
    LOG.info('Metrics:\n%s',metrics.to_string(index=False));LOG.info('Completed: %s',args.output_dir)


if __name__ == '__main__':
    main()
