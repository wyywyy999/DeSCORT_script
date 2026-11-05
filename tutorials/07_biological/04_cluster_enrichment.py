#!/usr/bin/env python3
"""Plot selected clusters' enrichment from an existing clustering run."""
from pathlib import Path
import logging
import sys

_SOURCE = Path(__file__).resolve().parents[2] / 'src'
if str(_SOURCE) not in sys.path:
    sys.path.insert(0, str(_SOURCE))

from descort.biological.cluster_enrichment import (
    parser, load_enrichment, select_cluster_enrichment, plot_cluster_enrichment_bubble,
)
from descort.biological.utils import save_json


def main():
    args = parser().parse_args()
    args.cluster = list(dict.fromkeys(args.cluster))
    if any(c < 0 for c in args.cluster) or not 0 < args.fdr < 1 or args.png_dpi < 1 or args.top_n < 1:
        raise ValueError('Clusters must be nonnegative, FDR in (0,1), and DPI/top-n positive')
    enrichment = load_enrichment(args.enrichment_file, args.cluster)
    out = args.output_dir.resolve()
    suffix = '_p.adjust' if args.p_value_mode == 'adjusted' else ''
    results, figures, qa = out/f'results{suffix}', out/f'figures{suffix}', out/f'qa{suffix}'
    for folder in [results, figures, qa]:
        folder.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s',
                        handlers=[logging.StreamHandler(), logging.FileHandler(out/f'analysis{suffix}.log')])
    logging.getLogger('fontTools').setLevel(logging.WARNING)
    save_json(out/f'parameters{suffix}.json', vars(args))
    summaries = []
    for cluster in args.cluster:
        selected = select_cluster_enrichment(enrichment, cluster, args.fdr, results)
        tested = enrichment.loc[(enrichment.representation == 'DeSCORT') & (enrichment.cluster == cluster)].copy()
        plot_cluster_enrichment_bubble(tested, cluster, args.fdr, figures, qa,
                                       args.figure_formats, args.png_dpi, top_n=args.top_n,
                                       p_value_mode=args.p_value_mode)
        eligible_count = len(selected) if args.p_value_mode == 'adjusted' else int((tested.p_value < .05).sum())
        summary = {
            'enrichment_file': str(args.enrichment_file.resolve()),
            'representation': 'DeSCORT', 'cluster': cluster, 'fdr': args.fdr,
            'selected_terms': len(selected),
            'raw_p_pass_terms': int((tested.p_value < .05).sum()),
            'displayed_terms': min(eligible_count, args.top_n),
            'top_n': args.top_n,
            'plot_ranking': (f'Filter p.adjust < {args.fdr:g}, then p.adjust ascending (ties: GeneRatio descending)'
                             if args.p_value_mode == 'adjusted' else
                             'Filter raw p_value < 0.05, then p_value ascending (ties: GeneRatio descending)'),
            'plot_color': '-log10(p.adjust)' if args.p_value_mode == 'adjusted' else '-log10(p_value)',
            'p_value_mode': args.p_value_mode,
            'database_counts': selected.database.value_counts().to_dict(),
            'fdr_column': 'p_adj_BH_within_database',
            'correction_scope': 'Retain database-wide BH correction from the input; no reclustering or FDR recalculation',
        }
        summaries.append(summary)
        save_json(results/f'cluster{cluster}_analysis_summary.json', summary)
        logging.info('Cluster %s: %s BH-significant terms; %s raw-P-passing terms; output: %s',
                     cluster, len(selected), summary['raw_p_pass_terms'], out)
    save_json(results/'analysis_summary.json', summaries[0] if len(summaries)==1 else
              {'clusters':args.cluster,'cluster_results':summaries})


if __name__ == '__main__':
    main()
