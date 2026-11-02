'Prepare CRISPR data with optional train/test gene-list splitting.'
import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source_data_paths', nargs='+', required=True)
    parser.add_argument('--train_output_dir', required=True, help='Temporary shards for this cell line')
    parser.add_argument('--train_save_dir', required=True, help='Merged dataset for this cell line')
    for name in ['gene2index_pkl', 'esm2_dict_pkl', 'feature2index_pkl', 'need_feature_list_pkl']:
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--train_gene_list_pkl', help='Optional gene subset; omit to use all mapped genes with ESM2')
    parser.add_argument('--test_gene_list_pkl', help='Enable train/test splitting with explicit gene lists')
    parser.add_argument('--test_output_dir', help='Temporary test shards')
    parser.add_argument('--test_save_dir', help='Merged test dataset')
    parser.add_argument('--key_gene_column', default='Metadata_Foci_Barcode_MatchedTo_GeneCode')
    parser.add_argument('--control_key_word', default='nontargeting')
    parser.add_argument('--train_data_names', nargs='*', help='Path substrings selecting parquet files; omit for all')
    parser.add_argument('--test_data_names', nargs='*', help='Path substrings selecting test parquet files; omit for all')
    parser.add_argument('--nproc', type=int, default=1)
    parser.add_argument('--merge_nproc', type=int, default=2)
    parser.add_argument('--verbosity', type=int, choices=[0, 1], default=1)
    parser.add_argument('--check-only', action='store_true', help='Check paths and gene selection without processing data')
    args = parser.parse_args()
    if min(args.nproc, args.merge_nproc) < 1:
        parser.error('nproc and merge_nproc must be positive')
    test_options = [args.test_gene_list_pkl, args.test_output_dir, args.test_save_dir]
    if any(test_options) and not (all(test_options) and args.train_gene_list_pkl):
        parser.error('Test processing requires train_gene_list_pkl, test_gene_list_pkl, test_output_dir and test_save_dir')
    if args.test_data_names and not all(test_options):
        parser.error('test_data_names requires test processing options')
    if not args.test_gene_list_pkl:
        args.train_data_names = [v for v in (args.train_data_names or []) if v.strip()] or None
        paths = [Path(p).resolve() for p in [args.train_output_dir, args.train_save_dir]]
        if paths[0] == paths[1] or any(a in b.parents for a, b in [paths, paths[::-1]]):
            parser.error('Temporary and merged output directories must be separate, non-nested paths')
    return args


def filter_genes_by_esm2(gene_list, esm2_dict):
    """Preserve the original unseen-generation order, duplicates and control handling."""
    return [gene for gene in gene_list if gene in esm2_dict]


def process_split(processor, args, genes, features, selected_features, selected, split):
    merged = processor.process_and_merge(
        parquet_dirs=args.source_data_paths, output_dir=getattr(args, f'{split}_output_dir'),
        save_dir=getattr(args, f'{split}_save_dir'), data_name=getattr(args, f'{split}_data_names'),
        key_gene_column=args.key_gene_column, GENE2INDEX_DICT=genes,
        FEATURE2INDEX_DICT=features, need_genes_list=selected,
        need_feature_list=selected_features, merge_nproc=args.merge_nproc)
    print(f'[INFO] {split.capitalize()} processing done. Merged data saved to: '
          f'{getattr(args, f"{split}_save_dir")}')
    return merged
