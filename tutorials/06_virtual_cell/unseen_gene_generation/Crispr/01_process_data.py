"""Prepare CRISPR data with optional train/test gene-list splitting."""

# Support both an installed package and direct execution from this checkout.
from pathlib import Path as _Path
import sys as _sys
_SOURCE = _Path(__file__).resolve().parents[4] / "src"
if str(_SOURCE) not in _sys.path:
    _sys.path.insert(0, str(_SOURCE))

import json
import sys
from pathlib import Path


from datasets import disable_progress_bars
from descort.data.dataset import DeSCORTPerturbPredictionProcessor

from descort.vc.preprocessing_crispr import parse_args, filter_genes_by_esm2, process_split
from descort.vc.utils import load_pickle


def main():
    args = parse_args()
    disable_progress_bars()
    if not args.test_gene_list_pkl or args.check_only:
        for source in args.source_data_paths:
            if not Path(source).is_dir():
                raise FileNotFoundError(source)
    genes = load_pickle(args.gene2index_pkl)
    esm2 = load_pickle(args.esm2_dict_pkl)
    features = load_pickle(args.feature2index_pkl)
    selected_features = load_pickle(args.need_feature_list_pkl)
    if args.test_gene_list_pkl:
        # Preserve unseen-generation selection exactly: ESM2 filtering only,
        # retaining input order and the control entries supplied in each list.
        train = filter_genes_by_esm2(load_pickle(args.train_gene_list_pkl), esm2)
        test = filter_genes_by_esm2(load_pickle(args.test_gene_list_pkl), esm2)
        if args.check_only:
            print(json.dumps(dict(mode='train_test', train_gene_entries=len(train),
                                  test_gene_entries=len(test), selected_features=len(selected_features),
                                  train_output=args.train_save_dir, test_output=args.test_save_dir), indent=2))
            return
        processor = DeSCORTPerturbPredictionProcessor(
            nproc=args.nproc, verbosity=args.verbosity, control_key_word=args.control_key_word)
        process_split(processor, args, genes, features, selected_features, train, 'train')
        process_split(processor, args, genes, features, selected_features, test, 'test')
        return
    candidates = (list(load_pickle(args.train_gene_list_pkl)) if args.train_gene_list_pkl
                  else sorted(set(genes) & set(esm2)))
    selected = list(dict.fromkeys(g for g in candidates
                                 if g in genes and g in esm2 and g != args.control_key_word))
    if not selected:
        raise ValueError('No perturbation genes have both an index and an ESM2 embedding')
    if not features or not selected_features:
        raise ValueError('Feature vocabulary and selected feature list must be nonempty')
    # Controls are retained for pairing even when a subset does not explicitly list them.
    selected.append(args.control_key_word)
    print(json.dumps(dict(source_data_paths=args.source_data_paths,
                          perturbation_genes=len(selected)-1,
                          gene_selection='optional list' if args.train_gene_list_pkl else 'all mapped genes with ESM2',
                          selected_features=len(selected_features), control=args.control_key_word,
                          data_names=args.train_data_names, output=args.train_save_dir,
                          check_only=args.check_only), indent=2), flush=True)
    if args.check_only:
        return
    processor = DeSCORTPerturbPredictionProcessor(
        nproc=args.nproc, verbosity=args.verbosity, control_key_word=args.control_key_word)
    merged = process_split(processor, args, genes, features, selected_features, selected, 'train')
    print(f'[INFO] Cell-line processing done: {len(merged)} rows saved to {args.train_save_dir}')


if __name__ == '__main__':
    main()
