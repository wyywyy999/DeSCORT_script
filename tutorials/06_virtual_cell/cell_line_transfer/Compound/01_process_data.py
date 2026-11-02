"""Build compound training and optional unseen-compound test datasets."""

# Support both an installed package and direct execution from this checkout.
from pathlib import Path as _Path
import sys as _sys
_SOURCE = _Path(__file__).resolve().parents[4] / "src"
if str(_SOURCE) not in _sys.path:
    _sys.path.insert(0, str(_SOURCE))


import sys
from pathlib import Path

from datasets import disable_progress_bars


from descort.data.dataset import DeSCORTPerturbPredictionProcessor

from descort.vc.preprocessing_compound import parse_args, load_smiles_list
from descort.vc.utils import load_pickle


def main():
    args = parse_args()
    disable_progress_bars()
    gene2index_dict = load_pickle(args.gene2index_pkl)
    feature2index_dict = load_pickle(args.feature2index_pkl)
    need_feature_list = load_pickle(args.need_feature_list_pkl)
    if not gene2index_dict:
        raise ValueError("The SMILES-to-index dictionary is empty")
    train_smiles = load_smiles_list(args.train_gene_list_pkl, gene2index_dict, args.control_key_word)
    test_smiles = load_smiles_list(args.test_gene_list_pkl, gene2index_dict, args.control_key_word)
    if test_smiles is not None:
        overlap = set(train_smiles) & set(test_smiles) - {args.control_key_word}
        if overlap:
            raise ValueError(f"Train/test perturbation SMILES overlap: {sorted(overlap)}")
    for name, values in [("Train", train_smiles), ("Test", test_smiles)]:
        if values is not None:
            print(f"[INFO] {name}: {len(set(values) - {args.control_key_word})} perturbations; controls retained for pairing")
    processor = DeSCORTPerturbPredictionProcessor(
        nproc=args.nproc, verbosity=args.verbosity,
        control_key_word=args.control_key_word,
    )
    merged = processor.process_and_merge(
        parquet_dirs=args.source_data_paths,
        output_dir=args.train_output_dir,
        save_dir=args.train_save_dir,
        data_name=args.train_data_names,
        key_gene_column=args.key_gene_column,
        GENE2INDEX_DICT=gene2index_dict,
        FEATURE2INDEX_DICT=feature2index_dict,
        need_genes_list=train_smiles,
        need_feature_list=need_feature_list,
        merge_nproc=args.merge_nproc,
    )
    print(f"[INFO] Train processing done: {len(merged)} rows saved to {args.train_save_dir}")
    if test_smiles is not None:
        merged = processor.process_and_merge(
            parquet_dirs=args.source_data_paths,
            output_dir=args.test_output_dir,
            save_dir=args.test_save_dir,
            data_name=args.test_data_names,
            key_gene_column=args.key_gene_column,
            GENE2INDEX_DICT=gene2index_dict,
            FEATURE2INDEX_DICT=feature2index_dict,
            need_genes_list=test_smiles,
            need_feature_list=need_feature_list,
            merge_nproc=args.merge_nproc,
        )
        print(f"[INFO] Test processing done: {len(merged)} rows saved to {args.test_save_dir}")


if __name__ == "__main__":
    main()
