# Support both an installed package and direct execution from this checkout.
from pathlib import Path as _Path
import sys as _sys
_SOURCE = _Path("/fse/home/wangyongyan/Project/CellPainting/DeSCORT/other_model/DeSCORT/src")
if str(_SOURCE) not in _sys.path:
    _sys.path.insert(0, str(_SOURCE))

import os
import argparse
from pathlib import Path
from datasets import load_from_disk, concatenate_datasets
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge train_merged_data datasets from multiple perturbation-type directories."
    )

    parser.add_argument(
        "--base_dir", required=True,
        help="Root directory containing perturbation-type subdirectories (e.g., Compound, Crispr, Orf)."
    )
    parser.add_argument(
        "--save_path", required=True,
        help="Output path for the merged dataset."
    )
    parser.add_argument(
        "--subdir_name", default="train_merged_data",
        help="Subdirectory name to search for within each perturbation directory (default: %(default)s)."
    )
    parser.add_argument(
        "--exclude", nargs="*", default=["all_data"],
        help="Directory names to exclude (default: %(default)s)."
    )

    return parser.parse_args()


def main():
    args = parse_args()

    base_dir = Path(args.base_dir)
    save_path = Path(args.save_path)
    os.makedirs(save_path, exist_ok=True)

    # Find all perturbation-type subdirectories.
    pert_dirs = sorted(
        [d for d in base_dir.iterdir()
         if d.is_dir() and d.name not in args.exclude]
    )
    print(f"Found {len(pert_dirs)} perturbation-type directories: {[d.name for d in pert_dirs]}")

    # Iterate over perturbation types and load the specified subdirectory.
    all_datasets = []
    for pert_dir in pert_dirs:
        direct_path = pert_dir / args.subdir_name
        if direct_path.exists():
            print(f"  ✓ {pert_dir.name}/{args.subdir_name}")
            ds = load_from_disk(str(direct_path))
            all_datasets.append(ds)
            continue

        # If the subdirectory is not at the current level, search the timepoint subdirectories.
        timepoint_dirs = sorted([d for d in pert_dir.iterdir() if d.is_dir()])
        if not timepoint_dirs:
            print(f"  ⚠️ {pert_dir.name}: No subdirectories found")
            continue
        for tp_dir in tqdm(timepoint_dirs, desc=f"Loading {pert_dir.name}"):
            train_path = tp_dir / args.subdir_name
            if train_path.exists():
                print(f"    ✓ {pert_dir.name}/{tp_dir.name}/{args.subdir_name}")
                ds = load_from_disk(str(train_path))
                all_datasets.append(ds)

    if not all_datasets:
        raise ValueError("No valid datasets were found")

    print(f"\nMerging {len(all_datasets)} datasets...")
    merged_dataset = concatenate_datasets(all_datasets)

    print(f"Saving to {save_path}")
    merged_dataset.save_to_disk(str(save_path))

    print(f"✅ Done! Total samples: {len(merged_dataset):,d}")
    print(f"Features: {merged_dataset.features}")


if __name__ == "__main__":
    main()