# Support both an installed package and direct execution from this checkout.
from pathlib import Path as _Path
import sys as _sys
_SOURCE = _Path(__file__).resolve().parents[2] / "src"
if str(_SOURCE) not in _sys.path:
    _sys.path.insert(0, str(_SOURCE))

import os
import argparse
from pathlib import Path
from datasets import load_from_disk, concatenate_datasets
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description="合并多个扰动类型目录下的 train_merged_data 数据集。"
    )

    parser.add_argument(
        "--base_dir", required=True,
        help="根目录，包含多个扰动类型子目录（如 Compound, Crispr, Orf 等）。"
    )
    parser.add_argument(
        "--save_path", required=True,
        help="合并后数据集的保存路径。"
    )
    parser.add_argument(
        "--subdir_name", default="train_merged_data",
        help="在每个扰动目录下查找的子目录名称。（default: %(default)s）"
    )
    parser.add_argument(
        "--exclude", nargs="*", default=["all_data"],
        help="需要排除的目录名列表。（default: %(default)s）"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    base_dir = Path(args.base_dir)
    save_path = Path(args.save_path)
    os.makedirs(save_path, exist_ok=True)

    # 查找所有扰动类型子目录
    pert_dirs = sorted(
        [d for d in base_dir.iterdir()
         if d.is_dir() and d.name not in args.exclude]
    )
    print(f"发现 {len(pert_dirs)} 个扰动类型目录: {[d.name for d in pert_dirs]}")

    # 遍历每个扰动类型，加载指定子目录
    all_datasets = []
    for pert_dir in pert_dirs:
        direct_path = pert_dir / args.subdir_name
        if direct_path.exists():
            print(f"  ✓ {pert_dir.name}/{args.subdir_name}")
            ds = load_from_disk(str(direct_path))
            all_datasets.append(ds)
            continue

        # 如果子目录不在当前层级，尝试在时间点子目录下查找
        timepoint_dirs = sorted([d for d in pert_dir.iterdir() if d.is_dir()])
        if not timepoint_dirs:
            print(f"  ⚠️ {pert_dir.name}: 无子目录")
            continue
        for tp_dir in tqdm(timepoint_dirs, desc=f"加载 {pert_dir.name}"):
            train_path = tp_dir / args.subdir_name
            if train_path.exists():
                print(f"    ✓ {pert_dir.name}/{tp_dir.name}/{args.subdir_name}")
                ds = load_from_disk(str(train_path))
                all_datasets.append(ds)

    if not all_datasets:
        raise ValueError("没有找到任何有效数据集")

    print(f"\n合并 {len(all_datasets)} 个数据集...")
    merged_dataset = concatenate_datasets(all_datasets)

    print(f"保存到 {save_path}")
    merged_dataset.save_to_disk(str(save_path))

    print(f"✅ 完成! 总样本数: {len(merged_dataset):,d}")
    print(f"特征: {merged_dataset.features}")


if __name__ == "__main__":
    main()