#!/usr/bin/env python3
"""Index local resources and source files without copying datasets or checkpoints."""
import argparse
import ast
import hashlib
import importlib.metadata
import json
from pathlib import Path
import pickle
import re

ROOT = Path(__file__).resolve().parents[1]


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def describe(value):
    info = {"python_type": type(value).__name__}
    if hasattr(value, "__len__"):
        info["entries"] = len(value)
    if hasattr(value, "shape"):
        info["shape"] = list(value.shape)
    if isinstance(value, dict) and value:
        first = next(iter(value.values()))
        info["example_value_type"] = type(first).__name__
        if hasattr(first, "shape"):
            info["example_value_shape"] = list(first.shape)
        elif isinstance(first, (list, tuple)):
            info["example_value_length"] = len(first)
    return info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-root", type=Path, default=ROOT.parents[1])
    parser.add_argument("--inspect-trusted-pickles", action="store_true",
                        help="Inspect only trusted local pickle resources up to 2 MiB")
    args = parser.parse_args()
    original = args.original_root.resolve()
    resource_root = original / "common_files"
    resources = []
    for path in sorted(resource_root.rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts:
            continue
        if path.suffix not in {".pkl", ".json", ".csv", ".txt", ".tsv", ".xlsx", ".npy", ".npz"}:
            continue
        size = path.stat().st_size
        small = size <= 2 * 1024 * 1024
        item = {"path": str(path), "relative_to_original": str(path.relative_to(original)),
                "bytes": size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest() if small else None,
                "source": "existing local project resource; upstream provenance not independently verified",
                "details": None}
        if args.inspect_trusted_pickles and small and path.suffix == ".pkl":
            try:
                with path.open("rb") as file:
                    item["details"] = describe(pickle.load(file))
            except Exception as error:
                item["inspection_error"] = f"{type(error).__name__}: {error}"
        resources.append(item)
    save(ROOT / "resources/manifest.json", {
        "original_resource_root": str(resource_root),
        "hash_policy": "SHA-256 for files up to 2 MiB; larger files are indexed without reading their payload",
        "resources": resources,
    })
    workflows, external = [], []
    for path in sorted((ROOT / "tutorials").rglob("*")):
        if not path.is_file() or path.suffix not in {".py", ".sh"}:
            continue
        source = path.read_text()
        entry = {"path": str(path.relative_to(ROOT)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        if path.suffix == ".py":
            tree = ast.parse(source)
            entry["functions"] = [node.name for node in tree.body if isinstance(node, ast.FunctionDef)]
            entry["imports"] = sorted({node.module for node in ast.walk(tree)
                                        if isinstance(node, ast.ImportFrom) and node.module})
        workflows.append(entry)
        literals = sorted(set(re.findall(r"/(?:fse|data)/[^\s\"'<>]+", source)))
        if literals:
            external.append({"path": str(path.relative_to(ROOT)), "absolute_path_literals": literals})
    save(ROOT / "docs/reorganization/workflow_inventory.json", workflows)
    save(ROOT / "docs/reorganization/external_paths.json", external)
    baseline_paths = ["tutorials/06_vc/MLP", "tutorials/06_vc/Rigde_model",
                      "tutorials/06_vc_compound/MLP", "tutorials/06_vc_compound/Ridge",
                      "tutorials/06_vc_compound/GEARS", "tutorials/06_vc_compound/STATE",
                      "tutorials/06_vc_compound/delta_transfer"]
    save(ROOT / "benchmarks/manifest.json", [{
        "original_path": str(original / relative), "exists": (original / relative).is_dir(),
        "entry_scripts": sorted(p.name for p in (original / relative).glob("*.py")),
    } for relative in baseline_paths])
    output_paths = ["result", "tutorials/06_vc/result", "tutorials/06_vc/test_anndata",
                    "tutorials/06_vc_compound/result", "tutorials/06_vc_compound/test_anndata",
                    "tutorials/07_biological_new/result", "tutorials/09_paper_plot_code/result"]
    save(ROOT / "docs/reorganization/outputs_index.json", [{
        "original_path": str(original / relative), "exists": (original / relative).is_dir(),
        "policy": "historical outputs retained in original project; read-only source for new experiments",
    } for relative in output_paths])
    packages = ["torch", "transformers", "datasets", "numpy", "pandas", "pyarrow", "scipy",
                "scikit-learn", "anndata", "accelerate", "safetensors", "deepspeed", "pycytominer",
                "matplotlib", "seaborn", "scanpy", "copairs", "duckdb", "gseapy", "umap-learn"]
    versions = []
    for package in packages:
        try:
            versions.append(f"{package}=={importlib.metadata.version(package)}")
        except importlib.metadata.PackageNotFoundError:
            pass
    (ROOT / "environments/requirements-tested.txt").write_text("\n".join(versions) + "\n")
    print(f"Indexed {len(resources)} resources and {len(workflows)} workflow files")


if __name__ == "__main__":
    main()
