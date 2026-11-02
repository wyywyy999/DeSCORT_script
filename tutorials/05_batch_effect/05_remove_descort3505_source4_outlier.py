"""Remove the detached source_4 UMAP cluster identified in DeSCORT-3505.

The reference cluster is selected once from DeSCORT10epochs3505_adata.h5ad.
The same observations are removed from every H5AD by unique obs_name so that
all batch-correction methods retain exactly matched samples.
"""

from __future__ import annotations

# Support both an installed package and direct execution from this checkout.
from pathlib import Path as _Path
import sys as _sys
_SOURCE = _Path(__file__).resolve().parents[2] / "src"
if str(_SOURCE) not in _sys.path:
    _sys.path.insert(0, str(_SOURCE))


import gc
import json
from pathlib import Path

import anndata as ad
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_DIR = (
    PROJECT_ROOT.parents[1]
    / "result/05_batch_effect/Target_2_Compoud/batch_adata"
)
OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs/05_batch_effect/filtered"
)
REFERENCE_NAME = "DeSCORT10epochs3505_adata.h5ad"
SOURCE_KEY = "Metadata_Source"
CLUSTER_KEY = "Metadata_Cluster"
SOURCE_VALUE = "source_4"
CLUSTER_VALUE = "26"
EXPECTED_REMOVALS = 244


def identify_reference_cluster() -> tuple[list[str], dict]:
    reference_path = INPUT_DIR / REFERENCE_NAME
    reference = ad.read_h5ad(reference_path, backed="r")
    try:
        source = reference.obs[SOURCE_KEY].astype(str).to_numpy()
        cluster = reference.obs[CLUSTER_KEY].astype(str).to_numpy()
        coordinates = np.asarray(reference.obsm["X_umap"])[:, :2]

        selected = (source == SOURCE_VALUE) & (cluster == CLUSTER_VALUE)
        if int(selected.sum()) != EXPECTED_REMOVALS:
            raise ValueError(
                f"Expected {EXPECTED_REMOVALS} reference observations, "
                f"found {int(selected.sum())}"
            )

        # Geometry guard: cluster 26 is exactly the detached source_4 island.
        detached_by_gap = (source == SOURCE_VALUE) & (coordinates[:, 0] > 9.0)
        if not np.array_equal(selected, detached_by_gap):
            raise ValueError(
                "Reference cluster no longer matches the detached UMAP island; "
                "review the selection before filtering"
            )

        remove_names = reference.obs_names[selected].astype(str).tolist()
        if len(remove_names) != len(set(remove_names)):
            raise ValueError("Reference obs_names are not unique")

        selected_obs = reference.obs.loc[selected].copy()
        selected_obs["Reference_UMAP1"] = coordinates[selected, 0]
        selected_obs["Reference_UMAP2"] = coordinates[selected, 1]
        selected_obs.to_csv(OUTPUT_DIR / "removed_observations.csv", index=True)

        summary = {
            "reference_file": REFERENCE_NAME,
            "selection": {
                SOURCE_KEY: SOURCE_VALUE,
                CLUSTER_KEY: CLUSTER_VALUE,
                "geometry_guard": "X_umap[:, 0] > 9.0 within source_4",
            },
            "removed_observations": len(remove_names),
            "reference_umap_range": {
                "UMAP1": [
                    float(coordinates[selected, 0].min()),
                    float(coordinates[selected, 0].max()),
                ],
                "UMAP2": [
                    float(coordinates[selected, 1].min()),
                    float(coordinates[selected, 1].max()),
                ],
            },
            "affected_plates": sorted(
                selected_obs["Metadata_Plate"].astype(str).unique().tolist()
            ),
            "affected_batches": sorted(
                selected_obs["Metadata_Batch"].astype(str).unique().tolist()
            ),
        }
        return remove_names, summary
    finally:
        reference.file.close()


def write_filtered_h5ad(input_path: Path, remove_names: set[str]) -> dict:
    output_path = OUTPUT_DIR / input_path.name
    temporary_path = OUTPUT_DIR / f".{input_path.stem}.tmp.h5ad"

    adata = ad.read_h5ad(input_path, backed="r")
    try:
        names = adata.obs_names.astype(str)
        if not adata.obs_names.is_unique:
            raise ValueError(f"{input_path.name}: obs_names are not unique")

        found = set(names).intersection(remove_names)
        if found != remove_names:
            raise ValueError(
                f"{input_path.name}: expected all {len(remove_names)} removal IDs, "
                f"found {len(found)}"
            )

        keep = ~names.isin(remove_names)
        n_before = int(adata.n_obs)
        n_after = int(keep.sum())
        if n_before - n_after != len(remove_names):
            raise ValueError(f"{input_path.name}: removal count mismatch")

        # Backed copy writes a matched subset while preserving X, obs/var,
        # layers, obsm/varm, obsp/varp and uns.
        filtered = adata[keep].copy(filename=temporary_path)
        filtered.file.close()
        temporary_path.replace(output_path)
    finally:
        adata.file.close()

    check = ad.read_h5ad(output_path, backed="r")
    try:
        remaining = set(check.obs_names.astype(str)).intersection(remove_names)
        if check.n_obs != n_after or remaining:
            raise ValueError(f"{input_path.name}: post-write validation failed")
        result = {
            "file": input_path.name,
            "n_before": n_before,
            "n_removed": len(remove_names),
            "n_after": int(check.n_obs),
            "n_vars": int(check.n_vars),
        }
    finally:
        check.file.close()

    del filtered
    gc.collect()
    return result


def main() -> None:
    global INPUT_DIR, OUTPUT_DIR
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    INPUT_DIR, OUTPUT_DIR = args.input_dir, args.output_dir
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    remove_name_list, summary = identify_reference_cluster()
    remove_names = set(remove_name_list)

    input_files = sorted(INPUT_DIR.glob("*.h5ad"))
    if not input_files:
        raise FileNotFoundError(f"No H5AD files found in {INPUT_DIR}")

    file_results = []
    for index, input_path in enumerate(input_files, start=1):
        print(f"[{index:02d}/{len(input_files):02d}] {input_path.name}", flush=True)
        result = write_filtered_h5ad(input_path, remove_names)
        file_results.append(result)
        print(
            f"    {result['n_before']} -> {result['n_after']} "
            f"(removed {result['n_removed']})",
            flush=True,
        )

    summary["files"] = file_results
    summary_path = OUTPUT_DIR / "filter_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Saved filtering summary: {summary_path}")


if __name__ == "__main__":
    main()
