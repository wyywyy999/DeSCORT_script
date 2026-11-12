#!/usr/bin/env python3
"""Cross-modality profiles, cosine rankings and UMAP for model and/or original data.

Configure inputs and analysis parameters in 06_umap.sh. Each input is analyzed
independently; model and original UMAP coordinate systems are not aligned.
"""

# Support direct execution from this checkout, independently of the working directory.
from pathlib import Path as _Path
import sys as _sys
_SOURCE = _Path("/fse/home/wangyongyan/Project/CellPainting/DeSCORT/other_model/DeSCORT/src")
if str(_SOURCE) not in _sys.path:
    _sys.path.insert(0, str(_SOURCE))

from pathlib import Path
import json
from threadpoolctl import threadpool_limits

from descort.multimodality.umap import (
    MODALITIES,
    CONTROL_NAMES,
    CRISPR_GUIDE,
    DMSO_UNIT,
    PROFILE_KEYS,
    parse_args,
    discover,
    select_features,
    clean_metadata,
    read_wells,
    display,
    analyze,
    analyze_profiles,
    compare_results,
)


def main():
    args = parse_args()
    if not args.dry_run:
        for name in ['audit_panel_alignment.py', 'audit_pdf_text.py', 'audit_figure_collisions.py']:
            if not (args.qa_tools_dir / name).is_file():
                raise FileNotFoundError(args.qa_tools_dir / name)
    jobs = []
    if args.model_data_dir:
        jobs.append(('model', {m: args.model_data_dir / m for m in MODALITIES}))
    if args.original_data_dir or any(getattr(args, f'original_{m.lower()}_dir') for m in MODALITIES):
        jobs.append(('original', {
            m: getattr(args, f'original_{m.lower()}_dir') or args.original_data_dir / m
            for m in MODALITIES
        }))
    completed = {}
    for kind, directories in jobs:
        print(f'[{kind}] Discovering files', flush=True)
        entries = discover(directories)
        features, schemas, exclusions = select_features(entries, kind, args.original_feature_list)
        wells, manifest = read_wells(entries, features, args.dry_run)
        print(json.dumps(dict(input_kind=kind, input_files=len(entries), input_wells=len(wells),
                              selected_features=len(features), excluded_features=len(exclusions),
                              repaired_crispr_dmso_rows=int(manifest.repaired_crispr_dmso_rows.sum())), indent=2), flush=True)
        if args.dry_run:
            continue
        result_dir = args.output_dir / kind / 'results'
        result_dir.mkdir(parents=True, exist_ok=True)
        schemas.to_csv(result_dir / 'source_schema_summary.csv', index=False)
        with threadpool_limits(limits=args.threads):
            completed[kind] = analyze(kind, entries, features, wells, manifest, exclusions, args)
        del wells
    if args.dry_run:
        print('DRY RUN PASSED: schemas and metadata validated; numerical feature values are checked during a full run.')
        return
    compare_results(completed, args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    settings = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    (args.output_dir / 'run_config.json').write_text(json.dumps(settings, indent=2))
    print(f'Finished: {args.output_dir}', flush=True)



if __name__ == "__main__":
    main()
