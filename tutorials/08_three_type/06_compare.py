#!/usr/bin/env python3
"""Compare cross-modality profiles and run the built-in temporal retrieval workflow.

Configure all paths and run options in 06_compare.sh. The pooled-profile and
time-stratified retrieval analyses intentionally retain separate definitions.
Profile comparison also plots Model–Original Top-k ranking overlap for
Compound–CRISPR, Compound–ORF and CRISPR–ORF (k=5,10,...,100 by default).
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

from descort.multimodality.comparison import (
    MODALITIES,
    CONTROL_NAMES,
    CRISPR_GUIDE,
    PROFILE_KEYS,
    _read_wells,
    load_model_profiles,
    pair_identities,
    sha256_file_set,
    run_comparison,
    load_profile_bundle,
    plot_profile_hits,
    audit_pdf,
    run_temporal,
    parse_args,
    validate_comparison,
    write_combined_report,
)


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name in ['audit_panel_alignment.py', 'audit_pdf_text.py', 'audit_figure_collisions.py']:
        if not (args.qa_tools_dir / name).is_file():
            raise FileNotFoundError(args.qa_tools_dir / name)
    manifest = {'status': 'running', 'stages': {}, 'parameters': {
        name: str(value) if isinstance(value, Path) else value for name, value in vars(args).items()}}
    manifest_path = args.output_dir / 'run_manifest.json'
    def save_status():
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    save_status()
    try:
        if 'comparison' in args.stages:
            validate_comparison(args)
            if not args.dry_run:
                with threadpool_limits(limits=args.threads):
                    run_comparison(args)
            manifest['stages']['comparison'] = 'validated' if args.dry_run else 'complete'
            save_status()
        if 'temporal' in args.stages:
            run_temporal(args)
            manifest['stages']['temporal'] = 'validated' if args.dry_run else 'complete'
            save_status()
        if not args.dry_run:
            write_combined_report(args)
        manifest['status'] = 'validated' if args.dry_run else 'complete'
        save_status()
        print(f'Finished ({manifest["status"]}): {args.output_dir}', flush=True)
    except Exception as error:
        manifest.update(status='failed', error=f'{type(error).__name__}: {error}')
        save_status()
        raise



if __name__ == "__main__":
    main()
