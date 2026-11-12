#!/usr/bin/env python3
"""Resolve JUMP pilot perturbations, download five channels, and render RGB images.

Configure query lists, absolute resource paths and display settings in 07_get_image.sh.
"""
from pathlib import Path as _Path
import sys as _sys
_SOURCE = _Path("/fse/home/wangyongyan/Project/CellPainting/DeSCORT/other_model/DeSCORT/src")
if str(_SOURCE) not in _sys.path:
    _sys.path.insert(0, str(_SOURCE))

import argparse
import csv
import json
from pathlib import Path
import re
import pandas as pd

from descort.multimodality.get_image import (
    CHANNELS, COLORS, INDEX_COLUMNS, S3_METADATA,
    aws_copy, build_index, find_candidates, download_channel,
    save_display_image, timepoint_tag, render_field,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    query = parser.add_mutually_exclusive_group(required=True)
    query.add_argument('--gene', help='Perturbed gene or annotated compound target')
    query.add_argument('--compound', help='Compound name from Pert_IName (e.g. CCG-50014)')
    parser.add_argument('--perturbation-type', type=str.lower, required=True,
                        choices=['crispr', 'orf', 'compound'])
    parser.add_argument('--cell-line', choices=['A549', 'U2OS'])
    parser.add_argument('--plate')
    parser.add_argument('--well')
    parser.add_argument('--batch')
    parser.add_argument('--timepoint', help='Exact timepoint in annotation, e.g. 144')
    parser.add_argument('--site', type=int, default=1)
    parser.add_argument('--all-sites', action='store_true',
                        help='Select all sites in each matching well; overrides --site')
    parser.add_argument('--max-fields', type=int, default=1,
                        help='Maximum matching fields to download; default 1')
    parser.add_argument('--all-fields', action='store_true',
                        help='Download all selected fields across matching wells; overrides --max-fields')
    parser.add_argument('--dry-run', action='store_true',
                        help='Write download_plan.csv without downloading or rendering images')
    parser.add_argument('--sync-metadata', action='store_true',
                        help='Run the recursive unsigned AWS copy of load_data_csv first')
    parser.add_argument('--rebuild-index', action='store_true')
    parser.add_argument('--annotation', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--load-data-dir', type=Path, required=True, help='Root containing Batch/Plate/load_data.csv')
    parser.add_argument('--index-dir', type=Path, required=True, help='Directory for cached well annotations')
    parser.add_argument('--low-percentile', type=float, default=1)
    parser.add_argument('--high-percentile', type=float, default=99.8)
    parser.add_argument('--dpi', type=int, default=600,
                        help='PNG resolution metadata and montage rendering DPI; default 600. '
                             'Standalone channel/merged PNGs retain native image dimensions.')
    args = parser.parse_args()
    if args.compound and args.perturbation_type != 'compound':
        parser.error('--compound requires --perturbation-type compound')
    if args.site < 1 or args.max_fields < 1 or not 0 <= args.low_percentile < args.high_percentile <= 100:
        parser.error('Require site/max-fields >= 1 and 0 <= low < high <= 100')
    if args.dpi < 1:
        parser.error('--dpi must be a positive integer')
    if not args.annotation.is_file():
        parser.error(f'Missing perturbation annotation: {args.annotation}')
    for name in ['annotation', 'output_dir', 'load_data_dir', 'index_dir']:
        setattr(args, name, getattr(args, name).expanduser().resolve())
    load_root = args.load_data_dir
    if args.sync_metadata or not load_root.exists():
        aws_copy(S3_METADATA, load_root, recursive=True)
    candidates = find_candidates(args)
    safe_gene = re.sub(r'[^A-Za-z0-9_.-]+', '_', args.compound or args.gene)
    output = args.output_dir / f'{safe_gene}_{args.perturbation_type}'
    output.mkdir(parents=True, exist_ok=True)
    candidates.to_csv(output / 'matching_wells.csv', index=False)
    limit = None if args.all_fields else args.max_fields
    plan = []
    seen = set()
    for annotation in candidates.to_dict('records'):
        batch, plate, well = (annotation[c] for c in ['Batch', 'Plate', 'Well'])
        load_csv = load_root / batch / plate / 'load_data.csv'
        if not load_csv.exists():
            load_csv.parent.mkdir(parents=True, exist_ok=True)
            aws_copy(f'{S3_METADATA}{batch}/{plate}/load_data.csv', load_csv)
        with load_csv.open(newline='') as handle:
            fields = [r for r in csv.DictReader(handle)
                      if r['Metadata_Plate'] == plate and r['Metadata_Well'] == well
                      and (args.all_sites or int(r['Metadata_Site']) == args.site)]
        fields.sort(key=lambda row: int(row['Metadata_Site']))
        for row in fields:
            identity = tuple(row[f'URL_Orig{c}'] for c in CHANNELS)
            if identity in seen:
                continue
            seen.add(identity)
            site = int(row['Metadata_Site'])
            folder = (output / annotation['Cell_Line'] / batch /
                      f'time_{timepoint_tag(annotation)}' / plate / well / f'site_{site:02d}')
            plan.append((row, annotation, folder))
            if limit is not None and len(plan) >= limit:
                break
        if limit is not None and len(plan) >= limit:
            break
    if not plan:
        raise ValueError('No matching image field in load_data.csv; check --site and matching_wells.csv')
    pd.DataFrame([dict(**annotation, site=row['Metadata_Site'],
                       output_directory=str(folder.resolve()),
                       **{f'URL_Orig{c}': row[f'URL_Orig{c}'] for c in CHANNELS})
                  for row, annotation, folder in plan]).to_csv(output / 'download_plan.csv', index=False)
    print(f'Plan: {len(plan)} field(s), {len(plan) * len(CHANNELS)} channel images.', flush=True)
    if args.dry_run:
        print(f'Dry run: no image downloads/rendering. Plan: {output / "download_plan.csv"}', flush=True)
        return
    completed = []
    for row, annotation, folder in plan:
        completed.append(render_field(row, annotation, folder, args))
    (output / 'run_summary.json').write_text(json.dumps(dict(
        gene=args.gene, compound=args.compound, perturbation_type=args.perturbation_type,
        annotation_file=str(args.annotation.resolve()), matching_wells=len(candidates),
        all_sites=args.all_sites, all_fields=args.all_fields,
        completed_fields=completed), indent=2) + '\n')
    print(f'Success: {len(completed)} field(s); {len(candidates)} matching wells. Results: {output}', flush=True)


if __name__ == '__main__':
    main()
