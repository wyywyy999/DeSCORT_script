#!/usr/bin/env python3
"""Resolve JUMP pilot perturbations to fields, download five channels, and render RGB."""

import csv
import json
import os
from pathlib import Path
import re
import subprocess
from urllib.parse import urlparse

os.environ.setdefault('MPLCONFIGDIR', '/tmp/jump_pilot_image_matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import tifffile

S3_METADATA = 's3://cellpainting-gallery/cpg0000-jump-pilot/source_4/workspace/load_data_csv/'
CHANNELS = ['DNA', 'ER', 'RNA', 'AGP', 'Mito']
COLORS = {'DNA': [0, 0, 1], 'ER': [0, 1, 0], 'RNA': [1, 1, 0],
          'AGP': [1, 0, 0], 'Mito': [1, 0, 1]}
INDEX_COLUMNS = ['Source', 'Batch', 'Plate', 'Well', 'Treatment', 'Pert_IName',
                 'Perturbation_Category', 'Cell_Line', 'Timepoint']


def aws_copy(source, destination, recursive=False):
    command = ['aws', 's3', 'cp']
    if recursive:
        command.append('--recursive')
    command += [source, str(destination), '--no-sign-request', '--only-show-errors']
    print('Downloading:', source, flush=True)
    subprocess.run(command, check=True)


def build_index(annotation, output):
    """Cache well-level annotations instead of rescanning the 2 GB channel table."""
    parts = []
    for chunk in pd.read_csv(annotation, usecols=INDEX_COLUMNS, dtype=str,
                             keep_default_na=False, chunksize=100_000):
        parts.append(chunk.loc[chunk.Source.eq('source_4')].drop_duplicates())
    frame = pd.concat(parts, ignore_index=True).drop_duplicates()
    temporary = output.with_suffix('.tmp')
    frame.to_csv(temporary, index=False)
    temporary.replace(output)
    print(f'Indexed {len(frame):,} perturbation wells: {output}', flush=True)
    return frame


def find_candidates(args):
    args.index_dir.mkdir(parents=True, exist_ok=True)
    index = args.index_dir / 'jump_pilot_well_index.csv'
    signature_file = args.index_dir / 'jump_pilot_well_index_source.json'
    stat = args.annotation.stat()
    signature = dict(path=str(args.annotation.resolve()), size=stat.st_size, mtime_ns=stat.st_mtime_ns)
    cached_signature = json.loads(signature_file.read_text()) if signature_file.exists() else None
    if args.rebuild_index or not index.exists() or signature != cached_signature:
        table = build_index(args.annotation, index)
        signature_file.write_text(json.dumps(signature, indent=2) + '\n')
    else:
        table = pd.read_csv(index, dtype=str, keep_default_na=False)
    if args.compound:
        # Permit spelling variants such as CCG50014 and CCG-50014.
        normalize = lambda value: re.sub(r'[^a-z0-9]', '', value.casefold())
        wanted = normalize(args.compound)
        match = table.Pert_IName.map(normalize).eq(wanted)
    else:
        wanted = args.gene.strip().casefold()
        # Compound target annotations may contain multiple symbols. Never split hyphens.
        match = table.Treatment.map(lambda value: wanted in
                                    [g.strip().casefold() for g in re.split(r'[;,|]', value)])
    match &= table.Perturbation_Category.str.casefold().eq(args.perturbation_type)
    if args.cell_line:
        match &= table.Cell_Line.str.casefold().eq(args.cell_line.casefold())
    for column, value in [('Plate', args.plate), ('Well', args.well),
                          ('Batch', args.batch), ('Timepoint', args.timepoint)]:
        if value is not None:
            match &= table[column].str.casefold().eq(str(value).casefold())
    selected = table.loc[match].sort_values(['Batch', 'Plate', 'Well']).drop_duplicates()
    if selected.empty:
        raise ValueError(f'No matching annotation for {args.compound or args.gene}/{args.perturbation_type}; '
                         'check cell line, gene symbol and other filters. '
                         'Compound results refer to annotated targets, not gene knockout.')
    # Some channel annotations repeat a physical well with/without a target label.
    keys = [c for c in selected.columns if c != 'Treatment']
    return selected.groupby(keys, as_index=False, sort=False, dropna=False).agg(
        {'Treatment': lambda values: ';'.join(sorted({v for v in values if v}))})


def download_channel(url, destination):
    if not url.startswith('s3://cellpainting-gallery/'):
        raise ValueError(f'Unexpected image URL: {url}')
    if not destination.exists():
        temporary = destination.with_suffix(destination.suffix + '.part')
        aws_copy(url, temporary)
        if temporary.stat().st_size == 0:
            raise ValueError(f'Empty download: {url}')
        temporary.replace(destination)
    image = tifffile.imread(destination)
    if image.ndim != 2 or not np.isfinite(image).all():
        raise ValueError(f'Expected a finite 2D channel image: {destination}, {image.shape}')
    return image


def save_display_image(values, stem, dpi):
    """Save an 8-bit color PNG with the requested DPI metadata."""
    values = np.clip(values, 0, 1)
    Image.fromarray(np.rint(values * 255).astype('uint8')).save(
        stem.with_suffix('.png'), dpi=(dpi, dpi))


def timepoint_tag(annotation):
    value = str(annotation.get('Timepoint', '')).strip()
    if not value:
        return 'time_unknown'
    try:
        value = format(float(value), 'g')
    except ValueError:
        pass
    return re.sub(r'[^A-Za-z0-9_-]', '_', value) + 'h'


def render_field(row, annotation, folder, args):
    folder.mkdir(parents=True, exist_ok=True)
    raw_folder = folder / 'raw'
    raw_folder.mkdir(exist_ok=True)
    scaled, channel_info = [], []
    time_tag = timepoint_tag(annotation)
    shape = None
    for channel in CHANNELS:
        url = row[f'URL_Orig{channel}']
        source_name = Path(urlparse(url).path)
        path = raw_folder / f'{source_name.stem}_{time_tag}{source_name.suffix}'
        # Reuse downloads from the earlier layout, without downloading them again.
        legacy = (folder.parents[3] / annotation['Plate'] / annotation['Well'] /
                  folder.name / 'raw' / source_name.name)
        if not path.exists() and legacy.is_file():
            os.link(legacy, path)
        image = download_channel(url, path)
        if shape is not None and image.shape != shape:
            raise ValueError('Channel image dimensions do not match')
        shape = image.shape
        low, high = np.percentile(image, [args.low_percentile, args.high_percentile])
        values = (np.clip((image.astype(np.float32) - low) / (high - low), 0, 1)
                  if high > low else np.zeros(shape, dtype=np.float32))
        scaled.append(values)
        rgb = values[:, :, None] * np.asarray(COLORS[channel], dtype=np.float32)
        save_display_image(rgb, folder / f'{channel}_color_{time_tag}', args.dpi)
        channel_info.append(dict(channel=channel, s3_url=url, local_tiff=str(path.resolve()),
                                 dtype=str(image.dtype), shape=list(shape),
                                 raw_min=float(image.min()), raw_max=float(image.max()),
                                 display_low=float(low), display_high=float(high),
                                 rgb_weights=COLORS[channel]))
    # Equal-weight additive pseudocoloring, followed by one global display stretch.
    colored = [values[:, :, None] * np.asarray(COLORS[c], dtype=np.float32)
               for c, values in zip(CHANNELS, scaled)]
    merged = sum(colored) / len(CHANNELS)
    maximum = float(merged.max())
    if maximum > 0:
        merged /= maximum
    save_display_image(merged, folder / f'merged_rgb_{time_tag}', args.dpi)
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    for ax, channel, rgb in zip(axes.flat, CHANNELS, colored):
        ax.imshow(rgb)
        ax.set_title(channel)
        ax.axis('off')
    axes.flat[-1].imshow(merged)
    axes.flat[-1].set_title('Merged RGB')
    axes.flat[-1].axis('off')
    fig.suptitle(f'{args.compound or args.gene} | {args.perturbation_type} | {annotation["Cell_Line"]} | {time_tag}\n'
                 f'{annotation["Plate"]} / {annotation["Well"]} / site {row["Metadata_Site"]}')
    fig.tight_layout(rect=[0, 0, 1, .94])
    panel_path = folder / f'channels_and_merged_{time_tag}.png'
    fig.savefig(panel_path, dpi=args.dpi)
    plt.close(fig)
    manifest = dict(annotation=annotation, site=row['Metadata_Site'],
                    load_data_row=row, channels=channel_info,
                    display=dict(dpi=args.dpi, native_image_shape=list(shape),
                                 montage_inches=[12, 8],
                                 output_format='PNG', bits_per_sample=8,
                                 percentiles=[args.low_percentile, args.high_percentile],
                                 per_channel_scaling=True, illumination_correction=False,
                                 merge='equal weights; additive RGB; global maximum stretch',
                                 note='Display intensities are independently scaled per field/channel; '
                                      'not suitable for quantitative brightness comparisons.'))
    (folder / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(f'Finished: {panel_path}', flush=True)
    return dict(batch=annotation['Batch'], plate=annotation['Plate'], well=annotation['Well'],
                timepoint_hours=annotation['Timepoint'],
                site=row['Metadata_Site'], output_directory=str(folder.resolve()))
