"""Download commands for the same Cell Painting Gallery sources as the original scripts."""
import argparse
from pathlib import Path
import shlex
import subprocess


def run_download(command, destination, dry_run):
    print(shlex.join(command), flush=True)
    if not dry_run:
        destination.mkdir(parents=True, exist_ok=True)
        subprocess.run(command, check=True)


def main(dataset):
    parser = argparse.ArgumentParser(description=f"Download {dataset} profiles from Cell Painting Gallery")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true", help="List commands without downloading or creating output directories")
    if dataset in {"cpg0021_A549", "cpg0021_HeLa"}:
        cell = dataset.split("_")[1]
        default_plates = list("ABCDEFGHN") if cell == "A549" else list("BDFHJKLN")
        parser.add_argument("--plates", nargs="+", default=default_plates)
    elif dataset == "cpg0016":
        parser.add_argument("--sources", nargs="+", default=["source_4"])
    elif dataset == "cpg0000":
        parser.add_argument("--metadata-csv", type=Path, required=True)
        parser.add_argument("--cell-lines", nargs="+", default=["A549", "U2OS"])
        parser.add_argument("--perturbation-types", nargs="+", choices=["compound", "orf", "crispr"],
                            default=["compound", "orf"])
    else:
        raise ValueError(dataset)
    args = parser.parse_args()
    if dataset.startswith("cpg0021"):
        prefix = "20200805_A549_WG_Screen" if cell == "A549" else "20210422_6W_CP257"
        plate_prefix = "CP186" if cell == "A549" else "CP257"
        for plate in args.plates:
            url = (f"https://cellpainting-gallery.s3.amazonaws.com/cpg0021-periscope/broad/workspace/profiles/"
                   f"{cell}/single_cell/{prefix}_single_cell_profiles_ALLBATCHES___{plate_prefix}{plate}___ALLWELLS.csv.gz")
            run_download(["wget", "-P", str(args.output_dir), url], args.output_dir, args.dry_run)
    elif dataset == "cpg0016":
        for source in args.sources:
            destination = args.output_dir / source
            url = f"s3://cellpainting-gallery/cpg0016-jump/{source}/workspace/profiles/"
            run_download(["aws", "s3", "cp", "--recursive", url, str(destination), "--no-sign-request"],
                         destination, args.dry_run)
    else:
        import pandas as pd
        metadata = pd.read_csv(args.metadata_csv)
        metadata.columns = [name if name.startswith("Metadata_") else "Metadata_" + name for name in metadata.columns]
        folders = {"compound": "Compoud", "orf": "Orf", "crispr": "Crispr"}
        for cell_line in args.cell_lines:
            for perturbation in args.perturbation_types:
                selected = metadata[(metadata.Metadata_Cell_Line == cell_line) &
                                    (metadata.Metadata_Perturbation_Category == perturbation)]
                for batch, plate in selected[["Metadata_Batch", "Metadata_Plate"]].drop_duplicates().itertuples(index=False, name=None):
                    destination = args.output_dir / cell_line / folders[perturbation] / str(batch) / str(plate)
                    url = f"s3://cellpainting-gallery/cpg0000-jump-pilot/source_4/workspace/profiles/{batch}/{plate}/"
                    run_download(["aws", "s3", "cp", "--recursive", url, str(destination), "--no-sign-request"],
                                 destination, args.dry_run)
