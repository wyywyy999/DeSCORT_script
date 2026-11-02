#!/usr/bin/env bash
# U2OS raw morphology features: eight sequential modality/time mAP runs.
# Run: bash 02_run_calculate_mAP_three_type_original_U2OS.sh
# Preview plate selection and commands: bash 02_run_calculate_mAP_three_type_original_U2OS.sh --dry-run
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "$SCRIPT_DIR/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

mAP_type="sister_consistency"
source_data_paths=(
    "/fse/DC/CP_data/feature_data/cpg000_jump_pilot/raw_data_bulk_download/U2OS/Orf/raw_data/normalization_clip_data"
    "/fse/DC/CP_data/feature_data/cpg000_jump_pilot/raw_data_bulk_download/U2OS/Crispr/raw_data/normalization_clip_data"
    "/fse/DC/CP_data/feature_data/cpg000_jump_pilot/raw_data_bulk_download/U2OS/Compound/raw_data/normalization_clip_data"
)
need_feature_list_pkl="$PROJECT_DIR/DeSCORT/common_files/feature_intersection_union/cpg0000_three_type/vocab_union_feature_3344.pkl"
feature2index_pkl="$PROJECT_DIR/Final_well_level_new_LLM/common_files/00_pretrain/feature2index_vocabulary_7366.pkl"
# Align with the current U2OS embedding launcher; optionally override the destination.
save_path="${MAP_ORIGINAL_SAVE_PATH:-$PROJECT_DIR/DeSCORT/tutorials/08_three_type/mAP/U2OS_50epochs/mAP_result_original}"
metadata_tsv="$PROJECT_DIR/Final_well_level_new_LLM/common_files/00_pretrain/cpg0000_experiment-metadata.tsv"
perturbation_gene_key="Metadata_Treatment"
sgRNA_barcode_key="Metadata_Broad_Sample"
need_columns=(Metadata_Broad_Sample Metadata_Treatment Metadata_Well)
group_by_columns=(Metadata_Broad_Sample Metadata_Treatment Metadata_Well)
# Match the current embedding launcher for a direct comparison.
random_seeds=(42)
n_jobs="${N_JOBS:-20}"
sample_method="all"
n_number=2
do_intersect_features=true

# Intersect raw numeric features across each comparison's selected files, using
# the requested feature list and the existing feature2index vocabulary.
# The list also includes metadata columns; preserve the three requested keys.
# Schema-only preflight checks the intersection without loading full tables.
# The existing sister_consistency algorithm pools both selected modalities:
# positives are all other wells with the same Treatment, including same-modality
# wells. Metadata_Broad_Sample is used for per-file well aggregation, not as a
# restriction to cross-modality-only positives. Negative controls are excluded
# by the existing calculator.

# 02_calculate_mAP.py uses relative imports, so launch it from its own directory.
cd -- "$SCRIPT_DIR"
"$PYTHON_BIN" - \
    --calculator "$SCRIPT_DIR/02_calculate_mAP.py" \
    --mAP_type "$mAP_type" \
    --source_data_paths "${source_data_paths[@]}" \
    --need_feature_list_pkl "$need_feature_list_pkl" \
    --feature2index_pkl "$feature2index_pkl" \
    --save_path "$save_path" \
    --metadata_tsv "$metadata_tsv" \
    --perturbation_gene_key "$perturbation_gene_key" \
    --sgRNA_barcode_key "$sgRNA_barcode_key" \
    --need_columns "${need_columns[@]}" \
    --group_by_columns "${group_by_columns[@]}" \
    --random_seeds "${random_seeds[@]}" \
    --n_jobs "$n_jobs" \
    --sample_method "$sample_method" \
    --n_number "$n_number" \
    --do_intersect_features "$do_intersect_features" \
    "$@" <<'PY'
import argparse
import csv
import pickle
import shlex
import subprocess
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

parser = argparse.ArgumentParser(description="Launch eight U2OS modality/time mAP comparisons.")
for name in ("calculator", "mAP_type", "need_feature_list_pkl",
             "feature2index_pkl", "save_path", "metadata_tsv", "perturbation_gene_key",
             "sgRNA_barcode_key", "sample_method"):
    parser.add_argument("--" + name, required=True)
for name in ("source_data_paths", "need_columns", "group_by_columns", "random_seeds"):
    parser.add_argument("--" + name, nargs="+", required=True)
parser.add_argument("--do_intersect_features", choices=["true", "false"], required=True)
parser.add_argument("--n_jobs", type=int, required=True)
parser.add_argument("--n_number", type=int, required=True)
parser.add_argument("--dry-run", action="store_true",
                    help="Validate metadata/file selection and print commands without writing or calculating mAP.")
args = parser.parse_args()

source_roots = {}
for value in args.source_data_paths:
    root = Path(value).resolve()
    if not root.is_dir():
        raise NotADirectoryError(root)
    # Layout: U2OS/<modality>/raw_data/normalization_clip_data/<batch>/<plate>.parquet
    modality = root.parent.parent.name.lower()
    if modality not in ("compound", "crispr", "orf"):
        raise ValueError(f"Cannot infer source modality from {root}")
    if modality in source_roots:
        raise ValueError(f"Duplicate source directory for {modality}")
    source_roots[modality] = root
if set(source_roots) != {"compound", "crispr", "orf"}:
    raise ValueError("Provide exactly one Compound, Crispr and Orf source directory")
output = Path(args.save_path).resolve()
for value in (args.calculator, args.need_feature_list_pkl,
              args.feature2index_pkl, args.metadata_tsv):
    if not Path(value).is_file():
        raise FileNotFoundError(value)
if args.n_jobs == 0:
    parser.error("--n_jobs cannot be zero")

# Time is the metadata Time column, including records from repeat-imaging batches.
# Match (modality, batch, full plate barcode) exactly. A substring such as
# BR00116991 would also select BR00116991A-F, so pass complete selected file paths
# as need_plate_list to the existing calculator's path-substring filter.
time_hours = {
    "compound": {"short": 24, "long": 48},
    "crispr": {"short": 96, "long": 144},
    "orf": {"short": 48, "long": 96},
}
metadata = {}
with Path(args.metadata_tsv).open(newline="", encoding="utf-8-sig") as handle:
    reader = csv.DictReader(handle, delimiter="\t")
    required = {"Batch", "Assay_Plate_Barcode", "Perturbation", "Cell_type", "Time"}
    missing = required.difference(reader.fieldnames or [])
    if missing:
        raise ValueError(f"Metadata TSV is missing columns: {sorted(missing)}")
    for row in reader:
        if row["Cell_type"].strip().upper() != "U2OS":
            continue
        modality = row["Perturbation"].strip().lower()
        if modality not in time_hours:
            continue
        key = (modality, row["Batch"].strip(), row["Assay_Plate_Barcode"].strip())
        hours = float(row["Time"])
        if key in metadata and metadata[key] != hours:
            raise ValueError(f"Conflicting Time values for {key}")
        metadata[key] = hours

selected = {(modality, period): [] for modality in time_hours for period in ("short", "long")}
unmatched = []
for modality, root in source_roots.items():
    for path in sorted(root.rglob("*.parquet")):
        key = (modality, path.parent.name, path.stem)
        if key not in metadata:
            unmatched.append(str(path))
            continue
        hours = metadata[key]
        for period, wanted_hours in time_hours[modality].items():
            if hours == wanted_hours:
                selected[(modality, period)].append({
                    "modality": modality, "period": period, "time_hours": wanted_hours,
                    "batch": key[1], "plate": key[2], "source_file": str(path),
                })
if unmatched:
    raise ValueError("Source files have no matching U2OS modality/batch/plate metadata: "
                     + ", ".join(unmatched))
for (modality, period), rows in selected.items():
    if not rows:
        raise ValueError(f"No available files for {modality}_{period} ({time_hours[modality][period]} h)")
    print(f"{modality}_{period}: {time_hours[modality][period]} h, "
          f"{len(rows)} files, {len({row['plate'] for row in rows})} unique plate barcodes", flush=True)

# Mirror the existing calculator's eligible feature list and metadata handling.
with Path(args.need_feature_list_pkl).open("rb") as handle:
    requested_features = pickle.load(handle)
with Path(args.feature2index_pkl).open("rb") as handle:
    vocabulary = pickle.load(handle)
eligible = {name for name in requested_features if name in vocabulary or "Meta" in name}
required_metadata = set(args.need_columns) | set(args.group_by_columns)
required_metadata.add(args.perturbation_gene_key)
schemas = {}
for rows in selected.values():
    for row in rows:
        filename = row["source_file"]
        schema = pq.read_schema(filename)
        missing = required_metadata.difference(schema.names)
        if missing:
            raise ValueError(f"{filename}: missing required metadata {sorted(missing)}")
        schemas[filename] = schema

jobs = []
for compound_period in ("short", "long"):
    for other_modality in ("crispr", "orf"):
        for other_period in ("short", "long"):
            name = f"compound_{compound_period}_vs_{other_modality}_{other_period}"
            rows = selected[("compound", compound_period)] + selected[(other_modality, other_period)]
            destination = output / name
            log_file = output / "logs" / f"{name}.log"
            command = [
                sys.executable, args.calculator,
                "--mAP_type", args.mAP_type,
                "--source_data_paths", str(source_roots["compound"]),
                str(source_roots[other_modality]),
                "--need_feature_list_pkl", args.need_feature_list_pkl,
                "--feature2index_pkl", args.feature2index_pkl,
                "--save_path", str(destination),
                "--perturbation_gene_key", args.perturbation_gene_key,
                "--sgRNA_barcode_key", args.sgRNA_barcode_key,
                "--negative_control_list", "nontargeting",
                "--choose_type", "well-level",
                "--need_columns", *args.need_columns,
                "--group_by_columns", *args.group_by_columns,
                "--do_groupby_mean",
                "--random_seeds", *args.random_seeds,
                "--n_jobs", str(args.n_jobs),
                "--sample_method", args.sample_method,
                "--n_number", str(args.n_number),
                "--log_file", str(log_file), "--log_level", "INFO",
                "--need_plate_list", *(row["source_file"] for row in rows),
            ]
            if args.do_intersect_features == "true":
                command.append("--do_intersect_features")
                common = eligible.intersection(*(set(schemas[row["source_file"]].names) for row in rows))
                missing = required_metadata.difference(common)
                if missing:
                    raise ValueError(f"{name}: feature intersection removes metadata {sorted(missing)}")
                features = sorted(column for column in common if not column.startswith("Meta"))
                if not features:
                    raise ValueError(f"{name}: no numeric features remain after intersection")
                for row in rows:
                    schema = schemas[row["source_file"]]
                    nonnumeric = [column for column in features
                                  if not (pa.types.is_floating(schema.field(column).type)
                                          or pa.types.is_integer(schema.field(column).type))]
                    if nonnumeric:
                        raise ValueError(f"{row['source_file']}: nonnumeric selected features {nonnumeric}")
            else:
                features = []
            jobs.append((name, destination, rows, command, features))

# All eight plate selections are validated before the first calculation starts.
for index, (name, destination, rows, command, features) in enumerate(jobs, 1):
    print(f"\n[{index}/8] {name}: {len(rows)} files -> {destination}", flush=True)
    print(f"Intersected numeric morphology features: {len(features)}", flush=True)
    if args.dry_run:
        print(shlex.join(command), flush=True)
        continue
    destination.mkdir(parents=True, exist_ok=True)
    (output / "logs").mkdir(parents=True, exist_ok=True)
    with (destination / "selected_plates.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (destination / "intersected_features.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["feature"])
        writer.writerows((column,) for column in features)
    # One comparison at a time; existing per-gene parallelism remains n_jobs.
    subprocess.run(command, cwd=str(Path(args.calculator).parent), check=True)

print("\nAll eight selections validated." if args.dry_run else "\nAll eight comparisons completed.", flush=True)
PY
