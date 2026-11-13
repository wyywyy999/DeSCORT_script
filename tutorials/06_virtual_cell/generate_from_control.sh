#!/usr/bin/env bash
set -euo pipefail

# Run from this tutorial directory. Use the matching trained model and resources.
# CRISPR: gene2index + index2esm2. Compound: smiles2index + index2morgan.
python generate_from_control.py \
  --pretrained_model_path /absolute/path/virtual_cell/models \
  --parquet_path /absolute/path/normalized_profiles.parquet \
  --csv_path /absolute/path/generation_requests.csv \
  --pert_col Metadata_Foci_Barcode_MatchedTo_GeneCode \
  --control_label nontargeting \
  --perturbation2index_pkl /absolute/path/label_vocab/gene2index.pkl \
  --index2condition_pkl /absolute/path/condition_vectors/index2esm2.pkl \
  --feature2index_pkl /absolute/path/feature_vocab/feature2index.pkl \
  --output_path /absolute/path/generated_profiles.parquet \
  --batch_size 64 \
  --device cuda:0 \
  --seed 42 \
  --replacement auto \
  "$@"
