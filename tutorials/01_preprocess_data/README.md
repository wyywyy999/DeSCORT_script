# Preprocessing and Vocabularies

Input and output: raw CSV / Parquet files → aggregated, normalized, and clipped features and vocabularies.

Follow the scripts in order: `01_csv2parquet.py` → `02_preprocess_data.py` → `03_make_label_vocab.py` → `04_select_intersection_union_feature.py` → `05_get_union_feature_data.py`. Complete feature preprocessing before proceeding to downstream tasks. Supplementary scripts for generating vocabularies and feature sets are documented in the resources provenance index.

```bash
bash /fse/home/wangyongyan/Project/CellPainting/DeSCORT/other_model/DeSCORT/tutorials/01_preprocess_data/run.sh --help
```

To run from any working directory, use the absolute path to this entry point. Refer to each script's `--help` output for the supported arguments. Update server paths in the historical experiment Bash scripts to match your local configuration.

Input, output, source-code, and log paths use absolute paths for the current server. Enter the tutorial directory before running the numbered Bash scripts; they launch Python using only the script filename (for example, `python 01_csv2parquet.py`) and do not change the working directory. CLI file and directory arguments are resolved to absolute paths. Edit the configured paths when moving this tutorial. The `python` command uses the active environment.
