# Tutorials and entry points

## Documentation

Full tutorials are available in the [DeSCORT documentation](https://descort-script.readthedocs.io/en/latest/?refresh=no-reference-v2). See [Documentation hosting](../docs/hosting.md) for deployment and local preview.

All entry points are contained in this independent copy. Specify absolute paths explicitly when reusing resources and models from the original project in read-only mode.

Use the task-specific Bash or Python commands shown in each tutorial README, from the directory specified there. A shared `run.sh` wrapper is not available for every task. Configure download paths inside the download scripts before running them directly. Use the documented argument options to override experiment inputs and outputs.

Existing task-specific Bash scripts retain editable experiment parameters and may still contain server data/model paths. Start new runs from `configs` or explicit arguments. Large datasets, checkpoints, exploratory experiment directories, and historical results are not copied into this repository.

| Order | Workflow documentation |
|---|---|
| 00 | [Cell Painting Gallery downloads](00_download/README.md) |
| 01 | [CSV conversion, preprocessing, label vocabulary, and feature sets](01_preprocess_data/README.md) |
| 02 | [Pretraining](02_pretrain/README.md) |
| 03 | [Representation fine-tuning and extraction](03_finetune/README.md) |
| 04 | [mAP](04_mAP/README.md) |
| 05 | [Batch effects](05_batch_effect/README.md) |
| 06 | [Virtual cells](06_virtual_cell/README.md) |
| 07 | [Biological interpretation](07_biological/README.md) |
| 08 | [Cross-perturbation types](08_three_type/README.md) |
| 09 | [Paper figures](09_paper_plot_code/README.md) |
