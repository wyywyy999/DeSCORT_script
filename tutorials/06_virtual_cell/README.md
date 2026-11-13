# 06 — Virtual Cell Perturbation Generation

## Pipeline Overview

```text
             Normalized profiles + perturbation conditions
                                  │
                                  ▼
              ┌───────────────────────────────────────┐
              │            01_process_data            │
              │ Select training and test profiles     │
              │ Pair perturbations with controls      │
              └───────────────────┬───────────────────┘
                                  │
                                  ▼
              ┌───────────────────────────────────────┐
              │              02_finetune              │
              │ Fine-tune a perturbation predictor    │
              │ using control profiles + conditions   │
              └───────────────────┬───────────────────┘
                                  │
                                  ▼
                         finetune/models/
                                  │
                                  ▼
              ┌───────────────────────────────────────┐
              │              03_inference             │
              │ Generate perturbed morphology         │
              │ Export truth, prediction, train mean  │
              └───────────────────┬───────────────────┘
                                  │
                                  ▼
              ┌───────────────────────────────────────┐
              │     04_calculate_vcc2026_metric        │
              │ Validate alignment and calculate      │
              │ model and baseline morphology metrics │
              └───────────────────┬───────────────────┘
                                  │
                                  ▼
                       model_vs_baseline.csv
```

## Before You Start

Cell-line transfer and unseen-gene generation follow the same workflow: prepare paired data, fine-tune a perturbation predictor, generate perturbed profiles, and evaluate predictions. Their main difference is the dataset and the definition of the training/test split.

| Task | Training/test definition | CRISPR entry directory |
|---|---|---|
| Unseen-gene generation | Separate non-control gene sets for training and testing | `unseen_gene_generation/Crispr/` |
| Cell-line transfer | Select training and test data according to the source/target cell-line experiment | `cell_line_transfer/Crispr/` |

Both tasks also provide `Compound/` entries. This tutorial gives **one example only: CRISPR unseen-gene generation**. The cell-line-transfer workflow uses its corresponding entries and datasets; directory names alone do not enforce data separation.

This task predicts continuous **Cell Painting morphological features**. Paths below are placeholders; replace them before running.

Activate the environment and enter the example directory. Run Steps 1–4 from this directory:

```bash
conda activate DeSCORT
cd /absolute/path/tutorials/06_virtual_cell/unseen_gene_generation/Crispr
```

## Step 1 — Prepare Paired Training and Test Datasets

### Required resources

| Resource | Purpose |
|---|---|
| Normalized Parquet files | Morphological values, perturbation labels, and within-file controls |
| `gene2index_pkl` | Gene name to integer ID mapping |
| `index2gene_pkl` | Integer ID to gene name mapping |
| `esm2_dict_pkl` | Gene-name-to-ESM2 dictionary used to filter the gene lists |
| `index2esm2_pkl` | Gene-ID-to-ESM2 dictionary used by the model; must agree with the name-based dictionary |
| `feature2index_pkl` | Feature name to integer ID mapping |
| `index2feature_pkl` | Integer ID to feature name mapping |
| `need_feature_list_pkl` | Selected morphological features, shared across training and inference |
| Training/test gene-list pickles | Explicit non-overlapping non-control gene sets |


### Key arguments

| Argument | Meaning |
|---|---|
| `--source_data_paths` | One or more normalized Parquet directories; searched recursively |
| `--train_output_dir` | Temporary training shards, separate from the test directory |
| `--test_output_dir` | Temporary test shards, separate from the training directory |
| `--train_save_dir` | Merged training Hugging Face Dataset directory |
| `--test_save_dir` | Merged test Hugging Face Dataset directory |
| `--gene2index_pkl` | Gene name to integer ID mapping |
| `--esm2_dict_pkl` | Condition dictionary indexed by name |
| `--feature2index_pkl` | Feature vocabulary |
| `--need_feature_list_pkl` | Selected feature list |
| `--train_gene_list_pkl` | Training gene-list pickle; supplied in this example |
| `--test_gene_list_pkl` | Test gene-list pickle; supplied in this example |
| `--key_gene_column` | `Metadata_Foci_Barcode_MatchedTo_GeneCode` |
| `--control_key_word` | `nontargeting`; controls must exist within each processed file |
| `--train_data_names` | Optional filename/path substring filters for training data |
| `--test_data_names` | Optional filename/path substring filters for test data |
| `--nproc` | 1 |
| `--merge_nproc` | 2 |
| `--verbosity` | 1 |
| `--check-only` | Check resources and gene-selection information without building datasets; does not verify actual train/test gene disjointness |

### Example

```bash
bash 01_process_data_unseen_gene_generation.sh \
  --source_data_paths /absolute/path/normalized_profiles \
  --train_output_dir /absolute/path/virtual_cell/unseen_crispr_50genes/tmp/train \
  --train_save_dir /absolute/path/virtual_cell/unseen_crispr_50genes/data/train \
  --test_output_dir /absolute/path/virtual_cell/unseen_crispr_50genes/tmp/test \
  --test_save_dir /absolute/path/virtual_cell/unseen_crispr_50genes/data/test \
  --gene2index_pkl /absolute/path/ESM2_files/gene2index_21611.pkl \
  --esm2_dict_pkl /absolute/path/ESM2_files/ESM2_pert_features_with_nontargeting.pkl \
  --feature2index_pkl /absolute/path/feature_vocab/feature2index_vocabulary_7366.pkl \
  --need_feature_list_pkl /absolute/path/selected_features/intersection_columns_1107.pkl \
  --train_gene_list_pkl /absolute/path/few_shot/gene_50_train.pkl \
  --test_gene_list_pkl /absolute/path/few_shot/gene_50_test.pkl \
  --key_gene_column Metadata_Foci_Barcode_MatchedTo_GeneCode \
  --control_key_word nontargeting \
  --train_data_names CP257A CP257B CP257D CP257F CP257H \
  --test_data_names CP257A CP257B CP257D CP257F CP257H \
  --nproc 1 \
  --merge_nproc 2 \
  --verbosity 1
```



### Output

```text
virtual_cell/unseen_crispr_50genes/
├── tmp/
│   ├── train/
│   └── test/
└── data/
    ├── train/
    └── test/
```

Each dataset row contains:

| Field | Content |
|---|---|
| `Perturb_Gene` | Perturbation gene name |
| `Perturb_expression` | Observed perturbed morphology, used as the target |
| `Control_expression` | Paired control morphology |
| `Feature_name_index` | Feature IDs aligned with both vectors |
| `Perturb_Gene_Index` | Gene ID used to retrieve the condition vector |

Inspect the saved datasets with `datasets.load_from_disk()`. Paired vectors and feature IDs must have equal lengths and matching order. Use new temporary directories for each experiment.

**The entry point does not guarantee an unseen-gene split automatically.** A successful `--check-only` run does not prove train/test disjointness. Check the original gene lists and, after Step 1, the non-control genes actually present in both datasets. Confirm that both contain valid perturbations:

```python
from datasets import load_from_disk

train = load_from_disk("/absolute/path/virtual_cell/unseen_crispr_50genes/data/train")
test = load_from_disk("/absolute/path/virtual_cell/unseen_crispr_50genes/data/test")
train_genes = set(train.unique("Perturb_Gene")) - {"nontargeting"}
test_genes = set(test.unique("Perturb_Gene")) - {"nontargeting"}
assert train_genes and test_genes, "The training or test dataset has no valid perturbation genes"
assert not train_genes & test_genes, "Training and test genes overlap"
```

The initialization checkpoint must also respect the training scope of the unseen-gene experiment. The current fine-tuning split alone cannot rule out earlier exposure to test genes.

## Step 2 — Fine-tune the Perturbation Predictor

The entry loads an initialization checkpoint and trains a conditional predictor using control morphology and gene ESM2 vectors. The supplied loss weights are:

```text
loss = 1.0 × mean_loss
     + 1.0 × direction_loss
     + 0.1 × marginal_quantile_loss
```


### Key arguments and presets

| Argument | Supplied Bash preset / meaning |
|---|---|
| `--train_dataset_path` | Training Dataset from Step 1; test data are not passed to training |
| `--pretrianed_model_path` | Initialization checkpoint; retain this exact CLI spelling |
| `--index2esm2_pkl` | Gene-ID-to-ESM2 dictionary |
| `--output_dir` | New fine-tuning experiment directory |
| `--deepspeed` | The example uses `/absolute/path/deepspeed/zero2.json`; supply an actual configuration file |
| `--gene_column` | `Perturb_Gene`; used for same-gene training batches |
| `--hidden_size` | 256 |
| `--n_layers` | 8 |
| `--num_attention_heads` | 8 |
| `--n_routed_experts` | 8 routed experts |
| `--num_experts_per_tok` | Select 4 experts per token |
| `--esm2_input_dim` | 5120; must match the condition vectors |
| `--max_seq_len` | 1107 |
| `--max_length` | 1107 |
| `--sampling` | `False` |
| `--pad_value` | 100 |
| `--num_train_epochs` | 50 |
| `--per_device_train_batch_size` | 64 |
| `--gradient_accumulation_steps` | 1 |
| `--learning_rate` | `2e-4` |
| `--lr_scheduler_type` | `cosine` |
| `--warmup_ratio` | 0.1 |
| `--seed` | 42 |
| `--bf16` | `True` |
| `--do_eval` | `False` |
| `--save_strategy` | `epoch` |
| `--save_total_limit` | 5 |

### Example

```bash
GPU_IDS=0,1 bash 02_finetune.sh \
  --train_dataset_path /absolute/path/virtual_cell/unseen_crispr_50genes/data/train \
  --pretrianed_model_path /absolute/path/initialization_checkpoint/models \
  --index2esm2_pkl /absolute/path/ESM2_files/Index2ESM2_pert_features_with_nontargeting_19791.pkl \
  --output_dir /absolute/path/virtual_cell/unseen_crispr_50genes/finetune \
  --deepspeed /absolute/path/deepspeed/zero2.json \
  --esm2_input_dim 5120 \
  --max_length 1107 \
  --sampling False \
  --num_train_epochs 50 \
  --per_device_train_batch_size 64 \
  --gradient_accumulation_steps 1 \
  --learning_rate 2e-4 \
  --bf16 True \
  --seed 42 \
  --overwrite_output_dir False
```

### Output

```text
virtual_cell/unseen_crispr_50genes/finetune/
├── models/
│   ├── config.json
│   └── model.safetensors    # May be sharded depending on settings
├── checkpoint-*/
├── runs/
└── trainer_state.json
```


## Step 3 — Generate Morphology for Unseen Genes

The model receives paired control morphology and the test gene's ESM2 condition vector. Observed test perturbation morphology is exported as ground truth; it is not the input used to generate predictions.

### Key arguments and presets

| Argument | Meaning |
|---|---|
| `--pretrained_model_path` | Fine-tuned `models/` directory from Step 2 |
| `--index2gene_pkl` | Gene ID to name mapping consistent with Step 1 |
| `--index2esm2_pkl` | Condition vectors consistent with Step 1 |
| `--index2feature_pkl` | Feature ID to name mapping consistent with Step 1 |
| `--train_dataset_path` | Training Dataset, used to build the train-mean baseline |
| `--test_dataset_path` | Held-out-gene Dataset to predict |
| `--output_dir` | Directory for the three H5AD files |
| `--prediction_filename` | Model prediction filename; set explicitly below |
| `--ground_truth_filename` | Ground-truth filename |
| `--mean_baseline_filename` | Training-mean baseline filename |
| `--device` | Logical device |
| `--batch_size` | Inference batch size |
| `--max_length` | Sequence length |
| `--no_sampling` | Preserve the preset's disabled sampling behavior |
| `--gene_column` | Output label column `Perturb_Gene` |
| `--control_label` | Control label `nontargeting` |

### Example

```bash
CUDA_VISIBLE_DEVICES=0 bash 03_run_inference.sh \
  --pretrained_model_path /absolute/path/virtual_cell/unseen_crispr_50genes/finetune/models \
  --index2gene_pkl /absolute/path/label_vocab/index2gene_21611.pkl \
  --index2esm2_pkl /absolute/path/ESM2_files/Index2ESM2_pert_features_with_nontargeting_19791.pkl \
  --index2feature_pkl /absolute/path/feature_vocab/index2feature_vocabulary_7366.pkl \
  --train_dataset_path /absolute/path/virtual_cell/unseen_crispr_50genes/data/train \
  --test_dataset_path /absolute/path/virtual_cell/unseen_crispr_50genes/data/test \
  --output_dir /absolute/path/virtual_cell/unseen_crispr_50genes/inference \
  --prediction_filename prediction_data.h5ad \
  --ground_truth_filename ground_truth.h5ad \
  --mean_baseline_filename perturb_mean_data.h5ad \
  --device cuda:0 \
  --batch_size 64 \
  --max_length 1107 \
  --pad_value 100 \
  --no_sampling \
  --gene_column Perturb_Gene \
  --control_label nontargeting \
  --log_file /absolute/path/virtual_cell/unseen_crispr_50genes/logs/inference.log \
  --log_level INFO
```

`CUDA_VISIBLE_DEVICES=0` selects physical GPU 0; `--device cuda:0` selects the first visible GPU. Use `--device cpu` for CPU execution. The exporter collects the complete test matrices in memory, so reducing batch size does not remove the memory requirement of the final outputs.

Training and inference must use the same feature axis. With sampling disabled, sequences longer than `max_length` are truncated; shorter sequences are padded. This exporter expects the complete original feature-name axis to match the returned matrix, so choose a length covering all selected features and keep the feature order consistent. Padding positions are excluded from exported morphology.

### Output

| File | Contents of `X` |
|---|---|
| `ground_truth.h5ad` | Observed test perturbation morphology, followed by paired controls |
| `prediction_data.h5ad` | Predicted test perturbation morphology, followed by the same controls |
| `perturb_mean_data.h5ad` | Repeated train-mean baseline, followed by the same controls |


## Step 4 — Evaluate Predictions and the Train-mean Baseline

This entry calculates seven raw VCC2026-style metrics for continuous morphology. Differential-feature testing uses Wilcoxon/Mann–Whitney tests with BH correction. Effect sizes are perturbation-minus-control mean differences; `lfc` in the metric names does not mean taking log2 fold changes of morphology values.

### Key arguments

| Argument | Meaning |
|---|---|
| `--ground_truth_path` | Ground-truth H5AD generated in Step 3 |
| `--prediction_path` | Prediction H5AD generated in Step 3 |
| `--baseline_path` | Training-mean baseline H5AD generated in Step 3 |
| `--outdir` | Metric output directory |
| `--cache_dir` | Reusable differential-test cache directory |
| `--pert_col` | `Perturb_Gene` |
| `--control_name` | `nontargeting` |
| `--dedup_controls` | `true`; deduplicate identical control vectors |
| `--strict_alignment` | `true`; require matching sample IDs/order and perturbation labels |
| `--copy_truth_controls` | `false`; do not replace the supplied controls |
| `--fdr_scope` | `per_pert`; correct across features within each perturbation |
| `--n_jobs` | 8 |
| `--blas_threads` | 4 |
| `--reuse_cache` | `true` |
| `--show_progress` | `true` |
| `--check-only` | Validate input alignment and run a small synthetic MWU check without calculating experiment metrics |

### Example

```bash
bash 04_calculate_vcc2026_metric.sh \
  --ground_truth_path /absolute/path/virtual_cell/unseen_crispr_50genes/inference/ground_truth.h5ad \
  --prediction_path /absolute/path/virtual_cell/unseen_crispr_50genes/inference/prediction_data.h5ad \
  --baseline_path /absolute/path/virtual_cell/unseen_crispr_50genes/inference/perturb_mean_data.h5ad \
  --outdir /absolute/path/virtual_cell/unseen_crispr_50genes/metrics \
  --cache_dir /absolute/path/virtual_cell/unseen_crispr_50genes/de_cache \
  --pert_col Perturb_Gene \
  --control_name nontargeting \
  --dedup_controls true \
  --strict_alignment true \
  --copy_truth_controls false \
  --fdr_scope per_pert \
  --n_jobs 8 \
  --blas_threads 4 \
  --scipy_checks 100 \
  --reuse_cache true \
  --show_progress true
```


### Output metrics

| Metric | Interpretation |
|---|---|
| `pds_cosine` | Perturbation retrieval using cosine distance between morphology effects; higher is better |
| `mse` | Mean-squared error of perturbation mean morphology; lower is better |
| `expr_mse_unbiased_capped_norm` | Noise-corrected, normalized, capped error; lower is better |
| `de_wilcoxon_direction_fidelity_yield_raw` | Direction agreement and yield of significant effects; higher is better |
| `de_wilcoxon_direction_reach_raw` | Effect coverage under the direction-agreement criterion; higher is better |
| `de_wilcoxon_sig_jaccard` | Jaccard similarity of significant feature sets; higher is better |
| `de_wilcoxon_lfc_nmae` | Normalized absolute error of significant morphology effects; lower is better |

These are raw metrics, not uniformly scaled scores relative to replicate upper bounds and baselines. Inspect `valid_metric_gene_counts` alongside the summary; undefined effects can produce NaN values and should not automatically be replaced with zero.

```text
virtual_cell/unseen_crispr_50genes/
├── tmp/
│   ├── train/
│   └── test/
├── data/
│   ├── train/
│   └── test/
├── finetune/models/
├── inference/
│   ├── ground_truth.h5ad
│   ├── prediction_data.h5ad
│   └── perturb_mean_data.h5ad
├── logs/inference.log
├── de_cache/
└── metrics/
    ├── input_provenance.json
    ├── cli_parameters.json
    ├── model_metrics.csv
    ├── model.parquet
    ├── cell_eval/
    │   ├── results.csv
    │   └── agg_results.csv
    ├── run_manifest.json
    ├── baseline/
    │   ├── baseline_metrics.csv
    │   ├── baseline.parquet
    │   └── run_manifest.json
    └── model_vs_baseline.csv
```

## Supplement — Generate profiles from controls with requested conditions and counts

The entry point is [generate_from_control.py](generate_from_control.py), launched through [generate_from_control.sh](generate_from_control.sh). Argument parsing and input checks are in [src/descort/vc/generate_from_control.py](../../src/descort/vc/generate_from_control.py). This section can run independently after model training, without preparing the Step 3 Arrow test dataset or running Step 4 evaluation.

Use a trained **DeSCORTForPerturbPrediction** model to read controls from normalized morphological feature Parquet and generate predictions conditioned on the perturbations requested in a CSV. This supports CRISPR and Compound models for cell line transfer and unseen gene generation; use the checkpoint and training resources for the intended task.

### Input

The Parquet must contain the perturbation label column specified by `--pert_col`, control rows, and the numerical morphological features used in training. Do not pass CLS embeddings. Normalization and feature processing must match training.

Example CSV:

```text
pert_col,number
PSMA1,100
ATP6V0C,50
```

The CSV column is named `pert_col` by default; the command-line option `--pert_col` specifies the **label column in the Parquet**. For a different CSV column name, use `--csv_pert_col`. For example, if the CSV uses `Metadata_Treatment`, pass `--csv_pert_col Metadata_Treatment`.

`number` is a nonnegative integer specifying how many samples to generate for that row. Zero generates no samples, but the perturbation name and condition vector are still validated; unknown perturbations or missing condition vectors still cause errors. At least one row must have a positive count. Duplicate perturbation rows are processed separately and their counts add up.

CRISPR perturbation names must match the training gene2index vocabulary. Compound conditions are usually SMILES from the training smiles2index vocabulary; unmapped drug names cannot be substituted directly. Quote SMILES containing commas according to CSV rules. This script does not compute new ESM2/Morgan vectors: conditions and corresponding vectors must already exist in the supplied resources. They may include perturbations unseen during training if condition vectors are available.

### Launch

Switch to the directory containing the shared generation scripts; this differs from the directory for Steps 1–4. Replace `/absolute/path` placeholders in the scripts. Keep Python filenames relative.

```bash
cd /absolute/path/tutorials/06_virtual_cell
```

### Checkpoints without a saved feature axis

The current CRISPR fine-tuning entry point does not explicitly save `population_feature_ids` in the configuration. For these checkpoints, pass `--need_feature_list_pkl` containing the **actual training feature order**. If the checkpoint has a saved feature axis, the generation script uses that axis preferentially.

Step 1 uses the selected-feature list to filter columns, but the actual order comes from Parquet column order. Do not assume that the selection list is in training order. For the consistently laid-out CRISPR data in this tutorial, recover and save ordered feature names from the training Arrow dataset. The following example also checks that every training row shares the same feature axis:

```python
from pathlib import Path
import pickle
import numpy as np
from datasets import load_from_disk

train = load_from_disk("/absolute/path/virtual_cell/unseen_crispr_50genes/data/train")
with open("/absolute/path/feature_vocab/index2feature_vocabulary_7366.pkl", "rb") as handle:
    index2feature = pickle.load(handle)
axes = train.select_columns(["Feature_name_index"]).with_format("numpy")
assert len(axes) > 0, "The training dataset is empty"
reference = np.asarray(axes[0]["Feature_name_index"], dtype=np.int64)
assert reference.ndim == 1 and len(reference) > 0
assert np.all(reference > 0) and len(np.unique(reference)) == len(reference)
for start in range(0, len(axes), 512):
    for axis in axes[start:start + 512]["Feature_name_index"]:
        assert np.array_equal(axis, reference), "Multiple training feature layouts exist; the first row cannot represent the entire dataset"
ordered_features = [index2feature[int(index)] for index in reference]
output = Path("/absolute/path/training_feature_order.pkl")
output.parent.mkdir(parents=True, exist_ok=True)
with output.open("wb") as handle:
    pickle.dump(ordered_features, handle)
```

The inverse vocabulary must match the training `feature2index` mapping. If the check finds multiple layouts, determine the feature axis from the actual training alignment strategy rather than applying this example directly.

CRISPR example:

```bash
bash generate_from_control.sh \
  --pretrained_model_path /absolute/path/crispr/models \
  --parquet_path /absolute/path/normalized_profiles.parquet \
  --csv_path /absolute/path/generation_requests.csv \
  --pert_col Metadata_Foci_Barcode_MatchedTo_GeneCode \
  --control_label nontargeting \
  --perturbation2index_pkl /absolute/path/gene2index.pkl \
  --index2condition_pkl /absolute/path/index2esm2.pkl \
  --feature2index_pkl /absolute/path/feature2index.pkl \
  --need_feature_list_pkl /absolute/path/training_feature_order.pkl \
  --output_path /absolute/path/crispr_generated.parquet
```

For Compound, use the same script and replace the model, label column, and two condition resources:

```bash
bash generate_from_control.sh \
  --pretrained_model_path /absolute/path/compound/models \
  --parquet_path /absolute/path/compound_normalized.parquet \
  --csv_path /absolute/path/compound_requests.csv \
  --pert_col Metadata_SMILES \
  --control_label nontargeting \
  --perturbation2index_pkl /absolute/path/smiles2index.pkl \
  --index2condition_pkl /absolute/path/index2morgan.pkl \
  --feature2index_pkl /absolute/path/feature2index.pkl \
  --output_path /absolute/path/compound_generated.parquet
```

`Metadata_SMILES` is only an example column name. Use the label column and control value actually present in the Parquet. The CSV still defaults to `pert_col,number`, with condition values matching SMILES in the training vocabulary.

### Parameters and behavior

| Argument | Meaning or default |
|---|---|
| `--pretrained_model_path` | Perturbation prediction model directory for this task; a foundation representation model cannot be substituted |
| `--parquet_path` | One normalized morphological feature Parquet |
| `--csv_path` | One generation-request CSV |
| `--pert_col` | Parquet column used to select controls and store generated perturbation labels |
| `--control_label` | `nontargeting`, matched exactly; replace with the actual control label |
| `--csv_pert_col` | CSV perturbation column; default `pert_col` |
| `--number_col` | CSV count column; default `number` |
| `--perturbation2index_pkl` | Perturbation name to integer ID mapping consistent with training |
| `--index2condition_pkl` | Integer ID to condition vector mapping; ESM2 for CRISPR, Morgan for Compound |
| `--feature2index_pkl` | Feature name to integer ID mapping consistent with training |
| `--need_feature_list_pkl` | Required ordered training feature list when an older checkpoint has no `population_feature_ids`; do not substitute an arbitrary Parquet column order |
| `--output_path` | Generated output Parquet path |
| `--batch_size` | 64 |
| `--device` | `cuda:0`; use `--device cpu` for CPU inference |
| `--seed` | 42 |
| `--replacement` | `auto`: sample without replacement within each request if its count fits the control pool, otherwise with replacement |
| `--overwrite` | Allow replacement of an existing output file; overwriting is refused by default |

`--replacement never` always samples without replacement and errors if the count exceeds the control pool. `always` always samples with replacement. Requests are sampled independently and may reuse controls. All rows matching the control label form one pool, without further grouping by plate, batch, or cell line; filter the input Parquet first to restrict their source.

Feature order comes preferentially from the checkpoint, or from the ordered training feature list for older models. Sequence length is the checkpoint's `max_seq_len`: excess features are truncated from the end with a log message, and shorter inputs are padded without exporting padding. Features are not randomly sampled. Missing input features, NaN, or infinite values cause errors. The model determines direct versus residual prediction; controls are not added again externally.

### Output

The output row count equals the sum of CSV `number`, preserving request order. Only generated samples are saved; original controls, true perturbation profiles, and mean baselines are not appended.

- Numerical feature columns: model predictions, ordered consistently with the effective input features.
- The column specified by `--pert_col`: target perturbation labels from the CSV.
- Other original `Metadata_` columns: inherited from sampled controls to describe their source. Barcode and similar metadata are not experimental annotations for the target perturbation.
- `Metadata_Generation_ControlRow`: zero-based row position of the sampled control in the original Parquet.
- `Metadata_Generation_ControlLabel`: sampled control label.
- `Metadata_Generation_RequestRow`: zero-based CSV request row, excluding the header.
- `Metadata_Generation_ConditionIndex`: integer ID of the requested condition.

The temporary output replaces the target only after successful completion. Failed runs do not leave a target file that appears complete. Identical inputs, seed, and runtime environment reproduce control sampling; model floating-point results may vary slightly across hardware.
