# Generate virtual cells with requested perturbations and counts

The entry point is [generate_from_control.py](generate_from_control.py), launched through [generate_from_control.sh](generate_from_control.sh). Argument parsing and input checks are in [src/descort/vc/generate_from_control.py](../../src/descort/vc/generate_from_control.py). This section can run independently after model training, without preparing the Step 3 Arrow test dataset or running Step 4 evaluation.

Use a trained **DeSCORTForPerturbPrediction** model to read controls from normalized morphological feature Parquet and generate predictions conditioned on the perturbations requested in a CSV. This supports CRISPR and Compound models for cell line transfer and unseen gene generation; use the checkpoint and training resources for the intended task.

## Input

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

## Launch

Switch to the directory containing the shared generation scripts; this differs from the directory for Steps 1–4. Replace `/absolute/path` placeholders in the scripts. Keep Python filenames relative.

```bash
cd /absolute/path/tutorials/06_virtual_cell
```

## Checkpoints without a saved feature axis

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

## Parameters and behavior

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

## Output

The output row count equals the sum of CSV `number`, preserving request order. Only generated samples are saved; original controls, true perturbation profiles, and mean baselines are not appended.

- Numerical feature columns: model predictions, ordered consistently with the effective input features.
- The column specified by `--pert_col`: target perturbation labels from the CSV.
- Other original `Metadata_` columns: inherited from sampled controls to describe their source. Barcode and similar metadata are not experimental annotations for the target perturbation.
- `Metadata_Generation_ControlRow`: zero-based row position of the sampled control in the original Parquet.
- `Metadata_Generation_ControlLabel`: sampled control label.
- `Metadata_Generation_RequestRow`: zero-based CSV request row, excluding the header.
- `Metadata_Generation_ConditionIndex`: integer ID of the requested condition.

The temporary output replaces the target only after successful completion. Failed runs do not leave a target file that appears complete. Identical inputs, seed, and runtime environment reproduce control sampling; model floating-point results may vary slightly across hardware.
