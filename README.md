<p align="center">
  <img src="assets/DeSCORT_logo_300dpi.svg" alt="DeSCORT logo" width="270">
</p>

# DeSCORT: Decoding Morphological Signatures of Cellular Responses to Perturbations

![DeSCORT model architecture](assets/workflow.png)

DeSCORT is a well-level morphological foundation model built from large-scale CRISPR perturbation data. Its Transformer architecture jointly encodes morphological feature identities and continuous values, combining masked feature reconstruction with perturbation-consistent contrastive learning to learn general morphological representations. This framework supports research into perturbation retrieval, batch correction, biological functional associations, morphological feature generation, and transfer across cell lines. It also enables exploration of functional relationships among chemical perturbations, CRISPR gene knockouts, and ORF overexpression for morphological data integration, virtual perturbation prediction, and mechanism-of-action analysis.



## Installation

### Step 1 — Create a Python environment

Use a dedicated Conda environment. The following example uses Python 3.9:

```bash
conda create -n DeSCORT python=3.9 pip -y
conda activate DeSCORT
python -m pip install --upgrade pip setuptools wheel
```

### Step 2 — Install PyTorch

Select a PyTorch CUDA build supported by your NVIDIA driver. The examples below install PyTorch **2.7.0**; **run only one option**. See the [official PyTorch installation instructions](https://pytorch.org/get-started/previous-versions/) for platform support and commands.



```bash
# CUDA 11.8
python -m pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cu118
# CUDA 12.6
python -m pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cu126
# CUDA 12.8
python -m pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cu128
```

For CPU-only use:

```bash
python -m pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cpu
```

After installation, check the version and GPU availability:

```bash
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA build:', torch.version.cuda); print('GPU available:', torch.cuda.is_available())"
```
For a CPU build, `GPU available: False` is expected. A GPU training environment must detect an available GPU.

### Step 3 — Install DeepSpeed

Install DeepSpeed before running tutorial pretraining or fine-tuning scripts launched with `deepspeed`. The example uses **0.18.0**, a version used in this project:

```bash
python -m pip install deepspeed==0.18.0
```

### Step 4 — Install DeSCORT

#### Option 1 — Install from a package index (currently TestPyPI)

`descort==0.1.7` is published on **TestPyPI**, not the production PyPI index. The command below makes TestPyPI and PyPI available to pip for package and dependency resolution.

```bash
python -m pip install \
  --index-url https://test.pypi.org/simple/ \
  --extra-index-url https://pypi.org/simple/ \
  descort==0.1.7
```

#### Option 2 — Clone and install from GitHub

Install Git, then clone the repository and install from its root directory:

```bash
git clone https://github.com/wyywyy999/DeSCORT_script.git
cd DeSCORT_script
python -m pip install .
```

Version 0.1.7 installs training, preprocessing, virtual-cell, and downstream-analysis dependencies by default; no dependency extras are required. If the repository is private, authenticate with a GitHub account that has access before cloning.

### Step 5 — Verify installation

Use the same verification commands for either installation method:

```bash
python -c "import descort; from descort import DeSCORTConfig, DeSCORTForContrastiveLM; print('DeSCORT:', descort.__version__); print('Package path:', descort.__file__)"
python -m pip show descort
```

Run `ds_report` separately to inspect the DeepSpeed environment. Prepare task-specific backends such as R, scIB, and DESC/TensorFlow for batch correction as needed; see the [installation guide](docs/installation.md).

`environments/environment.yml` and `environments/requirement.txt` record the full environment from the original server, including additional tools and known version-constraint conflicts. Neither is a required step for the installation methods above, which use the dependencies declared by this project.

## Documentation

For installation instructions and step-by-step tutorials, see the [DeSCORT documentation](https://descort-script.readthedocs.io/en/latest/?refresh=no-reference-v2).


## Task navigation

| Task | Entry point | Input → Output |
|---|---|---|
| Cell Painting downloads | [00_download](tutorials/00_download/README.md) | Cell Painting Gallery (cpg0021, cpg0000, cpg0016) → raw feature files |
| Preprocessing and vocabularies | [01_preprocess_data](tutorials/01_preprocess_data/README.md) | CSV / Parquet → aggregated, normalized, clipped data; label vocabularies; feature intersections/unions and filtered data |
| Pretraining | [02_pretrain](tutorials/02_pretrain/README.md) | Parquet → Arrow dataset → foundation model |
| Representation fine-tuning and embeddings | [03_finetune](tutorials/03_finetune/README.md) | Foundation model, labeled data → representation model and embeddings |
| mAP | [04_mAP](tutorials/04_mAP/README.md) | Embeddings → retrieval metrics |
| Batch effects | [05_batch_effect](tutorials/05_batch_effect/README.md) | AnnData → corrected outputs and metrics |
| Virtual cells | [06_virtual_cell](tutorials/06_virtual_cell/README.md) | Control profiles, perturbation vectors → predicted perturbed morphology |
| Biological analysis | [07_biological](tutorials/07_biological/README.md) | Morphological representations, annotations → pathways, clustering, and enrichment analysis |
| Cross-perturbation types | [08_three_type](tutorials/08_three_type/README.md) | CRISPR / Compound / ORF → shared-representation analysis |
| Paper figures | [09_paper_plot_code](tutorials/09_paper_plot_code/README.md) | Existing result tables → paper figures |

Configure paths inside the download scripts and run them directly. The preprocessing tutorial has five steps; launch its Bash scripts from `tutorials/01_preprocess_data`. See each task README for its entry points and parameters, and the [tutorial overview](tutorials/README.md) for the complete workflow.
