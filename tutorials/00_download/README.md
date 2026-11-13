# Data downloads

Input/output: Cell Painting Gallery sources → local raw files.

| Scripts                                                                   | Dataset                 | Configuration to verify before running                                       |
| -------------------------------------------------------------------- | ------------------ | ------------------------------------------------ |
| [01\_01\_download\_A549\_cpg0021.py](01_01_download_A549_cpg0021.py) | cpg0021 A549       | Output directory `path` and `Plate_Num`; default plates A/B/C/D/E/F/G/H/N     |
| [01\_02\_download\_HeLa\_cpg0021.py](01_02_download_HeLa_cpg0021.py) | cpg0021 HeLa       | Output directory `path` and `Plate_Num`; default plates B/D/F/H/J/K/L/N       |
| [01\_03\_download\_cpg0000.py](01_03_download_cpg0000.py)            | cpg0000 JUMP Pilot | Metadata CSV path, ESM2 pickle path, A549/U2OS output directories, and enabled download calls |
| [01\_04\_download\_cpg0016.py](01_04_download_cpg0016.py)            | cpg0016 JUMP       | `source_list` and `save_dir` inside the loop; the script's list determines which sources are downloaded |

The scripts retain absolute paths from the original server. Update the configuration in each script before running.

Python dependencies are `requests`, `pandas`, and `tqdm`. System commands require `wget` for cpg0021 or the AWS CLI command `aws` for cpg0000/cpg0016.

The four original scripts do not provide `--output-dir`, `--metadata-csv`, `--dry-run`, or Python `--help` options. Configure paths and filters directly in the scripts.

Run the required script from the repository root, for example:

```bash
python tutorials/00_download/01_01_download_A549_cpg0021.py
```
