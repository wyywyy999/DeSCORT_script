# 数据下载

输入输出：Cell Painting Gallery 数据源 → 本地原始文件。

| 脚本                                                                   | 数据                 | 执行前需要核对的配置                                       |
| -------------------------------------------------------------------- | ------------------ | ------------------------------------------------ |
| [01\_01\_download\_A549\_cpg0021.py](01_01_download_A549_cpg0021.py) | cpg0021 A549       | `path` 保存目录、`Plate_Num`；默认 A/B/C/D/E/F/G/H/N     |
| [01\_02\_download\_HeLa\_cpg0021.py](01_02_download_HeLa_cpg0021.py) | cpg0021 HeLa       | `path` 保存目录、`Plate_Num`；默认 B/D/F/H/J/K/L/N       |
| [01\_03\_download\_cpg0000.py](01_03_download_cpg0000.py)            | cpg0000 JUMP Pilot | 元数据 CSV 路径、ESM2 pickle 路径、A549/U2OS 保存目录及启用的下载调用 |
| [01\_04\_download\_cpg0016.py](01_04_download_cpg0016.py)            | cpg0016 JUMP       | `source_list`、循环内 `save_dir`；实际下载 source 以脚本列表为准 |

脚本沿用原服务器的绝对路径，使用前请直接修改本目录脚本中的配置。

Python 依赖为 `requests`、`pandas`、`tqdm`；系统命令需要 `wget`（cpg0021）或 AWS CLI 的 `aws`（cpg0000/cpg0016）。四个原脚本不提供 `--output-dir`、`--metadata-csv`、`--dry-run` 或 Python `--help` 参数，路径和筛选配置均在脚本内设置。

从仓库根目录运行所需脚本，例如：

```bash
python tutorials/00_download/01_01_download_A549_cpg0021.py
```
