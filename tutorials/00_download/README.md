# 数据下载

输入输出：原始数据源 → 本地原始文件。

`01_01_download_A549_cpg0021.py`、`01_02_download_HeLa_cpg0021.py`、`01_03_download_cpg0000.py`、`01_04_download_cpg0016.py`。新脚本保留原数据源，要求显式 `--output-dir`，可先 `--dry-run` 查看下载命令；cpg0000 还需 `--metadata-csv`，默认与原活动代码一样下载 Compound 和 ORF，`--perturbation-types crispr` 可显式选择 CRISPR。目录拼写 `Compoud` 为兼容原文件结构保留。下载依赖本机 wget / AWS CLI，本次未启动下载。

```bash
bash tutorials/00_download/run.sh --help
```

从任意工作目录执行时，可使用本入口的绝对路径。参数应以当前脚本 `--help` 为准；历史实验 Bash 中的服务器路径需要按本机配置调整。
