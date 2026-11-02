# 本地历史代码

- `original_code_20261102.tar.gz`：整理前的 Python/Bash/JSON/Markdown/环境文件快照；不包含数据、权重及结果。具体文件与 SHA-256 见 `docs/reorganization/original_files.json`。
- `legacy_models/`：原 `src/descort/modeling_descort.py` 和模型 `copy` 副本，仅供差异追溯，不属于安装包；根目录旧模型文件原先还引用了不存在的相邻配置模块。
- `zero2_source.json`：汇总到 `configs/deepspeed/zero2.json` 的原配置。

正式实现是 `src/descort/model/modeling_descort.py`。历史模型不视为与其数值等价，也不自动用于加载当前检查点。
