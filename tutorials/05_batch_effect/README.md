# 批次效应与校正

输入输出：已有 AnnData → 校正后的 AnnData → scIB / LISI / mAP。

`03_batch_method.py --data-path INPUT.h5ad --save-path NEW_OUTPUT` 调用原有七种方法适配器；`04_calculate_metrics.py --data-path DIR --save-path NEW_OUTPUT` 评估已有输出。方法前缀到 embedding 的映射可用 `--method-embeddings-json` 显式传入，默认保留 Bbknn 实验设定。BBKNN 的 X 不等于其校正图表征，需按原注释选择指标输入。`05_remove_descort3505_source4_outlier.py` 是特定实验的过滤脚本，不是所有数据的必需步骤。

```bash
bash tutorials/05_batch_effect/run.sh --help
```

从任意工作目录执行时，可使用本入口的绝对路径。参数应以当前脚本 `--help` 为准；历史实验 Bash 中的服务器路径需要按本机配置调整。
