# 路径与配置

JSON 配置由 `scripts/run_config.py` 真正读取；参数以原始 argv 字符串列表保存，避免隐式改写布尔值和空列表含义。

```bash
mkdir -p configs/local
cp configs/paths.example.sh configs/local/paths.sh
# 编辑 local/paths.sh 为真实路径，然后加载：
source configs/local/paths.sh
python scripts/run_config.py --config configs/pretrain/example.json --dry-run
# 审阅命令后运行：
python scripts/run_config.py --config configs/pretrain/example.json
# 命令行参数追加在配置之后，标量参数可覆盖配置：
python scripts/run_config.py --config configs/pretrain/example.json -- --num_train_epochs 2
```

配置相对路径以新仓库根目录为基准；环境变量没有设置时直接报错，不把占位符当真实路径。`--dry-run` 不导入训练模块，也不创建训练输出。

- `pretrain/example.json`：基础表征训练的可编辑例子。
- `finetune/example.json` / `embedding.example.json`：微调与提取。
- `virtual_cell/crispr_process.example.json`：显式训练/测试列表的 CRISPR 数据处理。
- `biological/umap.example.json`：输入检查阶段，先核对文件与特征。
- `deepspeed/zero2.json`：从原配置原样汇总。分布式 profile 可将 `launcher` 改成 `["deepspeed", "--include", "localhost:0,1"]`，并在 args 中加入 `--deepspeed configs/deepspeed/zero2.json`。

这些配置是启动模板，不宣称为论文最终参数。保留历史拼写 `--pretrianed_model_path`，避免破坏原训练参数接口。
