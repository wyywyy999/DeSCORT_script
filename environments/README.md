# 环境记录

本机使用 `DeSCORT` 环境，Python 3.9.15。`environment.yml` 和 `requirement.txt` 记录原服务器的完整环境快照，包含额外工具及已知版本约束冲突，不是跨平台锁文件。新建环境及 PyTorch、DeepSpeed 安装步骤见根目录 README。

```bash
conda activate DeSCORT
python -m pip install -e .
```

CUDA/DeepSpeed 单独按训练机器配置。scIB、R/rpy2、DESC/TensorFlow、GEARS、STATE 与 UniMol2 等后端使用各自原环境；不要把它们都塞进最小推理环境。本次未重新创建全部第三方方法环境。
