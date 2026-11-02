# 环境记录

本次验证复用了本机 `DeSCORT` 环境，Python 3.9.15。`requirements-tested.txt` 记录实际使用的关键依赖版本，`environment.yml` 提供核心环境模板；它不是完整的跨平台锁文件。

```bash
conda env create -f environments/environment.yml
conda activate descort-organized
python -m pip install -e '.[train,preprocess,vc,analysis]'
```

CUDA/DeepSpeed 单独按训练机器配置。scIB、R/rpy2、DESC/TensorFlow、GEARS、STATE 与 UniMol2 等后端使用各自原环境；不要把它们都塞进最小推理环境。本次未重新创建全部第三方方法环境。
