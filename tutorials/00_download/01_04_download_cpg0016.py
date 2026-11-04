import os
import subprocess

source_list = [
    'source_1', 'source_2', 'source_3',
    'source_4',
    'source_5',
    'source_6', 'source_7', 'source_8', 'source_9', 'source_10',
    'source_11',  'source_13', 'source_15'
]

# 记录成功和失败的 source
success_list = []
fail_list = []

for source in source_list:
    s3_path = f"s3://cellpainting-gallery/cpg0016-jump/{source}/workspace/profiles/"
    save_dir = f"/fse/DC/CP_data/feature_data/cpg0016_jump/{source}"

    # 创建目标目录
    os.makedirs(save_dir, exist_ok=True)

    # 构建命令
    cmd = f"aws s3 cp --recursive {s3_path} {save_dir} --no-sign-request"

    print(f"📥 Downloading {source}...")

    # 执行命令
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

    if result.returncode == 0:
        print(f"✅ {source} downloaded successfully")
        success_list.append(source)
    else:
        print(f"❌ {source} failed: {result.stderr}")
        fail_list.append(source)

# 总结
print(f"\n✅ Success: {len(success_list)} sources: {success_list}")
print(f"❌ Failed: {len(fail_list)} sources: {fail_list}")
