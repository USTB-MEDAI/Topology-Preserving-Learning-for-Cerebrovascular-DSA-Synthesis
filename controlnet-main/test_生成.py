from cldm.hack import disable_verbosity
disable_verbosity()

import os
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from tutorial_dataset2 import MyDataset
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict

import shutil
import time

# Configs
# resume_path = "/disk/ssy/pyproj/bishe/controlnet-main/lightning_logs/version_29/checkpoints/epoch=999-step=40999.ckpt"  #模型路径
resume_path = "/disk/ssy/controlnet-main/lightning_logs/version_34/checkpoints/epoch=999-step=82999.ckpt"  #模型路径
base_data_dir = "/disk/ssy/AAAIresults/Controlnet-LDM/test_cond/骨架hands412_test"  #测试数据集
log_image_dir = "/disk/ssy/pyproj/bishe/controlnet-main/image_log/test"  # 图像保存路径"test"
# resume_path = "/disk/ssy/pyproj/bishe/controlnet-main/lightning_logs/version_25_handle_DSA/checkpoints/epoch=999-step=55999.ckpt"  #模型路径
# base_data_dir = "/disk/ssy/data/DIAS/myDIAS/hand/test_predict514pk"  #测试数据集
# log_image_dir = "/disk/ssy/pyproj/bishe/controlnet-main/image_log/test"  # 图像保存路径"test"
# origin_dir = "/your/origin/dir/path"  # 👈 修改为你的 origin_dir 路径

# 生成目标目录名（当前时间）
timestamp = time.strftime("%Y%m%d_%H%M%S")
renamed_image_dir = f"/disk/ssy/pyproj/bishe/controlnet-main/image_log/test_{timestamp}"
os.makedirs(renamed_image_dir, exist_ok=True)

batch_size = 1
logger_freq = 1400
learning_rate = 1e-5
sd_locked = True
only_mid_control = False

# 模型初始化（加载一次即可）
model = create_model('./models/cldm_v15.yaml').cpu()
model.load_state_dict(load_state_dict(resume_path, location='cpu'))
model.learning_rate = learning_rate
model.sd_locked = sd_locked
model.only_mid_control = only_mid_control

# 遍历所有子文件夹，并按名称排序（确保与 VSCode 中文件显示一致）
subfolders = [f for f in os.listdir(base_data_dir) if os.path.isdir(os.path.join(base_data_dir, f))]
subfolders = sorted(subfolders)

# 显示将要处理的文件夹顺序
print(f"总共将处理 {len(subfolders)} 个子文件夹，顺序如下：")
for idx, folder in enumerate(subfolders, 1):
    print(f"{idx:2d}. {folder}")
# 记录顺序
folder_order = subfolders.copy()

# 开始依次处理每个子文件夹
for idx, folder in enumerate(subfolders, 1):
    data_path = os.path.join(base_data_dir, folder)
    print(f"\n=== 正在处理第 {idx}/{len(subfolders)} 个文件夹：{folder} ===")

    # 创建 dataset 和 dataloader
    dataset = MyDataset(source_dir=data_path, target_dir=data_path)
    dataloader = DataLoader(dataset, num_workers=0, batch_size=batch_size, shuffle=True)

    # 创建 logger 和 trainer
    logger = ImageLogger(batch_frequency=logger_freq)
    trainer = pl.Trainer(gpus=[0], precision=32, callbacks=[logger])

    # 执行 test
    trainer.test(model, dataloader)


import re
from collections import defaultdict

print("\n=== 开始重命名 image_log/test 中的图像文件 ===")

# 获取图像文件列表
image_files = [f for f in os.listdir(log_image_dir) if f.lower().endswith(('.jpg', '.png'))]

# 用正则提取编号，按编号分组
grouped_files = defaultdict(list)
for f in image_files:
    match = re.search(r"ee-(\d+)_", f)
    if match:
        group_id = match.group(1)
        grouped_files[group_id].append(f)

# 将 group_id 按编号升序排列，确保顺序一致
sorted_groups = sorted(grouped_files.items(), key=lambda x: int(x[0]))

if len(sorted_groups) != len(folder_order):
    print(f"⚠️ 警告：图像编号组数（{len(sorted_groups)}）与文件夹数（{len(folder_order)}）不一致，请检查！")

# 依次重命名每组图像
for i, (group_id, files) in enumerate(sorted_groups):
    folder_name = folder_order[i]
    for f in files:
        old_path = os.path.join(log_image_dir, f)
        new_filename = re.sub(r"(ee-)\d+", r"\1" + folder_name, f)
        # new_filename = re.sub(r"^\d+", folder_name, f)
        new_path = os.path.join(log_image_dir, new_filename)
        os.rename(old_path, new_path)
        print(f"[{i+1:2d}] 重命名：{f} → {new_filename}")

print("\n✅ 所有图像文件批量重命名完成（按组替换编号）。")

# # 移动所有图片到新目录
# for f in os.listdir(log_image_dir):
#     if f.lower().endswith(('.jpg', '.png')):
#         src = os.path.join(log_image_dir, f)
#         dst = os.path.join(renamed_image_dir, f)
#         shutil.move(src, dst)

# print("✅ 所有图像已移动到 test_renamed 文件夹。")


# 第一步：删除包含 "reconstruction" 的图像文件
for fname in os.listdir(log_image_dir):
    full_path = os.path.join(log_image_dir, fname)

    # 删除带有 "reconstruction" 的图像文件
    if "reconstruction" in fname and fname.lower().endswith(('.jpg', '.png')):
        os.remove(full_path)
        continue  # 删除后跳过重命名操作

    # 如果文件名以 "ee_" 开头，重命名去掉前缀
    if fname.startswith("ee-"):
        new_fname = fname[3:]  # 去掉前3个字符 "ee-"
        new_path = os.path.join(log_image_dir, new_fname)

        # 避免命名冲突
        if not os.path.exists(new_path):
            os.rename(full_path, new_path)


# 第二步：移动 test 下其余图片到新文件夹
for fname in os.listdir(log_image_dir):
    if fname.lower().endswith(('.jpg', '.png')):
        src = os.path.join(log_image_dir, fname)
        dst = os.path.join(renamed_image_dir, fname)
        shutil.move(src, dst)

# # 第三步：复制 origin_dir 中图片到新文件夹
# for fname in os.listdir(origin_dir):
#     if fname.lower().endswith(('.jpg', '.png')):
#         src = os.path.join(origin_dir, fname)
#         dst = os.path.join(renamed_image_dir, fname)
#         shutil.copy2(src, dst)

print(f"\n✅ 所有图片已移动至 {renamed_image_dir}，并从 origin_dir 复制完成。")