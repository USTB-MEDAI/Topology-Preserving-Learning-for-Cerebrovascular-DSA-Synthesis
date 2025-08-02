# import os
# import cv2
# import numpy as np
# from skimage.morphology import skeletonize
# from tqdm import tqdm

# # 图像骨架提取函数，统一大小为 512x512
# def extract_skeleton(image_path, target_size=(512, 512)):
#     # 读取灰度图
#     image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
#     if image is None:
#         print(f"无法读取图像: {image_path}")
#         return None

#     # 统一大小
#     image = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)

#     # 二值化
#     _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
#     binary_bool = binary == 255

#     # 骨架提取
#     skeleton = skeletonize(binary_bool)
#     return (skeleton * 255).astype(np.uint8)

# # 批处理函数
# def process_folder(input_dir, output_dir, target_size=(512, 512)):
#     os.makedirs(output_dir, exist_ok=True)
#     supported_ext = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']

#     for filename in tqdm(os.listdir(input_dir), desc="批量骨架提取中"):
#         if any(filename.lower().endswith(ext) for ext in supported_ext):
#             input_path = os.path.join(input_dir, filename)
#             output_path = os.path.join(output_dir, filename)

#             skeleton_img = extract_skeleton(input_path, target_size)
#             if skeleton_img is not None:
#                 cv2.imwrite(output_path, skeleton_img)

# # ---------- 使用示例 ----------
# # 输入文件夹：原图
# input_folder = "/disk/ssy/Xray-Diffsuion-main/datas/mask412_package"

# # 输出文件夹：骨架图
# output_folder = "/disk/ssy/Xray-Diffsuion-main/datas/skeleton412_512size_cond提取版"

# # 执行批处理（统一为 512×512）
# process_folder(input_folder, output_folder, target_size=(512, 512))


import os
import cv2
import numpy as np
from skimage.morphology import skeletonize
from tqdm import tqdm

# 图像骨架提取函数，统一大小为 512x512
def extract_skeleton(image_path, target_size=(512, 512)):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"无法读取图像: {image_path}")
        return None

    image = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
    _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    binary_bool = binary == 255
    skeleton = skeletonize(binary_bool)
    return (skeleton * 255).astype(np.uint8)

# 不保留子文件夹结构，全部输出到一个目录
def process_flat_output(input_dir, output_dir, target_size=(512, 512)):
    os.makedirs(output_dir, exist_ok=True)
    supported_ext = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']

    for root, _, files in os.walk(input_dir):
        for filename in files:
            if any(filename.lower().endswith(ext) for ext in supported_ext):
                input_path = os.path.join(root, filename)

                # 输出路径直接放在输出根目录中
                output_path = os.path.join(output_dir, filename)

                skeleton_img = extract_skeleton(input_path, target_size)
                if skeleton_img is not None:
                    cv2.imwrite(output_path, skeleton_img)

# ---------- 使用示例 ----------
input_folder = "/disk/ssy/Xray-Diffsuion-main/datas/mask412_package"
output_folder = "/disk/ssy/Xray-Diffsuion-main/datas/skeleton412_512size_flat版"

process_flat_output(input_folder, output_folder, target_size=(512, 512))

