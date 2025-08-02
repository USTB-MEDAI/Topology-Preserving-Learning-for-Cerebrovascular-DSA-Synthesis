# import cv2
# import numpy as np
# import os
# import random

# def load_binary_image(path):
#     img = cv2.imread(path, 0)
#     _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
#     return binary

# def extract_skeleton(binary):
#     return cv2.ximgproc.thinning(binary)

# def suppress_branches(skeleton, min_length=10):
#     """通过分支长度来抑制细小分支"""
#     # 查找所有连通区域
#     num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(skeleton, connectivity=8)
    
#     # 创建一个空白图像，用于保存主血管
#     main_vessel = np.zeros_like(skeleton)
    
#     for i in range(1, num_labels):  # 0为背景
#         if stats[i, cv2.CC_STAT_AREA] >= min_length:  # 只保留面积较大的连通区域
#             main_vessel[labels == i] = 255
#     return main_vessel

# def add_jitter(skeleton, max_offset=1):
#     """模拟手绘的抖动效果"""
#     height, width = skeleton.shape
#     jittered = np.zeros_like(skeleton)

#     coords = np.argwhere(skeleton > 0)
#     for y, x in coords:
#         dy = random.randint(-max_offset, max_offset)
#         dx = random.randint(-max_offset, max_offset)
#         ny, nx = np.clip(y + dy, 0, height - 1), np.clip(x + dx, 0, width - 1)
#         jittered[ny, nx] = 255

#     return jittered

# def apply_handdrawn_effect(skeleton):
#     # 抖动
#     jittered = add_jitter(skeleton, max_offset=1)

#     # 轻微模糊（像手绘铅笔痕迹）
#     blurred = cv2.GaussianBlur(jittered, (3, 3), sigmaX=0.5)

#     # 提升对比度（线条更加突出）
#     result = cv2.normalize(blurred, None, 0, 255, cv2.NORM_MINMAX)
#     return result

# def process_image(mask_path, save_path):
#     binary = load_binary_image(mask_path)
#     skeleton = extract_skeleton(binary)

#     # 只保留主血管部分
#     main_vessel = suppress_branches(skeleton, min_length=20)  # 根据需求调整min_length

#     # 生成手绘效果
#     handdrawn = apply_handdrawn_effect(main_vessel)
#     cv2.imwrite(save_path, handdrawn)
#     print(f"Saved: {save_path}")

# def batch_process(input_dir, output_dir):
#     os.makedirs(output_dir, exist_ok=True)
#     for fname in os.listdir(input_dir):
#         if fname.endswith((".png", ".jpg", ".jpeg")):
#             input_path = os.path.join(input_dir, fname)
#             output_path = os.path.join(output_dir, fname)
#             process_image(input_path, output_path)

# # 示例用法
# if __name__ == "__main__":
#     input_dir = "/disk/ssy/data/DIAS/DIAS/test/labels"      # 替换为你的mask图像文件夹
#     output_dir = "/disk/ssy/data/DIAS/myDIAS/hand/test_makehand" # 输出文件夹
#     batch_process(input_dir, output_dir)



import cv2
import numpy as np
import os

def process_image(image_path, output_path):
    # 读取图像（灰度模式）
    img = cv2.imread(image_path, 0)

    # 对图像进行二值化，保留最白部分
    _, binary = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY)  # 200是阈值，可以根据需要调整

    # 保存处理后的图像到新文件夹
    cv2.imwrite(output_path, binary)
    print(f"Saved: {output_path}")

def batch_process(input_dir, output_dir):
    # 创建输出目录（如果不存在）
    os.makedirs(output_dir, exist_ok=True)

    # 遍历输入文件夹中的所有文件
    for fname in os.listdir(input_dir):
        if fname.endswith((".png", ".jpg", ".jpeg")):
            input_path = os.path.join(input_dir, fname)
            output_path = os.path.join(output_dir, fname)
            process_image(input_path, output_path)

# 示例用法
if __name__ == "__main__":
    input_dir = "/disk/ssy/data/DIAS/myDIAS/hand/handtest"  # 替换为输入图片文件夹的路径
    output_dir = "/disk/ssy/data/DIAS/myDIAS/hand/handtesthand"  # 替换为输出图片文件夹的路径
    batch_process(input_dir, output_dir)
