import os
from PIL import Image

def convert_rgb_to_gray(root_dir, save_dir=None):
    """
    将 root_dir 下所有子文件夹中的 RGB 图像转换为灰度图像（单通道）。

    Args:
        root_dir (str): 原始图像的根目录。
        save_dir (str): 转换后的图像保存目录（若为 None 则覆盖原图）。
    """
    for subdir, _, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                full_path = os.path.join(subdir, file)
                try:
                    img = Image.open(full_path).convert("L")  # 转为灰度图
                    if save_dir:
                        # 计算保存路径
                        rel_path = os.path.relpath(subdir, root_dir)
                        save_subdir = os.path.join(save_dir, rel_path)
                        os.makedirs(save_subdir, exist_ok=True)
                        save_path = os.path.join(save_subdir, file)
                    else:
                        save_path = full_path  # 覆盖原图

                    img.save(save_path)
                    print(f"Converted: {full_path} -> {save_path}")
                except Exception as e:
                    print(f"Failed to process {full_path}: {e}")

# 用法示例
if __name__ == "__main__":
    input_folder = "/disk/ssy/pyproj/bishe/controlnet-main/image_log/pre/prejpg6"      # 修改为你自己的输入路径
    output_folder = None  # 或指定 "/path/to/output_folder"
    convert_rgb_to_gray(input_folder, output_folder)
