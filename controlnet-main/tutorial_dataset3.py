import os
import cv2
import numpy as np
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, source_dir, target_dir, prompt="生成脑血管图像", train_ratio=0.9, is_train=True):
        self.source_dir = source_dir
        self.target_dir = target_dir
        self.prompt = prompt

        # 获取所有 source 文件名，并确保 target 也有相应文件名
        source_files = sorted(os.listdir(source_dir))
        target_files = sorted(os.listdir(target_dir))

        assert len(source_files) == len(target_files), "源图像和目标图像数量不一致"
        
        # 计算划分点
        split_idx = int(len(source_files) * train_ratio)
        
        # 根据训练/验证模式选择文件
        if is_train:
            self.source_files = source_files[:split_idx]
            self.target_files = target_files[:split_idx]
            print(f"训练集: 使用 {len(self.source_files)} 张图像 (前{train_ratio*100:.0f}%)")
        else:
            self.source_files = source_files[split_idx:]
            self.target_files = target_files[split_idx:]
            print(f"验证集: 使用 {len(self.source_files)} 张图像 (后{(1-train_ratio)*100:.0f}%)")

    def __len__(self):
        return len(self.source_files)

    def __getitem__(self, idx):
        source_path = os.path.join(self.source_dir, self.source_files[idx])
        target_path = os.path.join(self.target_dir, self.target_files[idx])

        source = cv2.imread(source_path)
        target = cv2.imread(target_path)

        # OpenCV 读图是 BGR，转为 RGB
        source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

        # Resize the images to 512x512
        source = cv2.resize(source, (256, 256))
        target = cv2.resize(target, (256, 256))

        # 归一化 [-1, 1]
        source = (source.astype(np.float32) / 127.5) - 1.0
        target = (target.astype(np.float32) / 127.5) - 1.0

        return dict(jpg=target, txt=self.prompt, hint=source)    