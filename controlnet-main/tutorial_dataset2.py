import os
import cv2
import numpy as np
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, source_dir, target_dir, prompt="生成脑血管图像"):
        self.source_dir = source_dir
        self.target_dir = target_dir
        # self.source_dir = "/disk/ssy/data/DIAS/myDIAS/hand/handtesthand" #source_dir
        # self.target_dir = "/disk/ssy/data/DIAS/myDIAS/hand/handtesthand" #target_dir
        self.prompt = prompt

        # 获取所有 source 文件名，并确保 target 也有相应文件名
        self.source_files = sorted(os.listdir(source_dir))
        self.target_files = sorted(os.listdir(target_dir))

        assert len(self.source_files) == len(self.target_files), "源图像和目标图像数量不一致"

    def __len__(self):
        return len(self.source_files)

    def __getitem__(self, idx):
        source_path = os.path.join(self.source_dir, self.source_files[idx])
        target_path = os.path.join(self.target_dir, self.target_files[idx])

        # # test 单通道图 读取为单通道灰度图  失败！哈哈，必须匹配预训练模型3通道
        # source = cv2.imread(source_path, cv2.IMREAD_GRAYSCALE)  # shape: (H, W)
        # target = cv2.imread(target_path, cv2.IMREAD_GRAYSCALE)  # shape: (H, W)
        # # 如果后续模型期望的是 3D 张量 (C, H, W)，需要手动加 channel 维
        # source = np.expand_dims(source, axis=0)  # shape: (1, H, W)
        # target = np.expand_dims(target, axis=0)  # shape: (1, H, W)

        source = cv2.imread(source_path)
        target = cv2.imread(target_path)

        # OpenCV 读图是 BGR，转为 RGB
        source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

         # mdf  Resize the images to 512x512
        source = cv2.resize(source, (256, 256))
        target = cv2.resize(target, (256, 256))

        # print(f"Loaded source shape: {source.shape}, target shape: {target.shape}")


        # 归一化 [-1, 1]
        source = (source.astype(np.float32) / 127.5) - 1.0
        target = (target.astype(np.float32) / 127.5) - 1.0

        return dict(jpg=target, txt=self.prompt, hint=source)
