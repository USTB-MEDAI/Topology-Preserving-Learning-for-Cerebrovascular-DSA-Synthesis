import os
import cv2
import numpy as np
import torch
from torchvision.utils import save_image
from tqdm import tqdm

from cldm.model import create_model, load_state_dict

# === 模型加载 ===
config_path = './models/cldm_v15.yaml'
ckpt_path = '/disk/ssy/pyproj/bishe/controlnet-main/lightning_logs/version_19/checkpoints/epoch=999-step=55999.ckpt'
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

model = create_model(config_path).cpu()
model.load_state_dict(load_state_dict(ckpt_path, location='cpu'))
model = model.to(device)
model.eval()

# === 测试数据路径 ===
source_dir = "/disk/ssy/data/DIAS/myDIAS/hand/test_predict505"
output_dir = "/disk/ssy/data/DIAS/myDIAS/hand/test_outputs513"
os.makedirs(output_dir, exist_ok=True)

# === 固定 prompt ===
prompt = "生成脑血管图像"

# === 遍历每个 source 图像 ===
file_list = sorted(os.listdir(source_dir))

for filename in tqdm(file_list):
    if not filename.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp')):
        continue

    # === 读取并预处理 source 图像 ===
    source_path = os.path.join(source_dir, filename)
    source = cv2.imread(source_path)
    source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
    source = cv2.resize(source, (512, 512))
    source = (source.astype(np.float32) / 127.5) - 1.0
    source = torch.from_numpy(source).permute(2, 0, 1).unsqueeze(0).to(device)  # (1, 3, H, W)
    print(source.shape)


    # === 构造模型输入 ===
    batch = {
        "jpg": source,      # target 不参与推理，这里随便填即可
        "txt": prompt,
        "hint": source      # 控制输入
    }

    # === 获取 latent 和条件 ===
    z, c = model.get_input(batch, 'jpg', bs=1)
    c_cat, c_cross = c["c_concat"][0], c["c_crossattn"][0]

    # === 采样（生成 latent）===
    samples, _ = model.sample_log(
        cond={"c_concat": [c_cat], "c_crossattn": [c_cross]},
        batch_size=1,
        ddim=True,
        ddim_steps=50,
        eta=0.0
    )

    # === 解码生成图像 ===
    x_samples = model.decode_first_stage(samples)
    x_samples = (x_samples.clamp(-1.0, 1.0) + 1.0) / 2.0  # [0, 1]
    
    # === 保存图像 ===
    output_path = os.path.join(output_dir, filename)
    save_image(x_samples, output_path)
