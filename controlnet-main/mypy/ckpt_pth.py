import torch

ckpt = torch.load("/disk/ssy/pyproj/bishe/controlnet-main/lightning_logs/version_19/checkpoints/epoch=999-step=55999.ckpt", map_location='cpu')
state_dict = ckpt['state_dict']  # 只提取权重

# 可选：去掉前缀 'model.' 等
new_state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}

torch.save(new_state_dict, "/disk/ssy/pyproj/bishe/controlnet-main/models/model511_version19.pth")
