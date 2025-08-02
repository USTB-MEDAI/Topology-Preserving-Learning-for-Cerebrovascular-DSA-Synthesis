import torch

state_dict = torch.load("/disk/ssy/pyproj/bishe/controlnet-main/lightning_logs/version_19/checkpoints/epoch=999-step=55999.ckpt", map_location='cpu')

if "state_dict" in state_dict:
    state_dict = state_dict["state_dict"]

print("包含 control 模块的参数如下：")
print("\n".join([k for k in state_dict.keys() if k.startswith("control")]))

for name, param in state_dict.named_parameters():
    if "control_model" in name:
        print(f"{name} | shape: {param.shape} | requires_grad: {param.requires_grad}")
