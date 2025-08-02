
import cv2
import einops
import gradio as gr
import numpy as np
import torch

from cldm.hack import disable_verbosity
disable_verbosity()

from pytorch_lightning import seed_everything
from annotator.util import resize_image, HWC3
from cldm.model import create_model, load_state_dict
from ldm.models.diffusion.ddim import DDIMSampler


import os  #mdf my
import cv2
import torch
from tqdm import tqdm
from cldm.model import create_model, load_state_dict
from ldm.models.diffusion.ddim import DDIMSampler


def process_scribble(input_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, ddim_steps, scale, seed, eta, model, ddim_sampler):
    with torch.no_grad():
        img = resize_image(HWC3(input_image), image_resolution)  # input_image  # mdf 
        H, W, C = img.shape

        detected_map = np.zeros_like(img, dtype=np.uint8)
        detected_map[np.min(img, axis=2) < 127] = 255


        control = control.to(model.device)
        control = einops.rearrange(control, 'b h w c -> b c h w')
        control = control.to(memory_format=torch.contiguous_format).float()
        

        seed_everything(seed)

        cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning([prompt] * num_samples)]}
        un_cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning([n_prompt] * num_samples)]}
        shape = (4, H // 8, W // 8)

        samples, intermediates = ddim_sampler.sample(ddim_steps, num_samples,
                                                     shape, cond, verbose=False, eta=eta,
                                                     unconditional_guidance_scale=scale,
                                                     unconditional_conditioning=un_cond)
        x_samples = model.decode_first_stage(samples)
        x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)

        results = [x_samples[i] for i in range(num_samples)]
    return [255 - detected_map] + results



def smart_load_state_dict(path, model):
    ckpt = torch.load(path, map_location='cpu')
    if 'state_dict' in ckpt:
        state_dict = ckpt['state_dict']
        state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
    else:
        state_dict = ckpt
    model.load_state_dict(state_dict)
    return model

# 引入上面定义的 process_scribble 函数（假设你已经定义在同一个文件里）

def main(
    input_dir='/disk/ssy/data/DIAS/myDIAS/hand/test_predict505',
    output_dir='/disk/ssy/data/DIAS/myDIAS/hand/test_predict505_jieguo',
    model_config='./models/cldm_v15.yaml',
    model_ckpt='/disk/ssy/pyproj/bishe/controlnet-main/lightning_logs/version_19/checkpoints/epoch=999-step=55999.ckpt',
    #model_ckpt='/disk/ssy/pyproj/bishe/controlnet-main/models/model511_version19.pth',
    prompt="生成脑血管图像",
    a_prompt="",
    n_prompt="",
    num_samples=1,
    image_resolution=512,
    ddim_steps=100,
    scale=9.0,
    seed=42,
    eta=0.0
):
    os.makedirs(output_dir, exist_ok=True)

    # 1. 加载模型
    model = create_model(model_config).cpu()
    
    #model = smart_load_state_dict(model_ckpt, model)

    model.load_state_dict(load_state_dict(model_ckpt, location='cuda'))   #mdf smart
    model = model.cuda()
    ddim_sampler = DDIMSampler(model)

    # 2. 遍历输入图像
    for filename in tqdm(os.listdir(input_dir)):
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, os.path.splitext(filename)[0] + '_ldm.png')

        input_image = cv2.imread(input_path)

        # input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB) #mdf chuliimage
        # input_image = cv2.resize(input_image, (512, 512))
        # input_image = (input_image.astype(np.float32) / 127.5) - 1.0
       
        result_images = process_scribble(
            input_image=input_image,
            prompt=prompt,
            a_prompt=a_prompt,
            n_prompt=n_prompt,
            num_samples=num_samples,
            image_resolution=image_resolution,
            ddim_steps=ddim_steps,
            scale=scale,
            seed=seed,
            eta=eta,
            model=model,
            ddim_sampler=ddim_sampler
        )

        # result_images[1:] 是生成的图，result_images[0] 是草图
        for idx, img in enumerate(result_images[1:]):
            if num_samples == 1:
                cv2.imwrite(output_path, img)
            else:
                cv2.imwrite(output_path.replace('.png', f'_{idx}.png'), img)


if __name__ == '__main__':
    main()

















# import os
# import torch
# from torch.utils.data import DataLoader
# from torchvision.utils import save_image

# from cldm.model import create_model, load_state_dict
# from tutorial_dataset2 import MyDataset  # 保证你有这个文件

# # === 配置部分 ===
# config_path = './models/cldm_v15.yaml'  # 替换成你的配置文件路径
# model_ckpt = '/disk/ssy/pyproj/bishe/controlnet-main/models/epoch=419-step=23519.ckpt'  # 模型权重路径
# source_dir = '/disk/ssy/data/DIAS/myDIAS/hand/test_predict505'  # 测试集 hint 图像路径
# target_dir = '/disk/ssy/data/DIAS/myDIAS/hand/test_predict505'      # 测试集 gt 图像路径（也可以设成 source_dir 占位）
# save_dir = './output'  # 生成图像的保存路径
# batch_size = 1  # 每次处理一张图，确保一一对应
# device = 'cuda' if torch.cuda.is_available() else 'cpu'

# # === 模型加载 ===
# model = create_model(config_path).cpu()
# model.load_state_dict(load_state_dict(model_ckpt, location='cpu'))
# model = model.to(device)
# model.eval()

# # === 数据加载 ===
# dataset = MyDataset(source_dir=source_dir, target_dir=target_dir)
# dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

# # === 推理 & 保存 ===
# os.makedirs(save_dir, exist_ok=True)

# # with torch.no_grad():
# #     for i, batch in enumerate(dataloader):
# #         hint = batch["hint"].to(device)  # (B, 3, H, W)
# #         cond = {"c_concat": [hint], "c_crossattn": [model.get_learned_conditioning([batch["txt"][0]])]}

# #         # 生成图像
# #         samples = model.sample(cond=cond, batch_size=batch_size, ddim_steps=50, eta=0.0, verbose=False)
# #         samples = (samples + 1.0) / 2.0  # [-1, 1] → [0, 1]
# #         samples.clamp_(0.0, 1.0)

# #         # 保存图像
# #         # save_path = os.path.join(save_dir, f"gen_{i:03d}.png")
# #         # save_image(samples, save_path)
# #         # print(f"Saved: {save_path}")
# import datetime

# # === 生成时间戳文件夹 ===
# timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
# save_dir = os.path.join("./testoutput", f"test_{timestamp}")
# os.makedirs(save_dir, exist_ok=True)

# # === 推理 & 保存 ===
# with torch.no_grad():
#     for i, batch in enumerate(dataloader):
#         hint = batch["hint"].to(device)
#         cond = {"c_concat": [hint], "c_crossattn": [model.get_learned_conditioning([batch["txt"][0]])]}

#         samples = model.sample(cond=cond, batch_size=batch_size, ddim_steps=50, eta=0.0, verbose=False)
#         samples = (samples + 1.0) / 2.0
#         samples.clamp_(0.0, 1.0)

#         save_path = os.path.join(save_dir, f"gen_{i:03d}.png")
#         save_image(samples, save_path)
#         print(f"Saved: {save_path}")






# # from torch.utils.data import DataLoader
# # from my_dataset import MyDataset  # 你上面那段代码的文件路径
# # from ldm.models.diffusion.ddim import DDIMSampler
# # from ldm.util import instantiate_from_config
# # from omegaconf import OmegaConf
# # import os
# # import torch
# # from torchvision.utils import save_image

# # @torch.no_grad()
# # def predict(model, sampler, dataloader, out_dir, ddim_steps=50, scale=9.0):
# #     os.makedirs(out_dir, exist_ok=True)
# #     for idx, batch in enumerate(dataloader):
# #         cond_image = batch["hint"].cuda()           # shape: (1, 3, 512, 512)
# #         prompt = batch["txt"]

# #         cond = {
# #             "c_concat": [cond_image], 
# #             "c_crossattn": [model.get_learned_conditioning(prompt)]
# #         }
# #         uc = {
# #             "c_concat": [cond_image], 
# #             "c_crossattn": [model.get_unconditional_conditioning(1)]
# #         }

# #         shape = (3, 64, 64)  # = 512 // 8

# #         samples, _ = sampler.sample(
# #             S=ddim_steps,
# #             batch_size=1,
# #             shape=shape,
# #             cond=cond,
# #             verbose=False,
# #             unconditional_guidance_scale=scale,
# #             unconditional_conditioning=uc
# #         )
# #         x_samples = model.decode_first_stage(samples)
# #         x_samples = torch.clamp((x_samples + 1.0) / 2.0, 0.0, 1.0)

# #         save_image(x_samples, os.path.join(out_dir, f"{idx:03d}_gen.png"))
# #         save_image((cond_image + 1.0) / 2.0, os.path.join(out_dir, f"{idx:03d}_cond.png"))
# #         print(f"[{idx+1}] Done")

# # def load_model(config_path, ckpt_path):
# #     config = OmegaConf.load(config_path)
# #     model = instantiate_from_config(config.model)
# #     model.load_state_dict(torch.load(ckpt_path)["state_dict"], strict=False)
# #     model.cuda().eval()
# #     return model

# # if __name__ == "__main__":
# #     config_path = "configs/your_config.yaml"
# #     ckpt_path = "logs/your_model.ckpt"
# #     source_dir = "test_source/"
# #     target_dir = "test_target/"
# #     out_dir = "outputs/"

# #     model = load_model(config_path, ckpt_path)
# #     sampler = DDIMSampler(model)

# #     dataset = MyDataset(source_dir, target_dir)
# #     dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

# #     predict(model, sampler, dataloader, out_dir)


# # # import os
# # # from pathlib import Path
# # # from predict import Predictor
# # # from cog import BasePredictor
# # # from PIL import Image

# # # # 设置输入输出路径
# # # input_path = "/disk/ssy/data/DIAS/myDIAS/hand/test_predict505"   # 替换为你存放scribble图像的文件夹路径
# # # output_path = "/disk/ssy/data/DIAS/myDIAS/hand/test_predict505_jieguo" # 替换为你想保存生成图像的路径
# # # os.makedirs(output_path, exist_ok=True)

# # # # 初始化 Predictor
# # # predictor = Predictor()
# # # predictor.setup()

# # # # 模型生成的固定参数（可以修改）
# # # prompt = "生成脑血管图像"  #a fantasy landscape
# # # num_samples = '1'
# # # image_resolution = '512'
# # # ddim_steps = 20
# # # scale = 9.0
# # # eta = 0.0
# # # a_prompt = "best quality, extremely detailed"
# # # n_prompt = "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality"
# # # detect_resolution = 512
# # # low_threshold = 100
# # # high_threshold = 200

# # # # 遍历输入文件夹中的所有图片文件
# # # for file_name in os.listdir(input_path):
# # #     if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
# # #         input_file = Path(os.path.join(input_path, file_name))

# # #         # 进行预测
# # #         output_files = predictor.predict(
# # #             image=input_file,
# # #             prompt=prompt,
# # #             num_samples=num_samples,
# # #             image_resolution=image_resolution,
# # #             ddim_steps=ddim_steps,
# # #             scale=scale,
# # #             seed=None,
# # #             eta=eta,
# # #             a_prompt=a_prompt,
# # #             n_prompt=n_prompt,
# # #             detect_resolution=detect_resolution,
# # #             low_threshold=low_threshold,
# # #             high_threshold=high_threshold,
# # #         )

# # #         # 移动输出图像到目标目录
# # #         for i, out_path in enumerate(output_files):
# # #             new_name = f"{Path(file_name).stem}_gen_{i}.png"
# # #             os.rename(out_path, os.path.join(output_path, new_name))
# # #             print(f"Saved: {new_name}")

# # # # import torch

# # # # ckpt_path = "/disk/ssy/pyproj/bishe/controlnet-main/lightning_logs/version_17/checkpoints/epoch=94-step=5319.ckpt"
# # # # ckpt = torch.load(ckpt_path, map_location='cuda')

# # # # print(ckpt.keys())  # 打印顶层字典的 key，一般只有 4-5 个，不会太多
# # # # print(list(ckpt['state_dict'].keys())[:10])  # 打印 state_dict 的前10个键，观察是否有前缀
