# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/resolve/main/docs/python.md

from cog import BasePredictor, Input, Path
import os
from subprocess import call
from cldm.model import create_model, load_state_dict
from ldm.models.diffusion.ddim import DDIMSampler
from PIL import Image
import numpy as np
from typing import List
from utils import get_state_dict_path, download_model, model_dl_urls, annotator_dl_urls

import torch

# MODEL_TYPE = "openpose"
MODEL_TYPE = "scribble"

if MODEL_TYPE == "canny":
    from gradio.gradio_canny2image import process_canny
elif MODEL_TYPE == "depth":
    from gradio.gradio_depth2image import process_depth
elif MODEL_TYPE == "hed":
    from gradio.gradio_hed2image import process_hed
elif MODEL_TYPE == "normal":
    from gradio.gradio_normal2image import process_normal
elif MODEL_TYPE == "mlsd":
    from gradio.gradio_hough2image import process_mlsd
elif MODEL_TYPE == "scribble":
    from gradio_scribble2image import process_scribble
elif MODEL_TYPE == "seg":
    from gradio.gradio_seg2image import process_seg
elif MODEL_TYPE == "openpose":
    from gradio.gradio_pose2image import process_pose

class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        self.model = create_model('./models/cldm_v15.yaml').cuda()  
        # self.model = create_model('./models/cldm_v15.yaml').to("cpu")  #先不占用cuda


        # self.model.load_state_dict(load_state_dict(get_state_dict_path(MODEL_TYPE), location='cuda'))
        self.model.load_state_dict(load_state_dict("/disk/ssy/pyproj/bishe/controlnet-main/models/control_sd15_scribble.pth", location='cuda'))
        self.ddim_sampler = DDIMSampler(self.model)
    # def setup(self):
    #     """Load the model into memory to make running multiple predictions efficient"""
    #     self.model = create_model('./models/cldm_v15.yaml').cuda()
    #     # self.model.load_state_dict(load_state_dict(get_state_dict_path(MODEL_TYPE), location='cuda'))  #mdf

    #     # 读取 checkpoint 文件
    #     ckpt = torch.load('/disk/ssy/pyproj/bishe/controlnet-main/lightning_logs/version_17/checkpoints/epoch=104-step=5879.ckpt', map_location='cuda')
    #     # 提取 state_dict，去掉 model 前缀（如果需要）
    #     state_dict = {
    #         k.replace("model.", ""): v  # 如果有 'model.' 前缀
    #         for k, v in ckpt["state_dict"].items()
    #     }
    #     # 加载模型
    #     self.model.load_state_dict(state_dict)

        # self.ddim_sampler = DDIMSampler(self.model)

    def predict(
        self,
        image: Path = Input(description="/disk/ssy/pyproj/bishe/controlnet-main/Input_jpg"), #  mdf Input image
        prompt: str = Input(description="Prompt for the model"),
        num_samples: str = Input(
            description="Number of samples (higher values may OOM)",
            choices=['1', '4'],
            default='1'
        ),
        image_resolution: str = Input(
            description="Image resolution to be generated",
            choices = ['256', '512', '768'],
            default='512'
        ),
        low_threshold: int = Input(description="Canny line detection low threshold", default=100, ge=1, le=255), # only applicable when model type is 'canny'
        high_threshold: int = Input(description="Canny line detection high threshold", default=200, ge=1, le=255), # only applicable when model type is 'canny'
        ddim_steps: int = Input(description="Steps", default=20),
        scale: float = Input(description="Scale for classifier-free guidance", default=9.0, ge=0.1, le=30.0),
        seed: int = Input(description="Seed", default=None),
        eta: float = Input(description="Controls the amount of noise that is added to the input data during the denoising diffusion process. Higher value -> more noise", default=0.0),
        a_prompt: str = Input(description="Additional text to be appended to prompt", default="best quality, extremely detailed"),
        n_prompt: str = Input(description="Negative Prompt", default="longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality"),
        detect_resolution: int = Input(description="Resolution at which detection method will be applied)", default=512, ge=128, le=1024), # only applicable when model type is 'HED', 'seg', or 'MLSD'
        # bg_threshold: float = Input(description="Background Threshold (only applicable when model type is 'normal')", default=0.0, ge=0.0, le=1.0), # only applicable when model type is 'normal'
        # value_threshold: float = Input(description="Value Threshold (only applicable when model type is 'MLSD')", default=0.1, ge=0.01, le=2.0), # only applicable when model type is 'MLSD'
        # distance_threshold: float = Input(description="Distance Threshold (only applicable when model type is 'MLSD')", default=0.1, ge=0.01, le=20.0), # only applicable when model type is 'MLSD'
    ) -> List[Path]:
        """Run a single prediction on the model"""
        num_samples = int(num_samples)
        image_resolution = int(image_resolution)
        if not seed:
            seed = np.random.randint(1000000)
        else:
            seed = int(seed)

        # load input_image
        input_image = Image.open(image)
        # convert to numpy
        input_image = np.array(input_image)

        if MODEL_TYPE == "canny":
            outputs = process_canny(
                input_image,
                prompt,
                a_prompt,
                n_prompt,
                num_samples,
                image_resolution,
                ddim_steps,
                scale,
                seed,
                eta,
                low_threshold,
                high_threshold,
                self.model,
                self.ddim_sampler,
            )
        elif MODEL_TYPE == "depth":
            outputs = process_depth(
                input_image,
                prompt,
                a_prompt,
                n_prompt,
                num_samples,
                image_resolution,
                detect_resolution,
                ddim_steps,
                scale,
                seed,
                eta,
                self.model,
                self.ddim_sampler,
            )
        elif MODEL_TYPE == "hed":
            outputs = process_hed(
                input_image,
                prompt,
                a_prompt,
                n_prompt,
                num_samples,
                image_resolution,
                detect_resolution,
                ddim_steps,
                scale,
                seed,
                eta,
                self.model,
                self.ddim_sampler,
            )
        # elif MODEL_TYPE == "normal":
        #     outputs = process_normal(
        #         input_image,
        #         prompt,
        #         a_prompt,
        #         n_prompt,
        #         num_samples,
        #         image_resolution,
        #         ddim_steps,
        #         scale,
        #         seed,
        #         eta,
        #         bg_threshold,
        #         self.model,
        #         self.ddim_sampler,
        #     )
        # elif MODEL_TYPE == "mlsd":
        #     outputs = process_mlsd(
        #         input_image,
        #         prompt,
        #         a_prompt,
        #         n_prompt,
        #         num_samples,
        #         image_resolution,
        #         detect_resolution,
        #         ddim_steps,
        #         scale,
        #         seed,
        #         eta,
        #         value_threshold,
        #         distance_threshold,
        #         self.model,
        #         self.ddim_sampler,
        #     )
        elif MODEL_TYPE == "scribble":
            outputs = process_scribble(
                input_image,
                prompt,
                a_prompt,
                n_prompt,
                num_samples,
                image_resolution,
                ddim_steps,
                scale,
                seed,
                eta,
                self.model,
                self.ddim_sampler,
            )
        elif MODEL_TYPE == "seg":
            outputs = process_seg(
                input_image,
                prompt,
                a_prompt,
                n_prompt,
                num_samples,
                image_resolution,
                detect_resolution,
                ddim_steps,
                scale,
                seed,
                eta,
                self.model,
                self.ddim_sampler,
            )
        elif MODEL_TYPE == "openpose":
            outputs = process_pose(
                input_image,
                prompt,
                a_prompt,
                n_prompt,
                num_samples,
                image_resolution,
                detect_resolution,
                ddim_steps,
                scale,
                seed,
                eta,
                self.model,
                self.ddim_sampler,
            )
        
        # outputs from list to PIL
        outputs = [Image.fromarray(output) for output in outputs]
        # save outputs to file
        outputs = [output.save(f"tmp/output_{i}.png") for i, output in enumerate(outputs)]
        # return paths to output files
        return [Path(f"tmp/output_{i}.png") for i in range(len(outputs))]
    

# if __name__ == "__main__":
#     predictor = Predictor()
#     predictor.setup()

#     from pathlib import Path as Path_
#     outputs = predictor.predict(
#         image=Path_("/disk/ssy/pyproj/bishe/controlnet-main/Input_jpg/菜.jpg"),
#         prompt="菜",
#         num_samples='1',
#         image_resolution='512',
#         low_threshold=100,
#         high_threshold=200,
#         ddim_steps=20,
#         scale=9.0,
#         seed=42,
#         eta=0.0,
#         a_prompt="best quality, extremely detailed",
#         n_prompt="bad quality",
#         # a_prompt="",
#         # n_prompt="",
#         detect_resolution=512,
#     )

#     print("Generated outputs:")
#     for out in outputs:
#         print(out)

if __name__ == "__main__":
    predictor = Predictor()
    predictor.setup()

    from pathlib import Path as Path_
    outputs = predictor.predict(
        image=Path_("/disk/ssy/data/DIAS/myDIAS/hand/first20/原mask前20/pro_CXL_L_39.jpg"),  # 你可以换成你想用的参考图，或者同一张图
        prompt="river",  # 重点改这里，写详细点更好
        num_samples='1',
        image_resolution='512',
        low_threshold=100,
        high_threshold=200,
        ddim_steps=20,          # 多点步数让画质更细腻，20->30
        scale=9.0,              # 保持不变，scale越大越重视prompt
        seed=42,                # 固定种子方便复现，也可换成None随机
        eta=0.0,
        a_prompt="best quality, extremely detailed, masterpiece, photo-realistic",  # 正面加强词
        n_prompt="bad quality, lowres, blurry, cartoon, deformed",                  # 负面词，避免劣质图
        detect_resolution=512,
    )

    print("Generated outputs:")
    for out in outputs:
        print(out)

