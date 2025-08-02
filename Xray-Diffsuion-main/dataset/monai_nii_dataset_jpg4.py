import hydra
import numpy as np

from pathlib import Path 
from torch.utils.data import DataLoader, Dataset

from monai.utils import set_determinism
from monai.transforms import (
    AsDiscrete,
    LoadImage,
    EnsureChannelFirstd,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandCropByPosNegLabeld,
    ScaleIntensityRanged,
    ScaleIntensityd,
    NormalizeIntensityd,
    Spacingd,
    EnsureType,
    Resized,
    SaveImage,
)
import monai.transforms as mt

from monai.networks.layers import Norm
from monai.data import CacheDataset, list_data_collate, decollate_batch, Dataset

#替换DataLoader为MultiEpochsDataLoader，使得训练数据加快
class MultiEpochsDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._DataLoader__initialized = False
        self.batch_sampler = _RepeatSampler(self.batch_sampler)
        self._DataLoader__initialized = True
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
          yield next(self.iterator)


class _RepeatSampler(object):
    """ Sampler that repeats forever.

    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
          yield from iter(self.sampler)

def prepare_dataset(data_path, resize_size, img_resize_size=None, batch_size=1, cond_path=None, split="train",cond_nums=[1],skeleton_path=None):  #cond_nums=[1,2]
    """
    Prepare dataset for training
    data_path: str, path to nii data(3D)
    cond_path: str, path to x-ray 2d png images, if None means only conduct autoencoder process
    resize_size: tuple, (x, y, z)
    split: str, "train" or "val"
    """
    # if "data_path" in config.keys():
    data_list = sorted(Path(data_path).glob("*.jpg"))

    # 初始化条件图和骨架图列表
    cond_dirs = []
    skeleton_list = []


    if cond_path:
        cond_dirs = sorted(Path(cond_path).glob("*"))
    # 新增：加载骨架图路径列表（与原图一一对应）  #mdf ske 7.23
    if skeleton_path:
        skeleton_list = sorted(Path(skeleton_path).glob("*.jpg"))  # 骨架图需与原图同名同格式
     
    # 加载骨架图路径（增加健壮性检查）
    if skeleton_path:
        skeleton_list = sorted(Path(skeleton_path).glob("*.jpg"))
        # 验证骨架图数量与原图一致
        if len(skeleton_list) == 0:
            print(f"警告：骨架图路径 {skeleton_path} 下没有找到 .jpg 文件！")
            skeleton_path = None  # 视为无骨架图
        elif len(skeleton_list) != len(data_list):
            raise ValueError(f"骨架图数量({len(skeleton_list)})与原图数量({len(data_list)})不匹配！")
    
    # 打印数据集信息（修正后的版本）
    print(f"原图数量: {len(data_list)}, 条件图目录数量: {len(cond_dirs)}, 骨架图数量: {len(skeleton_list)}")
    
    cond_path=cond_dirs
    skeleton_path=skeleton_list

    # * create data_dicts, a list of dictionary with keys "image" and "cond",  cond means x-ray 2d png image
    data_dicts = []
    if cond_path and skeleton_path:  # 同时有条件图和骨架图
        for image, cond, skeleton in zip(data_list, cond_path, skeleton_path):
            tmp = {"image": image}  # 原图
            # 处理条件图（原有逻辑）
            cond_png = list(sorted(Path(cond).glob("*.png"))) + list(sorted(Path(cond).glob("*.jpg")))
            if 1 in cond_nums:
                tmp["cond1"] = cond_png[0]
            if 2 in cond_nums:
                tmp["cond2"] = cond_png[1]
            if 3 in cond_nums:
                tmp["cond3"] = cond_png[2]
            # 新增：添加骨架图路径
            tmp["skeleton"] = skeleton  # 骨架图与原图一一对应
            data_dicts.append(tmp)
    elif cond_path:
        for image, cond in zip(data_list, cond_path):
            tmp = {"image": image}
            #cond_png = list(sorted(Path(cond).glob("*.png")))  #mdf jpg
            cond_png = list(sorted(Path(cond).glob("*.png"))) + list(sorted(Path(cond).glob("*.jpg")))

            # print(f"cond_png: {cond_png}")  # 打印变量内容     mdf
            tmp["cond1"] = cond_png[0]

            if 1 in cond_nums:
                tmp["cond1"] = cond_png[0]
            if 2 in cond_nums:
                tmp["cond2"] = cond_png[1]
            if 3 in cond_nums:
                tmp["cond3"] = cond_png[2]
            
            # tmp["skeleton"] = None
            data_dicts.append(tmp)
    else:
        for image in data_list:
            tmp = {"image": image}
            data_dicts.append(tmp)
    cond_keys = []
    load_keys = ["image"]
    if 1 in cond_nums:
        cond_keys.append("cond1")
        load_keys.append("cond1")
    if 2 in cond_nums:
        cond_keys.append("cond2")
        load_keys.append("cond2")
    if 3 in cond_nums:
        cond_keys.append("cond3")
        load_keys.append("cond3")
    # print(cond_keys)
    # load_keys.append("image")
    # print(load_keys)

    if skeleton_path:  # 新增：加载骨架图
        load_keys.append("skeleton")
    

    if split == "train":
        data_dicts = data_dicts[: int(len(data_dicts) * 0.8)]  #mdf
    else:
        # data_dicts = data_dicts[int(len(data_dicts) * 0.9) :]
        data_dicts = data_dicts[int(len(data_dicts) * 0.8) : ]

    # if split == "train":
    #     data_dicts = data_dicts[: int(len(data_dicts) * 0.7)]
    # elif split == "test":
    #     data_dicts = data_dicts[int(len(data_dicts) * 0.8) : ]
    # elif split == "val":
    #     data_dicts = data_dicts[int(len(data_dicts) * 0.7) : int(len(data_dicts) * 0.8)]

    set_determinism(seed=0)

    if cond_path and skeleton_path:
        train_transforms = Compose(
            [
                LoadImaged(keys=load_keys, ensure_channel_first=True),
                # Orientationd(keys=["image"], axcodes="RAS"),
                # Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear")),
                Resized(keys=["image", "skeleton"], spatial_size=resize_size),
                Resized(keys=cond_keys, spatial_size=img_resize_size),
                # NormalizeIntensityd(keys=used_keys),
                # NormalizeIntensityd(keys=["image"]),
                ############## else ###############
                # ScaleIntensityd(keys=["imgae"]),
                ############## Fei  ###############
                ScaleIntensityRanged(keys=["image"], a_min=-1000, a_max=400, b_min=-1, b_max=1, clip=True),
                ScaleIntensityd(keys=cond_keys, minv=-1, maxv=1),
                # 新增：骨架图强度缩放（假设骨架图是二值图，缩放到[-1,1]）
                ScaleIntensityd(keys=["skeleton"], minv=-1, maxv=1),
            ]
        )
    elif cond_path:
        train_transforms = Compose(
            [
                LoadImaged(keys=load_keys, ensure_channel_first=True),
                # Orientationd(keys=["image"], axcodes="RAS"),
                # Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear")),
                Resized(keys=["image"], spatial_size=resize_size),
                Resized(keys=cond_keys, spatial_size=img_resize_size),
                # NormalizeIntensityd(keys=used_keys),
                # NormalizeIntensityd(keys=["image"]),
                ############## else ###############
                # ScaleIntensityd(keys=["imgae"]),
                ############## Fei  ###############
                ScaleIntensityRanged(keys=["image"], a_min=-1000, a_max=400, b_min=-1, b_max=1, clip=True),
                ScaleIntensityd(keys=cond_keys, minv=-1, maxv=1),
                # 新增：骨架图强度缩放（假设骨架图是二值图，缩放到[-1,1]）
                # ScaleIntensityd(keys=["skeleton"], minv=-1, maxv=1),
            ]
        )
    else:
        train_transforms = Compose(
            [
                LoadImaged(keys=["image"], ensure_channel_first=True),
                # Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear")),
                Resized(keys=["image"], spatial_size=resize_size),
                ScaleIntensityRanged(keys=["image"], a_min=-1000, a_max=400, b_min=-1, b_max=1, clip=True),
            ]
        )
    if split == "train":
        train_ds = CacheDataset(data=data_dicts, transform=train_transforms, cache_rate=1.0, num_workers=8)
    else:
        train_ds = Dataset(data=data_dicts, transform=train_transforms)
    shuffle = False if split == "test" else True
    train_dl = MultiEpochsDataLoader(train_ds, batch_size=batch_size, num_workers=4, shuffle=shuffle)
    return train_dl


@hydra.main(config_path="../conf", config_name="config/TEST.yaml", version_base="1.3")
def main(config):
    config = config["config"]
    # train_dl = prepare_dataset(
    #     data_path=config.data_path, resize_size=config.resize_size, cond_path=config.cond_path, split="test"
    # )
    test_dl = prepare_dataset(
        cond_path=config.cond_path,
        data_path=config.data_path,
        resize_size=config.resize_size,
        img_resize_size=config.img_resize_size,
        split="test",
    )

    for i in test_dl:
        # print(i["image"])
        # continue
        # print(i["cond1"].shape)
        # cond1 = i["cond1"]
        # cond1 = i["cond1"].permute(0, 2, 3, 1)
        # cond1 = cond1 * 255
        # cond = cond[:, :, :, :3]
        # print(f"max value {max(cond)}, min value {min(cond)}")
        # cond = cond * 255
        # print(cond.shape)
        img = i["image"]
        # img = img.squeeze(0)
        # img = img * 255
        img = (img + 1) * 127.5
        print(img.shape)
        # print(img.affine)
        saver_origin = SaveImage(
            output_dir="./",
            output_ext=".jpg",           #modify .nii.gz
            output_postfix="cache",
            separate_folder=False,
            output_dtype=np.uint8,
            # scale=255,
            resample=False,
            squeeze_end_dims=True,
            writer="PILWriter",    #modify NibabelWriter
        )
        saver_origin(img)

        # saver = SaveImage(
        #     output_dir="./",
        #     output_ext=".png",
        #     output_postfix="PIL",
        #     output_dtype=np.uint8,
        #     resample=False,
        #     squeeze_end_dims=True,
        #     writer="PILWriter",
        # )
        # img = saver(cond1)
        break
        # break


def test_save_image():
    # img = LoadImage()
    # path = "/disk/ssy/data/drr/feijiejie/all/LNDb-0210.nii"
    path = "/disk/ssy/data/DIAS/myDIAS/training/imagesjpg/image_s0_i3.jpg"
    trans = Compose(
        [
            LoadImaged(keys="image", ensure_channel_first=True),
            lambda d: print(f"After LoadImaged - shape: {d['image'].shape}, dtype: {d['image'].dtype}") or d,
            
            # Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear")),
            Resized(keys="image", spatial_size=(128, 128)),
            lambda d: print(f"After Resized - shape: {d['image'].shape}, dtype: {d['image'].dtype}") or d,
            
            # ScaleIntensityd(keys="image"),
            ScaleIntensityRanged(keys="image", a_min=-1000, a_max=400, b_min=-1, b_max=1, clip=True),
            lambda d: print(f"After ScaleIntensityRanged - shape: {d['image'].shape}, dtype: {d['image'].dtype}") or d,
        
        ]
    )
    d = {"image": path}
    img = trans(d)
    img = img["image"]
    # print(img.shape)
    # print(img.affine)
    # print(img.meta.keys())
    img = (img + 1) * 127.5
    # print(img.shape)
    saver_origin = SaveImage(
        output_dir="./",
        output_ext=".jpg",
        output_postfix="origin",
        separate_folder=False,
        output_dtype=np.uint8,
        # scale=255,
        resample=False,
        squeeze_end_dims=True,
        writer="PILWriter",
    )
    saver_origin(img)


if __name__ == "__main__":
    # main()
    test_save_image()
