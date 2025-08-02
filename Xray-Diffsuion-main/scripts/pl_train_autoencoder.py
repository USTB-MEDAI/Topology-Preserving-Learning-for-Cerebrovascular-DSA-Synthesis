import hydra
import os
import torch
from torch.utils.data import DataLoader
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from lightning.pytorch.loggers import NeptuneLogger, CometLogger

# from dataset.med_3Ddataset import ImageDataset
from dataset.MhdDataset import MhdDataset
import lightning as pl
from ldm.autoencoderkl.autoencoder import AutoencoderKL
from lightning.pytorch.callbacks import ModelCheckpoint
from dataset.monai_nii_dataset_jpg2 import prepare_dataset
# from dataset.monai_nii_dataset import prepare_dataset

torch.set_float32_matmul_precision("high")


@hydra.main(config_path="../conf", config_name="config", version_base="1.3")
def train(config):
    config = config["config"]
    checkpoint_callback = ModelCheckpoint(
        monitor=config["model"].monitor,
        dirpath=config.hydra_path,
        filename="pl_train_autoencoder-epoch{epoch:02d}-val_rec_loss{val/rec_loss:.2f}",
        save_top_k=1,
        mode="min",
        auto_insert_metric_name=False,
    )
    checkpoint_callback_latest = ModelCheckpoint(
        dirpath=config.hydra_path,
        filename="latest",
    )
    train_dl = prepare_dataset(
        data_path=config.data_path, resize_size=config.resize_size, split="train"
    )
    val_dl = prepare_dataset(
        data_path=config.data_path, resize_size=config.resize_size, split="val"
    )



    # 打印训练集和验证集的大小
    print(f"Train dataset size: {len(train_dl.dataset)}")
    print(f"Validation dataset size: {len(val_dl.dataset)}")

    # 获取一个批次的数据并打印
    train_batch = next(iter(train_dl))
    val_batch = next(iter(val_dl))

    # 打印训练集批次中的数据和标签
    print(f"Train batch data shape: {train_batch['image'].shape}")
    print(f"Train batch label shape: {train_batch.get('label', 'No label in batch')}")

    # 打印验证集批次中的数据和标签
    print(f"Validation batch data shape: {val_batch['image'].shape}")
    print(f"Validation batch label shape: {val_batch.get('label', 'No label in batch')}")

    # 打印图像的最小值和最大值，确认预处理是否正常
    train_image = train_batch['image']
    val_image = val_batch['image']

    print(f"Train image min: {train_image.min()}, max: {train_image.max()}")
    print(f"Validation image min: {val_image.min()}, max: {val_image.max()}")
    



    # * dataset and dataloader
    # ds = MhdDataset(config, split="train", mean=-464.24, std=558.47)
    # dataloader = DataLoader(
    #     dataset=ds,
    #     shuffle=True,
    #     pin_memory=True,
    #     drop_last=True,
    #     num_workers=config.num_workers,
    #     batch_size=config.batch_size,
    # )
    # val_dataset = MhdDataset(config, split="val", mean=-464.24, std=558.47)
    # val_dataloader = DataLoader(
    #     dataset=val_dataset,
    #     shuffle=False,
    #     pin_memory=True,
    #     drop_last=True,
    #     num_workers=config.num_workers,
    #     batch_size=config.batch_size,
    # )

    # * model
    model = AutoencoderKL(save_path=config.hydra_path, **config["model"])

    # * trainer fit
    trainer = pl.Trainer(
        **config["trainer"],
        callbacks=[checkpoint_callback, checkpoint_callback_latest],
        default_root_dir=config.hydra_path,
    )
    trainer.fit(model=model, train_dataloaders=train_dl, val_dataloaders=val_dl)


if __name__ == "__main__":
    train()
