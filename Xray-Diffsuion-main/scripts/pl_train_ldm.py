import hydra
import torch
from torch.utils.data import DataLoader
import sys
import os
import lightning as pl
from lightning.pytorch.callbacks import ModelCheckpoint


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
# from dataset.monai_nii_dataset_jpg2 import prepare_dataset                     #mdf
from dataset.monai_nii_dataset_jpg4 import prepare_dataset                     #mdf

# from ldm.autoencoderkl.autoencoder import AutoencoderKL
from ldm.ddpm import LatentDiffusion

torch.set_float32_matmul_precision("high")


@hydra.main(config_path="../conf", config_name="config", version_base="1.3")
def train(config):
    config = config["config"]
    checkpoint_callback = ModelCheckpoint(
        monitor=config["latent_diffusion"].monitor,
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
    # * dataset and dataloader
    
    skeleton_path = config.skeleton_path if "skeleton_path" in config.keys() else None
    
    train_dl = prepare_dataset(
        cond_path=config.cond_path,
        data_path=config.data_path,
        skeleton_path=skeleton_path,
        resize_size=config.resize_size,
        img_resize_size=config.img_resize_size,
        split="train",
        cond_nums=config.latent_diffusion.cond_nums,
        batch_size=config.batch_size,
    )
    val_dl = prepare_dataset(
        cond_path=config.cond_path,
        data_path=config.data_path,
        skeleton_path=skeleton_path,
        resize_size=config.resize_size,
        img_resize_size=config.img_resize_size,
        cond_nums=config.latent_diffusion.cond_nums,
        batch_size=config.batch_size,
    )

    # * model
    model = LatentDiffusion(root_path=config.hydra_path, **config["latent_diffusion"])

    # * trainer fit
    trainer = pl.Trainer(
        **config["trainer"], callbacks=[checkpoint_callback, checkpoint_callback_latest], default_root_dir=config.hydra_path
    )
    trainer.fit(model=model, train_dataloaders=train_dl, val_dataloaders=val_dl)


if __name__ == "__main__":
    train()
