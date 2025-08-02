from cldm.hack import disable_verbosity
disable_verbosity()

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from tutorial_dataset2 import MyDataset
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict


# Configs
# resume_path ='/disk/ssy/pyproj/bishe/controlnet-main/lightning_logs/version_17/checkpoints/epoch=928-step=52023.ckpt'
# resume_path ='/disk/ssy/pyproj/bishe/controlnet-main/models/control_sd15_scribble.pth'
resume_path="/disk/ssy/pyproj/bishe/controlnet-main/lightning_logs/version_19/checkpoints/epoch=999-step=55999.ckpt"
#'./models/control_sd15_ini.ckpt'
batch_size = 1  #4
logger_freq = 1400#1400  #mdf 300
learning_rate = 1e-5  #1e-5 5e-6 2e-6
sd_locked = True
only_mid_control = False


# First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
model = create_model('./models/cldm_v15.yaml').cpu()
model.load_state_dict(load_state_dict(resume_path, location='cpu'))
# model = create_model('./models/cldm_v15.yaml')
# model = model.load_from_checkpoint(resume_path)
model.learning_rate = learning_rate
model.sd_locked = sd_locked
model.only_mid_control = only_mid_control


# Misc
#dataset = MyDataset()
dataset = MyDataset(source_dir="/disk/ssy/data/DIAS/myDIAS/hand/test_predict505", target_dir="/disk/ssy/data/DIAS/myDIAS/hand/test_predict505")
dataloader = DataLoader(dataset, num_workers=0, batch_size=batch_size, shuffle=True)
dataset2 = MyDataset(source_dir="/disk/ssy/data/DIAS/myDIAS/hand/test_predict505", target_dir="/disk/ssy/data/DIAS/myDIAS/hand/test_predict505")
dataloader2 = DataLoader(dataset2, num_workers=0, batch_size=batch_size, shuffle=True)
logger = ImageLogger(batch_frequency=logger_freq)
trainer = pl.Trainer(gpus=[1], precision=32, callbacks=[logger])


# Train!
trainer.test(model, dataloader)
trainer.test(model, dataloader2)
