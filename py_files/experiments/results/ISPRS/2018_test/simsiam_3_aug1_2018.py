# compare the crop type classification of RF and SimSiam
import sys
import os
sys.path.append(os.path.realpath('../../../../'))

from model import *
from processing import *
import math

import torch.nn as nn
import torchvision
import lightly

import matplotlib.pyplot as plt
import torch
import seaborn as sns
import numpy as np
import pandas as pd
import geopandas as gpd
import torch
import pandas as pd
import numpy as np
from pytorch_lightning import Trainer, seed_everything
import copy
from pytorch_lightning.loggers import TensorBoardLogger

################################
#IARAI / ESA
IARAI = True
no_gpus = 16
# no_gpus = [0,1,2,3,4,5,6]
PA = False
data_path = '../../../../../data/cropdata/Bavaria/sentinel-2/Training_bavaria.xlsx'

#directory with logs
pretrained_dir = "../logs/pretrained_model3/aug1"

try:
    # Create target Directory
    os.makedirs(pretrained_dir)
    print("Directory " , pretrained_dir ,  " Created ") 
except FileExistsError:
    print("Directory " , pretrained_dir ,  " already exists")


logger1 = TensorBoardLogger("../logs/ssl3", name="aug1")
logger2 = TensorBoardLogger("../logs/ssl3", name="aug1")

logger3 = TensorBoardLogger("../logs/ssl3", name="aug2")
logger4 = TensorBoardLogger("../logs/ssl3", name="aug2")

logger_sup1 = TensorBoardLogger("../logs/ssl3", name="supervised3")
logger_sup2 = TensorBoardLogger("../logs/ssl3", name="supervised3")



################################


#some definitions for Transformers
batch_size = 1349
test_size = 0.25
num_workers=4
shuffle_dataset =True
_epochs = 600
_epochs_fine = 300
input_dim = 9
lr =  0.0016612

#definitions for simsiam
num_ftrs = 64
proj_hidden_dim =6
pred_hidden_dim =6
out_dim =14
batch_size_sim = 256

# scale the learning rate
#lr = 0.05 * batch_size / 256

def seed_torch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

seed_torch()


# %%
#load data for bavaria

#extra test for only 2018 5% + 10%
dm_bavaria5 = BavariaDataModule(data_dir = data_path, batch_size = batch_size, num_workers = num_workers, experiment='Experiment5')
dm_bavaria6 = BavariaDataModule(data_dir = data_path, batch_size = batch_size, num_workers = num_workers, experiment='Experiment6')

dm_crops5 = AugmentationExperiments(data_dir = data_path, batch_size = batch_size_sim, num_workers = num_workers, experiment='Experiment16')
dm_crops6 = AugmentationExperiments(data_dir = data_path, batch_size = batch_size_sim, num_workers = num_workers, experiment='Experiment17')

dm_crops7 = AugmentationExperiments(data_dir = data_path, batch_size = batch_size_sim, num_workers = num_workers, experiment='Experiment18')
dm_crops8 = AugmentationExperiments(data_dir = data_path, batch_size = batch_size_sim, num_workers = num_workers, experiment='Experiment19')


# %%
# Vorgehen:
# 1. Pre-Train transformer unsupervised mit allen Daten (typische Augmentation + physikalisch)
# 2. Finetune

#####################
transformer = Attention(input_dim=input_dim,num_classes = 6, n_head=4, nlayers=3)
backbone = nn.Sequential(*list(transformer.children())[-2])
model_sim = SimSiam_LM(backbone,num_ftrs=num_ftrs,proj_hidden_dim=proj_hidden_dim,pred_hidden_dim=pred_hidden_dim,out_dim=out_dim,lr=lr)
if IARAI:
    trainer1 = pl.Trainer(gpus=no_gpus, strategy='ddp', deterministic=True, max_epochs = _epochs, logger=logger1)
else:
    trainer1 = pl.Trainer(deterministic=True, max_epochs = _epochs, logger=logger1)
#fit the first time with one augmentation
trainer1.fit(model_sim, datamodule=dm_crops5)
#torch.save(backbone, pretrained_dir +"/pretraining1.ckpt")

#####################
transformer2 = Attention(input_dim=input_dim,num_classes = 6, n_head=4, nlayers=3)
backbone2 = nn.Sequential(*list(transformer2.children())[-2])
model_sim2 = SimSiam_LM(backbone2,num_ftrs=num_ftrs,proj_hidden_dim=proj_hidden_dim,pred_hidden_dim=pred_hidden_dim,out_dim=out_dim,lr=lr)

if IARAI:
    trainer2 = pl.Trainer(gpus=no_gpus, strategy='ddp', deterministic=True, max_epochs = _epochs, logger=logger2)
else:
    trainer2 = pl.Trainer(deterministic=True, max_epochs = _epochs, logger=logger2)
trainer2.fit(model_sim2, datamodule=dm_crops6)
#torch.save(backbone2, pretrained_dir +"/pretraining2.ckpt")

#####################
transformer3 = Attention(input_dim=input_dim,num_classes = 6, n_head=4, nlayers=3)
backbone3 = nn.Sequential(*list(transformer3.children())[-2])
model_sim3 = SimSiam_LM(backbone3,num_ftrs=num_ftrs,proj_hidden_dim=proj_hidden_dim,pred_hidden_dim=pred_hidden_dim,out_dim=out_dim,lr=lr)

if IARAI:
    trainer3 = pl.Trainer(gpus=no_gpus, strategy='ddp', deterministic=True, max_epochs = _epochs, logger=logger3)
else:
    trainer3 = pl.Trainer(deterministic=True, max_epochs = _epochs, logger=logger3)
trainer3.fit(model_sim3, datamodule=dm_crops7)

#####################
transformer4 = Attention(input_dim=input_dim,num_classes = 6, n_head=4, nlayers=3)
backbone4 = nn.Sequential(*list(transformer4.children())[-2])
model_sim4 = SimSiam_LM(backbone4,num_ftrs=num_ftrs,proj_hidden_dim=proj_hidden_dim,pred_hidden_dim=pred_hidden_dim,out_dim=out_dim,lr=lr)

if IARAI:
    trainer4 = pl.Trainer(gpus=no_gpus, strategy='ddp', deterministic=True, max_epochs = _epochs, logger=logger4)
else:
    trainer4 = pl.Trainer(deterministic=True, max_epochs = _epochs, logger=logger4)
trainer4.fit(model_sim4, datamodule=dm_crops8)

#####################
#%%
#backbone = torch.load("../model/pretrained/backbone_3_aug_17.2.ckpt")
#copy pretrained backbone for experiments
backbone_copy1 = copy.deepcopy(backbone)
backbone_copy2 = copy.deepcopy(backbone2)
backbone_copy3 = copy.deepcopy(backbone3)
backbone_copy4 = copy.deepcopy(backbone4)


#backbone_copy1 = torch.load(pretrained_dir + "/pretraining1.ckpt")
#backbone_copy2 = torch.load(pretrained_dir + "/pretraining2.ckpt")
#backbone_copy3 = torch.load(pretrained_dir + "/pretraining3.ckpt")
#backbone_copy4 = torch.load(pretrained_dir + "/pretraining4.ckpt")


# %%
#####################################
#use pretrained backbone and finetune 
#####################################

transformer1 = Attention(input_dim=input_dim,num_classes = 6, n_head=4, nlayers=3)
head = nn.Sequential(*list(transformer1.children())[-1])
transfer_model = Attention_Transfer(input_dim=input_dim, num_classes = 6, d_model=num_ftrs, backbone = backbone_copy1, head=head, batch_size = batch_size, finetune=True, lr=lr)

transformer2 = Attention(input_dim=input_dim, num_classes = 6, n_head=4, nlayers=3)
head2 = nn.Sequential(*list(transformer2.children())[-1])
#use pretrained backbone and finetune 
transfer_model2 = Attention_Transfer(input_dim=input_dim, num_classes = 6, d_model=num_ftrs, backbone = backbone_copy2, head=head2, batch_size = batch_size, finetune=True, lr=lr)

transformer3 = Attention(input_dim=input_dim, num_classes = 6, n_head=4, nlayers=3)
head3 = nn.Sequential(*list(transformer3.children())[-1])
#use pretrained backbone and finetune 
transfer_model3 = Attention_Transfer(input_dim=input_dim, num_classes = 6, d_model=num_ftrs, backbone = backbone_copy3, head=head3, batch_size = batch_size, finetune=True, lr=lr)

transformer4 = Attention(input_dim=input_dim, num_classes = 6, n_head=4, nlayers=3)
head4 = nn.Sequential(*list(transformer4.children())[-1])
#use pretrained backbone and finetune 
transfer_model4 = Attention_Transfer(input_dim=input_dim, num_classes = 6, d_model=num_ftrs, backbone = backbone_copy4, head=head4, batch_size = batch_size, finetune=True, lr=lr)


#####################################
if IARAI:
    trainer = pl.Trainer(gpus=no_gpus, strategy='ddp', deterministic=True, max_epochs = _epochs_fine, logger=logger1)
else:
    trainer = pl.Trainer(deterministic=True, max_epochs= _epochs_fine, logger=logger1)
trainer.fit(transfer_model, datamodule = dm_bavaria5)
trainer.test(transfer_model, datamodule = dm_bavaria5)

#####################################

if IARAI:
    trainer = pl.Trainer(gpus=no_gpus, strategy='ddp', deterministic=True, max_epochs = _epochs_fine, logger=logger2)
else:
    trainer = pl.Trainer(deterministic=True, max_epochs= _epochs_fine, logger=logger2)
trainer.fit(transfer_model2, datamodule = dm_bavaria6)
trainer.test(transfer_model2, datamodule = dm_bavaria6)

#####################################

if IARAI:
    trainer = pl.Trainer(gpus=no_gpus, strategy='ddp', deterministic=True, max_epochs = _epochs_fine, logger=logger3)
else:
    trainer = pl.Trainer(deterministic=True, max_epochs= _epochs_fine, logger=logger3)
trainer.fit(transfer_model3, datamodule = dm_bavaria5)
trainer.test(transfer_model3, datamodule = dm_bavaria5)

#####################################

if IARAI:
    trainer = pl.Trainer(gpus=no_gpus, strategy='ddp', deterministic=True, max_epochs = _epochs_fine, logger=logger4)
else:
    trainer = pl.Trainer(deterministic=True, max_epochs= _epochs_fine, logger=logger4)
trainer.fit(transfer_model4, datamodule = dm_bavaria6)
trainer.test(transfer_model4, datamodule = dm_bavaria6)

# %%
################################################################
# E3/E4 supervised
################################################################

model1 = Attention_LM(input_dim=input_dim, num_classes = 6, n_head=4, nlayers=3, batch_size = batch_size, lr=lr, PositonalEncoding=PA)
model2 = Attention_LM(input_dim=input_dim, num_classes = 6, n_head=4, nlayers=3, batch_size = batch_size, lr=lr, PositonalEncoding=PA)

trainer = pl.Trainer( deterministic=True, max_epochs= _epochs_fine, logger=logger_sup1)
trainer.fit(model1, datamodule = dm_bavaria5)
trainer.test(model1, datamodule = dm_bavaria5)

trainer = pl.Trainer( deterministic=True, max_epochs= _epochs_fine, logger=logger_sup2)
trainer.fit(model2, datamodule = dm_bavaria6)
trainer.test(model2, datamodule = dm_bavaria6)

