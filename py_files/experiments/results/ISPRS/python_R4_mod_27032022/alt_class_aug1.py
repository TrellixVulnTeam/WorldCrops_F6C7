#%%
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


data_path = '../../../../../data/cropdata/Bavaria/sentinel-2/Training_bavaria.xlsx'

#directory with logs
pretrained_dir = "../logs/pretrained_model4/aug1"

try:
    # Create target Directory
    os.makedirs(pretrained_dir)
    print("Directory " , pretrained_dir ,  " Created ") 
except FileExistsError:
    print("Directory " , pretrained_dir ,  " already exists")


logger1 = TensorBoardLogger("../logs/ssl4", name="aug1")
logger2 = TensorBoardLogger("../logs/ssl4", name="aug1")
logger3 = TensorBoardLogger("../logs/ssl4", name="aug1")
logger4 = TensorBoardLogger("../logs/ssl4", name="aug1")


################################


#some definitions for Transformers
batch_size = 1349
test_size = 0.25
num_workers=4
shuffle_dataset =True
_epochs = 300
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
#experiment with train/test split for all data
dm_bavaria = BavariaDataModule(data_dir = data_path, batch_size = batch_size, num_workers = num_workers, experiment='Experiment1')
#experiment with 16/17 train and 2018 test
dm_bavaria2 = BavariaDataModule(data_dir = data_path, batch_size = batch_size, num_workers = num_workers, experiment='Experiment2')
#experiment with 16/17 + 5% 2018 train and 2018 test
dm_bavaria3 = BavariaDataModule(data_dir = data_path, batch_size = batch_size, num_workers = num_workers, experiment='Experiment3')
#experiment with 16/17 + 10% 2018 train and 2018 test
dm_bavaria4 = BavariaDataModule(data_dir = data_path, batch_size = batch_size, num_workers = num_workers, experiment='Experiment4')

#%%
dm_crops1 = AugmentationExperiments(data_dir = data_path, batch_size = batch_size_sim, num_workers = num_workers, experiment='Experiment3')
dm_crops2 = AugmentationExperiments(data_dir = data_path, batch_size = batch_size_sim, num_workers = num_workers, experiment='Experiment5')
dm_crops3 = AugmentationExperiments(data_dir = data_path, batch_size = batch_size_sim, num_workers = num_workers, experiment='Experiment6')
dm_crops4 = AugmentationExperiments(data_dir = data_path, batch_size = batch_size_sim, num_workers = num_workers, experiment='Experiment7')


#%%
TRAIN = True
if TRAIN==True:
    no_gpus = 1

    transformer = Attention(input_dim=input_dim,num_classes = 6, n_head=4, nlayers=3)
    backbone = nn.Sequential(*list(transformer.children())[-2])
    model_sim = SimSiam_LM(backbone,num_ftrs=num_ftrs,proj_hidden_dim=proj_hidden_dim,pred_hidden_dim=pred_hidden_dim,out_dim=out_dim,lr=lr)
    trainer = pl.Trainer(gpus=no_gpus, strategy='ddp', deterministic=True, max_epochs = _epochs, logger=logger1)
    trainer.fit(model_sim, datamodule=dm_crops1)
    torch.save(model_sim, 'model_sim1_R4_aug1.ckpt')


    transformer2 = Attention(input_dim=input_dim,num_classes = 6, n_head=4, nlayers=3)
    backbone2 = nn.Sequential(*list(transformer2.children())[-2])
    model_sim2 = SimSiam_LM(backbone2,num_ftrs=num_ftrs,proj_hidden_dim=proj_hidden_dim,pred_hidden_dim=pred_hidden_dim,out_dim=out_dim,lr=lr)
    trainer2 = pl.Trainer(gpus=no_gpus, strategy='ddp', deterministic=True, max_epochs = _epochs)
    trainer2.fit(model_sim2, datamodule=dm_crops2)
    torch.save(model_sim2, 'model_sim2_R4_aug1.ckpt')


    transformer3 = Attention(input_dim=input_dim,num_classes = 6, n_head=4, nlayers=3)
    backbone3 = nn.Sequential(*list(transformer3.children())[-2])
    model_sim3 = SimSiam_LM(backbone3,num_ftrs=num_ftrs,proj_hidden_dim=proj_hidden_dim,pred_hidden_dim=pred_hidden_dim,out_dim=out_dim,lr=lr)
    trainer3 = pl.Trainer(gpus=no_gpus, strategy='ddp', deterministic=True, max_epochs = _epochs, logger=logger3)
    trainer3.fit(model_sim3, datamodule=dm_crops3)
    torch.save(model_sim3, 'model_sim3_R4_aug1.ckpt')


    transformer4 = Attention(input_dim=input_dim,num_classes = 6, n_head=4, nlayers=3)
    backbone4 = nn.Sequential(*list(transformer4.children())[-2])
    model_sim4 = SimSiam_LM(backbone4,num_ftrs=num_ftrs,proj_hidden_dim=proj_hidden_dim,pred_hidden_dim=pred_hidden_dim,out_dim=out_dim,lr=lr)
    trainer4 = pl.Trainer(gpus=no_gpus, strategy='ddp', deterministic=True, max_epochs = _epochs, logger=logger4)
    trainer4.fit(model_sim4, datamodule=dm_crops4)
    torch.save(model_sim4, 'model_sim4_R4_aug1.ckpt')
