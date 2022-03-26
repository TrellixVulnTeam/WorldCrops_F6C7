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
pretrained_dir = "../logs/pretrained_model3/aug3"

try:
    # Create target Directory
    os.makedirs(pretrained_dir)
    print("Directory " , pretrained_dir ,  " Created ") 
except FileExistsError:
    print("Directory " , pretrained_dir ,  " already exists")


logger1 = TensorBoardLogger("../logs/ssl3", name="aug3")
logger2 = TensorBoardLogger("../logs/ssl3", name="aug3")
logger3 = TensorBoardLogger("../logs/ssl3", name="aug3")
logger4 = TensorBoardLogger("../logs/ssl3", name="aug3")


################################


#some definitions for Transformers
batch_size = 1349
test_size = 0.25
num_workers=4
shuffle_dataset =True
_epochs = 1800
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
dm_crops1 = AugmentationExperiments(data_dir = data_path, batch_size = batch_size_sim, num_workers = num_workers, experiment='Experiment12')
dm_crops2 = AugmentationExperiments(data_dir = data_path, batch_size = batch_size_sim, num_workers = num_workers, experiment='Experiment13')
dm_crops3 = AugmentationExperiments(data_dir = data_path, batch_size = batch_size_sim, num_workers = num_workers, experiment='Experiment14')
dm_crops4 = AugmentationExperiments(data_dir = data_path, batch_size = batch_size_sim, num_workers = num_workers, experiment='Experiment15')


#%%
TRAIN = True
if TRAIN==True:
    no_gpus = 1

    transformer = Attention(input_dim=input_dim,num_classes = 6, n_head=4, nlayers=3)
    backbone = nn.Sequential(*list(transformer.children())[-2])
    model_sim = SimSiam_LM(backbone,num_ftrs=num_ftrs,proj_hidden_dim=proj_hidden_dim,pred_hidden_dim=pred_hidden_dim,out_dim=out_dim,lr=lr)
    trainer = pl.Trainer(gpus=no_gpus, strategy='ddp', deterministic=True, max_epochs = _epochs, logger=logger1)
    trainer.fit(model_sim, datamodule=dm_crops1)
    torch.save(model_sim, 'model_sim1_R1_aug3.ckpt')


    transformer2 = Attention(input_dim=input_dim,num_classes = 6, n_head=4, nlayers=3)
    backbone2 = nn.Sequential(*list(transformer2.children())[-2])
    model_sim2 = SimSiam_LM(backbone2,num_ftrs=num_ftrs,proj_hidden_dim=proj_hidden_dim,pred_hidden_dim=pred_hidden_dim,out_dim=out_dim,lr=lr)
    trainer2 = pl.Trainer(gpus=no_gpus, strategy='ddp', deterministic=True, max_epochs = _epochs)
    trainer2.fit(model_sim2, datamodule=dm_crops2)
    torch.save(model_sim2, 'model_sim2_R1_aug3.ckpt')


    transformer3 = Attention(input_dim=input_dim,num_classes = 6, n_head=4, nlayers=3)
    backbone3 = nn.Sequential(*list(transformer3.children())[-2])
    model_sim3 = SimSiam_LM(backbone3,num_ftrs=num_ftrs,proj_hidden_dim=proj_hidden_dim,pred_hidden_dim=pred_hidden_dim,out_dim=out_dim,lr=lr)
    trainer3 = pl.Trainer(gpus=no_gpus, strategy='ddp', deterministic=True, max_epochs = _epochs, logger=logger3)
    trainer3.fit(model_sim3, datamodule=dm_crops3)
    torch.save(model_sim3, 'model_sim3_R1_aug3.ckpt')


    transformer4 = Attention(input_dim=input_dim,num_classes = 6, n_head=4, nlayers=3)
    backbone4 = nn.Sequential(*list(transformer4.children())[-2])
    model_sim4 = SimSiam_LM(backbone4,num_ftrs=num_ftrs,proj_hidden_dim=proj_hidden_dim,pred_hidden_dim=pred_hidden_dim,out_dim=out_dim,lr=lr)
    trainer4 = pl.Trainer(gpus=no_gpus, strategy='ddp', deterministic=True, max_epochs = _epochs, logger=logger4)
    trainer4.fit(model_sim4, datamodule=dm_crops4)
    torch.save(model_sim4, 'model_sim4_R1_aug3.ckpt')



    exit()






### CLASSIFICATION PART

#%%
train,test = dm_bavaria2.experiment2()
print(train.shape)
print(test.shape)

#%%
feature_list = ['B4_mean','B5_mean','B6_mean','B7_mean','B8_mean','B8A_mean','B9_mean','B11_mean','B12_mean']
time_steps = 11

data_dict = {}
data_dict['2016'] = {}
data_dict['2017'] = {}
data_dict['2018'] = {}
for n in range(6):
    data_dict['2016'][n] = []
    data_dict['2017'][n] = []
    data_dict['2018'][n] = []
    # for c in range(len(channel_idx)):
    #     data_dict['2016'][n][c] = []
    #     data_dict['2017'][n][c] = []
    #     data_dict['2018'][n][c] = []

data = train
keys = data.keys()
channel_idx = []
for n in range(keys.shape[0]):
    for m in range(len(feature_list)):
        if (feature_list[m]==keys[n]):
            channel_idx.append(n)
time_idx = [0,time_steps]
data2 = np.array(data)
tot_samples = int(data2.shape[0]/time_steps)
for n in range(tot_samples):
    data_dict[str(data2[n*time_steps,20])][data2[n*time_steps,3]].append(data2[n*time_steps+time_idx[0]:n*time_steps+time_idx[1], channel_idx])

data = test
keys = data.keys()
channel_idx = []
for n in range(keys.shape[0]):
    for m in range(len(feature_list)):
        if (feature_list[m]==keys[n]):
            channel_idx.append(n)
time_idx = [0,time_steps]
data2 = np.array(data)
tot_samples = int(data2.shape[0]/time_steps)
for n in range(tot_samples):
    data_dict[str(data2[n*time_steps,20])][data2[n*time_steps,3]].append(data2[n*time_steps+time_idx[0]:n*time_steps+time_idx[1], channel_idx])


#%%
print(np.array(data_dict['2018'][1]).shape)


#%%
# path = '/iarai/home/daniel.springer/Projects/WorldCrops/WorldCrops/py_files/experiments/results/ISPRS/logs_R2_23032022/pretrained_model/aug1/pretraining2.ckpt'
# backbone2 = torch.load(path)
# model_sim2 = SimSiam_LM(backbone2,num_ftrs=num_ftrs,proj_hidden_dim=proj_hidden_dim,pred_hidden_dim=pred_hidden_dim,out_dim=out_dim,lr=lr)
# model_sim2 = model_sim2.eval()
# model_sim2 = SimSiam_LM.load_from_checkpoint('model1.ckpt')
model_sim2 = torch.load('../NEW_2/model_R1.ckpt')
model_sim2 = model_sim2.eval()
model_sim2 = model_sim2.cuda(1)
# print(model_sim2)
#%%
in1 = data_dict['2016'][0][0]
in2 = data_dict['2018'][0][0]
in1 = np.array(in1, dtype=np.float32)
in2 = np.array(in2, dtype=np.float32)

ce = lightly.loss.NegativeCosineSimilarity()
(z0, p0),(z1, p1), embedding = model_sim2.forward(torch.tensor(in1).cuda(1)[None], torch.tensor(in2).cuda(1)[None])
loss = 0.5 * (ce(z0, p1) + ce(z1, p0))
print(loss)

    # %%
ce = lightly.loss.NegativeCosineSimilarity()
results_dict = {}
results_dict['2016'] = {}
results_dict['2017'] = {}
results_dict['2018'] = {}
for n in range(6):
    results_dict['2016'][n] = []
    results_dict['2017'][n] = []
    results_dict['2018'][n] = []

prediction_year = ['2018']

for y in range(len(prediction_year)):
    year = prediction_year[y]
    for k in range(6):
        gd = 0
        bd = 0
        # print(k, len(data_dict[year][k]))
        for l in range(len(data_dict[year][k])):
            in2 = data_dict[year][k][l]
            in2 = np.array(in2, dtype=np.float32)
            loss_tot = np.zeros(6)

            for m in range(6):
                loss = 0
                for n in range(len(data_dict['2016'][m])):
                    in1 = data_dict['2016'][m][n]
                    in1 = np.array(in1, dtype=np.float32)
                    (z0, p0),(z1, p1), embedding = model_sim2.forward(torch.tensor(in1).cuda(1)[None], torch.tensor(in2).cuda(1)[None])
                    loss += 0.5 * (ce(z0, p1) + ce(z1, p0))
                for n in range(len(data_dict['2017'][m])):
                    in1 = data_dict['2017'][m][n]
                    in1 = np.array(in1, dtype=np.float32)
                    (z0, p0),(z1, p1), embedding = model_sim2.forward(torch.tensor(in1).cuda(1)[None], torch.tensor(in2).cuda(1)[None])
                    loss += 0.5 * (ce(z0, p1) + ce(z1, p0))
                loss_tot[m] = loss.detach()

            if (np.argmin(loss_tot)==k):
                results_dict['2018'][k].append(1)
                gd += 1
                print(prediction_year[y], k, l+1, '/',len(data_dict[year][k]),  'Correct:', gd, 'Wrong:', bd)
            else:
                results_dict['2018'][k].append(0)
                bd += 1
                print(prediction_year[y], k, l+1, '/',len(data_dict[year][k]),  'Correct:', gd, 'Wrong:', bd)

            # break
        # break


# %%
print(results_dict['2018'])

print(sum(results_dict['2018'][0]))

#%%
torch.save(results_dict, 'Results')

#%%
fff = torch.load('Results')

#%%
print(len(fff['2018'][0]))
