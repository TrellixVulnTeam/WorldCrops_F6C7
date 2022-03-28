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
batch_size = 1349
num_workers=4
batch_size_sim = 256

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
dm_crops1 = AugmentationExperiments(data_dir = data_path, batch_size = batch_size_sim, num_workers = num_workers, experiment='Experiment8')
dm_crops2 = AugmentationExperiments(data_dir = data_path, batch_size = batch_size_sim, num_workers = num_workers, experiment='Experiment9')
dm_crops3 = AugmentationExperiments(data_dir = data_path, batch_size = batch_size_sim, num_workers = num_workers, experiment='Experiment10')
dm_crops4 = AugmentationExperiments(data_dir = data_path, batch_size = batch_size_sim, num_workers = num_workers, experiment='Experiment11')


#%% ALL DATA FOR EVALUATION

train,test = dm_bavaria2.experiment2()

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
for n in range(4):
    FNAME = f'model_sim{n+1}_R4_aug3.ckpt'
    SNAME = f'results_sim{n+1}_R4_aug3'
    print(FNAME)

    model_sim2 = torch.load(FNAME)
    model_sim2 = model_sim2.eval()
    model_sim2 = model_sim2.cuda(1)

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
                    results_dict['2018'][k].append(-np.argmin(loss_tot))
                    bd += 1
                    print(prediction_year[y], k, l+1, '/',len(data_dict[year][k]),  'Correct:', gd, 'Wrong:', bd, 'Predicted Type:', np.argmin(loss_tot))

    torch.save(results_dict, SNAME)

#%%
# import numpy as np
import torch

for n in range(4):
    SNAME = f'results_sim{n+1}_R4_aug3'
    ttt = torch.load(SNAME)
    acc = 0
    inter = 0
    for n in range(6):
        print(ttt['2018'][n].count(1))
        acc += ttt['2018'][n].count(1)
    print(acc/6)

# ttt = torch.load('results_sim1_R1_aug1')
# acc = 0
# inter = 0
# for n in range(6):
#     #print(ttt['2018'][n].count(1))
#     acc += ttt['2018'][n].count(1)
# print(acc/6)

# ttt = torch.load('results_sim2_R1_aug1')
# acc = 0
# for n in range(6):
#     #print(ttt['2018'][n].count(1))
#     acc += ttt['2018'][n].count(1)
# print(acc/6)

# ttt = torch.load('results_sim3_R1_aug1')
# acc = 0
# for n in range(6):
#     #print(ttt['2018'][n].count(1))
#     acc += ttt['2018'][n].count(1)
# print(acc/6)
# # print(ttt)

# ttt = torch.load('results_sim4_R1_aug1')
# acc = 0
# for n in range(6):
#     #print(ttt['2018'][n].count(1))
#     acc += ttt['2018'][n].count(1)
# print(acc/6)

