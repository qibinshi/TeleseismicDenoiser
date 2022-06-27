"""
Transfer learning for WaveDecompNet
with extended 2-branch U-net layers,
Data being augmented on the fly.

@auther: Qibin Shi (qibins@uw.edu)
"""
import h5py
import torch
import random
import numpy as np
from matplotlib import pyplot as plt
from numpy.random import default_rng
from torch.utils.data import DataLoader
from utilities import mkdir, write_progress
from sklearn.model_selection import train_test_split
from torch_tools import WaveformDataset, try_gpu, CCMSELoss, MSELossOnly
from torch_tools import training_loop_branches_augmentation
from autoencoder_1D_models_torch import T_model, W_model

# %%
weighted_loss = False
gpu_num = 0
npts = 3000
devc = try_gpu(i=gpu_num)
bottleneck_name = 'LSTM'
model_dir = 'Freeze_Middle_augmentation'
model_structure = "Branch_Encoder_Decoder"
progress_file = model_dir + '/Running_progress.txt'
model_name = model_structure + "_" + bottleneck_name
datadir = '/mnt/DATA0/qibin_data/matfiles_for_denoiser/'
wave_mat = datadir + 'Alldepths_snr25_2000_21_sample10_lpass2_P_preP_MP.hdf5'
mkdir(model_dir)

# %% Read the pre-processed datasets
print("#" * 12 + " Loading data " + "#" * 12)

with h5py.File(wave_mat, 'r') as f:
    X_all = f['pwave'][:]
    Y_all = f['noise'][:]
X_train = X_all[:, 7000:13000, :]
Y_train = Y_all[:, 7000:13000, :]

X_train = (X_train - np.mean(X_train, axis=1, keepdims=True)) / (np.std(X_train, axis=1, keepdims=True) + 1e-12)
Y_train = (Y_train - np.mean(Y_train, axis=1, keepdims=True)) / (np.std(Y_train, axis=1, keepdims=True) + 1e-12)



X = np.sum(np.square(X_train), axis=1)
Y = np.sum(np.square(Y_train), axis=1)

ind1 = np.where(X == 0)[0]
ind2 = np.where(X == 0)[1]
ind3 = np.where(Y == 0)[0]
ind4 = np.where(Y == 0)[1]
print(ind1)
print(ind2)
print(ind3)
print(ind4)
