"""
Apply the model to noise data
e.g. deep earthquakes

author: Qibin Shi
"""
import os
import time
import h5py
import torch
import matplotlib
import numpy as np
from utilities import mkdir
from functools import partial
from multiprocessing import Pool
from numpy.random import default_rng
from torch.utils.data import DataLoader
from source_util import plot_application
from torch_tools import WaveformDataset, try_gpu

matplotlib.rcParams.update({'font.size': 12})

# %%
dt = 0.1
npts = 3000
batch_size = 100
gpu_num = 1
devc = try_gpu(i=gpu_num)
datadir = '/mnt/DATA0/qibin_data/matfiles_for_denoiser/'
wave_mat = datadir + 'M5_deep500km_SNRmax3_2000_21_sample10_lpass2_P_mpi.hdf5'
model_dir = 'Freeze_Middle_augmentation'
fig_dir = model_dir + '/figures_apply_mpi'
mkdir(fig_dir)

# %% Read in the noisy data
print("+_+Reading data ...")
with h5py.File(wave_mat, 'r') as f:
    X_train = f['pwave'][:]
print("%.0f traces have been read." % len(X_train))

test_data = WaveformDataset(X_train, X_train)

# %% Model
print(">_<Loading model ...")
model = torch.load(model_dir + '/Branch_Encoder_Decoder_LSTM_Model.pth', map_location=devc)
test_iter = DataLoader(test_data, batch_size=batch_size, shuffle=False)
model.eval()

print("-_-Processing data ...")
with torch.no_grad():
    data_iter = iter(test_iter)
    X0, y0 = data_iter.next()

    nbatch = X0.size(0)
    rng = default_rng(17)
    # start_pt = rng.choice(X0.size(2) - npts, nbatch)
    X = torch.zeros(nbatch, X0.size(1), npts, dtype=torch.float64)

    for i in np.arange(nbatch):
        # X[i] = X0[i, :, start_pt[i]:start_pt[i] + npts]
        quake_one = X0[i, :, 1500:1500+npts]
        scale_mean = torch.mean(quake_one, dim=1)
        scale_std = torch.std(quake_one, dim=1) + 1e-12

        for j in np.arange(X0.size(1)):
            quake_one[j] = torch.div(torch.sub(quake_one[j], scale_mean[j]), scale_std[j])
        X[i] = quake_one
    noisy_input = X.to(devc)

    # %% prediction
    since = time.time()
    print("o_oDenoising the data ...")
    quake_denoised, noise_output = model(noisy_input)
    elapseT = time.time() - since
    print("#@_@#All traces are denoised in < %.2f s > !" % elapseT)

    noisy_signal = noisy_input.cpu().numpy()
    separated_noise = noise_output.cpu().numpy()
    denoised_signal = quake_denoised.cpu().numpy()

# %% Parallel Plotting
partial_func = partial(plot_application, directory=fig_dir, dt=dt, npts=npts)
num_proc = min(os.cpu_count(), batch_size)
pool = Pool(processes=num_proc)
print("Total number of threads for plotting: ", num_proc)

pool.starmap(partial_func, zip(noisy_signal, denoised_signal, separated_noise, np.arange(nbatch)))

elapseT = time.time() - since
print("All are plotted. Time elapsed: %.2f s" % elapseT)
