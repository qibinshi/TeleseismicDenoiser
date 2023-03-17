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
import pandas as pd
from functools import partial
from multiprocessing import Pool
from torch.utils.data import DataLoader
from torch_tools import WaveformDataset, try_gpu
from scipy.integrate import cumulative_trapezoid
from denoiser_util import plot_application, mkdir

matplotlib.use('Agg')
matplotlib.rcParams.update({'font.size': 12})

# %%
dt = 0.1
npts = 3000  #P
npts_trim = 1000
# npts = 7500  #S
# npts_trim = 1000
devc = try_gpu(i=10)  # for parallel plotting, let's only use CPUs
datadir = '/fd1/QibinShi_data/matfiles_for_denoiser/'
csv_file = datadir + "metadata_M6_deep200km_SNRmax3_2000_21_sample10_lpass2_P_mpi_both_BH_HH.csv"
wave_mat = datadir + 'M6_deep200km_SNRmax3_2000_21_sample10_lpass2_P_mpi_both_BH_HH.hdf5'
model_dir = 'Release_Middle_augmentation'
fig_dir = model_dir + '/Apply_releaseWDN_M6_deep200km_SNRmax3_P_mp'
mkdir(fig_dir)

# %% Read the noisy waveforms of multiple events
with h5py.File(wave_mat, 'r') as f:
    X_train = f['pwave'][:]

# %% Load Denoiser
model = torch.load(model_dir + '/Branch_Encoder_Decoder_LSTM_Model.pth')
model = model.module.to(devc)
model.eval()

# %% Read the metadata
meta_all = pd.read_csv(csv_file, low_memory=False)

since = time.time()
# %% loop of events
for evid in meta_all.source_id.unique():

    ev_dir = fig_dir + '/quake_' + str(evid)
    mkdir(ev_dir)

    meta = meta_all[(meta_all.source_id == evid)]
    dist_az = meta[['distance', 'azimuth', 'trace_snr_db']].to_numpy()
    idx_list = meta.index.values
    evdp = meta.source_depth_km.unique()[0]
    evmg = meta.source_magnitude.unique()[0]
    batch_size = len(idx_list)

    # %% extract event data
    X_1 = X_train[idx_list]
    test_data = WaveformDataset(X_1, X_1)
    test_iter = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    data_iter = iter(test_iter)
    X0, y0 = data_iter.next()
    nbatch = X0.size(0)

    X = torch.zeros(nbatch, X0.size(1), npts, dtype=torch.float64)
    for i in np.arange(nbatch):
        quake_one = X0[i, :, 1500:1500 + npts]  # P wave
        # quake_one = X0[i, :, :]  # S wave

        scale_mean = torch.mean(quake_one, dim=1)
        scale_std = torch.std(quake_one, dim=1) + 1e-12
        for j in np.arange(X0.size(1)):
            quake_one[j] = torch.div(torch.sub(quake_one[j], scale_mean[j]), scale_std[j])
        X[i] = quake_one

    # %% prediction
    with torch.no_grad():
        quake_denoised, noise_output = model(X)

    noisy_signal = X.numpy()
    separated_noise = noise_output.numpy()
    denoised_signal = quake_denoised.numpy()

    ####################### Save the denoised traces to disk #########################
    with h5py.File(ev_dir + '/' + str(evid) + '_denoised.hdf5', 'w') as f:
        f.create_dataset("pwave", data=denoised_signal)
        f.create_dataset("noise", data=separated_noise)

    # %% integrate and trim
    timex = np.arange(0, npts_trim) * dt
    startpt = int((npts - npts_trim)/2)
    endpt = int((npts + npts_trim)/2)

    noisy_signal = cumulative_trapezoid(noisy_signal[:, :, startpt:endpt], timex, axis=-1, initial=0)
    separated_noise = cumulative_trapezoid(separated_noise[:, :, startpt:endpt], timex, axis=-1, initial=0)
    denoised_signal = cumulative_trapezoid(denoised_signal[:, :, startpt:endpt], timex, axis=-1, initial=0)

    noisy_signal = (noisy_signal - np.mean(noisy_signal, axis=-1, keepdims=True)) / (
                np.std(noisy_signal, axis=-1, keepdims=True) + 1e-12)
    separated_noise = (separated_noise - np.mean(separated_noise, axis=-1, keepdims=True)) / (
                np.std(separated_noise, axis=-1, keepdims=True) + 1e-12)
    denoised_signal = (denoised_signal - np.mean(denoised_signal, axis=-1, keepdims=True)) / (
                np.std(denoised_signal, axis=-1, keepdims=True) + 1e-12)

    ####################### Plot time & spec figures for stations in parallel #########################
    partial_func = partial(plot_application, directory=ev_dir, dt=dt, npts=npts_trim)
    num_proc = min(os.cpu_count(), batch_size)
    with Pool(processes=num_proc) as pool:
        pool.starmap(partial_func, zip(noisy_signal, denoised_signal, separated_noise, np.arange(nbatch)))

    print('Quake ID %s has been denosied, saved and plotted. Time elapsed: %.0f s' % (evid, time.time()-since))
