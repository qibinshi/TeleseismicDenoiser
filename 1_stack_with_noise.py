# %%
import os
import h5py
import obspy
import torch
import datetime
import numpy as np
import pandas as pd
from source_util import trim_align
from matplotlib import pyplot as plt
from numpy.random import default_rng
from torch.utils.data import DataLoader
from scipy.interpolate import interp1d
from scipy.fft import fft, fftfreq, ifft
from scipy.io import savemat, loadmat
from utilities import downsample_series, mkdir, randomization_noise
from torch_tools import WaveformDataset, try_gpu

# %%
in_pts = 2400
shuffle_phase = False
model_dataset = './data_stacked_M6_plus_POHA_2Hz.mat'
cleanwave_mat = './wave_double_include_S_snr_100_2Hz.mat'
waveform_mseed = '../WaveDecompNet-paper/work/continuous_waveforms/IU.POHA.00.20210731-20210901.mseed'

# %% Load continuous waveform
tr = obspy.read(waveform_mseed)
tr.filter("lowpass", freq=0.99)
tr.resample(2)
tr.merge(fill_value=np.nan)
tr = trim_align(tr)
dt = tr[0].stats.delta
npts = tr[0].stats.npts-1
waveform0 = np.zeros((npts, 3))
for i in range(3):
    waveform0[:, i] = np.array(tr[i].data)[0:npts]

# %% Extract 600 multiples of noise points by median amplitude
amplitude_series = np.sqrt(np.sum(waveform0 ** 2, axis=1))
amplitude_median = np.nanmedian(amplitude_series)
noise0 = waveform0[amplitude_series < (6 * amplitude_median), :]
noise0 = noise0[np.newaxis, :(noise0.shape[0] // in_pts * in_pts), :]
if shuffle_phase:
    noise = randomization_noise(noise0)
else:
    noise = noise0
noise = np.reshape(noise, (-1, in_pts, 3))
noise = np.append(noise, noise, axis=0)

# %% Stack with M6 teleseismic waveforms
wv = loadmat(cleanwave_mat)["allwv"]
N_traces = min(noise.shape[0], wv.shape[0])
print(N_traces, "traces totally")
snr_seed = 111
rng_snr = default_rng(snr_seed)
snr = 10 ** rng_snr.uniform(-1, 0.5, N_traces)
stack_waves = np.zeros((N_traces, in_pts, 3), dtype=np.double)
quake_waves = np.zeros((N_traces, in_pts, 3), dtype=np.double)
for i in range(N_traces):
    quake_one = wv[i, :, :]
    noise_one = noise[i, :, :]
    quake_one[np.isnan(quake_one)] = 0
    noise_one[np.isnan(noise_one)] = 0
    quake_one = (quake_one - np.mean(quake_one, axis=0)) / (np.std(quake_one, axis=0) + 1e-12) * snr[i]
    noise_one = (noise_one - np.mean(noise_one, axis=0)) / (np.std(noise_one, axis=0) + 1e-12)
    stack_one = quake_one + noise_one
    scale_mean = np.mean(stack_one, axis=0)
    scale_std  = np.std(stack_one, axis=0) + 1e-12
    stack_waves[i, :, :] = (stack_one - scale_mean) / scale_std
    quake_waves[i, :, :] = (quake_one - scale_mean) / scale_std

savemat(model_dataset, {"stack_waves":stack_waves, "quake_waves":quake_waves})

