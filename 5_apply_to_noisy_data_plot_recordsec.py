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
from torch_tools import try_gpu
from denoiser_util import mkdir, plot_record_section
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rcParams.update({'font.size': 24})

# %% CPU/ GPU
devc = try_gpu(i=10)

# %% data format
dt = 0.1
npts = 3000  # P
npts_trim = 1200
# npts = 7500  # S
# npts_trim = 1000

# %% directories
model_dir = 'Release_Middle_augmentation'
fig_dir = model_dir + '/Apply_releaseWDN_M5.5-8_deep100km_allSNR'
datadir = '/fd1/QibinShi_data/matfiles_for_denoiser/'
mkdir(fig_dir)

print("#" * 12 + " Loading quake waveforms " + "#" * 12)
# %% Read the raw data of all events
csv_file = datadir + "metadata_M6_deep100km_SNR20_P.csv"
wave_mat = datadir + 'M6_deep100km_SNR20_P.hdf5'
with h5py.File(wave_mat, 'r') as f:
    X_train = f['pwave'][:]
meta_all = pd.read_csv(csv_file, low_memory=False)

csv_file = datadir + "metadata_M55_deep100km_SNR20_P.csv"
wave_mat = datadir + 'M55_deep100km_SNR20_P.hdf5'
with h5py.File(wave_mat, 'r') as f:
    X_train1 = f['pwave'][:]
meta_all1 = pd.read_csv(csv_file, low_memory=False)

print("#" * 12 + " Merging data files " + "#" * 12)
X_train = np.append(X_train, X_train1, axis=0)
meta_all = pd.concat([meta_all, meta_all1], ignore_index=True)
evids = meta_all.source_id.unique()

print("#" * 12 + " Loading denoiser " + "#" * 12)
# %% Load Denoiser
model = torch.load(model_dir + '/Branch_Encoder_Decoder_LSTM_Model.pth')
model = model.module.to(devc)
model.eval()

since = time.time()
####################### Plot record sections #########################
partial_func = partial(plot_record_section, meta_all=meta_all, X_train=X_train, model=model, fig_dir=fig_dir, dt=dt, npts=npts, npts_trim=npts_trim, normalized_stack=True)
num_proc = min(os.cpu_count(), 10)
print('---- %d threads for plotting %d record sections\n' % (num_proc, len(evids)))
with Pool(processes=num_proc) as pool:
    result = pool.map(partial_func, evids)

####################### Saving duration estimates  ###################
meta_result = pd.DataFrame(columns=[
    "source_magnitude",
    "source_depth_km",
    "duration_denoised",
    "centroid_speed_denoised",
    "centroid_direction_denoised",
    "duration_noisy",
    "centroid_speed_noisy",
    "centroid_direction_noisy",
    "Es_denoised",
    "Es_noisy",
    "falloff_denoised",
    "falloff_noisy",
    "corner_freq_denoised",
    "corner_freq_noisy",
    "num_station"])

for i in range(len(evids)):
    meta_result = pd.concat([meta_result, pd.DataFrame(data={
        "source_magnitude": result[i][0],
        "source_depth_km": result[i][1],
        "duration_denoised": "%.3f" % result[i][2],
        "centroid_speed_denoised": "%.3f" % result[i][3],
        "centroid_direction_denoised": "%.3f" % result[i][4],
        "duration_noisy": "%.3f" % result[i][5],
        "centroid_speed_noisy": "%.3f" % result[i][6],
        "centroid_direction_noisy": "%.3f" % result[i][7],
        "Es_denoised": result[i][8],
        "Es_noisy": result[i][9],
        "falloff_denoised": result[i][10],
        "falloff_noisy": result[i][11],
        "corner_freq_denoised": result[i][12],
        "corner_freq_noisy": result[i][13],
        "num_station": result[i][14]}, index=[0])], ignore_index=True)

meta_result.to_csv(fig_dir + "/source_measurements.csv", sep=',', index=False)

print('---- All are plotted, using %.2f s\n' % (time.time() - since))
