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
from denoiser_util import mkdir, denoise_cc_stack
matplotlib.use('Agg')
matplotlib.rcParams.update({'font.size': 24})

# %% CPU/ GPU
devc = try_gpu(i=10)

# %% data format
snr = 2
dt = 0.1
npts = 1500
npts_trim = 1200
start_pt = int(2500 - npts/2)

# %% directories
model_dir = 'Release_Middle_augmentation_S4Hz_150s_removeCoda_SoverCoda25'
datadir = '/fd1/QibinShi_data/matfiles_for_denoiser/'
fig_dir = model_dir + '/Apply_releaseWDN_M5.5-8_deep100km_SNR' + str(snr) + '_azimuthBined_fixedWindow'
mkdir(fig_dir)

print("#" * 12 + " Loading quake waveforms " + "#" * 12)
# %% Read data of M6
csv_file = datadir + "metadata_M6_deep100km_allSNR_S_rot.csv"
wave_mat = datadir + 'M6_deep100km_allSNR_S_rot.hdf5'
with h5py.File(wave_mat, 'r') as f:
    X_train = f['pwave'][:, start_pt:start_pt+npts, :]
meta_all = pd.read_csv(csv_file, low_memory=False)

# %% Read data of M5
csv_file = datadir + "metadata_M55_deep100km_allSNR_S_rot.csv"
wave_mat = datadir + 'M55_deep100km_allSNR_S_rot.hdf5'
with h5py.File(wave_mat, 'r') as f:
    X_train1 = f['pwave'][:, start_pt:start_pt+npts, :]
meta_all1 = pd.read_csv(csv_file, low_memory=False)


print("#" * 12 + " Merging data files " + "#" * 12)
X_train = np.append(X_train, X_train1, axis=0)
meta_all = pd.concat([meta_all, meta_all1], ignore_index=True)
evids = meta_all.source_id.unique()

print("#" * 12 + " Loading denoiser " + "#" * 12)
# %% Load Denoiser
model = torch.load(model_dir + '/Branch_Encoder_Decoder_LSTM_Model.pth')
# model = torch.load('Release_Middle_augmentation_P4Hz_150s/Branch_Encoder_Decoder_LSTM_Model.pth')
model = model.module.to(devc)
model.eval()

since = time.time()
################ Measure and plot in parallel #################
partial_func = partial(denoise_cc_stack, meta_all=meta_all, X_train=X_train, model=model, fig_dir=fig_dir, dt=dt, npts=npts, npts_trim=npts_trim, normalized_stack=True, min_snr=snr, cmp=0, tstar=1.2)
num_proc = min(os.cpu_count(), 10)
print('---- %d threads for plotting %d record sections\n' % (num_proc, len(evids)))
with Pool(processes=num_proc) as pool:
    result = pool.map(partial_func, evids)

####################### Save measurements  ###################
meta_result = pd.DataFrame(columns=[
    "source_id",
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
        "source_id": result[i][0],
        "source_magnitude": result[i][1],
        "source_depth_km": result[i][2],
        "duration_denoised": "%.3f" % result[i][3],
        "centroid_speed_denoised": "%.3f" % result[i][4],
        "centroid_direction_denoised": "%.3f" % result[i][5],
        "duration_noisy": "%.3f" % result[i][6],
        "centroid_speed_noisy": "%.3f" % result[i][7],
        "centroid_direction_noisy": "%.3f" % result[i][8],
        "Es_denoised": result[i][9],
        "Es_noisy": result[i][10],
        "falloff_denoised": result[i][11],
        "falloff_noisy": result[i][12],
        "corner_freq_denoised": result[i][13],
        "corner_freq_noisy": result[i][14],
        "num_station": result[i][15]}, index=[0])], ignore_index=True)

meta_result.to_csv(fig_dir + "/source_measurements.csv", sep=',', index=False)

print('---- All are plotted, using %.2f s\n' % (time.time() - since))
