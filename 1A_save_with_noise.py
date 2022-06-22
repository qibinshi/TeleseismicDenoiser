"""
Stack clean earthquake signal with POHA noise

@author: Qibin Shi (qibins@uw.edu)
"""
import h5py
import obspy
import numpy as np
import pandas as pd
from source_util import trim_align
from numpy.random import default_rng
from scipy.io import savemat, loadmat
from utilities import randomization_noise, downsample_series

# %%
in_pts = 6000
sample_rate = 10
maxfreq = 2.0

datadir = '/mnt/DATA0/qibin_data/matfiles_for_denoiser/'
cleanwave_mat = datadir + 'wave_Ponly_2004_18_alldepth_snr_25_sample10Hz_lowpass2Hz.hdf5'
model_dataset = datadir + 'STEAD_POHA_and_Ponly_2004_18_alldepth_snr_25_sample10Hz_lowpass2Hz.hdf5'

# %% POHA
shuffle_phase = False
waveform_mseed = '../WaveDecompNet-paper/work/continuous_waveforms/IU.POHA.00.20210731-20210901.mseed'

# %% STEAD
file_name = '/mnt/DATA0/qibin_data/STEAD/merge.hdf5'
csv_file  = '/mnt/DATA0/qibin_data/STEAD/merge.csv'

############## Load POHA noise ##############
print('--- Reading POHA waveform, it may take a few minutes ...')
tr = obspy.read(waveform_mseed)
tr.filter("lowpass", freq=maxfreq)
tr.resample(sample_rate)
tr.merge(fill_value=np.nan)
tr = trim_align(tr)
dt = tr[0].stats.delta
npts = tr[0].stats.npts-1
waveform0 = np.zeros((npts, 3))
for i in range(3):
    waveform0[:, i] = np.array(tr[i].data)[0:npts]

# %% Extract POHA noise based on the median amplitude
print(f'----- Extracting noise from POHA ...')
amplitude_series = np.sqrt(np.sum(waveform0 ** 2, axis=1))
amplitude_median = np.nanmedian(amplitude_series)
noise0 = waveform0[amplitude_series < (5 * amplitude_median), :]
noise0 = noise0[np.newaxis, :(noise0.shape[0] // in_pts * in_pts), :]
if shuffle_phase:
    noise = randomization_noise(noise0, dt)
else:
    noise_POHA = noise0
noise_POHA = np.reshape(noise_POHA, (-1, in_pts, 3))
print(f'------- {len(noise_POHA)} POHA waveforms has been processed')

############## Load STEAD noise ###############
# %% Set the total number of noise records
N_events = 12000
noise_seed = 87
df = pd.read_csv(csv_file, low_memory=False)
df_noise = df[(df.trace_category == 'noise')]

# %% Randomly select noise records
rng_noise = default_rng(noise_seed)
noise_index = rng_noise.choice(len(df_noise), N_events)
df_noise = df_noise.iloc[noise_index]
noise_list = df_noise['trace_name'].to_list()

# %% Read STEAD file
print('--- Reading STEAD waveform, it may take a few minutes ...')
dtfl = h5py.File(file_name, 'r')
print(f'----- Random selecting {N_events} sample noises from STEAD ...')

# %% Loop over list
time = np.arange(0, 6000) * 0.01
stead_noise = np.zeros((N_events, 600, 3))
for i, noise in enumerate(noise_list):

    noise_waveform = np.array(dtfl.get('data/' + str(noise)))

    # %% Down-sample and normalize the records
    time_new, noise_waveform, dt_new = downsample_series(time, noise_waveform, sample_rate)
    noise_waveform = (noise_waveform - np.mean(noise_waveform, axis=0)) / (np.std(noise_waveform, axis=0) + 1e-12)
    stead_noise[i] = noise_waveform

stead_noise = np.reshape(stead_noise, (-1, in_pts, 3))
print(f'------- {len(stead_noise)} STEAD waveforms has been processed')

############## Add together POHA and STEAD noises
noise_all = np.append(noise_POHA, stead_noise, axis=0)
noise_all = np.append(noise_all, noise_all, axis=0)
noise_all = np.append(noise_all, noise_all, axis=0)

print('--- Reading quake waveform, it is quick')
# %% Save together with earthquake waveforms
# wv = loadmat(cleanwave_mat)["allwv"]
with h5py.File(cleanwave_mat, 'r') as f:
    wv = f['allwv'][:]

N_traces = min(noise_all.shape[0], wv.shape[0])
print(f'----- {N_traces} traces totally')

# snr_seed = 111
# rng_snr = default_rng(snr_seed)
# snr = 10 ** rng_snr.uniform(-1, 1, N_traces)
# stack_waves = np.zeros((N_traces, in_pts, 3), dtype=np.double)

quake_waves = np.zeros((N_traces, in_pts, 3), dtype=np.double)
noise_waves = np.zeros((N_traces, in_pts, 3), dtype=np.double)
for i in range(N_traces):
    quake_one = wv[i, :, :]
    noise_one = noise_all[i, :, :]
    quake_one[np.isnan(quake_one)] = 0
    noise_one[np.isnan(noise_one)] = 0
    # quake_one = (quake_one - np.mean(quake_one, axis=0)) / (np.std(quake_one, axis=0) + 1e-12) * snr[i]
    quake_one = (quake_one - np.mean(quake_one, axis=0)) / (np.std(quake_one, axis=0) + 1e-12)
    noise_one = (noise_one - np.mean(noise_one, axis=0)) / (np.std(noise_one, axis=0) + 1e-12)
    # stack_one = quake_one + noise_one
    # scale_mean = np.mean(stack_one, axis=0)
    # scale_std  = np.std(stack_one, axis=0) + 1e-12
    # stack_waves[i, :, :] = (stack_one - scale_mean) / scale_std
    # quake_waves[i, :, :] = (quake_one - scale_mean) / scale_std
    quake_waves[i, :, :] = quake_one
    noise_waves[i, :, :] = noise_one

print(f'----- Saving ...')
# savemat(model_dataset, {"stack_waves":stack_waves, "quake_waves":quake_waves})
# savemat(model_dataset, {"quake_waves":quake_waves, "noise_waves":noise_waves})
with h5py.File(model_dataset, 'w') as f:
    f.create_dataset("quake_waves", data=quake_waves)
    f.create_dataset("noise_waves", data=noise_waves)

print('Done!')
