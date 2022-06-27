"""
Save clean earthquake signal with STEAD noise

@author: Qibin Shi (qibins@uw.edu)
"""
import h5py
import obspy
import numpy as np
import pandas as pd
from source_util import trim_align
from numpy.random import default_rng
from utilities import randomization_noise, downsample_series

# %%
in_pts = 3000
sample_rate = 10
maxfreq = 2.0

datadir = '/mnt/DATA0/qibin_data/matfiles_for_denoiser/'
cleanwave_mat = datadir + 'Alldepths_snr25_2000_21_sample10_lpass2_P_preP_MP1.hdf5'
model_dataset = datadir + 'Alldepths_snr25_2000_21_sample10_lpass2_P_STEAD_MP1.hdf5'

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
noise0 = waveform0[amplitude_series < (4 * amplitude_median), :]
noise0 = noise0[np.newaxis, :(noise0.shape[0] // in_pts * in_pts), :]
if shuffle_phase:
    noise = randomization_noise(noise0, dt)
else:
    noise_POHA = noise0
noise_POHA = np.reshape(noise_POHA, (-1, in_pts, 3))
print(f'------- {len(noise_POHA)} POHA waveforms has been processed')

############## Load STEAD noise ###############
# %% Set the total number of noise records
N_events = 15000
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

############## %% Save together with earthquake waveforms
print('--- Reading quake waveform, it is quick')
with h5py.File(cleanwave_mat, 'r') as f:
    wv = f['pwave'][:]
X_sum = np.sum(np.square(wv), axis=1)
indX = np.where(X_sum == 0)[0]
wv = np.delete(wv, indX, 0)

N_traces = min(noise_all.shape[0], wv.shape[0])
print(f'----- {N_traces} traces totally')

quake_waves = np.zeros((N_traces, 30000, 3), dtype=np.double)
noise_waves = np.zeros((N_traces, in_pts, 3), dtype=np.double)
for i in range(N_traces):
    noise_one = noise_all[i, :, :]
    noise_one[np.isnan(noise_one)] = 0
    noise_one = (noise_one - np.mean(noise_one, axis=0)) / (np.std(noise_one, axis=0) + 1e-12)
    quake_waves[i, :, :] = wv[i, :, :]
    noise_waves[i, :, :] = noise_one

print(f'----- Saving ...')
with h5py.File(model_dataset, 'w') as f:
    f.create_dataset("pwave", data=quake_waves)
    f.create_dataset("noise", data=noise_waves)

print('Done!')