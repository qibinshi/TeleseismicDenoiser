import glob
import torch
import matplotlib
import numpy as np
from obspy import read, UTCDateTime
from torch_tools import WaveformDataset, try_gpu
from torch.utils.data import DataLoader
from denoiser_util import mkdir
matplotlib.use('Agg')
import matplotlib.pyplot as plt

mkdir('denoised_groun_vel_aceh_S')
#npts = 7500
npts = 3000
preP = 50.0
time_after = npts * 0.1 - preP
devc = try_gpu(i=10)
# sac_dir = 'Petrolia_tel'
# sac_dir = 'aceh_data_for_denoise'
sac_dir = 'groun_vel_aceh'
# sac_dir = 'calving3'
model_dir = 'Release_Middle_augmentation'
all_wave = np.zeros((0, npts, 3), dtype=np.double)
scaling = np.zeros((0, 1, 3), dtype=np.double)
sta_idx = []

stalist = glob.glob(sac_dir + '/*.BHZ')
print('Number of stations: ', len(stalist))
for j in range(len(stalist)):
    sta = stalist[j]
    try:
        st0 = read(sta[:-1]+'?')
        st = st0.copy()
    except:
        continue

    if len(st) >= 3:
        tp = UTCDateTime(st[2].stats.starttime - st[2].stats.sac.b + st[2].stats.sac.a)
        #tp = UTCDateTime(st[2].stats.starttime - st[2].stats.sac.b + st[2].stats.sac.t8)
        st.trim(tp - preP, tp + time_after)

        one_wave = np.zeros((npts, 3), dtype=np.double)
        for i in range(3):
            one_wave[:, i] = np.array(st[i].data)[0:npts]
        one_wave[np.isnan(one_wave)] = 0
        scale_mean = np.mean(one_wave, axis=0, keepdims=True)
        scale_stdv = np.std(one_wave, axis=0, keepdims=True) + 1e-12
        one_wave = (one_wave - scale_mean) / scale_stdv

        all_wave = np.append(all_wave, one_wave[np.newaxis, :, :], axis=0)
        scaling = np.append(scaling, scale_stdv[np.newaxis, :, :], axis=0)
        sta_idx.append(j)
    else:
        print('Failed to convert station: ', sta[:-4])

# %% Load Denoiser
model = torch.load(model_dir + '/Branch_Encoder_Decoder_LSTM_Model.pth', map_location=devc)
model = model.module.to(devc)
model.eval()

# %% after all stations are converted to numpy
batch_size = all_wave.shape[0]
test_data = WaveformDataset(all_wave, all_wave)
test_iter = DataLoader(test_data, batch_size=batch_size, shuffle=False)
data_iter = iter(test_iter)
X, y = data_iter.next()

# %% Prediction
with torch.no_grad():
    quake_denoised, noise_output = model(X)
noisy_signal = X.numpy()
denoised_signal = quake_denoised.numpy()

# %% rewrite data
print('Number of denoised stations: ', denoised_signal.shape[0])
for j in range(denoised_signal.shape[0]):
    idx = sta_idx[j]
    sta = stalist[idx]

    try:
        st0 = read(sta[:-1]+'?')
        st = st0.copy()
    except:
        continue

    for i in range(3):
        st[i].data = denoised_signal[j, i, :] * scaling[j, 0, i]
        nt = st[i].stats.network
        nm = st[i].stats.station
        lc = st[i].stats.location
        ch = st[i].stats.channel

        st[i].stats.starttime = UTCDateTime(st[i].stats.starttime - st[i].stats.sac.b + st[i].stats.sac.a - preP)
        #st[i].stats.starttime = UTCDateTime(st[i].stats.starttime - st[i].stats.sac.b + st[i].stats.sac.t8 - preP)

        st[i].write('denoised_groun_vel_aceh_S/'+nt+'.'+nm+'.'+lc+'.'+ch, format='SAC')




