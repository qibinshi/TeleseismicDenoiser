import glob
import torch
import matplotlib
import numpy as np
import pandas as pd
from obspy import read, UTCDateTime
from torch_tools import WaveformDataset, try_gpu
from torch.utils.data import DataLoader
from denoiser_util import plot_application
matplotlib.use('Agg')
import matplotlib.pyplot as plt

npts_trim = 1000
npts = 3000
preP = 145.0
preP = 45.0
dt = 0.1
time_after = npts * 0.1 - preP
devc = try_gpu(i=10)
sac_dir = '/Users/qibin/Work/RF-denoise/IA_SISI'
model_dir = 'Release_Middle_augmentation'
all_wave = np.zeros((0, npts, 3), dtype=np.double)
scaling = np.zeros((0, 1, 3), dtype=np.double)


sta_idx = []

meta = pd.DataFrame(columns=[
        "source_id",
        "source_latitude_deg",
        "source_longitude_deg",
        "source_depth_km",
        "source_magnitude",
        "station_latitude_deg",
        "station_longitude_deg",
        "trace_snr_before",
        "trace_snr_after",
        "distance",
        "azimuth"])


# stalist = glob.glob(sac_dir + '/*.?HZ')

stalist = glob.glob(sac_dir + '/*.?HZ')
print('Number of stations: ', len(stalist))
snr_before = np.ones((len(stalist), 3), dtype=np.double)
snr_after = np.ones((len(stalist), 3), dtype=np.double)
snr_ratio = np.ones((len(stalist), 3), dtype=np.double)
for j in range(len(stalist)):
    sta = stalist[j]
    try:
        st0 = read(sta[:-1]+'?')
        st = st0.copy()
    except:
        continue

    if len(st) >= 3:
        tp = UTCDateTime(st[2].stats.starttime - st[2].stats.sac.b + st[2].stats.sac.a)
        st.trim(tp - preP, tp + time_after)

        one_wave = np.zeros((npts, 3), dtype=np.double)
        for i in range(3):
            one_wave[:, i] = np.array(st[i].data)[0:npts]
        one_wave[np.isnan(one_wave)] = 0
        scale_mean = np.mean(one_wave, axis=0, keepdims=True)
        scale_stdv = np.std(one_wave, axis=0, keepdims=True) + 1e-12
        one_wave = (one_wave - scale_mean) / scale_stdv

        noise_amp = np.std(one_wave[100:int((preP-5)/dt), :], axis=0)
        pwave_amp = np.std(one_wave[int(preP/dt):int(preP/dt*2)-100, :], axis=0)
        snr_before[j, :] = 20*np.log10(np.divide(pwave_amp, noise_amp))

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
separated_noise = noise_output.numpy()

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

    one_wave = denoised_signal[j, :, :]
    noise_amp = np.std(one_wave[:, 100:int((preP - 5) / dt)], axis=1)
    pwave_amp = np.std(one_wave[:, int(preP/dt):int(preP/dt*2)-100], axis=1)
    snr_after[j, :] = 20*np.log10(np.divide(pwave_amp, noise_amp))
    snr_ratio[j, :] = np.divide(snr_after[j], snr_before[j])

    for i in range(3):
        st[i].data = denoised_signal[j, i, :] * scaling[j, 0, i]
        ch = st[i].stats.channel
        st[i].stats.starttime = UTCDateTime(st[i].stats.starttime - st[i].stats.sac.b + st[i].stats.sac.a - preP)
        st[i].write(sta[:-3] + ch + '.denoised', format='SAC')

    aa = noisy_signal[j, :, :npts_trim]
    bb = separated_noise[j, :, :npts_trim]
    cc = denoised_signal[j, :, :npts_trim]
    plot_application(aa, cc, bb, sta[-24:-4], directory=sac_dir, npts=npts_trim)


plt.close("all")
fig, ax = plt.subplots(1, 2, figsize=(8, 4))
# bins = np.logspace(np.log10(0.01),np.log10(1000),20)
bins = np.linspace(0, 50, 20)
bin1 = np.linspace(0, 20, 20)
ax[0].hist(snr_before.flatten(), bins=bins, density=True, histtype='step', color='r', rwidth=0.1, label='noisy')
ax[0].hist(snr_after.flatten(),  bins=bins, density=True, histtype='step', color='k', rwidth=0.1, label='denoised')
ax[1].hist(snr_ratio.flatten(),  bins=bin1, density=True, histtype='step', color='b', rwidth=0.1, label='Denoised-SNR/ noisy-SNR')

ax[0].set_xlabel('SNR', fontsize=14)
ax[0].set_ylabel('density', fontsize=14)
ax[1].set_xlabel('SNR ratio', fontsize=14)
ax[0].legend(loc=1)
ax[1].legend(loc=1)
# ax[1].set_xscale('log')
# ax[0].set_xscale('log')



plt.savefig(sac_dir + f'/statistics_of_SNR.pdf')




