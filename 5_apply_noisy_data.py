"""
Apply the model to noise data
e.g. deep earthquakes

Qibin Shi uses Jiuxun Yin's plotting code
"""
import h5py
import torch
import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from numpy.random import default_rng
from torch.utils.data import DataLoader
from utilities import waveform_fft, mkdir
from torch_tools import Explained_Variance_score
from torch_tools import WaveformDataset, try_gpu, CCLoss
from sklearn.model_selection import train_test_split

matplotlib.rcParams.update({'font.size': 9})

# %%
dt = 0.1
npts = 3000
batch_size = 128
gpu_num = 1
devc = try_gpu(i=gpu_num)
datadir = '/mnt/DATA0/qibin_data/matfiles_for_denoiser/'
wave_mat = datadir + 'deep500km_SNRmax10_2000_21_sample10_lpass2_P.hdf5'
model_name = "Branch_Encoder_Decoder_LSTM"
model_dir = 'Freeze_Middle_augmentation'
comps = ['E', 'N', 'Z']
mkdir(model_dir + '/figures_apply')

with h5py.File(wave_mat, 'r') as f:
    X_train = f['allwv'][:]

test_data = WaveformDataset(X_train, X_train)

# %% Model
model = torch.load(model_dir + '/Branch_Encoder_Decoder_LSTM_Model.pth', map_location=devc)
test_iter = DataLoader(test_data, batch_size=batch_size, shuffle=False)
model.eval()

with torch.no_grad():
    # %% Shift and stack noise for the 1st batch
    data_iter = iter(test_iter)
    X0, y0 = data_iter.next()

    nbatch = X0.size(0)
    rng = default_rng(17)
    # start_pt = rng.choice(X0.size(2) - npts, nbatch)
    X = torch.zeros(nbatch, X0.size(1), npts, dtype=torch.float64)

    for i in np.arange(nbatch):
        # X[i] = X0[i, :, start_pt[i]:start_pt[i] + npts]
        quake_one = X0[i]

        scale_mean = torch.mean(quake_one, dim=1)
        scale_std = torch.std(quake_one, dim=1) + 1e-12

        for j in np.arange(X0.size(1)):
            quake_one[j] = torch.div(torch.sub(quake_one[j], scale_mean[j]), scale_std[j])
        X[i] = quake_one[:, 1500:1500+npts]
    noisy_input = X.to(devc)

    # %% prediction
    quake_denoised, noise_output = model(noisy_input)
    noisy_signal = noisy_input.cpu().numpy()
    separated_noise = noise_output.cpu().numpy()
    denoised_signal = quake_denoised.cpu().numpy()

# %% Plot time series
time = np.arange(0, npts) * dt
loss_fn = CCLoss()
ev_score = Explained_Variance_score()
num_sample = batch_size

for i_model in range(num_sample):
    print(i_model, 'time series')
    plt.close("all")
    fig, ax = plt.subplots(denoised_signal.shape[1], 3, sharex=True, sharey=True, num=1, figsize=(12, 5))

    for i in range(noisy_signal.shape[1]):
        scaling_factor = np.max(abs(noisy_signal[i_model, i, :]))
        ax[i, 0].plot(time, noisy_signal[i_model, i, :]/scaling_factor, '-k', label='Noisy signal', linewidth=1)
        ax[i, 0].plot(time, denoised_signal[i_model, i, :]/scaling_factor, '-r', label='Predicted signal', linewidth=1)
        ax[i, 1].plot(time, denoised_signal[i_model, i, :]/scaling_factor, '-r', label='Predicted signal', linewidth=1)
        ax[i, 2].plot(time, separated_noise[i_model, i, :]/scaling_factor, '-b', label='Predicted noise', linewidth=1)
     
    ## Title, label and axes
    ax[0, 0].set_title("Original signal")
    ax[0, 1].set_title("Denoised P-wave")
    ax[0, 2].set_title("Separated noise")

    for i in range(noisy_signal.shape[1]):
        ax[i, 0].set_ylabel(comps[i])
        for j in range(3):
            ax[i, j].xaxis.set_visible(False)
            ax[i, j].yaxis.set_ticks([])
            ax[i, j].spines['right'].set_visible(False)
            ax[i, j].spines['left'].set_visible(False)
            ax[i, j].spines['top'].set_visible(False)
            ax[i, j].spines['bottom'].set_visible(False)

            if i == noisy_signal.shape[1]-1:
                ax[i, j].xaxis.set_visible(True)
                ax[i, j].set_xlim(0, npts*dt)
                ax[i, j].spines['bottom'].set_visible(True)
                ax[i, j].set_xlabel('time (s)')

    plt.figure(1)
    plt.savefig(model_dir + f'/figures_apply/{model_name}_Prediction_waveform_model_{i_model}.pdf', bbox_inches='tight')

# %% Plot spectrum
for i_model in range(num_sample):
    print(i_model, 'spectrum')
    plt.close("all")
    fig, ax = plt.subplots(1, denoised_signal.shape[1], sharex=True, sharey=True, num=1, figsize=(12, 5))

    for i in range(noisy_signal.shape[1]):
        scaling_factor = np.max(abs(noisy_signal[i_model, i, :]))
        _, spect_noisy_signal = waveform_fft(noisy_signal[i_model, i, :]/scaling_factor, dt)
        _, spect_noise = waveform_fft(separated_noise[i_model, i, :] / scaling_factor, dt)
        freq, spect_denoised_signal = waveform_fft(denoised_signal[i_model, i, :]/scaling_factor, dt)

        ax[i].loglog(freq, spect_noisy_signal, '-k', label='raw signal', linewidth=0.5, alpha=1)
        ax[i].loglog(freq, spect_denoised_signal, '-r', label='separated earthquake', linewidth=0.5, alpha=1)
        ax[i].loglog(freq, spect_noise, '-', color='b', label='noise', linewidth=0.5, alpha=0.8)
        ax[i].set_xlabel('Frequency (Hz)', fontsize=14)
        ax[i].set_title(comps[i], fontsize=16)
        ax[i].grid(alpha=0.2)


    ax[0].set_ylabel('velocity spectra', fontsize=14)
    plt.figure(1)
    plt.savefig(model_dir + f'/figures_apply/{model_name}_spectrum_{i_model}.pdf', bbox_inches='tight')