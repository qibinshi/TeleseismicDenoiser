"""
Test the model performance with data
shifted and noise stacked on the fly

Qibin Shi uses Jiuxun Yin's plotting code
"""
import h5py
import torch
import matplotlib
import numpy as np
from scipy.io import loadmat
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
wave_mat = './POHA_and_Ponly_2004_18_shallow_snr_25_sample10Hz_lowpass2Hz.mat'
model_name = "Branch_Encoder_Decoder_LSTM"
model_dir = 'Freeze_Middle_augmentation'
comps = ['E', 'N', 'Z']
mkdir(model_dir + '/figures')

# %% Read mat data
X_train = loadmat(wave_mat)["quake_waves"]
Y_train = loadmat(wave_mat)["noise_waves"]

with h5py.File(model_dir + f'/{model_name}_Dataset_split.hdf5', 'r') as f:
    train_size = f.attrs['train_size']
    test_size = f.attrs['test_size']
    rand_seed1 = f.attrs['rand_seed1']
    rand_seed2 = f.attrs['rand_seed2']

X_training,X_test,Y_training,Y_test=train_test_split(X_train,Y_train,train_size=train_size,random_state=rand_seed1)
X_validate,X_test,Y_validate,Y_test=train_test_split(X_test, Y_test,  test_size=test_size, random_state=rand_seed2)
test_data = WaveformDataset(X_test, Y_test)

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
    rng_snr = default_rng(23)
    start_pt = rng.choice(y0.size(2) - npts, nbatch)
    snr = 10 ** rng_snr.uniform(-1, 1, nbatch)

    stack = torch.zeros(X0.size(), dtype=torch.float64)
    quake = torch.zeros(X0.size(), dtype=torch.float64)
    X = torch.zeros(nbatch, X0.size(1), npts, dtype=torch.float64)
    y = torch.zeros(nbatch, y0.size(1), npts, dtype=torch.float64)

    for i in np.arange(nbatch):
        quake[i] = X0[i] * snr[i]
        stack[i] = quake[i] + y0[i]
        scale_mean = torch.mean(stack[i], dim=1)
        scale_std = torch.std(stack[i], dim=1) + 1e-12
        X[i] = stack[i, :, start_pt[i]:start_pt[i] + npts]
        y[i] = quake[i, :, start_pt[i]:start_pt[i] + npts]
        for j in np.arange(X0.size(1)):
            X[i, j] = torch.div(torch.sub(X[i, j], scale_mean[j]), scale_std[j])
            y[i, j] = torch.div(torch.sub(y[i, j], scale_mean[j]), scale_std[j])

    noisy_input, quake_label = X.to(devc), y.to(devc)

    # noisy_signal = torch.flip(noisy_signal, dims=[2])
    # clean_signal = torch.flip(clean_signal, dims=[2])

    # %% prediction
    quake_denoised, noise_output = model(noisy_input)
    noise_label = noisy_input - quake_label

    noisy_signal = noisy_input.cpu().numpy()
    clean_signal = quake_label.cpu().numpy()
    separated_noise = noise_output.cpu().numpy()
    denoised_signal = quake_denoised.cpu().numpy()
    true_noise = noise_label.cpu().numpy()

# %% Plot time series
time = np.arange(0, npts) * dt
loss_fn = CCLoss()
ev_score = Explained_Variance_score()
num_sample = batch_size
scores = np.zeros((num_sample, noisy_signal.shape[1], 4))
for i_model in range(num_sample):
    print(i_model, 'time series')
    plt.close("all")
    fig, ax = plt.subplots(denoised_signal.shape[1], 3, sharex=True, sharey=True, num=1, figsize=(12, 5))

    for i in range(noisy_signal.shape[1]):
        scaling_factor = np.max(abs(noisy_signal[i_model, i, :]))
        evs_earthquake = ev_score(quake_denoised[i_model, i, :], quake_label[i_model, i, :])
        evs_noise = ev_score(noise_output[i_model, i, :], noise_label[i_model, i, :])
        cc_quake = 1 - loss_fn(quake_denoised[i_model, i, :], quake_label[i_model, i, :])
        cc_noise = 1 - loss_fn(noise_output[i_model, i, :], noise_label[i_model, i, :])
        ax[i, 0].plot(time, noisy_signal[i_model, i, :]/scaling_factor, '-k', label='Noisy signal', linewidth=1)
        ax[i, 0].plot(time, clean_signal[i_model, i, :]/scaling_factor, '-r', label='True signal', linewidth=1)
        ax[i, 1].plot(time, clean_signal[i_model, i, :]/scaling_factor, '-r', label='True signal', linewidth=1)
        ax[i, 1].plot(time, denoised_signal[i_model, i, :]/scaling_factor, '-b', label='Predicted signal', linewidth=1)
        ax[i, 2].plot(time, true_noise[i_model, i, :] / scaling_factor, '-', color='gray', label='True noise', linewidth=1)
        ax[i, 2].plot(time, separated_noise[i_model, i, :] / scaling_factor, '-b',  label='Predicted noise', linewidth=1)
        ax[i, 1].text(0, 0.8, f'EV: {evs_earthquake:.2f}/ CC: {cc_quake:.2f}')
        ax[i, 2].text(0, 0.8, f'EV: {evs_noise:.2f}/ CC: {cc_noise:.2f}')
        scores[i_model, i, 0] = evs_earthquake
        scores[i_model, i, 1] = evs_noise
        scores[i_model, i, 2] = cc_quake
        scores[i_model, i, 3] = cc_noise
     
    ## Title, label and axes
    ax[0, 0].set_title("Original signal")
    ax[0, 1].set_title("Earthquake signal")
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

            if i==noisy_signal.shape[1]-1:
                ax[i, j].xaxis.set_visible(True)
                ax[i, j].set_xlim(0, npts*dt)
                # ax[i, j].set_xlim(250, 400)
                ax[i, j].spines['bottom'].set_visible(True)
                ax[i, j].set_xlabel('time (s)')

    plt.figure(1)
    plt.savefig(model_dir + f'/figures/{model_name}_Prediction_waveform_model_{i_model}.pdf', bbox_inches='tight')

# %% Plot spectrum
for i_model in range(num_sample):
    print(i_model, 'spectrum')
    plt.close("all")
    fig, ax = plt.subplots(2, denoised_signal.shape[1], sharex=True, sharey=True, num=1, figsize=(12, 5))

    for i in range(noisy_signal.shape[1]):
        scaling_factor = np.max(abs(noisy_signal[i_model, i, :]))
        _, spect_noisy_signal = waveform_fft(noisy_signal[i_model, i, :]/scaling_factor, dt)
        _, spect_clean_signal = waveform_fft(clean_signal[i_model, i, :]/scaling_factor, dt)
        _, spect_noise = waveform_fft(separated_noise[i_model, i, :] / scaling_factor, dt)
        _, spect_true_noise = waveform_fft(true_noise[i_model, i, :] / scaling_factor, dt)
        freq, spect_denoised_signal = waveform_fft(denoised_signal[i_model, i, :]/scaling_factor, dt)
        ax[0, i].loglog(freq, spect_noisy_signal, '-k', label='raw signal', linewidth=0.5, alpha=1)
        ax[0, i].loglog(freq, spect_clean_signal, '-r', label='true earthquake', linewidth=0.5, alpha=1)
        ax[0, i].loglog(freq, spect_denoised_signal, '-b', label='separated earthquake', linewidth=0.5, alpha=1)
        ax[1, i].loglog(freq, spect_noise, '-', color='b', label='noise', linewidth=0.5, alpha=0.8)
        ax[1, i].loglog(freq, spect_true_noise, 'r', label='orginal noise', linewidth=0.5, alpha=1)
        ax[1, i].set_xlabel('Frequency (Hz)', fontsize=14)
        ax[0, i].set_title(comps[i], fontsize=16)
        ax[1, i].grid(alpha=0.2)
        ax[0, i].grid(alpha=0.2)

    ax[0, 0].set_ylabel('velocity spectra', fontsize=14)
    ax[1, 0].set_ylabel('velocity spectra', fontsize=14)
    plt.figure(1)
    plt.savefig(model_dir + f'/figures/{model_name}_spectrum_{i_model}.pdf', bbox_inches='tight')

# %% Plot statistics of scores
plt.close("all")
fig, ax = plt.subplots(1, 2, sharex=False, sharey=False, figsize=(8, 4))
ax[0].hist(scores[:, :, 0].flatten(), bins=10, density=True, histtype='step', color='r', rwidth=0.1, label='EV_quake')
ax[0].hist(scores[:, :, 1].flatten(), bins=10, density=True, histtype='step', color='k', rwidth=0.1, label='EV_noise')
ax[0].set_xlabel('explained variance', fontsize=14)
ax[0].set_ylabel('density', fontsize=14)
ax[0].legend(loc=2)
ax[1].hist(scores[:, :, 2].flatten(), bins=10, density=True, histtype='step', color='r', rwidth=0.1, label='CC_quake')
ax[1].hist(scores[:, :, 3].flatten(), bins=10, density=True, histtype='step', color='k', rwidth=0.1, label='CC_noise')
ax[1].set_xlabel('cross-correlation coefficient', fontsize=14)
ax[1].legend(loc=2)

plt.figure(1)
plt.savefig(model_dir + f'/figures/statistics_of_scores.pdf', bbox_inches='tight')