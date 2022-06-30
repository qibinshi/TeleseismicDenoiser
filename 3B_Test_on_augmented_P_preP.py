"""
Test the model performance with earthquake
signal squeezed, shifted and noise stacked

author: Qibin Shi
"""
import os
import time
import h5py
import torch
import matplotlib
import numpy as np
from functools import partial
from multiprocessing import Pool
from matplotlib import pyplot as plt
from source_util import plot_testing
from numpy.random import default_rng
from torch.utils.data import DataLoader
from utilities import waveform_fft, mkdir
from torch_tools import WaveformDataset, try_gpu
from sklearn.model_selection import train_test_split


matplotlib.rcParams.update({'font.size': 12})

# %%
dt = 0.1
npts = 3000
pttp = 10000
batch_size = 100
gpu_num = 1
devc = try_gpu(i=gpu_num)
datadir = '/mnt/DATA0/qibin_data/matfiles_for_denoiser/'
wave_preP = datadir + 'Alldepths_snr25_2000_21_sample10_lpass2_P_preP_MP1.hdf5'
wave_stead = datadir + 'Alldepths_snr25_2000_21_sample10_lpass2_P_STEAD_MP1.hdf5'
model_name = "Branch_Encoder_Decoder_LSTM"
model_dir = 'Freeze_Middle_augmentation'
fig_dir = model_dir + '/figures_mpi'
mkdir(fig_dir)

# %% Read the pre-processed datasets
print("#" * 12 + " Loading P wave and pre-P noises " + "#" * 12)
with h5py.File(wave_preP, 'r') as f:
    X_train = f['pwave'][:]
    Y_train = f['noise'][:, (0 - npts):, :]

X_sum = np.sum(np.square(X_train), axis=1)
indX = np.where(X_sum == 0)[0]
X_train = np.delete(X_train, indX, 0)
Y_train = np.delete(Y_train, indX, 0)

print("#" * 12 + " Loading P wave and STEAD noises " + "#" * 12)
with h5py.File(wave_stead, 'r') as f:
    X_stead = f['pwave'][:]
    Y_stead = f['noise'][:, (0 - npts):, :]
X_train = np.append(X_train, X_stead, axis=0)
Y_train = np.append(Y_train, Y_stead, axis=0)

print("#" * 12 + " Normalizing P wave and noises " + "#" * 12)
X_train = (X_train - np.mean(X_train, axis=1, keepdims=True)) / (np.std(X_train, axis=1, keepdims=True) + 1e-12)
Y_train = (Y_train - np.mean(Y_train, axis=1, keepdims=True)) / (np.std(Y_train, axis=1, keepdims=True) + 1e-12)

with h5py.File(model_dir + f'/{model_name}_Dataset_split.hdf5', 'r') as f:
    train_size = f.attrs['train_size']
    test_size = f.attrs['test_size']
    rand_seed1 = f.attrs['rand_seed1']
    rand_seed2 = f.attrs['rand_seed2']

X_training,X_test,Y_training,Y_test=train_test_split(X_train,Y_train,train_size=train_size,random_state=rand_seed1)
X_validate,X_test,Y_validate,Y_test=train_test_split(X_test, Y_test,  test_size=test_size, random_state=rand_seed2)
test_data = WaveformDataset(X_test, Y_test)

# %% Model
print(">_<Loading model ...")
model = torch.load(model_dir + '/Branch_Encoder_Decoder_LSTM_Model.pth', map_location=devc)
test_iter = DataLoader(test_data, batch_size=batch_size, shuffle=False)
model.eval()

print("-_-Processing data ...")
with torch.no_grad():
    data_iter = iter(test_iter)
    X0, y0 = data_iter.next()

    # %% Augmentation: 1-squeeze 2-shift 3-stack
    nbatch = X0.size(0)
    rng = default_rng(17)
    rng_snr = default_rng(23)
    rng_sqz = default_rng(11)
    start_pt = rng.choice(npts - int(npts * 0.05), nbatch) + int(npts * 0.05)
    snr = 10 ** rng_snr.uniform(-1, 1, nbatch)
    sqz = rng_sqz.choice(2, nbatch) + 1
    pt1 = pttp - sqz * npts
    pt2 = pttp + sqz * npts

    quak2 = torch.zeros(nbatch, y0.size(1), npts * 2, dtype=torch.float64)
    quake = torch.zeros(y0.size(), dtype=torch.float64)
    stack = torch.zeros(y0.size(), dtype=torch.float64)

    for i in np.arange(nbatch):
        # %% squeeze earthquake signal
        quak2[i] = X0[i, :, pt1[i]:pt2[i]:sqz[i]]
        # %% shift earthquake signal
        tmp = quak2[i, :, start_pt[i]:start_pt[i] + npts]
        for j in np.arange(X0.size(1)):
            quake[i, j] = torch.div(torch.sub(tmp[j], torch.mean(tmp[j], dim=-1)),
                                    torch.std(tmp[j], dim=-1) + 1e-12) * snr[i]
        # %% stack signal and noise
        stack[i] = quake[i] + y0[i]
        # %% normalize
        scale_mean = torch.mean(stack[i], dim=1)
        scale_std = torch.std(stack[i], dim=1) + 1e-12
        for j in np.arange(X0.size(1)):
            stack[i, j] = torch.div(torch.sub(stack[i, j], scale_mean[j]), scale_std[j])
            quake[i, j] = torch.div(torch.sub(quake[i, j], scale_mean[j]), scale_std[j])

    noisy_input, quake_label = stack.to(devc), quake.to(devc)
    noise_label = noisy_input - quake_label

    # %% prediction
    since = time.time()
    print("o_oDenoising the data ...")
    quake_denoised, noise_output = model(noisy_input)
    elapseT = time.time() - since
    print("#@_@#All traces are denoised in < %.2f s > !" % elapseT)

    noisy_signal = noisy_input.cpu().numpy()
    clean_signal = quake_label.cpu().numpy()
    separated_noise = noise_output.cpu().numpy()
    denoised_signal = quake_denoised.cpu().numpy()
    true_noise = noise_label.cpu().numpy()

# %% Parallel Plotting
partial_func = partial(plot_testing, directory=fig_dir, dt=dt, npts=npts)
num_proc = min(os.cpu_count(), batch_size)
pool = Pool(processes=num_proc)
print("Total number of threads for plotting: ", num_proc)

result = pool.starmap(partial_func, zip(noisy_signal, denoised_signal, separated_noise, clean_signal, true_noise, np.arange(nbatch), sqz))

elapseT = time.time() - since
print("All are plotted. Time elapsed: %.2f s" % elapseT)

# %% get the scores for each thread
scores = np.zeros((0, 3, 4), dtype=np.double)
for i in range(batch_size):
    scores = np.append(scores, result[i], axis=0)
    print(i, 'th score added')

# %% Plot statistics of scores
plt.close("all")
fig, ax = plt.subplots(1, 2, figsize=(8, 4))
ax[0].hist(scores[:, :, 0].flatten(), bins=10, density=True, histtype='step', color='r', rwidth=0.1, label='EV_quake')
ax[0].hist(scores[:, :, 1].flatten(), bins=10, density=True, histtype='step', color='k', rwidth=0.1, label='EV_noise')
ax[0].set_xlabel('explained variance', fontsize=14)
ax[0].set_ylabel('density', fontsize=14)
ax[0].legend(loc=2)
ax[1].hist(scores[:, :, 2].flatten(), bins=10, density=True, histtype='step', color='r', rwidth=0.1, label='CC_quake')
ax[1].hist(scores[:, :, 3].flatten(), bins=10, density=True, histtype='step', color='k', rwidth=0.1, label='CC_noise')
ax[1].set_xlabel('cross-correlation coefficient', fontsize=14)
ax[1].legend(loc=2)

plt.savefig(fig_dir + f'/statistics_of_scores.pdf')