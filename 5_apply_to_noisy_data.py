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
from numpy.random import default_rng
from torch.utils.data import DataLoader
from torch_tools import WaveformDataset, try_gpu
from scipy.integrate import cumulative_trapezoid
from denoiser_util import plot_application, mkdir, write_progress, shift2maxcc, dura_amp, dura_cc
matplotlib.use('Agg')
import matplotlib.pyplot as plt

matplotlib.rcParams.update({'font.size': 18})

# %%
dt = 0.1
npts = 3000
npts_trim = 500
devc = try_gpu(i=0)
datadir = '/mnt/DATA0/qibin_data/matfiles_for_denoiser/'
csv_file = datadir + "metadata_M6_deep200km_SNRmax3_2000_21_sample10_lpass2_P_mpi_both_BH_HH.csv"
wave_mat = datadir + 'M6_deep200km_SNRmax3_2000_21_sample10_lpass2_P_mpi_both_BH_HH.hdf5'
# model_dir = 'Freeze_Middle_augmentation'
model_dir = 'Release_Middle_augmentation'
fig_dir = model_dir + '/Apply_releaseWDN_M6_deep200km_SNRmax3'
log = fig_dir + '/progress.txt'
mkdir(fig_dir)

# %% Read the noisy waveforms of multiple events
write_progress(log, 'Reading the entire data file ...\n')
with h5py.File(wave_mat, 'r') as f:
    X_train = f['pwave'][:]
write_progress(log, '%.0f traces have been read.\n' % len(X_train))

# %% Load Denoiser
write_progress(log, 'Loading Denoiser model ...\n')
model = torch.load(model_dir + '/Branch_Encoder_Decoder_LSTM_Model.pth')
model = model.module.to(devc)
model.eval()

# %% Read the metadata
write_progress(log, 'Reading all metadata ...\n')
meta_all = pd.read_csv(csv_file, low_memory=False)

# %% loop of events
for evid in meta_all.source_id.unique():
    write_progress(log, '-- Donoising event %d ...\n' % evid)
    ev_dir = fig_dir + '/' + str(evid)
    mkdir(ev_dir)

    meta = meta_all[(meta_all.source_id == evid)]
    dist_az = meta[['distance', 'azimuth', 'trace_snr_db']].to_numpy()
    idx_list = meta.index.values
    evdp = meta.source_depth_km.unique()[0]
    evmg = meta.source_magnitude.unique()[0]
    batch_size = len(idx_list)
    # %% extract data for this event
    X_1 = X_train[idx_list]
    test_data = WaveformDataset(X_1, X_1)
    test_iter = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    write_progress(log, '---- Processing event data ...\n')

    data_iter = iter(test_iter)
    X0, y0 = data_iter.next()

    nbatch = X0.size(0)
    rng = default_rng(17)
    # start_pt = rng.choice(X0.size(2) - npts, nbatch)
    X = torch.zeros(nbatch, X0.size(1), npts, dtype=torch.float64)

    for i in np.arange(nbatch):
        # X[i] = X0[i, :, start_pt[i]:start_pt[i] + npts]
        quake_one = X0[i, :, 1500:1500+npts]
        scale_mean = torch.mean(quake_one, dim=1)
        scale_std = torch.std(quake_one, dim=1) + 1e-12

        for j in np.arange(X0.size(1)):
            quake_one[j] = torch.div(torch.sub(quake_one[j], scale_mean[j]), scale_std[j])
        X[i] = quake_one
    noisy_input = X.to(devc)

    # %% prediction
    since = time.time()
    write_progress(log, '---- Denoising the data ...\n')
    with torch.no_grad():
        quake_denoised, noise_output = model(noisy_input)
    elapseT = time.time() - since
    write_progress(log, '---- All traces are denoised in < %.2f s > !\n' % elapseT)

    noisy_signal = noisy_input.cpu().numpy()
    separated_noise = noise_output.cpu().numpy()
    denoised_signal = quake_denoised.cpu().numpy()

    # write_progress(log, '---- Saving denoised waveforms to hdf5 ...\n')
    # with h5py.File(ev_dir + '/' + str(evid) + '_quake_and_noise.hdf5', 'w') as f:
    #     f.create_dataset("pwave", data=denoised_signal)
    #     f.create_dataset("noise", data=separated_noise)
    # write_progress(log, '---- Saved!\n')

    ###################### Plotting figures #########################
    # %% integrate and trim
    timex = np.arange(0, npts_trim) * dt
    startpt = int((npts - npts_trim)/2)
    endpt = int((npts + npts_trim)/2)

    noisy_signal = cumulative_trapezoid(noisy_signal[:, :, startpt:endpt], timex, axis=-1, initial=0)
    separated_noise = cumulative_trapezoid(separated_noise[:, :, startpt:endpt], timex, axis=-1, initial=0)
    denoised_signal = cumulative_trapezoid(denoised_signal[:, :, startpt:endpt], timex, axis=-1, initial=0)

    noisy_signal = (noisy_signal - np.mean(noisy_signal, axis=-1, keepdims=True)) / (
                np.std(noisy_signal, axis=-1, keepdims=True) + 1e-12)
    separated_noise = (separated_noise - np.mean(separated_noise, axis=-1, keepdims=True)) / (
                np.std(separated_noise, axis=-1, keepdims=True) + 1e-12)
    denoised_signal = (denoised_signal - np.mean(denoised_signal, axis=-1, keepdims=True)) / (
                np.std(denoised_signal, axis=-1, keepdims=True) + 1e-12)

    # %% Plot record section
    write_progress(log, '---- Plotting record section ...\n')
    plt.close('all')
    color = ['k', 'k']
    cc = np.zeros((batch_size, 2))
    flip = np.zeros((batch_size, 2), dtype=np.int32)
    ratio = np.zeros((batch_size, 2))
    shift = np.zeros((batch_size, 2), dtype=np.int32)
    pt_st = np.zeros((batch_size, 2), dtype=np.int32)
    pt_en = np.zeros((batch_size, 2), dtype=np.int32)
    id_az_min = np.argmin(dist_az[:, 1])
    if dist_az[id_az_min, 2] > 1.5:
        id_snr_max = id_az_min
    else:
        id_snr_max = np.argmax(dist_az[:, 2])
    ref_noisy = noisy_signal[id_snr_max, 2, :]
    ref_clean = denoised_signal[id_snr_max, 2, :]
    fig, ax = plt.subplots(4, 2, figsize=(30, 40), constrained_layout=True)
    ax[2, 1] = plt.subplot(426, projection='polar')
    ax[2, 0] = plt.subplot(425, projection='polar')
    ax[3, 1] = plt.subplot(428, projection='polar')
    ax[3, 0] = plt.subplot(427, projection='polar')
    amp_azi = 12
    amp_dis = 2

    for i in range(batch_size):

        # %% Re-align traces and mark amplitude-based duration
        tr_noisy, shift[i, 0], flip[i, 0] = shift2maxcc(ref_noisy, noisy_signal[i, 2, :], maxshift=30, flip_thre=-0.3)
        tr_clean, shift[i, 1], flip[i, 1] = shift2maxcc(ref_clean, denoised_signal[i, 2, :], maxshift=30, flip_thre=-0.3)

        for k in range(2):
            if flip[i, k]:
                color[k] = 'b'  # plot the flipped traces in blue
            else:
                color[k] = 'k'

        ax[0, 0].plot(timex + shift[i, 0] * dt, tr_noisy * amp_azi + dist_az[i, 1], color[0], linewidth=1)
        ax[0, 1].plot(timex + shift[i, 1] * dt, tr_clean * amp_azi + dist_az[i, 1], color[1], linewidth=1)

        # %% 1) STA/LTA
        pt_st[i, 1], pt_en[i, 1], sta_lta1, sta_lta2, sta, lta = dura_amp(denoised_signal[i, 2, :])
        pt_st[i, 0], pt_en[i, 0], sta_lta1, sta_lta2, sta, lta = dura_amp(noisy_signal[i, 2, :])
        for k in range(2):
            pt1 = (pt_st[i, k] + shift[i, k]) * dt
            pt2 = (pt_en[i, k] + shift[i, k]) * dt
            ax[0, k].plot(pt1, tr_clean[pt_st[i, k]]*amp_azi+dist_az[i, 1], 'or', pt2, tr_clean[pt_en[i, k]]*amp_azi+dist_az[i, 1], 'og')

        ax[3, 1].plot(dist_az[i, 1] / 180 * np.pi, shift[i, 1] * dt, marker='o', mfc=color[1], mec=color[1], ms=10)
        ax[3, 0].plot(dist_az[i, 1] / 180 * np.pi, shift[i, 0] * dt, marker='o', mfc=color[0], mec=color[0], ms=10)

        # %% 2) stretching ratio
        time_clean, wave_clean, ratio[i, 1], cc[i, 1], flip[i, 1] = dura_cc(ref_clean, denoised_signal[i, 2, :], timex, maxshift=30, max_ratio=2)
        time_noisy, wave_noisy, ratio[i, 0], cc[i, 0], flip[i, 0] = dura_cc(ref_noisy, noisy_signal[i, 2, :], timex, maxshift=30, max_ratio=2)

        for k in range(2):
            if flip[i, k]:
                color[k] = 'b'  # plot the flipped traces in blue
            else:
                color[k] = 'k'

        ax[1, 1].plot(time_clean, wave_clean * amp_azi + dist_az[i, 1], color=color[1], linestyle='-', linewidth=1)
        if np.fabs(cc[i, 1]) > 0.5:
            ax[2, 1].plot(dist_az[i, 1]/180*np.pi, ratio[i, 1], marker='o', mfc=color[1], mec=color[1], ms=cc[i, 1] * 20)

        ax[1, 0].plot(time_noisy, wave_noisy * amp_azi + dist_az[i, 1], color=color[0], linestyle='-', linewidth=1)
        if np.fabs(cc[i, 0]) > 0.5:
            ax[2, 0].plot(dist_az[i, 1]/180*np.pi, ratio[i, 0], marker='o', mfc=color[0], mec=color[0], ms=cc[i, 0] * 20)

    ax[2, 1].plot(dist_az[id_snr_max, 1]/180*np.pi, ratio[id_snr_max, 1], marker='o', mfc='y', mec='y', ms=cc[id_snr_max, 1] * 20)
    ax[2, 0].plot(dist_az[id_snr_max, 1]/180*np.pi, ratio[id_snr_max, 0], marker='o', mfc='y', mec='y', ms=cc[id_snr_max, 0] * 20)

    for j in range(2):
        ax[2, j].set_ylim(0, 2.2)
        ax[3, j].set_ylim(-3.3, 3.3)
        ax[j, 0].set_ylabel('azimuth', fontsize=20)
        for k in range(2):
            ax[j, k].set_xlim(timex[0], timex[-1])
            ax[j, k].set_ylim(np.min(dist_az[:, 1]) - (50 * 1 + 10), np.max(dist_az[:, 1]) + (50 * 1 + 10))

    ax[1, 1].plot(timex, ref_clean * amp_azi + dist_az[id_snr_max, 1], '--y', dashes=(5, 7), linewidth=4)
    ax[1, 0].plot(timex, ref_noisy * amp_azi + dist_az[id_snr_max, 1], '--y', dashes=(5, 7), linewidth=4)
    ax[0, 0].set_title(f'Noisy Event {evid} depth={evdp} km / M={evmg}', fontsize=20)
    ax[0, 1].set_title('Denoised P waves', fontsize=20)


    plt.savefig(fig_dir + '/' + str(evid) + '_record_section.png')

    ####################### %% Plot each station as a figure in parallel #########################
    # partial_func = partial(plot_application, directory=ev_dir, dt=dt, npts=npts_trim)
    # num_proc = min(os.cpu_count(), batch_size)
    # pool = Pool(processes=num_proc)
    # write_progress(log, '---- %d threads for plotting event ID %d \n' % (num_proc, evid))
    #
    # pool.starmap(partial_func, zip(noisy_signal, denoised_signal, separated_noise, np.arange(nbatch)))
    ####################### Plot each station as a figure in parallel %% #########################

    elapseT = time.time() - since
    write_progress(log, '---- All are plotted. Time for this event: %.2f s\n' % elapseT)
