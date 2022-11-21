"""
Test the model performance with earthquake
signal squeezed, shifted and noise stacked

author: Qibin Shi
"""
import os
import time
import h5py
import torch
import argparse
import matplotlib
import numpy as np
from functools import partial
from multiprocessing import Pool
from numpy.random import default_rng
from torch.utils.data import DataLoader
from denoiser_util import mkdir, waveform_fft
from torch_tools import WaveformDataset, try_gpu, Explained_Variance_score, CCLoss
from sklearn.model_selection import train_test_split
matplotlib.use('Agg')
from matplotlib import pyplot as plt

matplotlib.rcParams.update({'font.size': 12})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--phase', default='P', type=str, help='earthquake phase')
    parser.add_argument('-g', '--gpu', default=9, type=int, help='main gpu: 0-3/ no gpu: 9')
    parser.add_argument('-T', '--transfer', default=1, type=int, help='transfer learning')
    args = parser.parse_args()

    phase = args.phase
    if phase == 'P':
        mid_pt = 25000
        npts = 1500
        strmax = 6
    else:
        mid_pt = 20000
        npts = 1500
        strmax = 4

    dt = 0.1
    frac = 0.45  # smallest window start
    batch_size = 100
    devc = try_gpu(i=args.gpu)
    model_name = "Branch_Encoder_Decoder_LSTM"
    model_dir = 'Release_Middle_augmentation_S4Hz_150s_removeCoda_SoverCoda25_1980_2021'
    data_dir = '/fd1/QibinShi_data/matfiles_for_denoiser/'
    wave_raw = data_dir + 'Ssnr25_lp4_1980-2021.hdf5'
    fig_dir = model_dir + '/figures'
    mkdir(fig_dir)

    # %% Read the pre-processed datasets
    print("#" * 12 + " Loading quake signals and pre-P noises " + "#" * 12)
    with h5py.File(wave_raw, 'r') as f:
        X_train = f['quake'][:]
        Y_train = f['noise'][:, (0 - npts):, :]

    X_sum = np.sum(np.square(X_train), axis=1)
    ind_X = np.where(X_sum == 0)[0]
    X_train = np.delete(X_train, ind_X, 0)
    Y_train = np.delete(Y_train, ind_X, 0)

    print("#" * 12 + " Normalizing signal and noises " + "#" * 12)
    X_train = (X_train - np.mean(X_train, axis=1, keepdims=True)) / (np.std(X_train, axis=1, keepdims=True) + 1e-12)
    Y_train = (Y_train - np.mean(Y_train, axis=1, keepdims=True)) / (np.std(Y_train, axis=1, keepdims=True) + 1e-12)

    with h5py.File(model_dir + f'/{model_name}_Dataset_split.hdf5', 'r') as f:
        train_size = f.attrs['train_size']
        test_size = f.attrs['test_size']
        rand_seed1 = f.attrs['rand_seed1']
        rand_seed2 = f.attrs['rand_seed2']

    X_training, X_test, Y_training, Y_test=train_test_split(X_train,
                                                            Y_train,
                                                            train_size=train_size,
                                                            random_state=rand_seed1)
    X_validate, X_test, Y_validate, Y_test=train_test_split(X_test,
                                                            Y_test,
                                                            test_size=test_size,
                                                            random_state=rand_seed2)
    test_data = WaveformDataset(X_test, Y_test)
    test_iter = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    # %% Model
    print(">_<Loading model ...")
    model = torch.load(model_dir + '/Branch_Encoder_Decoder_LSTM_Model.pth')
    model = model.module.to(devc)
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
        start_pt = rng.choice(npts - int(npts * frac * 2), nbatch) + int(npts * frac)
        snr = 10 ** rng_snr.uniform(-0.3, 0.5, nbatch)
        sqz = rng_sqz.choice(strmax, nbatch) + 1
        pt1 = mid_pt - sqz * npts
        pt2 = mid_pt + sqz * npts

        quak2 = torch.zeros(nbatch, y0.size(1), npts * 2, dtype=torch.float64)
        quake = torch.zeros(y0.size(), dtype=torch.float64)
        stack = torch.zeros(y0.size(), dtype=torch.float64)

        for i in np.arange(nbatch):
            # %% squeeze earthquake signal
            print(pt1[i],pt2[i],sqz[i])
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

    result = pool.starmap(partial_func, zip(noisy_signal,
                                            denoised_signal,
                                            separated_noise,
                                            clean_signal,
                                            true_noise,
                                            np.arange(nbatch),
                                            sqz))

    elapseT = time.time() - since
    print("All are plotted. Time elapsed: %.2f s" % elapseT)

    # %% get the scores for each thread
    # scores = np.zeros((0, 3, 4), dtype=np.double)
    #################################################
    scores = np.zeros((0, 3, 6), dtype=np.double)   #
    #################################################
    for i in range(batch_size):
        scores = np.append(scores, result[i], axis=0)
        print(i, 'th score added')

    # %% Plot statistics of scores
    plt.close("all")
    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    ax[0].hist(scores[:, :, 0].flatten(), bins=10, density=True, histtype='step', color='r', rwidth=0.1, label='EV_quake')
    ax[0].hist(scores[:, :, 1].flatten(), bins=10, density=True, histtype='step', color='k', rwidth=0.1, label='EV_noise')
    ###########################################################
    ax[0].hist(scores[:, :, 4].flatten(), bins=10, density=True, histtype='step', color='g', rwidth=0.1, label='EV_resquake')
    ###########################################################
    ax[0].set_xlabel('explained variance', fontsize=14)
    ax[0].set_ylabel('density', fontsize=14)
    ax[0].legend(loc=2)
    ax[1].hist(scores[:, :, 2].flatten(), bins=10, density=True, histtype='step', color='r', rwidth=0.1, label='CC_quake')
    ax[1].hist(scores[:, :, 3].flatten(), bins=10, density=True, histtype='step', color='k', rwidth=0.1, label='CC_noise')
    ###########################################################
    ax[1].hist(scores[:, :, 5].flatten(), bins=10, density=True, histtype='step', color='g', rwidth=0.1, label='CC_resquake')
    ###########################################################
    ax[1].set_xlabel('cross-correlation coefficient', fontsize=14)
    ax[1].legend(loc=2)

    plt.savefig(fig_dir + f'/statistics_of_scores.pdf')


def plot_testing(noisy_signal, denoised_signal, separated_noise, clean_signal, true_noise, idx, sqz, directory=None, dt=0.1, npts=None):
    quake_denoised = torch.from_numpy(denoised_signal)
    quake_label = torch.from_numpy(clean_signal)
    noise_output = torch.from_numpy(separated_noise)
    noise_label = torch.from_numpy(true_noise)
    #######################################################
    residual_signal = noisy_signal - separated_noise      #
    quake_residual = torch.from_numpy(residual_signal)    #
    #######################################################
    ev_score = Explained_Variance_score()
    loss_fn = CCLoss()
    comps = ['T', 'R', 'Z']
    # scores = np.zeros((1, 3, 4))
    ##################################
    scores = np.zeros((1, 3, 6))     #
    ##################################
    time = np.arange(0, npts) * dt
    gs_kw = dict(height_ratios=[1, 1, 1, 2, 2])

    plt.close("all")
    fig, ax = plt.subplots(5, 3, gridspec_kw=gs_kw, figsize=(12, 12), constrained_layout=True)

    for i in range(3):
        scaling_factor = np.max(abs(noisy_signal[i, :]))
        _, spect_noisy_signal = waveform_fft(noisy_signal[i, :] / scaling_factor, dt)
        _, spect_clean_signal = waveform_fft(clean_signal[i, :] / scaling_factor, dt)
        _, spect_true_noise = waveform_fft(true_noise[i, :] / scaling_factor, dt)
        _, spect_noise = waveform_fft(separated_noise[i, :] / scaling_factor, dt)
        freq, spect_denoised_signal = waveform_fft(denoised_signal[i, :] / scaling_factor, dt)
        ###############################################################################
        _, spect_resquake = waveform_fft(residual_signal[i, :] / scaling_factor, dt)  #
        ###############################################################################
        evs_earthquake = ev_score(quake_denoised[i, :], quake_label[i, :])
        evs_noise = ev_score(noise_output[i, :], noise_label[i, :])
        cc_quake = 1 - loss_fn(quake_denoised[i, :], quake_label[i, :])
        cc_noise = 1 - loss_fn(noise_output[i, :], noise_label[i, :])

        ############################################################################
        evs_resquake = ev_score(quake_residual[i, :], quake_label[i, :])           #
        cc_resquake = 1 - loss_fn(quake_residual[i, :], quake_label[i, :])         #
        ############################################################################

        scores[0, i, 0] = evs_earthquake
        scores[0, i, 1] = evs_noise
        scores[0, i, 2] = cc_quake
        scores[0, i, 3] = cc_noise
        ##################################
        scores[0, i, 4] = evs_resquake   #
        scores[0, i, 5] = cc_resquake    #
        ##################################

        ax[i, 0].plot(time, noisy_signal[i, :] / scaling_factor, '-k', label='Noisy signal', linewidth=1)
        ax[i, 0].plot(time, clean_signal[i, :] / scaling_factor, '-r', label='True signal', linewidth=1)
        ax[i, 1].plot(time, clean_signal[i, :] / scaling_factor, '-r', label='True signal', linewidth=1)
        ax[i, 1].plot(time, denoised_signal[i, :] / scaling_factor, '-b', label='Predicted signal', linewidth=1)
        #################################################################################################
        ax[i, 1].plot(time, residual_signal[i, :] / scaling_factor, '-', color='orange', linewidth=0.3)   #
        #################################################################################################
        ax[i, 2].plot(time, true_noise[i, :] / scaling_factor, '-', color='gray', label='True noise', linewidth=1)
        ax[i, 2].plot(time, separated_noise[i, :] / scaling_factor, '-b', label='Predicted noise', linewidth=1)
        ax[3, i].loglog(freq, spect_noisy_signal, '-k', label='raw signal', linewidth=0.5, alpha=1)
        ax[3, i].loglog(freq, spect_clean_signal, '-r', label='true earthquake', linewidth=0.5, alpha=1)
        ax[3, i].loglog(freq, spect_denoised_signal, '-b', label='separated earthquake', linewidth=0.5, alpha=1)
        ############################################################################################################
        ax[3, i].loglog(freq, spect_resquake, '-', color='orange', label='residual quake', linewidth=0.5, alpha=0.3) #
        ############################################################################################################

        ax[4, i].loglog(freq, spect_true_noise, '-r', label='orginal noise', linewidth=0.5, alpha=1)
        ax[4, i].loglog(freq, spect_noise, '-b', label='noise', linewidth=0.5, alpha=0.8)
        ax[i, 1].text(0, 0.8, f'EV: {evs_earthquake:.2f}/ CC: {cc_quake:.2f}')
        ax[i, 2].text(0, 0.8, f'EV: {evs_noise:.2f}/ CC: {cc_noise:.2f}')

        ax[i, 0].set_ylabel(comps[i], fontsize=16)
        ax[4, i].set_xlabel('Frequency (Hz)', fontsize=14)
        ax[3, i].set_title(comps[i], fontsize=16)
        ax[3, i].grid(alpha=0.2)
        ax[4, i].grid(alpha=0.2)

        for j in range(3):
            ax[i, j].xaxis.set_visible(False)
            ax[i, j].yaxis.set_ticks([])
            ax[i, j].spines['right'].set_visible(False)
            ax[i, j].spines['left'].set_visible(False)
            ax[i, j].spines['top'].set_visible(False)
            ax[i, j].spines['bottom'].set_visible(False)

            if i == 2:
                ax[i, j].xaxis.set_visible(True)
                ax[i, j].spines['bottom'].set_visible(True)
                ax[i, j].set_xlabel('time (s)', fontsize=14)
            if i <= 2:
                ax[i, j].set_xlim(0, npts * dt)
                ax[i, j].set_ylim(-1, 1)

    ax[0, 0].set_title("Original signal", fontsize=16)
    ax[0, 1].set_title(f"P wave squeezed x {sqz}")
    ax[0, 2].set_title("Separated noise", fontsize=16)
    ax[3, 0].set_ylabel('velocity spectra', fontsize=14)
    ax[4, 0].set_ylabel('velocity spectra', fontsize=14)
    ax[3, 2].legend(loc=3)
    ax[4, 2].legend(loc=3)

    plt.savefig(directory + f'/time_spec_{idx}.pdf')

    return scores


if __name__ == '__main__':
    main()