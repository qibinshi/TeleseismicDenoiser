"""
Test the model performance with earthquake
signal squeezed, shifted and noise stacked

author: Qibin Shi
"""
import os
import time
import h5py
import torch
import configparser
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
from Train import WDN_compose, read_split_data

matplotlib.rcParams.update({'font.size': 12})


def test(configure_file='config.ini'):
    config = configparser.ConfigParser()
    config.read(configure_file)

    gpu = config.getint('training', 'gpu')
    gpu_ids = config.get('training', 'gpu_ids')
    gpu_ids = [int(x) for x in gpu_ids.split(',')]
    storage_home = config.get('directories', 'storage_home')
    data_dir = storage_home + config.get('directories', 'data_dir')
    model_dir = storage_home + config.get('directories', 'save_model_dir')
    data_file = config.get('data', 'data_file')
    use_demo = config.getint('data', 'use_demo')
    half_length = config.getint('data', 'half_length')
    strmax = config.getint('data', 'stretch_max')
    npts = config.getint('data', 'npts')
    branch_signal = config.get('data', 'branch_signal')
    branch_noise = config.get('data', 'branch_noise')
    train_size = config.getfloat('training', 'train_size')
    test_size = config.getfloat('training', 'test_size')
    rand_seed1 = config.getint('training', 'rand_seed1')
    rand_seed2 = config.getint('training', 'rand_seed2')
    batch_size = config.getint('training', 'batch_size')
    lr = config.getfloat('training', 'learning_rate')
    epochs = config.getint('training', 'epochs')
    minimum_epochs = config.getint('training', 'minimum_epochs')
    patience = config.getint('training', 'patience')
    pre_trained_denote = pkg_resources.resource_filename(__name__, 'pretrained_models/Denote_weights.pth')
    demo_train_data = pkg_resources.resource_filename(__name__, 'datasets/demo_train_dataset.hdf5')

    if use_demo:
        wave_raw = demo_train_data
    else:
        wave_raw = data_dir + data_file

    print('gpu', gpu)
    print('use demo data', use_demo)
    print('dataset path', wave_raw)
    print('directory to load your model', model_dir)
    print('half of the length of waveform', half_length)
    print('fraction for training', train_size)
    print('fraction for testing', test_size * (1 - train_size))
    print('random seeds', rand_seed1, rand_seed2)
    print('batch size', batch_size)
    print('learning rate', lr)
    print('# epochs', epochs)
    print('min. # epochs', minimum_epochs)
    print('patience before stopping', patience)
    print('gpu IDs', gpu_ids)

    mid_pt = half_length

    fig_dir = model_dir + '/figures'
    mkdir(fig_dir)
    dt = 0.1
    frac = 0.45  # smallest window start
    batch_size = 100
    bottleneck_name = 'LSTM'
    model_structure = "Branch_Encoder_Decoder"
    model_name = model_structure + "_" + bottleneck_name

    print("#" * 12 + " Load and split data " + "#" * 12)
    # %% Read the pre-processed dataset and split into train, validate and test sets
    training_data, validate_data, test_data = read_split_data(wave_raw, branch_signal, branch_noise, npts,
                                                              train_size, test_size, rand_seed1, rand_seed2)

    test_iter = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    ############ %% Neural Net structure %% ###############
    print("#" * 12 + " Loading model " + model_name + " " + "#" * 12)
    devc = torch.device('cpu')

    # %% construct a WaveDecompNet kernel first
    model = WDN_compose()

    # %% keep constructing for DenoTe
    model = T_model(model, half_insize=int(npts / 2))

    # %% load pre-trained weights for DenoTe
    model.load_state_dict(torch.load(pre_trained_denote, map_location=devc))
    model = model.module.to(devc)
    model.eval()


    print("#" * 12 + " Augment data and feed to the model " + "#" * 12)
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

        # %% denoise
        quake_denoised, noise_output = model(noisy_input)
        print("Denoised!")

        noisy_signal = noisy_input.cpu().numpy()
        clean_signal = quake_label.cpu().numpy()
        separated_noise = noise_output.cpu().numpy()
        denoised_signal = quake_denoised.cpu().numpy()
        true_noise = noise_label.cpu().numpy()

    # %% Parallel Plotting
    partial_func = partial(plot_testing, directory=fig_dir, dt=dt, npts=npts)
    num_proc = min(os.cpu_count(), batch_size)
    pool = Pool(processes=num_proc)
    print("#" * 12 + " Plotting waveforms using " + str(num_proc) + "CPUs" + "#" * 12)

    result = pool.starmap(partial_func, zip(noisy_signal,
                                            denoised_signal,
                                            separated_noise,
                                            clean_signal,
                                            true_noise,
                                            np.arange(nbatch),
                                            sqz))

    print("#" * 12 + " Plotting statistics of testing performance " + "#" * 12)
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
    test()