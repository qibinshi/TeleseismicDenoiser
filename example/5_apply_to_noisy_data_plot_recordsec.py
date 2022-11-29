"""
Denoise and measure source characteristics

author: Qibin Shi
"""
import os
import time
import h5py
import torch
import argparse
import matplotlib
import numpy as np
import pandas as pd
from functools import partial
from multitaper import MTSpec
from scipy.signal import detrend
from scipy.integrate import cumulative_trapezoid
from multiprocessing import Pool
from torch.utils.data import DataLoader
from torch_tools import WaveformDataset, try_gpu
from denoiser_util import shift_pad_fix_time
from denoiser_util import shift_pad_stretch_time
from denoiser_util import mkdir, fit_spec, flux_int
from denoiser_util import dura_cc, ellipse_directivity
from denoiser_util import p_rad_pat, sh_rad_pat, shift2maxcc
matplotlib.use('Agg')
matplotlib.rcParams.update({'font.size': 24})
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, MaxNLocator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--phase', default='P', type=str, help='earthquake phase')
    parser.add_argument('-r', '--minsnr', default=2, type=int, help='minimum raw signal-noise ratio')
    parser.add_argument('-n', '--threads', default=10, type=int, help='number of processes')
    args = parser.parse_args()

    data_dir = '/fd1/QibinShi_data/matfiles_for_denoiser/'
    phase = args.phase
    if phase == 'P':
        comp = 2  # Z
        tstar = 0.3
        npts = 1500
        npts_trim = 600
        start_pt = int(1500 - npts / 2)
        model_dir = 'Release_Middle_augmentation_P4Hz_150s'
        csv1 = data_dir + 'metadata_M6_deep100km_allSNR_P.csv'
        wav1 = data_dir + 'M6_deep100km_allSNR_P.hdf5'
        csv2 = data_dir + 'metadata_M55_deep100km_allSNR_P.csv'
        wav2 = data_dir + 'M55_deep100km_allSNR_P.hdf5'
    else:
        comp = 0  # T
        tstar = 1.2
        npts = 1500
        npts_trim = 600
        start_pt = int(2500 - npts / 2)
        model_dir = 'Release_Middle_augmentation_S4Hz_150s_removeCoda_SoverCoda25'
        csv1 = data_dir + 'metadata_M6_deep100km_allSNR_S_rot.csv'
        wav1 = data_dir + 'M6_deep100km_allSNR_S_rot.hdf5'
        csv2 = data_dir + 'metadata_M55_deep100km_allSNR_S_rot.csv'
        wav2 = data_dir + 'M55_deep100km_allSNR_S_rot.hdf5'

    fig_dir = model_dir + '/apply_minSNR' + str(args.minsnr) + '_fixWin60'
    mkdir(fig_dir)
    since = time.time()

    # %% Read multiple data and merge
    print("#" * 12 + " Loading quake data " + "#" * 12)
    with h5py.File(wav1, 'r') as f:
        data1 = f['pwave'][:, start_pt:start_pt+npts, :]
    meta1 = pd.read_csv(csv1, low_memory=False)
    with h5py.File(wav2, 'r') as f:
        data2 = f['pwave'][:, start_pt:start_pt+npts, :]
    meta2 = pd.read_csv(csv2, low_memory=False)

    print("#" * 12 + " Merging data chunks " + "#" * 12)
    input_raw = np.append(data1, data2, axis=0)
    meta_all = pd.concat([meta1, meta2], ignore_index=True)
    quake_ids = meta_all.source_id.unique()

    # %% Load Denoiser
    print("#" * 12 + " Loading " + args.phase + " denoiser " + "#" * 12)
    model = torch.load(model_dir + '/Branch_Encoder_Decoder_LSTM_Model.pth')
    model = model.module.to(try_gpu(i=10))
    model.eval()

    # %% Denoise, measure and plot in parallel
    partial_func = partial(denoise_cc_stack,
                           meta_all=meta_all,
                           input_raw=input_raw,
                           fig_dir=fig_dir,
                           model=model,
                           npts=npts,
                           cmp=comp,
                           tstar=tstar,
                           min_snr=args.minsnr,
                           npts_trim=npts_trim)

    num_proc = min(os.cpu_count(), args.threads)
    print('---- %d threads for plotting %d record sections\n' % (num_proc, len(quake_ids)))
    with Pool(processes=num_proc) as pool:
        result = pool.map(partial_func, quake_ids)

    # %% Save measurements
    meta_result = pd.DataFrame(columns=[
        "source_id",
        "source_magnitude",
        "source_depth_km",
        "duration_denoised",
        "centroid_speed_denoised",
        "centroid_direction_denoised",
        "duration_noisy",
        "centroid_speed_noisy",
        "centroid_direction_noisy",
        "Es_denoised",
        "Es_noisy",
        "falloff_denoised",
        "falloff_noisy",
        "corner_freq_denoised",
        "corner_freq_noisy",
        "num_station",
        "num_bin"])

    for i in range(len(quake_ids)):
        meta_result = pd.concat([meta_result, pd.DataFrame(data={
            "source_id": result[i][0],
            "source_magnitude": result[i][1],
            "source_depth_km": result[i][2],
            "duration_denoised": "%.3f" % result[i][3],
            "centroid_speed_denoised": "%.3f" % result[i][4],
            "centroid_direction_denoised": "%.3f" % result[i][5],
            "duration_noisy": "%.3f" % result[i][6],
            "centroid_speed_noisy": "%.3f" % result[i][7],
            "centroid_direction_noisy": "%.3f" % result[i][8],
            "Es_denoised": result[i][9],
            "Es_noisy": result[i][10],
            "falloff_denoised": result[i][11],
            "falloff_noisy": result[i][12],
            "corner_freq_denoised": result[i][13],
            "corner_freq_noisy": result[i][14],
            "num_station": result[i][15],
            "num_bin": result[i][16]}, index=[0])], ignore_index=True)

    meta_result.to_csv(fig_dir + "/source_measurements.csv", sep=',', index=False)

    print('---- All are plotted, using %.2f s\n' % (time.time() - since))


def denoise_cc_stack(evid, meta_all=None, input_raw=None, model=None, normalized_stack=True, fix_win=True,
                     npts=None, npts_trim=None, fig_dir=None, dt=0.1, min_snr=5, cmp=2, tstar=0.3):
    ####################################
    # %%  Process and Denoise the data #
    ####################################

    # %% Read event info from metadata
    meta = meta_all[(meta_all.source_id == evid)]
    dist_az = meta[['distance', 'azimuth', 'takeoff_phase']].to_numpy()
    src_meca = meta[['source_strike', 'source_dip', 'source_rake']].to_numpy()
    trace_amp = meta[['trace_snr_db', 'trace_stdv_2']].to_numpy()
    dist_az = dist_az.astype(np.float64)
    src_meca = src_meca.astype(np.float64)
    trace_amp = trace_amp.astype(np.float64)
    ind_highsnr = np.where(trace_amp[:, 0] > min_snr)[0]
    num_tr = len(ind_highsnr)
    # %% get azimuthal coverage
    bin_count, bin_edges = np.histogram(dist_az[ind_highsnr, 1], bins=8, range=(0, 360))
    bin_num = np.count_nonzero(bin_count)

    if num_tr >= 1:
        idx_list = meta.index.values
        batch_size = len(idx_list)
        evdp = meta.source_depth_km.unique()[0]
        evmg = meta.source_magnitude.unique()[0]

        # %% load event data
        test_data = WaveformDataset(input_raw[idx_list], input_raw[idx_list])
        test_iter = DataLoader(test_data, batch_size=batch_size, shuffle=False)
        data_iter = iter(test_iter)
        x0, y0 = data_iter.next()

        x = torch.zeros(batch_size, x0.size(1), npts, dtype=torch.float64)
        for i in np.arange(batch_size):
            quake_one = x0[i, :, :]
            scale_mean = torch.mean(quake_one, dim=1)
            scale_std = torch.std(quake_one, dim=1) + 1e-12
            trace_amp[i, 1] = trace_amp[i, 1] * scale_std[cmp]
            for j in np.arange(x0.size(1)):
                quake_one[j] = torch.div(torch.sub(quake_one[j], scale_mean[j]), scale_std[j])
            x[i] = quake_one

        # %% Denoise
        with torch.no_grad():
            quake_denoised, noise_output = model(x)
        noisy_signal = x.numpy()
        denoised_signal = quake_denoised.numpy()

        # %% signal-noise ratio of velocity waveforms
        noise_amp = np.std(noisy_signal[:, :, int(npts / 2) - 250: int(npts / 2) - 50], axis=-1)
        signl_amp = np.std(noisy_signal[:, :, int(npts / 2): int(npts / 2) + 200], axis=-1)
        snr_before = 20 * np.log10(np.divide(signl_amp, noise_amp + 1e-12) + 1e-12)

        noise_amp = np.std(denoised_signal[:, :, int(npts / 2) - 250: int(npts / 2) - 50], axis=-1)
        signl_amp = np.std(denoised_signal[:, :, int(npts / 2): int(npts / 2) + 200], axis=-1)
        snr_after = 20 * np.log10(np.divide(signl_amp, noise_amp + 1e-12) + 1e-12)

        # %% trim
        if not fix_win:
            npts_trim = min(npts_trim, int(evdp/2/dt))
        timex = np.arange(0, npts_trim) * dt
        startpt = int((npts - npts_trim) / 2)
        endpt = int((npts + npts_trim) / 2)

        # %% integrate velocity to displacement
        noisy_signal = cumulative_trapezoid(noisy_signal[:, :, startpt:endpt], timex, axis=-1, initial=0)
        denoised_signal = cumulative_trapezoid(denoised_signal[:, :, startpt:endpt], timex, axis=-1, initial=0)

        # %% normalize clean displacement
        z_std = np.zeros((batch_size, 2), dtype=np.float64)
        scale_std = np.std(denoised_signal, axis=-1, keepdims=True) + 1e-12
        denoised_signal = (denoised_signal - np.mean(denoised_signal, axis=-1, keepdims=True)) / scale_std
        z_std[:, 1] = trace_amp[:, 1] * scale_std[:, cmp, 0]

        # %% normalize noisy displacement
        scale_std = np.std(noisy_signal, axis=-1, keepdims=True) + 1e-12
        noisy_signal = (noisy_signal - np.mean(noisy_signal, axis=-1, keepdims=True)) / scale_std
        z_std[:, 0] = trace_amp[:, 1] * scale_std[:, cmp, 0]

        # %% Discard abnormally large amplitudes
        ave_z_std = np.mean(z_std, axis=0)
        for k in range(2):
            z_std[np.where(z_std[:, 0] > 20 * ave_z_std[0])[0], k] = 0
        total_std_z = np.sum(z_std, axis=0)

        # %% pre-stack reference: max-SNR trace
        id_ref = np.argmax(trace_amp[:, 0])
        ref_noisy = noisy_signal[id_ref, cmp, :]
        ref_clean = denoised_signal[id_ref, cmp, :]

        #############################
        # %% Source characteristics #
        #############################
        # %% 0 noisy; 1 clean
        f_int = ['', '']
        n_best = ['', '']
        fc_best = ['', '']
        color = ['k', 'b']
        vr = [0.0, 0.0]
        med = [1.0, 1.0]
        dir = [0.0, 0.0]
        duration = [1.0, 1.0]
        azi_bins_deno = ['', '', '', '', '', '', '', '']
        azi_bins_noisy = ['', '', '', '', '', '', '', '']
        cc = np.zeros((batch_size, 2), dtype=np.float64)
        flip = np.zeros((batch_size, 2), dtype=np.int32)
        shift = np.zeros((batch_size, 2), dtype=np.int32)
        ratio = np.zeros((batch_size, 2), dtype=np.float64)
        stack = np.zeros((2, timex.shape[-1]), dtype=np.float64)
        stack_stretch = np.zeros((2, timex.shape[-1]), dtype=np.float64)

        plt.close('all')
        fig, ax = plt.subplots(3, 4, figsize=(40, 30), constrained_layout=True)
        ax[1, 0] = plt.subplot(3, 4, 5, projection='polar')
        ax[1, 1] = plt.subplot(3, 4, 6, projection='polar')
        ax[2, 0] = plt.subplot(3, 4, 9, projection='polar')
        ax[2, 1] = plt.subplot(3, 4, 10, projection='polar')
        ax[2, 3] = plt.subplot(3, 4, 12, projection='polar')
        cmap = plt.get_cmap('PiYG')  # colormap of radiation pattern
        cmap1 = plt.get_cmap('Oranges')  # colormap of stretch ratio
        amp_azi = 12  # amplifying factor for plotting

        # %% lower-hemisphere radiation patten
        y, x = np.mgrid[slice(0, 90 + 5, 5), slice(0, 360 + 5, 5)]
        if cmp == 2:
            r, z = p_rad_pat(src_meca[0, 0], src_meca[0, 1], src_meca[0, 2], y, x)
        else:
            r, z = sh_rad_pat(src_meca[0, 0], src_meca[0, 1], src_meca[0, 2], y, x)
        levels = MaxNLocator(nbins=50).tick_values(z.min(), z.max())
        im = ax[2, 3].contourf((x + 2.5) / 180 * np.pi, r, z, levels=levels, cmap=cmap)
        cb = fig.colorbar(im, ax=ax[2, 3])
        cb.set_label('radiation pattern')
        ax[2, 3].set_ylim(0, 0.6)
        ax[2, 3].set_theta_zero_location('N')
        ax[2, 3].set_theta_direction(-1)
        ax[2, 3].yaxis.set_major_locator(MultipleLocator(2))
        ax[2, 3].set_title('Radiation Pattern and Ray directions', fontsize=30)

        ############################################################### 1st stack Loop ###########
        for i in ind_highsnr:
            tmp_tr = np.zeros((2, timex.shape[-1]), dtype=np.float64)

            # %% Re-align traces
            tr_noisy, shift[i, 0], flip[i, 0] = shift2maxcc(ref_noisy, noisy_signal[i, cmp, :],
                                                            maxshift=30, flip_thre=-0.3)
            tr_clean, shift[i, 1], flip[i, 1] = shift2maxcc(ref_clean, denoised_signal[i, cmp, :],
                                                            maxshift=30, flip_thre=-0.3)
            tmp_tr[0, :] = shift_pad_fix_time(tr_noisy, shift[i, 0])
            tmp_tr[1, :] = shift_pad_fix_time(tr_clean, shift[i, 1])

            # %% Pre-stack noisy and clean waves
            for k in range(2):
                if normalized_stack:
                    stack[k, :] = stack[k, :] + tmp_tr[k, :] / num_tr
                else:
                    stack[k, :] = stack[k, :] + tmp_tr[k, :] * z_std[i, k] / total_std_z[k]

        #################################################### Stretch and 2nd stack Loop ###########
        count = 0
        for i in ind_highsnr:
            tmp_tr = np.zeros((2, timex.shape[-1]), dtype=np.float64)

            # %% Stretch and align with the pre-stacked wave
            time_clean, wave_clean, ratio[i, 1], cc[i, 1], flip[i, 1] = dura_cc(stack[1], denoised_signal[i, cmp, :],
                                                                                timex, maxshift=30, max_ratio=1.5)
            time_noisy, wave_noisy, ratio[i, 0], cc[i, 0], flip[i, 0] = dura_cc(stack[0], noisy_signal[i, cmp, :],
                                                                                timex, maxshift=30, max_ratio=1.5)
            ax[0, 1].plot(time_clean, wave_clean * amp_azi + dist_az[i, 1], color=color[flip[i, 1]], ls='-', lw=1)
            ax[0, 0].plot(time_noisy, wave_noisy * amp_azi + dist_az[i, 1], color=color[flip[i, 0]], ls='-', lw=1)
            tmp_tr[1, :] = shift_pad_stretch_time(wave_clean, timex, time_clean)
            tmp_tr[0, :] = shift_pad_stretch_time(wave_noisy, timex, time_noisy)
            for k in range(2):
                # %% Stack the stretched time series
                if normalized_stack:
                    stack_stretch[k, :] = stack_stretch[k, :] + tmp_tr[k, :] / num_tr
                else:
                    stack_stretch[k, :] = stack_stretch[k, :] + tmp_tr[k, :] * z_std[i, k] / total_std_z[k]

                # %% plot stretch ratio with azimuth and take-off angle
                clr = color[flip[i, k]]
                ax[1, k].plot(dist_az[i, 1]/180*np.pi, ratio[i, k], marker='o', mfc=clr, mec=clr, ms=cc[i, k] * 20)
                ax[2, k].plot(dist_az[i, 2]/180*np.pi, ratio[i, k], marker='o', mfc=clr, mec=clr, ms=cc[i, k] * 20)

                # %% Spectra of un-stretched traces
                if k == 0:
                    Py = MTSpec(detrend(noisy_signal[i, cmp, int(npts_trim/2-50):]), 4.0, 6, dt)
                else:
                    Py = MTSpec(detrend(denoised_signal[i, cmp, int(npts_trim/2-50):]), 4.0, 6, dt)

                freq, spec = Py.rspec()
                freq = np.squeeze(freq)
                spec = np.squeeze(spec) * np.exp(freq * tstar / 2.0)  # correct for attenuation
                spec_norm = spec / np.nanmax(spec[freq < 0.05])  # normalize spectra

                # %% stack spectra in each azimuth bin
                if count == 0 and k == 0:
                    for m in range(8):
                        azi_bins_noisy[m] = np.zeros(len(spec_norm), dtype=np.float64)
                        azi_bins_deno[m] = np.zeros(len(spec_norm), dtype=np.float64)

                bin_no = int(dist_az[i, 1] // 45)
                if k == 0:
                    azi_bins_noisy[bin_no] = azi_bins_noisy[bin_no] + spec_norm
                else:
                    azi_bins_deno[bin_no] = azi_bins_deno[bin_no] + spec_norm

            count += 1

        ####################################################### After Loop ###########
        # %% plot stretch ratio against radiation pattern
        takeoffs_rad = np.sin(dist_az[ind_highsnr, 2] / 360 * np.pi) / np.cos(dist_az[ind_highsnr, 2] / 360 * np.pi)
        ray = ax[2, 3].scatter(dist_az[ind_highsnr, 1] / 180 * np.pi, takeoffs_rad, marker='o', c=ratio[ind_highsnr, 1],
                               s=cc[ind_highsnr, 1] * 100, edgecolors='w', cmap=cmap1, vmin=0.8, vmax=1.2)
        cb = plt.colorbar(ray, ax=ax[2, 3])
        cb.set_label('stretching ratio')

        # %% Average spectrum of all bins
        stack_spec = np.zeros((2, len(spec_norm)), dtype=np.double)
        for m in range(8):
            azi_bins_noisy[m] = np.divide(azi_bins_noisy[m], bin_count[m]+0.01)
            stack_spec[0, :] = stack_spec[0, :] + azi_bins_noisy[m]
            ax[0, 2].loglog(freq, azi_bins_noisy[m], ls='-', color='0.8', lw=2)
            azi_bins_deno[m] = np.divide(azi_bins_deno[m], bin_count[m]+0.01)
            stack_spec[1, :] = stack_spec[1, :] + azi_bins_deno[m]
            ax[0, 3].loglog(freq, azi_bins_deno[m], ls='-', color='0.8', lw=2)

        # %% Model fitting the stacked spectrum
        for k in range(2):
            stack_spec[k, :] = stack_spec[k, :] / bin_num
            lowband = np.logical_and(freq > 0, freq < 1)
            n_best[k], fc_best[k] = fit_spec(freq[lowband], stack_spec[k, lowband])
            f_int[k] = flux_int(freq, stack_spec[k, :], fc_best[k], n_best[k])
            ax[0, 2+k].loglog(freq, stack_spec[k, :], ls='-', color='g', lw=5)
            ax[0, 2+k].loglog(freq, 1.0 / (1.0 + (freq/fc_best[k])**n_best[k]), ls='--', color='b', lw=5)
            ax[0, 2+k].loglog(fc_best[k], 0.5, 'o', markersize=15, mec='k', mfc='r')
            ax[0, 2+k].set_ylim(1e-15, 1e2)
            ax[0, 2+k].set_xlabel('frequency (Hz)', fontsize=24)
            ax[0, 2+k].set_ylabel('amplitude spectrum', fontsize=24)
            ax[0, 2+k].set_title(f'falloff {n_best[k]:.1f} fc {fc_best[k]:.2f} Er/M^2 {f_int[k]:.1e}', fontsize=30)

            # %% Fit directivity ellipse
            vr[k], med[k], dir[k] = ellipse_directivity(dist_az[:, 1], ratio[:, k], cc[:, k])
            azimuth = np.arange(0.0, 360.0, 2.0, dtype=np.float64)
            r_pred = med[k] / (1 - vr[k] * np.cos((azimuth - dir[k]) / 180.0 * np.pi))
            if vr[k] > 0.1:
                clr = '-r'
            else:
                clr = '-y'
            ax[1, k].plot(azimuth / 180.0 * np.pi, r_pred, clr, linewidth=12, alpha=0.2)

            # %% Plot stacked displacement
            stacked_disp = stack_stretch[k, :] * amp_azi
            stacked_velo = np.gradient(stacked_disp, timex)
            stacked_ener = cumulative_trapezoid(np.square(stacked_velo), timex, axis=-1, initial=0)
            stacked_ener = stacked_ener / (np.std(stacked_ener) + 1e-12) * amp_azi * 3
            ax[0, k].plot(timex, stacked_ener + np.max(dist_az[:, 1]) + (50 * 1 + 15), '-r', linewidth=5)
            ax[0, k].plot(timex, stacked_disp + np.max(dist_az[:, 1]) + (50 * 1 + 10), '-g', linewidth=5)
            ax[0, k].set_xlim(timex[0], timex[-1])

            # %% 0.05 and 0.95 of max energy
            max_energy = np.nanmax(stacked_ener[int(npts_trim/3):0-int(5/dt)])
            pre = timex[stacked_ener < 0.05 * max_energy]
            pos = timex[stacked_ener > 0.90 * max_energy]
            pt_start = int(pre[-1] / dt)
            pt_end = int(pos[0] / dt)
            duration[k] = pos[0] - pre[-1]
            ax[0, k].plot(pt_start * dt, stacked_ener[pt_start] + np.max(dist_az[:, 1]) + (50 * 1 + 15), 'ok', ms=10)
            ax[0, k].plot(pt_end * dt, stacked_ener[pt_end] + np.max(dist_az[:, 1]) + (50 * 1 + 15), 'oy', ms=10)
            ax[0, k].plot(pt_start * dt, stacked_disp[pt_start] + np.max(dist_az[:, 1]) + (50 * 1 + 10), 'ok', ms=10)
            ax[0, k].plot(pt_end * dt, stacked_disp[pt_end] + np.max(dist_az[:, 1]) + (50 * 1 + 10), 'oy', ms=10)

            # %% axes limits, titles and labels
            ax[0, k].set_xlim(timex[0], timex[-1])
            ax[0, k].set_ylim(np.min(dist_az[:, 1]) - (50 * 1 + 10), np.max(dist_az[:, 1]) + (50 * 1 + 120))
            ax[0, k].set_ylabel('azimuth', fontsize=24)
            ax[1, k].yaxis.set_major_locator(MultipleLocator(0.5))
            ax[1, k].yaxis.set_minor_locator(MultipleLocator(0.25))
            ax[1, k].set_theta_zero_location('N')
            ax[1, k].set_theta_direction(-1)
            ax[1, k].set_ylim(0, 2.2)
            ax[2, k].yaxis.set_major_locator(MultipleLocator(0.5))
            ax[2, k].yaxis.set_minor_locator(MultipleLocator(0.25))
            ax[2, k].set_theta_zero_location('S')
            ax[2, k].set_ylim(0, 2.2)
            ax[2, k].set_xlim(0, np.pi/2)
            ax[1, k].set_title(f'Vrup/V = {vr[k]:.3f} \n'
                               f'directivity = {dir[k]:.0f} \n'
                               f'duration = {duration[k]:.1f} s /{med[k]:.2f}',
                               fontsize=30)

        # %% SNR histograms
        bins = np.linspace(0, 100, 20)
        ax[2, 2].hist(snr_before.flatten(), bins=bins, density=True, histtype='stepfilled',
                      color='0.9', alpha=0.5, label='noisy', lw=2)
        ax[2, 2].hist(snr_after.flatten(), bins=bins, density=True, histtype='stepfilled',
                      color='r', alpha=0.3, label='denoised', lw=2)
        ax[2, 2].set_xlabel('SNR', fontsize=24)
        ax[2, 2].set_ylabel('density', fontsize=24)
        ax[2, 2].legend(loc=1)

        # %% azimuth histograms
        bins = np.linspace(0, 360, 8)
        ax[1, 2].hist(dist_az[ind_highsnr, 1], bins=bins, density=True, histtype='stepfilled',
                      color='0.2', alpha=1, label=f'SNR>{min_snr}', lw=2, orientation='horizontal')
        ax[1, 2].set_xlabel('density', fontsize=24)
        ax[1, 2].set_ylabel('azimuth', fontsize=24)

        # %%
        ax[0, 0].set_title(f'Noisy Event {evid} depth={evdp:.0f} km / M{evmg:.1f}', fontsize=30)
        ax[0, 1].set_title('Denoised waves', fontsize=30)

        # %%
        for j in range(4):
            ax[0, j].grid(which='major', color='#DDDDDD', linewidth=3)
            ax[0, j].grid(which='minor', color='#EEEEEE', linestyle='--', linewidth=2)
            ax[0, j].minorticks_on()

        # %% Save as a figure
        plt.savefig(fig_dir + '/quake_' + str(evid) + '_record_section.png')

        # %% Save denoised signal and spectra
        print(f'{evid} saving the spec')
        with h5py.File(fig_dir + '/' + str(evid) + '_timeDenoised_and_specStack.hdf5', 'w') as f:
            f.create_dataset("time_signal", data=denoised_signal)
            f.create_dataset("stack_spec_deno", data=stack_spec[1, :])
            f.create_dataset("stack_spec_noisy", data=stack_spec[0, :])
            f.create_dataset("stack_spec_freq", data=freq)

        return evid, evmg, evdp, \
               duration[1] / med[1], vr[1], dir[1], \
               duration[0] / med[0], vr[0], dir[0], \
               f_int[1], f_int[0], n_best[1], n_best[0], \
               fc_best[1], fc_best[0], num_tr, bin_num
    else:
        return evid, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1


if __name__ == '__main__':
    main()
