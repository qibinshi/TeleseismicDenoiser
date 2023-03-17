"""
Denoise deep earthquakes to
study source characteristics

author: Qibin Shi
"""
import os
import time
import h5py
import torch
import configparser
import numpy as np
import pandas as pd
from functools import partial
from multitaper import MTSpec
from scipy.signal import detrend, find_peaks
from scipy.integrate import cumulative_trapezoid
from multiprocessing import Pool
from scipy.interpolate import interp1d
from torch.utils.data import DataLoader
from torch_tools import WaveformDataset, try_gpu
from denoiser_util import shift_pad_fix_time
from denoiser_util import shift_pad_stretch_time
from denoiser_util import mkdir, fit_spec, flux_int
from denoiser_util import dura_cc, directivity3d_free
from denoiser_util import p_rad_pat, sh_rad_pat, shift2maxcc
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams.update({'font.size': 24})
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, MaxNLocator


def DenoTe(configure_file='config.ini'):
    config = configparser.ConfigParser()
    config.read(configure_file)

    storage_home = config.get('directories', 'storage_home')
    data_wave = storage_home + config.get('application', 'data_wave')
    data_meta = storage_home + config.get('application', 'data_meta')
    model_dir = config.get('application', 'model_dir')
    rslt_dir = storage_home + config.get('application', 'result_dir')
    phase = config.get('application', 'phase')
    npts = config.getint('application', 'npts')
    start_pt = config.getint('application', 'start_point')
    npts_trim = config.getint('application', 'npts_trim')
    tstar = config.getfloat('application', 'tstar')
    comp = config.getint('application', 'component')
    minsnr = config.getint('application', 'minsnr')
    threads = config.getint('application', 'threads')
    num_event = config.getint('application', 'num_event')

    mkdir(rslt_dir)
    since = time.time()

    # %% Load Denoiser
    print("#" * 12 + " Loading " + phase + " denoiser " + model_dir + " " + "#" * 12)
    model = torch.load(model_dir, map_location=try_gpu(i=1))
    model = model.module.to(try_gpu(i=10))
    model.eval()

    # %% Read multiple data and merge
    print("#" * 12 + " Loading quake data " + "#" * 12)
    with h5py.File(data_wave, 'r') as f:
        input_raw = f['pwave'][:, start_pt:start_pt+npts, :]
    meta_all = pd.read_csv(data_meta, low_memory=False)
    quake_ids = meta_all.source_id.unique()[:num_event]

    # %% Denoise, measure and plot in parallel
    partial_func = partial(denoise_cc_stack,
                           meta_all=meta_all,
                           input_raw=input_raw,
                           fig_dir=rslt_dir,
                           model=model,
                           npts=npts,
                           cmp=comp,
                           tstar=tstar,
                           min_snr=minsnr,
                           npts_trim=npts_trim)

    num_proc = min(os.cpu_count(), threads)
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
        "centroid_inclination_denoised",
        "duration_noisy",
        "centroid_speed_noisy",
        "centroid_direction_noisy",
        "centroid_inclination_noisy",
        "Es_denoised",
        "Es_noisy",
        "falloff_denoised",
        "falloff_noisy",
        "corner_freq_denoised",
        "corner_freq_noisy",
        "num_station",
        "num_bin",
        "num_peaks",
        "num_peaks_dir"])

    for i in range(len(quake_ids)):
        meta_result = pd.concat([meta_result, pd.DataFrame(data={
            "source_id": result[i][0],
            "source_magnitude": result[i][1],
            "source_depth_km": result[i][2],
            "duration_denoised": "%.3f" % result[i][3],
            "centroid_speed_denoised": "%.3f" % result[i][4],
            "centroid_direction_denoised": "%.3f" % result[i][5],
            "centroid_inclination_denoised": "%.3f" % result[i][6],
            "duration_noisy": "%.3f" % result[i][7],
            "centroid_speed_noisy": "%.3f" % result[i][8],
            "centroid_direction_noisy": "%.3f" % result[i][9],
            "centroid_inclination_noisy": "%.3f" % result[i][10],
            "Es_denoised": result[i][11],
            "Es_noisy": result[i][12],
            "falloff_denoised": result[i][13],
            "falloff_noisy": result[i][14],
            "corner_freq_denoised": result[i][15],
            "corner_freq_noisy": result[i][16],
            "num_station": result[i][17],
            "num_bin": result[i][18],
            "num_peaks": result[i][19],
            "num_peaks_dir": result[i][20]}, index=[0])], ignore_index=True)

    meta_result.to_csv(rslt_dir + "/source_measurements.csv", sep=',', index=False)

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
        noise_mean=np.mean(noisy_signal[:, :, int(npts / 2) - 300: int(npts / 2) - 100], axis=-1, keepdims=True)
        noise_amp = np.std(noisy_signal[:, :, int(npts / 2) - 300: int(npts / 2) - 100] - noise_mean, axis=-1)
        signl_amp = np.std(noisy_signal[:, :, int(npts / 2): int(npts / 2) + 300], axis=-1)
        snr_before = 20 * np.log10(np.divide(signl_amp, noise_amp + 1e-12) + 1e-12)

        noise_mean=np.mean(denoised_signal[:, :, int(npts / 2) - 300: int(npts / 2) - 100], axis=-1, keepdims=True)
        noise_amp = np.std(denoised_signal[:, :, int(npts / 2) - 300: int(npts / 2) - 100] - noise_mean, axis=-1)
        signl_amp = np.std(denoised_signal[:, :, int(npts / 2): int(npts / 2) + 300], axis=-1)
        snr_after = 20 * np.log10(np.divide(signl_amp, noise_amp + 1e-12) + 1e-12)

        # %% trim
        if not fix_win:
            npts_trim = min(npts_trim, int(evdp/2/dt))

        # %% big eqs need longer time window
        if evmg >= 7.0:
            npts_trim = npts

        timex = np.arange(0, npts_trim) * dt
        startpt = int((npts - npts_trim) / 2)
        endpt = int((npts + npts_trim) / 2)

        # %% integrate velocity to displacement
        noisy_signal = cumulative_trapezoid(noisy_signal[:, :, startpt:endpt], timex, axis=-1, initial=0)
        denoised_signal = cumulative_trapezoid(denoised_signal[:, :, startpt:endpt], timex, axis=-1, initial=0)

        # %% normalize clean displacement
        z_std = np.zeros((batch_size, 2), dtype=np.float64)
        scale_std = np.std(denoised_signal, axis=-1, keepdims=True) + 1e-12
        denoised_signal = denoised_signal / scale_std
        z_std[:, 1] = trace_amp[:, 1] * scale_std[:, cmp, 0]

        # %% normalize noisy displacement
        scale_std = np.std(noisy_signal, axis=-1, keepdims=True) + 1e-12
        noisy_signal = noisy_signal / scale_std
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
        pre = ['', '']
        pos = ['', '']
        f_int = ['', '']
        n_best = ['', '']
        fc_best = ['', '']
        color = ['k', 'b']
        vr = [0.0, 0.0]
        med = [1.0, 1.0]
        dir = [0.0, 0.0]
        inc = [0.0, 0.0]
        duration = [1.0, 1.0]
        azi_bins_deno = ['', '', '', '', '', '', '', '']
        azi_bins_noisy = ['', '', '', '', '', '', '', '']
        azi_bins_deno_noise = ['', '', '', '', '', '', '', '']
        azi_bins_noisy_noise = ['', '', '', '', '', '', '', '']
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

            for j in range(noisy_signal.shape[1]):
                noisy_signal[i, j, :] = noisy_signal[i, j, :] - noisy_signal[i, j, 0]
                denoised_signal[i, j, :] = denoised_signal[i, j, :] - denoised_signal[i, j, 0]

            # %% Re-align traces
            tr_noisy,shift[i, 0],flip[i, 0] =shift2maxcc(ref_noisy,noisy_signal[i, cmp, :],maxshift=30,flip_thre=-0.3)
            tr_clean,shift[i, 1],flip[i, 1] =shift2maxcc(ref_clean,denoised_signal[i, cmp, :],maxshift=30,flip_thre=-0.3)

            tmp_tr[0, :] = shift_pad_fix_time(tr_noisy, shift[i, 0])
            tmp_tr[1, :] = shift_pad_fix_time(tr_clean, shift[i, 1])

            # %% Pre-stack noisy and clean waves
            for k in range(2):
                if normalized_stack:
                    stack[k, :] = stack[k, :] + tmp_tr[k, :] / num_tr
                else:
                    stack[k, :] = stack[k, :] + tmp_tr[k, :] * z_std[i, k] / total_std_z[k]

        for k in range(2):
            stack_velo = np.gradient(stack[k, :], timex)
            squar_velo = np.square(stack_velo)
            stack_ener = cumulative_trapezoid(squar_velo, timex, axis=-1, initial=0)

            # %% 0.05 and 0.9 of max energy
            max_energy = np.nanmax(stack_ener[int(npts_trim / 3): 0 - int(10 / dt)])
            min_energy = np.nanmin(stack_ener[int(npts_trim / 3): 0 - int(10 / dt)])
            pre[k] = timex[stack_ener < 0.05 * (max_energy - min_energy) + min_energy]
            pos[k] = timex[stack_ener > 0.90 * (max_energy - min_energy) + min_energy]

            pt_st = int(pre[k][-1] / dt)
            pt_en = min(int((pos[k][0] + 2.0) / dt), npts_trim)

            stack_disp = stack[k, :] * amp_azi
            prom = np.nanmax(np.fabs(stack_disp[pt_st:pt_en])) / 5
            peaks_dir, _ = find_peaks(np.fabs(stack_disp[pt_st:pt_en]), prominence=prom)
            peaks_dir = peaks_dir + pt_st
            ax[0, k].plot(timex, stack_disp + np.max(dist_az[:, 1]) + (50 * 1 + 90), '-', color='0.4', linewidth=5)
            ax[0, k].plot(timex[peaks_dir], stack_disp[peaks_dir] + np.max(dist_az[:, 1]) + (50 * 1 + 90), 'x', ms=10, lw=3,
                          color='b')

        pt_st0 = max(int((pre[0][-1] - 10.0) / dt), 0)
        pt_en0 = min(int((pos[0][0] + 10.0) / dt), npts_trim)
        pt_st1 = max(int((pre[1][-1] - 10.0) / dt), 0)
        pt_en1 = min(int((pos[1][0] + 10.0) / dt), npts_trim)

        #################################################### Stretch and 2nd stack Loop ###########
        count = 0
        for i in ind_highsnr:
            tmp_tr = np.zeros((2, timex.shape[-1]), dtype=np.float64)

            # %% Stretch and align with the pre-stacked wave
            # time_clean, wave_clean, ratio[i, 1], cc[i, 1], flip[i, 1] = dura_cc(stack[1], denoised_signal[i, cmp, :],
            #                                                                     timex, maxshift=60, max_ratio=1.9, flip_thre=-0.5)
            # time_noisy, wave_noisy, ratio[i, 0], cc[i, 0], flip[i, 0] = dura_cc(stack[0], noisy_signal[i, cmp, :],
            #                                                                     timex, maxshift=60, max_ratio=1.9, flip_thre=-0.5)

            time_clean_cut, wave_clean, ratio[i, 1], cc[i, 1], flip[i, 1] = dura_cc(stack[1, pt_st1:pt_en1],
                                                                                denoised_signal[i, cmp, pt_st1:pt_en1],
                                                                                timex[pt_st1:pt_en1],
                                                                                maxshift=50, max_ratio=1.9,
                                                                                flip_thre=-0.4)
            time_noisy_cut, wave_noisy, ratio[i, 0], cc[i, 0], flip[i, 0] = dura_cc(stack[0, pt_st0:pt_en0],
                                                                                noisy_signal[i, cmp, pt_st0:pt_en0],
                                                                                timex[pt_st0:pt_en0],
                                                                                maxshift=50, max_ratio=1.9,
                                                                                flip_thre=-0.4)

            ### stretch and shift again with full time window
            time_new = np.arange(timex[0], timex[-1], dt / ratio[i, 1])
            interp_f = interp1d(timex, denoised_signal[i, cmp, :], bounds_error=False, fill_value=0.)
            wave_clean = interp_f(time_new) * (1 - 2 * flip[i, 1])
            num_pts = len(wave_clean)
            starttime = time_clean_cut[0] - timex[pt_st1] * ratio[i, 1]
            time_clean = np.linspace(starttime, starttime + num_pts * dt, num=num_pts)
            ###
            time_new = np.arange(timex[0], timex[-1], dt / ratio[i, 0])
            interp_f = interp1d(timex, noisy_signal[i, cmp, :], bounds_error=False, fill_value=0.)
            wave_noisy = interp_f(time_new) * (1 - 2 * flip[i, 0])
            num_pts = len(wave_noisy)
            starttime = time_noisy_cut[0] - timex[pt_st0] * ratio[i, 0]
            time_noisy = np.linspace(starttime, starttime + num_pts * dt, num=num_pts)

            ax[0, 1].plot(time_clean, wave_clean * amp_azi + dist_az[i, 1], color=color[flip[i, 1]], ls='-', lw=1)
            ax[0, 0].plot(time_noisy, wave_noisy * amp_azi + dist_az[i, 1], color=color[flip[i, 0]], ls='-', lw=1)

            ax[0, 1].plot(time_clean_cut[0],  dist_az[i, 1], color='orange', marker='o', ms=10)
            ax[0, 1].plot(time_clean_cut[-1], dist_az[i, 1], color='purple', marker='o', ms=10)
            ax[0, 0].plot(time_noisy_cut[0],  dist_az[i, 1], color='orange', marker='o', ms=10)
            ax[0, 0].plot(time_noisy_cut[-1], dist_az[i, 1], color='purple', marker='o', ms=10)

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
                    Py=MTSpec(detrend(noisy_signal[i, cmp, int(npts_trim/2-10/dt): 0 - int(5/dt)]), 4.0, 6, dt)
                    Py_noise = MTSpec(noisy_signal[i, cmp, int(5/dt): int(npts_trim/2-10/dt)], 4.0, 6, dt)
                else:
                    Py=MTSpec(detrend(denoised_signal[i, cmp, int(npts_trim/2-10/dt): 0 - int(5/dt)]), 4.0, 6, dt)
                    Py_noise = MTSpec(denoised_signal[i, cmp, int(5/dt): int(npts_trim/2-10/dt)], 4.0, 6, dt)

                freq, spec = Py.rspec()
                freq = np.squeeze(freq)
                spec = np.squeeze(spec) * np.exp(freq * tstar / 2.0)  # correct for attenuation
                spec_norm = spec / np.nanmax(spec[freq < 0.05])  # normalize spectra
                ##########
                freq_noise, spec_noise = Py_noise.rspec()  # spectra of pure noise
                freq_noise = np.squeeze(freq_noise)
                spec_noise = np.squeeze(spec_noise) * np.exp(freq_noise * tstar / 2.0)
                spec_noise_norm = spec_noise / np.nanmax(spec[freq < 0.05])
                ##########

                # %% stack spectra in each azimuth bin
                if count == 0 and k == 0:
                    for m in range(8):
                        azi_bins_noisy[m] = np.zeros(len(spec_norm), dtype=np.float64)
                        azi_bins_deno[m] = np.zeros(len(spec_norm), dtype=np.float64)
                        ##########
                        azi_bins_noisy_noise[m] = np.zeros(len(spec_noise_norm), dtype=np.float64)
                        azi_bins_deno_noise[m] = np.zeros(len(spec_noise_norm), dtype=np.float64)
                        ##########


                bin_no = int(dist_az[i, 1] // 45)
                if k == 0:
                    azi_bins_noisy[bin_no] = azi_bins_noisy[bin_no] + spec_norm
                    ##########
                    azi_bins_noisy_noise[bin_no] = azi_bins_noisy_noise[bin_no] + spec_noise_norm
                    ##########
                else:
                    azi_bins_deno[bin_no] = azi_bins_deno[bin_no] + spec_norm
                    ##########
                    azi_bins_deno_noise[bin_no] = azi_bins_deno_noise[bin_no] + spec_noise_norm
                    ##########

            count += 1

        ####################################################### After Loop ###########

        for k in range(2):
            # %% Plot stacked displacement
            stacked_disp = stack_stretch[k, :] * amp_azi
            stacked_velo = np.gradient(stacked_disp, timex)
            squared_velo = np.square(stacked_velo)
            stacked_ener = cumulative_trapezoid(squared_velo, timex, axis=-1, initial=0)
            stacked_ener = stacked_ener / (np.std(stacked_ener) + 1e-12) * amp_azi * 3

            ax[0, k].plot(timex, stacked_ener + np.max(dist_az[:, 1]) + (50 * 1 + 15), '-', color='0.0', linewidth=5)
            ax[0, k].plot(timex, stacked_disp + np.max(dist_az[:, 1]) + (50 * 1 + 10), '-', color='0.8', linewidth=5)

            # %% 0.05 and 0.9 of max energy
            max_energy = np.nanmax(stacked_ener[int(npts_trim / 3): 0 - int(10 / dt)])
            min_energy = np.nanmin(stacked_ener[int(npts_trim / 3): 0 - int(10 / dt)])
            pre = timex[stacked_ener < 0.05 * (max_energy - min_energy) + min_energy]
            pos = timex[stacked_ener > 0.90 * (max_energy - min_energy) + min_energy]
            pt_start = int(pre[-1] / dt)
            pt_end = int(pos[0] / dt)
            duration[k] = pos[0] - pre[-1]
            ax[0, k].plot(pt_start * dt, stacked_ener[pt_start] + np.max(dist_az[:, 1]) + (50 * 1 + 15), 'ok', ms=10)
            ax[0, k].plot(pt_end * dt, stacked_ener[pt_end] + np.max(dist_az[:, 1]) + (50 * 1 + 15), 'oy', ms=10)
            ax[0, k].plot(pt_start * dt, stacked_disp[pt_start] + np.max(dist_az[:, 1]) + (50 * 1 + 10), 'ok', ms=10)
            ax[0, k].plot(pt_end * dt, stacked_disp[pt_end] + np.max(dist_az[:, 1]) + (50 * 1 + 10), 'oy', ms=10)

            prom = np.nanmax(np.fabs(stacked_disp[pt_start:pt_end])) / 5
            peaks, _ = find_peaks(np.fabs(stacked_disp[pt_start:pt_end]), prominence=prom)
            peaks = peaks + pt_start
            ax[0, k].plot(timex[peaks], stacked_disp[peaks]+np.max(dist_az[:, 1])+(50*1+15),'x', ms=10, lw=3, color='orange')


        # %% plot stretch ratio against radiation pattern
        takeoffs_rad = np.sin(dist_az[ind_highsnr, 2] / 360 * np.pi) / np.cos(dist_az[ind_highsnr, 2] / 360 * np.pi)
        ray = ax[2, 3].scatter(dist_az[ind_highsnr, 1] / 180 * np.pi, takeoffs_rad, marker='o', c=ratio[ind_highsnr, 1],
                               s=cc[ind_highsnr, 1] * 100, edgecolors='w', cmap=cmap1, vmin=0.8, vmax=1.2)
        cb = plt.colorbar(ray, ax=ax[2, 3])
        cb.set_label('stretching ratio')

        # %% Average spectrum of all bins
        stack_spec = np.zeros((2, len(spec_norm)), dtype=np.double)
        ##########
        stack_spec_noise = np.zeros((2, len(spec_noise_norm)), dtype=np.double)
        ##########
        for m in range(8):
            azi_bins_noisy[m] = np.divide(azi_bins_noisy[m], bin_count[m]+0.01)
            stack_spec[0, :] = stack_spec[0, :] + azi_bins_noisy[m]
            ax[0, 2].loglog(freq, azi_bins_noisy[m], ls='-', color='0.8', lw=2)
            azi_bins_deno[m] = np.divide(azi_bins_deno[m], bin_count[m]+0.01)
            stack_spec[1, :] = stack_spec[1, :] + azi_bins_deno[m]
            ax[0, 3].loglog(freq, azi_bins_deno[m], ls='-', color='0.8', lw=2)

            ##########
            azi_bins_noisy_noise[m] = np.divide(azi_bins_noisy_noise[m], bin_count[m] + 0.01)
            stack_spec_noise[0, :] = stack_spec_noise[0, :] + azi_bins_noisy_noise[m]
            ax[0, 2].loglog(freq_noise, azi_bins_noisy_noise[m], ls='--', color='0.2', lw=1)
            azi_bins_deno_noise[m] = np.divide(azi_bins_deno_noise[m], bin_count[m] + 0.01)
            stack_spec_noise[1, :] = stack_spec_noise[1, :] + azi_bins_deno_noise[m]
            ax[0, 3].loglog(freq_noise, azi_bins_deno_noise[m], ls='--', color='0.2', lw=1)
            ##########

        # %% Model fitting the stacked spectrum
        for k in range(2):
            stack_spec[k, :] = stack_spec[k, :] / bin_num
            stack_spec_noise[k, :] = stack_spec_noise[k, :] / bin_num
            lowband = np.logical_and(freq > 0, freq < 1.0)
            n_best[k], fc_best[k] = fit_spec(freq[lowband], stack_spec[k, lowband])
            f_int[k] = flux_int(freq, stack_spec[k, :], fc_best[k], n_best[k])

            ##########
            ax[0, 2 + k].loglog(freq_noise, stack_spec_noise[k, :], ls='-', color='r', lw=5)
            ##########

            ax[0, 2+k].loglog(freq, stack_spec[k, :], ls='-', color='g', lw=5)
            ax[0, 2+k].loglog(freq, 1.0 / (1.0 + (freq/fc_best[k])**n_best[k]), ls='--', color='b', lw=5)
            ax[0, 2+k].loglog(fc_best[k], 0.5, 'o', markersize=15, mec='k', mfc='r')
            ax[0, 2+k].set_ylim(1e-15, 1e2)
            ax[0, 2+k].set_xlabel('frequency (Hz)', fontsize=24)
            ax[0, 2+k].set_ylabel('amplitude spectrum', fontsize=24)
            ax[0, 2+k].set_title(f'falloff {n_best[k]:.1f} fc {fc_best[k]:.2f} Er/M^2 {f_int[k]:.1e}', fontsize=30)

            # %% 3D direcivity fitting
            # vr[k], dir[k], inc[k], med[k] = directivity3d(dist_az[ind_highsnr, 1], dist_az[ind_highsnr, 2],
            #                                                ratio[ind_highsnr, k], src_meca[0, 0], src_meca[0, 1],
            #                                                cc[ind_highsnr, k])

            vr[k], dir[k], inc[k], med[k] = directivity3d_free(dist_az[ind_highsnr, 1],
                                                               dist_az[ind_highsnr, 2],
                                                               ratio[ind_highsnr, k],
                                                               cc[ind_highsnr, k])

            # %% axes limits, titles and labels
            ax[0, k].set_xlim(timex[0], timex[-1])
            ax[0, k].set_ylim(np.min(dist_az[:, 1]) - (50 * 1 + 10), np.max(dist_az[:, 1]) + (50 * 1 + 180))
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
                               f'inclination = {inc[k]:.0f} \n'
                               f'duration = {duration[k]:.1f} s /{med[k]:.2f}',
                               fontsize=30)

        # %% SNR histograms
        bins = np.linspace(0, 100, 20)
        ax[2, 2].hist(snr_before.flatten(), bins=bins, density=True,
                      histtype='stepfilled', color='0.9', alpha=0.5, label='noisy', lw=2)
        ax[2, 2].hist(snr_after.flatten(), bins=bins, density=True,
                      histtype='stepfilled', color='r', alpha=0.3, label='denoised', lw=2)
        ax[2, 2].set_xlabel('SNR', fontsize=24)
        ax[2, 2].set_ylabel('density', fontsize=24)
        ax[2, 2].legend(loc=1)

        # %% azimuth histograms
        bins = np.linspace(0, 360, 9)
        ax[1, 2].hist(dist_az[ind_highsnr, 1], bins=bins, density=True, histtype='stepfilled',
                      color='0.2', alpha=1, label=f'SNR>{min_snr}', lw=2, orientation='horizontal')
        ax[1, 2].set_xlabel('density', fontsize=24)
        ax[1, 2].set_ylabel('azimuth', fontsize=24)

        # %%
        ax[0, 0].set_title(f'Noisy Event {evid} depth={evdp:.0f} km / M{evmg:.1f}', fontsize=30)
        ax[0, 1].set_title('Denoised waves', fontsize=30)

        # %%
        for j in range(2):
            ax[0, 2+j].grid(which='major', color='#DDDDDD', linewidth=3)
            ax[0, 2+j].grid(which='minor', color='#EEEEEE', linestyle='--', linewidth=2)
            ax[0, 2+j].minorticks_on()

        # %% Save as a figure
        plt.savefig(fig_dir + '/quake_' + str(evid) + '_record_section.pdf')

        # %% Save denoised signal and spectra
        print(f'{evid} window {npts_trim} points | saving the spec')
        with h5py.File(fig_dir + '/' + str(evid) + '_timeDenoised_and_specStack.hdf5', 'w') as f:
            f.create_dataset("time_signal", data=denoised_signal)
            f.create_dataset("stack_spec_deno", data=stack_spec[1, :])
            f.create_dataset("stack_spec_noisy", data=stack_spec[0, :])
            f.create_dataset("stack_spec_freq", data=freq)

        return evid, evmg, evdp, \
               duration[1] / med[1], vr[1], dir[1], inc[1], \
               duration[0] / med[0], vr[0], dir[0], inc[0], \
               f_int[1], f_int[0], n_best[1], n_best[0], \
               fc_best[1], fc_best[0], num_tr, bin_num, \
               len(peaks), len(peaks_dir)

    else:
        return evid, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1


if __name__ == '__main__':
    DenoTe()
