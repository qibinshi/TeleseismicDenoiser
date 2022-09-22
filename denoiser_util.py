"""
@author: Qibin Shi
qibins@uw.edu
"""
import os
import glob
import torch
import matplotlib
import numpy as np
import pandas as pd
import scipy.signal as sgn
from scipy.signal import detrend
from distaz import DistAz
from obspy.taup import TauPyModel
from obspy import read_inventory
from obspy import read, UTCDateTime
from numpy.random import default_rng
from multitaper import MTSpec, MTCross
from scipy.interpolate import interp1d
from scipy.fft import fft, fftfreq, ifft
from scipy.integrate import cumulative_trapezoid, trapezoid
from torch_tools import Explained_Variance_score, CCLoss
from torch_tools import WaveformDataset
from torch.utils.data import DataLoader
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator

# Trim downloaded three components for same time range
def trim_align(trace):
    startt = trace[0].stats.starttime
    if startt < trace[1].stats.starttime:
        startt = trace[1].stats.starttime
    if startt < trace[2].stats.starttime:
        startt = trace[2].stats.starttime

    endt = trace[0].stats.endtime
    if endt > trace[1].stats.endtime:
        endt = trace[1].stats.endtime
    if endt > trace[2].stats.endtime:
        endt = trace[2].stats.endtime

    return trace.trim(startt, endt)


# Process each event for both signal and noises.
# This is to prepare big training data.
def process_single_event(ev, directory=None, npts=30000, noise_pts=36000, tp_after=2000.0):
    """Default 3000s window and 10Hz sampling rate
    for signal and preP noises. Our raw data starts
    3600s before and 3600s after the origin time.
    So it is safe to first cut P-3600s, P+2000s"""
    full_len = int(noise_pts + tp_after * 10)
    # initialize the metadata
    meta = pd.DataFrame(columns=[
        "source_id",
        "source_origin_time",
        "source_latitude_deg",
        "source_longitude_deg",
        "source_depth_km",
        "source_magnitude",
        "station_network_code",
        "station_code",
        "station_location_code",
        "station_latitude_deg",
        "station_longitude_deg",
        "trace_snr_db"])

    allpwave = np.zeros((0, npts, 3), dtype=np.double)
    allnoise = np.zeros((0, npts, 3), dtype=np.double)
    one_pwave = np.zeros((npts, 3), dtype=np.double)
    one_noise = np.zeros((npts, 3), dtype=np.double)
    pre_noise = np.zeros((noise_pts, 3), dtype=np.double)

    evnm = str(ev.resource_id)[13:]
    evlo = ev.origins[0].longitude
    evla = ev.origins[0].latitude
    evdp = ev.origins[0].depth / 1000.0
    org_t = ev.origins[0].time
    evmg = ev.magnitudes[0].mag

    # %% Loop over stations
    for sta in glob.glob(directory + evnm + 'stas/*xml'):
        inv = read_inventory(sta)
        stnw = inv[0].code
        stco = inv[0][0].code
        stla = inv[0][0].latitude
        stlo = inv[0][0].longitude
        stlc = inv[0][0][0].location_code
        result = DistAz(stla, stlo, evla, evlo)
        distdeg = result.getDelta()
        # backazi = result.getBaz()

        try:
            st0 = read(directory + evnm + "waves/" + stnw + "." + stco + "." + stlc + ".?H?_*")
            st = st0.copy()
        except:
            continue
        # try:
        #     st.remove_response(inventory=inv, output='VEL', pre_filt=pre_filt)
        # except:
        #     continue
        st.filter("lowpass", freq=2.0)
        st.resample(10)
        st.merge(fill_value=np.nan)
        model = TauPyModel(model="iasp91")
        arrivals = model.get_travel_times(source_depth_in_km=evdp, distance_in_degree=distdeg, phase_list=['P'])
        tp = UTCDateTime(org_t + arrivals[0].time)
        st.trim(tp - noise_pts / 10, tp + tp_after)

        if len(st) > 3:
            print(len(st), "channels", stco)
        if len(st) >= 3 and len(st[0].data) >= full_len and len(st[1].data) >= full_len and len(st[2].data) >= full_len:
            noise_amp = np.std(np.array(st[0].data)[noise_pts-1050: noise_pts-50])
            pwave_amp = np.std(np.array(st[0].data)[noise_pts: noise_pts+1000])

            if pwave_amp > (noise_amp * 25):
                for i in range(3):
                    one_pwave[:, i] = np.array(st[i].data)[full_len-npts: full_len]
                    pre_noise[:, i] = np.array(st[i].data)[0:noise_pts]

                # %% extract low amplitudes for noise
                amplitude_series = np.sqrt(np.sum(pre_noise ** 2, axis=1))
                amplitude_median = np.nanmedian(amplitude_series)
                noise0 = pre_noise[amplitude_series < (4 * amplitude_median), :]
                # %% make sure noise is long enough
                if noise0.shape[0] > npts / 2:
                    if noise0.shape[0] < npts:
                        noise0 = np.append(noise0, noise0, axis=0)
                    one_noise[:, :] = noise0[:npts, :]

                    # %% normalize p wave and noise
                    one_pwave[np.isnan(one_pwave)] = 0
                    one_noise[np.isnan(one_noise)] = 0
                    one_pwave = (one_pwave - np.mean(one_pwave, axis=0)) / (np.std(one_pwave, axis=0) + 1e-12)
                    one_noise = (one_noise - np.mean(one_noise, axis=0)) / (np.std(one_noise, axis=0) + 1e-12)

                    # %% Store the waveform pairs
                    allpwave = np.append(allpwave, one_pwave[np.newaxis, :, :], axis=0)
                    allnoise = np.append(allnoise, one_noise[np.newaxis, :, :], axis=0)

                    # %% Store the metadata of event-station
                    meta = pd.concat([meta, pd.DataFrame(data={
                        "source_id": evnm,
                        "source_origin_time": org_t,
                        "source_latitude_deg": "%.3f" % evla,
                        "source_longitude_deg": "%.3f" % evlo,
                        "source_depth_km": "%.3f" % evdp,
                        "source_magnitude": evmg,
                        "station_network_code": stnw,
                        "station_code": stco,
                        "station_location_code": stlc,
                        "station_latitude_deg": stla,
                        "station_longitude_deg": stlo,
                        "trace_snr_db": "%.3f" % (pwave_amp / (noise_amp + 1e-12))}, index=[0])], ignore_index=True)

    return allpwave, allnoise, meta


# Process each event for only signal.
# This is to prepare big training data.
def process_single_event_only(ev, directory=None, halftime=None, freq=4, rate=10, maxsnr=10, mindep=250, phase=1):
    # initialize the metadata
    meta = pd.DataFrame(columns=[
        "source_id",
        "source_origin_time",
        "source_latitude_deg",
        "source_longitude_deg",
        "source_depth_km",
        "source_magnitude",
        "source_strike",
        "source_dip",
        "source_rake",
        "station_network_code",
        "station_code",
        "station_location_code",
        "station_latitude_deg",
        "station_longitude_deg",
        "trace_snr_db",
        "trace_mean_0",
        "trace_stdv_0",
        "trace_mean_1",
        "trace_stdv_1",
        "trace_mean_2",
        "trace_stdv_2",
        "distance",
        "takeoff_p",
        "takeoff_phase",
        "azimuth"])

    npts = int(rate * halftime * 2)
    allpwave = np.zeros((0, npts, 3), dtype=np.double)
    one_pwave = np.zeros((npts, 3), dtype=np.double)

    evnm = str(ev.resource_id)[13:]
    evlo = ev.origins[0].longitude
    evla = ev.origins[0].latitude
    evdp = ev.origins[0].depth / 1000.0
    org_t = ev.origins[0].time
    evmg = ev.magnitudes[0].mag
    strike = ev.focal_mechanisms[0].nodal_planes.nodal_plane_1.strike
    dip = ev.focal_mechanisms[0].nodal_planes.nodal_plane_1.dip
    rake = ev.focal_mechanisms[0].nodal_planes.nodal_plane_1.rake
    pre_filt = (0.004, 0.005, 10.0, 12.0)

    if evdp > mindep:
        # %% Loop over stations
        for sta in glob.glob(directory + evnm + 'stas/*xml'):
            inv = read_inventory(sta)
            stnw = inv[0].code
            stco = inv[0][0].code
            stla = inv[0][0].latitude
            stlo = inv[0][0].longitude
            stlc = inv[0][0][0].location_code
            result = DistAz(stla, stlo, evla, evlo)
            distdeg = result.getDelta()
            azimuth = result.getAz()

            try:
                st0 = read(directory + evnm + "waves/" + stnw + "." + stco + "." + stlc + ".?H?_*")
                st = st0.copy()
                st.filter("lowpass", freq=freq)
            except:
                continue
            # try:
            #     st.remove_response(inventory=inv, output='VEL', pre_filt=pre_filt)
            # except:
            #     continue
            # st.filter("lowpass", freq=freq)
            st.resample(rate)
            st.merge(fill_value=np.nan)
            st1 = st.copy()
            model = TauPyModel(model="iasp91")
            arrivals = model.get_travel_times(source_depth_in_km=evdp, distance_in_degree=distdeg, phase_list=['P', 'S'])
            tp = UTCDateTime(org_t + arrivals[0].time)
            tphase = UTCDateTime(org_t + arrivals[phase].time)
            takeoff_p = arrivals[0].takeoff_angle
            takeoff_phase = arrivals[phase].takeoff_angle
            st.trim(tphase - halftime, tphase + halftime)
            st1.trim(tp - 50.0, tp + 50.0)

            if len(st) > 3:
                print(len(st), "channels", stco)
            if len(st) >= 3 and len(st[0].data) >= npts and len(st[1].data) >= npts and len(st[2].data) >= npts:
                noise_amp = np.std(np.array(st1[2].data)[0:int(rate*45.0)])
                pwave_amp = np.std(np.array(st1[2].data)[int(rate*50.0):])

                if pwave_amp < (noise_amp * maxsnr):
                # if pwave_amp > (noise_amp * 20):
                    for i in range(3):
                        one_pwave[:, i] = np.array(st[i].data)[0:npts]
                    one_pwave[np.isnan(one_pwave)] = 0
                    scale_mean = np.mean(one_pwave, axis=0, keepdims=True)
                    scale_stdv = np.std(one_pwave, axis=0, keepdims=True) + 1e-12
                    one_pwave = (one_pwave - scale_mean) / scale_stdv
                    allpwave = np.append(allpwave, one_pwave[np.newaxis, :, :], axis=0)

                    # %% Store the metadata of event-station
                    meta = pd.concat([meta, pd.DataFrame(data={
                        "source_id": evnm,
                        "source_origin_time": org_t,
                        "source_latitude_deg": "%.3f" % evla,
                        "source_longitude_deg": "%.3f" % evlo,
                        "source_depth_km": "%.3f" % evdp,
                        "source_magnitude": evmg,
                        "source_strike": strike,
                        "source_dip": dip,
                        "source_rake": rake,
                        "station_network_code": stnw,
                        "station_code": stco,
                        "station_location_code": stlc,
                        "station_latitude_deg": stla,
                        "station_longitude_deg": stlo,
                        "trace_snr_db": "%.3f" % (pwave_amp / (noise_amp + 1e-12)),
                        "trace_mean_0": scale_mean[0, 0],
                        "trace_stdv_0": scale_stdv[0, 0],
                        "trace_mean_1": scale_mean[0, 1],
                        "trace_stdv_1": scale_stdv[0, 1],
                        "trace_mean_2": scale_mean[0, 2],
                        "trace_stdv_2": scale_stdv[0, 2],
                        "distance": distdeg,
                        "takeoff_p": takeoff_p,
                        "takeoff_phase": takeoff_phase,
                        "azimuth": azimuth}, index=[0])], ignore_index=True)

    return allpwave, meta


"""
#################################################################
###################    Plotting figures    ######################
###################    1) record section    #####################
###################    2) time & spec      ######################
#################################################################
"""


def plot_record_section(evid, meta_all=None, X_train=None, model=None, fig_dir=None, dt=0.1, npts=None, npts_trim=None, normalized_stack=True):

    meta = meta_all[(meta_all.source_id == evid)]
    dist_az = meta[['distance', 'azimuth', 'takeoff_phase']].to_numpy()
    trace_amp = meta[['trace_snr_db', 'trace_stdv_2']].to_numpy()
    src_meca = meta[['source_strike', 'source_dip', 'source_rake']].to_numpy()
    idx_list = meta.index.values
    batch_size = len(idx_list)
    evdp = meta.source_depth_km.unique()[0]
    evmg = meta.source_magnitude.unique()[0]
    npts_trim = min(npts_trim, int(evdp/2.2/dt))

    # %% load the waveforms of the event
    X_1 = X_train[idx_list]
    test_data = WaveformDataset(X_1, X_1)
    test_iter = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    data_iter = iter(test_iter)
    X0, y0 = data_iter.next()
    nbatch = X0.size(0)

    X = torch.zeros(nbatch, X0.size(1), npts, dtype=torch.float64)
    z_std = np.zeros((batch_size, 2), dtype=np.float64)

    for i in np.arange(nbatch):
        quake_one = X0[i, :, :]
        scale_mean = torch.mean(quake_one, dim=1)
        scale_std = torch.std(quake_one, dim=1) + 1e-12
        trace_amp[i, 1] = trace_amp[i, 1] * scale_std[2]
        for j in np.arange(X0.size(1)):
            quake_one[j] = torch.div(torch.sub(quake_one[j], scale_mean[j]), scale_std[j])
        X[i] = quake_one

    # %% apply denoising
    with torch.no_grad():
        quake_denoised, noise_output = model(X)
    noisy_signal = X.numpy()
    denoised_signal = quake_denoised.numpy()

    # %% SNR stats of velocity waveforms
    noise_amp = np.std(noisy_signal[:, :, int(npts/2)-250:int(npts/2)-50], axis=-1)
    signl_amp = np.std(noisy_signal[:, :, int(npts/2):int(npts/2)+200], axis=-1)
    snr_before = 20 * np.log10(np.divide(signl_amp, noise_amp))

    noise_amp = np.std(denoised_signal[:, :, int(npts / 2) - 250:int(npts / 2) - 50], axis=-1)
    signl_amp = np.std(denoised_signal[:, :, int(npts / 2):int(npts / 2) + 200], axis=-1)
    snr_after = 20 * np.log10(np.divide(signl_amp, noise_amp))

    ####################### Save the denoised traces to disk #########################
    # with h5py.File(ev_dir + '/' + str(evid) + '_quake_and_noise.hdf5', 'w') as f:
    #     f.create_dataset("pwave", data=denoised_signal)
    #     f.create_dataset("noise", data=separated_noise)

    # %% trim
    timex = np.arange(0, npts_trim) * dt
    startpt = int((npts - npts_trim) / 2)
    endpt = int((npts + npts_trim) / 2)

    # %% integrate
    noisy_signal = cumulative_trapezoid(noisy_signal[:, :, startpt:endpt], timex, axis=-1, initial=0)
    denoised_signal = cumulative_trapezoid(denoised_signal[:, :, startpt:endpt], timex, axis=-1, initial=0)

    # %% normalized the displacement waveforms
    scale_std = np.std(denoised_signal, axis=-1, keepdims=True) + 1e-12
    denoised_signal = (denoised_signal - np.mean(denoised_signal, axis=-1, keepdims=True)) / scale_std
    z_std[:, 1] = trace_amp[:, 1] * scale_std[:, 2, 0]
    scale_std = np.std(noisy_signal, axis=-1, keepdims=True) + 1e-12
    noisy_signal = (noisy_signal - np.mean(noisy_signal, axis=-1, keepdims=True)) / scale_std
    z_std[:, 0] = trace_amp[:, 1] * scale_std[:, 2, 0]

    # %% Discard outliers for stacking the relative amplitudes
    ave_z_std = np.mean(z_std, axis=0)
    for k in range(2):
        indX = np.where(z_std[:, k] > 20 * ave_z_std[k])[0]
        z_std[indX, k] = 0
    total_std_z = np.sum(z_std, axis=0)

    # %% Reference for pre-stack
    id_az_min = np.argmin(dist_az[:, 1])
    if trace_amp[id_az_min, 0] > 1.5:
        id_ref = id_az_min
    else:
        id_ref = np.argmax(trace_amp[:, 0])
    ref_noisy = noisy_signal[id_ref, 2, :]
    ref_clean = denoised_signal[id_ref, 2, :]

    ###############################################################
    ###############################################################
    # %% Plot record sections and key measurements
    plt.close('all')
    color = ['k', 'b']
    f_int = ['', '']
    n_best = ['', '']
    fc_best = ['', '']
    f_int_st = ['', '']
    n_best_st = ['', '']
    fc_best_st = ['', '']
    stack_spec = ['', '']
    stack_spec_st = ['', '']
    vr = [0.0, 0.0]
    med = [1.0, 1.0]
    dir = [0.0, 0.0]
    duration = [1.0, 1.0]
    cc = np.zeros((batch_size, 2))
    ratio = np.zeros((batch_size, 2))
    flip = np.zeros((batch_size, 2), dtype=np.int32)
    shift = np.zeros((batch_size, 2), dtype=np.int32)
    stack = np.zeros((2, timex.shape[-1]), dtype=np.float64)
    stack_stretch = np.zeros((2, timex.shape[-1]), dtype=np.float64)

    fig, ax = plt.subplots(3, 4, figsize=(40, 30), constrained_layout=True)
    ax[1, 1] = plt.subplot(346, projection='polar')
    ax[1, 0] = plt.subplot(345, projection='polar')
    ax[2, 1] = plt.subplot(3, 4, 10, projection='polar')
    ax[2, 0] = plt.subplot(349, projection='polar')
    ax[2, 3] = plt.subplot(3, 4, 12, projection='polar')
    amp_azi = 12
    cmap = plt.get_cmap('PiYG')
    cmap1 = plt.get_cmap('Oranges')

    # %% Radiation patten projected to the lower hemisphere
    y, x = np.mgrid[slice(0, 90 + 5, 5), slice(0, 360 + 5, 5)]
    r, z = p_rad_pat(src_meca[0, 0], src_meca[0, 1], src_meca[0, 2], y, x)
    levels = MaxNLocator(nbins=50).tick_values(z.min(), z.max())
    im = ax[2, 3].contourf((x + 2.5) / 180 * np.pi, r, z, levels=levels, cmap=cmap)  ## as contours
    cb = fig.colorbar(im, ax=ax[2, 3])
    cb.set_label('radiation pattern')
    ax[2, 3].set_theta_zero_location('N')
    ax[2, 3].set_theta_direction(-1)
    ax[2, 3].yaxis.set_major_locator(MultipleLocator(2))
    ax[2, 3].set_ylim(0, 0.6)
    ax[2, 3].set_title('Radiation Pattern and Ray directions', fontsize=30)

    ############################################################### Stack Loop ###########
    for i in range(batch_size):
        tmp_tr = np.zeros((2, timex.shape[-1]), dtype=np.float64)

        # %% Re-align traces
        tr_noisy, shift[i, 0], flip[i, 0] = shift2maxcc(ref_noisy, noisy_signal[i, 2, :], maxshift=30, flip_thre=-0.3)
        tr_clean, shift[i, 1], flip[i, 1] = shift2maxcc(ref_clean, denoised_signal[i, 2, :],maxshift=30,flip_thre=-0.3)
        tmp_tr[0, :] = shift_pad_fix_time(tr_noisy, shift[i, 0])
        tmp_tr[1, :] = shift_pad_fix_time(tr_clean, shift[i, 1])

        # %% Pre-stack noisy and clean waves
        for k in range(2):
            if normalized_stack:
                stack[k, :] = stack[k, :] + tmp_tr[k, :] / batch_size
            else:
                stack[k, :] = stack[k, :] + tmp_tr[k, :] * z_std[i, k] / total_std_z[k]

    ############################################################# Stretch Loop ###########
    for i in range(batch_size):
        tmp_tr = np.zeros((2, timex.shape[-1]), dtype=np.float64)

        # %% Stretch and align with the pre-stacked wave
        time_clean,wave_clean,ratio[i, 1],cc[i, 1],flip[i, 1]=dura_cc(stack[1],denoised_signal[i, 2, :],timex,maxshift=30,max_ratio=2)
        time_noisy,wave_noisy,ratio[i, 0],cc[i, 0],flip[i, 0]=dura_cc(stack[0],noisy_signal[i, 2, :], timex,maxshift=30, max_ratio=2)
        ax[0, 1].plot(time_clean, wave_clean * amp_azi + dist_az[i, 1], color=color[flip[i, 1]], ls='-', lw=1)
        ax[0, 0].plot(time_noisy, wave_noisy * amp_azi + dist_az[i, 1], color=color[flip[i, 0]], ls='-', lw=1)
        tmp_tr[1, :] = shift_pad_stretch_time(wave_clean, timex, time_clean)
        tmp_tr[0, :] = shift_pad_stretch_time(wave_noisy, timex, time_noisy)

        for k in range(2):
            # %% Stack the stretched time series
            if normalized_stack:
                stack_stretch[k, :] = stack_stretch[k, :] + tmp_tr[k, :] / batch_size
            else:
                stack_stretch[k, :] = stack_stretch[k, :] + tmp_tr[k, :] * z_std[i, k] / total_std_z[k]

            clr = color[flip[i, k]]
            ax[1, k].plot(dist_az[i, 1]/180*np.pi, ratio[i, k], marker='o', mfc=clr, mec=clr, ms=cc[i, k] * 20)
            ax[2, k].plot(dist_az[i, 2]/180*np.pi, ratio[i, k], marker='o', mfc=clr, mec=clr, ms=cc[i, k] * 20)

            # %% MTSpec --Un-stretched
            if k == 0:
                Py = MTSpec(detrend(noisy_signal[i, 2, int(npts_trim / 2 - 10): 0 - int(npts_trim / 6)]), 4.0, 6, dt)
            else:
                Py = MTSpec(detrend(denoised_signal[i, 2, int(npts_trim / 2 - 10): 0 - int(npts_trim / 6)]), 4.0, 6, dt)
            freq, spec = Py.rspec()
            max_spec = np.nanmax(spec[freq < 0.05])
            spec_norm = spec / max_spec * np.exp(freq * 0.15)
            ax[0, 2 + k].loglog(freq, spec_norm, ls='-', color='0.8', lw=2)
            # spec_crct = spec * distance_factor / (radiation_pattern + 1e-12)
            # ax[1, 2+k].loglog(freq, spec_crct, ls='-', color='0.8', lw=2)

            if i == 0:
                stack_spec[k] = spec_norm
            else:
                stack_spec[k] = spec_norm + stack_spec[k]

            # %% MTSpec --stretched
            # Py_stretch = MTSpec(tmp_tr[k, int(npts_trim / 5): 0 - int(npts_trim / 6)], 4.0, 6, dt)
            # freq_st, spec_st = Py_stretch.rspec()
            # max_spec = np.nanmax(spec_st[freq_st < 0.05])
            # spec_norm_st = spec_st / max_spec * np.exp(freq_st * 0.15)
            # ax[1, 2+k].loglog(freq_st, spec_norm_st, ls='-', color='0.8', lw=2)
            #
            # if i == 0:
            #     stack_spec_st[k] = spec_norm_st
            # else:
            #     stack_spec_st[k] = spec_norm_st + stack_spec_st[k]

    ####################################################### After Stretch Loop ###########
    # %% project duration ratios to radiation pattern
    takeoffs_rad = np.sin(dist_az[:, 2]/360*np.pi) / np.cos(dist_az[:, 2]/360*np.pi)
    ray = ax[2, 3].scatter(dist_az[:, 1]/180*np.pi, takeoffs_rad, marker='o', c=ratio[:, 1], s=cc[:, 1] * 100, edgecolors='w', cmap=cmap1, vmin=0.8, vmax=1.2)
    cb = plt.colorbar(ray, ax=ax[2, 3])
    cb.set_label('relative duration')

    for k in range(2):
        # %% Model the stacked spectrum --Un-stretched
        stack_spec[k] = stack_spec[k] / batch_size
        n_best[k], fc_best[k] = fit_spec(freq[np.logical_and(freq < 1, freq > 0)], stack_spec[k][np.logical_and(freq < 1, freq > 0)])
        f_int[k] = flux_int(freq, stack_spec[k], fc_best[k], n_best[k])
        ax[0, 2+k].loglog(freq, stack_spec[k], ls='-', color='g', lw=5)
        ax[0, 2+k].loglog(freq, 1.0 / (1.0 + (freq/fc_best[k])**n_best[k]), ls='--', color='b', lw=5)
        ax[0, 2+k].set_ylim(1e-15, 1e2)
        ax[0, 2+k].set_xlabel('frequency (Hz)', fontsize=24)
        ax[0, 2+k].set_ylabel('amplitude spectrum', fontsize=24)
        ax[0, 2+k].set_title(f'falloff {n_best[k]:.1f} fc {fc_best[k]:.2f} Er/M^2 {f_int[k]:.1e}', fontsize=30)

        # %% Model the stacked spectrum --stretched
        # stack_spec_st[k] = stack_spec_st[k] / batch_size
        # n_best_st[k], fc_best_st[k] = fit_spec(freq_st[np.logical_and(freq_st < 1, freq_st > 0)], stack_spec_st[k][np.logical_and(freq_st < 1, freq_st > 0)])
        # f_int_st[k] = flux_int(freq_st, stack_spec_st[k], fc_best_st[k], n_best_st[k])
        # ax[1, 2+k].loglog(freq_st, stack_spec_st[k], ls='-', color='g', lw=5)
        # ax[1, 2+k].loglog(freq_st, 1.0 / (1.0 + (freq_st / fc_best_st[k]) ** n_best_st[k]), ls='--', color='b', lw=5)
        # ax[1, 2+k].set_ylim(1e-15, 1e2)
        # ax[1, 2+k].set_xlabel('frequency (Hz)', fontsize=24)
        # ax[1, 2+k].set_ylabel('amplitude spectrum', fontsize=24)
        # ax[1, 2+k].set_title(f'(Stretched)falloff {n_best_st[k]:.1f} fc {fc_best_st[k]:.2f} Er/M {f_int_st[k]:.1e}', fontsize=30)

        # %% Model the directivity ellipse
        vr[k], med[k], dir[k] = ellipse_directivity(dist_az[:, 1], ratio[:, k], cc[:, k])
        azimuth = np.arange(0.0, 360.0, 2.0, dtype=np.float64)
        r_pred = med[k] / (1 - vr[k] * np.cos((azimuth - dir[k]) / 180.0 * np.pi))
        if vr[k] > 0.1:
            clr = '-r'
        else:
            clr = '-y'
        ax[1, k].plot(azimuth / 180.0 * np.pi, r_pred, clr, linewidth=12, alpha=0.2)

        # %% Plot the stacked time series
        stacked_disp = stack_stretch[k, :] * amp_azi
        stacked_velo = np.gradient(stacked_disp, timex)
        stacked_acce = np.gradient(stacked_velo, timex)
        stacked_ener = cumulative_trapezoid(np.square(stacked_velo), timex, axis=-1, initial=0)
        stacked_ener = stacked_ener / (np.std(stacked_ener) + 1e-12) * amp_azi * 3
        ax[0, k].plot(timex, stacked_ener + np.max(dist_az[:, 1]) + (50 * 1 + 15), '-r', linewidth=5)
        ax[0, k].plot(timex, stacked_disp + np.max(dist_az[:, 1]) + (50 * 1 + 10), '-g', linewidth=5)
        ax[0, k].set_xlim(timex[0], timex[-1])

        # %% MTSpec stacked velocity
        # Py_v= MTSpec(stacked_velo[int(npts_trim / 5): 0 - int(npts_trim / 6)], 4.0, 6, dt)
        # Py_a = MTSpec(stacked_acce[int(npts_trim / 5): 0 - int(npts_trim / 6)], 4.0, 6, dt)
        # Py_d = MTSpec(stacked_disp[int(npts_trim / 5): 0 - int(npts_trim / 6)], 4.0, 6, dt)
        # freq_v, spec_v = Py_v.rspec()
        # freq_a, spec_a = Py_a.rspec()
        # freq_d, spec_d = Py_d.rspec()
        # max_a = np.nanmax(spec_a)
        # corner = freq_a[spec_a > 0.85 * max_a]
        #
        # if k == 0:
        #     ax[0, 2].loglog(freq_v, spec_v, ls='-', color='0.3', label='noisy stacked velocity', lw=2)
        #     ax[0, 2].loglog(freq_a, spec_a, ls='-', color='0.0', label='noisy stacked acceleration', lw=2)
        #     ax[0, 2].loglog(freq_d, spec_d, ls='-', color='0.6', label='noisy stacked displacement', lw=2)
        #     org_ener_noisy = trapezoid(np.square(spec_v[freq_v < 5]), freq_v[freq_v < 5], axis=-1)
        #     Res_ener_noisy = np.sum(np.square(spec_v[freq_v == corner[0]]) * corner[0])
        #     Nor_ener_noisy = trapezoid(np.square(spec_v[freq_v < corner[0]]), freq_v[freq_v < corner[0]], axis=-1)
        #
        # else:
        #     ax[0, 2].loglog(freq_v, spec_v, '-r', label='denoised stacked velocity', lw=2)
        #     ax[0, 2].loglog(freq_a, spec_a, '-c', label='denoised stacked acceleration', lw=2)
        #     ax[0, 2].loglog(freq_d, spec_d, '-g', label='denoised stacked displacement', lw=2)
        #     org_ener_clean = trapezoid(np.square(spec_v[freq_v < 5]), freq_v[freq_v < 5], axis=-1)
        #     Res_ener_clean = np.sum(np.square(spec_v[freq_v == corner[0]]) * corner[0])
        #     Nor_ener_clean = trapezoid(np.square(spec_v[freq_v < corner[0]]), freq_v[freq_v < corner[0]], axis=-1)
        #
        #     print(Res_ener_clean, Nor_ener_clean, org_ener_clean)
        #     ax[0, 2].legend(loc=3)
        #     ax[0, 2].set_xlabel('frequency (Hz)', fontsize=24)
        #     ax[0, 2].set_ylabel('amplitude spectrum', fontsize=24)
            # ax[0, 2].axvline(corner[0], linestyle='--', color='k')
            # ax[0, 2].axvline(corner[-1], linestyle='--', color='k')
            # ax[0, 2].set_title(f'Res-E={Res_ener_clean:.0f} low-f-E={Nor_ener_clean:.0f}', fontsize=30)

        # # %% STA/LTA of stacked waves
        # pt_start, pt_end, sta_lta1, sta_lta2, sta, lta = dura_amp(stacked_disp)
        # %% 0.05 and 0.95 of max energy
        max_energy = np.nanmax(stacked_ener[int(npts_trim/3):0-int(npts_trim/6)])
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
        ax[1, k].set_title(f'Vrup/V = {vr[k]:.3f} \ndirectivity = {dir[k]:.0f} \nduration = {duration[k]:.1f} s /{med[k]:.2f}', fontsize=30)

    # %% SNR histograms
    bins = np.linspace(0, 100, 20)
    ax[2, 2].hist(snr_before.flatten(), bins=bins, density=True, histtype='stepfilled', color='0.6', alpha=0.5, label='noisy', lw=2)
    ax[2, 2].hist(snr_after.flatten(), bins=bins, density=True, histtype='stepfilled', color='r', alpha=0.5, label='denoised', lw=2)
    ax[2, 2].set_xlabel('SNR', fontsize=24)
    ax[2, 2].set_ylabel('density', fontsize=24)
    ax[2, 2].legend(loc=1)

    ax[0, 0].set_title(f'Noisy Event {evid} depth={evdp:.0f} km / M{evmg:.1f}', fontsize=30)
    ax[0, 1].set_title('Denoised waves', fontsize=30)

    plt.savefig(fig_dir + '/quake_' + str(evid) + '_record_section_P.png')

    # return evmg, evdp, duration[1]/med[1], vr[1], dir[1], duration[0]/med[0], vr[0], dir[0], Res_ener_clean, Nor_ener_clean, org_ener_clean, Res_ener_noisy, Nor_ener_noisy, org_ener_noisy
    return evmg, evdp, duration[1] / med[1], vr[1], dir[1], duration[0] / med[0], vr[0], dir[0], f_int[1], f_int[0], n_best[1], n_best[0], fc_best[1], fc_best[0], batch_size

def plot_application(noisy_signal, denoised_signal, separated_noise, idx, directory=None, dt=0.1, npts=None):
    plt.close("all")
    comps = ['E', 'N', 'Z']
    gs_kw = dict(height_ratios=[1, 1, 1, 3])
    time = np.arange(0, npts) * dt
    fig, ax = plt.subplots(4, 3, gridspec_kw=gs_kw, figsize=(12, 10), constrained_layout=True)

    for i in range(3):
        scaling_factor = np.max(abs(noisy_signal[i, :]))
        _, spect_noisy_signal = waveform_fft(noisy_signal[i, :] / scaling_factor, dt)
        _, spect_noise = waveform_fft(separated_noise[i, :] / scaling_factor, dt)
        freq, spect_denoised_signal = waveform_fft(denoised_signal[i, :] / scaling_factor, dt)

        ax[i, 0].plot(time, noisy_signal[i, :] / scaling_factor, '-k', label='Noisy signal', linewidth=1)
        ax[i, 0].plot(time, denoised_signal[i, :] / scaling_factor, '-r', label='Predicted signal', linewidth=1)
        ax[i, 1].plot(time, denoised_signal[i, :] / scaling_factor, '-r', label='Predicted signal', linewidth=1)
        ax[i, 2].plot(time, separated_noise[i, :] / scaling_factor, '-b', label='Predicted noise', linewidth=1)
        ax[3, i].loglog(freq, spect_noisy_signal, '-k', label='raw signal', linewidth=0.5, alpha=1)
        ax[3, i].loglog(freq, spect_denoised_signal, '-r', label='separated earthquake', linewidth=0.5, alpha=1)
        ax[3, i].loglog(freq, spect_noise, '-', color='b', label='noise', linewidth=0.5, alpha=0.8)

        ax[i, 0].set_ylabel(comps[i], fontsize=16)
        ax[3, i].set_xlabel('Frequency (Hz)', fontsize=14)
        ax[3, i].set_title(comps[i], fontsize=16)
        ax[3, i].grid(alpha=0.2)

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
    ax[0, 1].set_title("Denoised P-wave", fontsize=16)
    ax[0, 2].set_title("Separated noise", fontsize=16)
    ax[3, 0].set_ylabel('velocity spectra', fontsize=14)
    plt.legend()
    plt.savefig(directory + f'/time_spec_{idx}.pdf')


def plot_testing(noisy_signal, denoised_signal, separated_noise, clean_signal, true_noise, idx, sqz, directory=None, dt=0.1, npts=None):
    quake_denoised = torch.from_numpy(denoised_signal)
    quake_label = torch.from_numpy(clean_signal)
    noise_output = torch.from_numpy(separated_noise)
    noise_label = torch.from_numpy(true_noise)
    ev_score = Explained_Variance_score()
    loss_fn = CCLoss()
    comps = ['E', 'N', 'Z']
    scores = np.zeros((1, 3, 4))
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

        evs_earthquake = ev_score(quake_denoised[i, :], quake_label[i, :])
        evs_noise = ev_score(noise_output[i, :], noise_label[i, :])
        cc_quake = 1 - loss_fn(quake_denoised[i, :], quake_label[i, :])
        cc_noise = 1 - loss_fn(noise_output[i, :], noise_label[i, :])
        scores[0, i, 0] = evs_earthquake
        scores[0, i, 1] = evs_noise
        scores[0, i, 2] = cc_quake
        scores[0, i, 3] = cc_noise

        ax[i, 0].plot(time, noisy_signal[i, :] / scaling_factor, '-k', label='Noisy signal', linewidth=1)
        ax[i, 0].plot(time, clean_signal[i, :] / scaling_factor, '-r', label='True signal', linewidth=1)
        ax[i, 1].plot(time, clean_signal[i, :] / scaling_factor, '-r', label='True signal', linewidth=1)
        ax[i, 1].plot(time, denoised_signal[i, :] / scaling_factor, '-b', label='Predicted signal', linewidth=1)
        ax[i, 2].plot(time, true_noise[i, :] / scaling_factor, '-', color='gray', label='True noise', linewidth=1)
        ax[i, 2].plot(time, separated_noise[i, :] / scaling_factor, '-b', label='Predicted noise', linewidth=1)
        ax[3, i].loglog(freq, spect_noisy_signal, '-k', label='raw signal', linewidth=0.5, alpha=1)
        ax[3, i].loglog(freq, spect_clean_signal, '-r', label='true earthquake', linewidth=0.5, alpha=1)
        ax[3, i].loglog(freq, spect_denoised_signal, '-b', label='separated earthquake', linewidth=0.5, alpha=1)
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


def shift_pad_fix_time(wave, shift_pt=0):
    tmp_tr = np.zeros(wave.shape, dtype=np.float64)
    if shift_pt > 0:
        tmp_tr[shift_pt:] = wave[0:0 - shift_pt]
    elif shift_pt < 0:
        tmp_tr[0:shift_pt] = wave[0 - shift_pt:]
    else:
        tmp_tr[:] = wave[:]
    return tmp_tr


def shift_pad_stretch_time(wave, time1, time2):
    tmp_tr = np.zeros(time1.shape, dtype=np.float64)
    dt = time1[1] - time1[0]
    left_time = max(time1[0], time2[0])
    right_time = min(time1[-1], time2[-1])
    stack_len = int((right_time - left_time) / dt)
    idx1 = int((left_time - time1[0]) / dt)
    idx2 = int((left_time - time2[0]) / dt)
    tmp_tr[idx1: idx1 + stack_len] = wave[idx2: idx2 + stack_len]

    return tmp_tr


def shift2maxcc(wave1, wave2, correction=0, maxshift=500, flip_thre=-0.3):
    n1 = np.sum(np.square(wave1))
    n2 = np.sum(np.square(wave2))
    corr = sgn.correlate(wave1, wave2) / np.sqrt(n1 * n2)
    lags = sgn.correlation_lags(len(wave1), len(wave2))

    l_maxshift = min(len(wave2) + correction, maxshift)
    r_maxshift = min(len(wave1) - correction, maxshift)

    st_pt = len(wave2) + correction - l_maxshift
    en_pt = len(wave2) + correction + r_maxshift

    ind1 = np.argmax(corr[st_pt: en_pt]) + st_pt
    ind2 = np.argmin(corr[st_pt: en_pt]) + st_pt

    flipsign = 0

    if 0 - corr[ind1] > corr[ind2] and corr[ind2] < flip_thre:
        ind1 = ind2
        wave2 = 0 - wave2
        flipsign = 1
    elif corr[ind1] < 0.3:
        ind1 = len(wave2) + correction

    return wave2, lags[ind1], flipsign


def dura_cc(wave1, wave2, time, maxshift=30, max_ratio=2, flip_thre=-0.3):

    interp_f = interp1d(time, wave2, bounds_error=False, fill_value=0.)
    n1 = np.sum(np.square(wave1))
    dt = time[1] - time[0]
    cc = 0
    relative_ratio = 1
    npts = len(time)

    for ratio in np.arange(1/max_ratio, max_ratio, 0.01):
        dt_new = dt / ratio
        time_new = np.arange(time[0], time[-1], dt_new)
        wave_new = interp_f(time_new)
        correction = int((npts - len(wave_new)) / 2)
        n2 = np.sum(np.square(wave_new))
        corr = sgn.correlate(wave1, wave_new) / np.sqrt(n1 * n2)

        l_maxshift = min(len(wave_new) + correction, maxshift)
        r_maxshift = min(len(wave1) - correction, maxshift)

        st_pt = len(wave_new) + correction - l_maxshift
        en_pt = len(wave_new) + correction + r_maxshift

        cc_max = np.nanmax(corr[st_pt: en_pt])
        cc_min = np.nanmin(corr[st_pt: en_pt])

        if 0 - cc_max > cc_min and cc_min < flip_thre:
            cc_best = 0 - cc_min
        else:
            cc_best = cc_max

        if cc < cc_best:
            cc = cc_best
            relative_ratio = ratio

    dt_new = dt / relative_ratio
    time_new = np.arange(time[0], time[-1], dt_new)
    wave_new = interp_f(time_new)
    num_pts = len(wave_new)
    correct = int((npts - num_pts) / 2)

    # %% shift the new trace for the max cc
    wave_out, shift, flipsign = shift2maxcc(wave1, wave_new, correction=correct, maxshift=maxshift, flip_thre=flip_thre)
    timex = np.linspace(time[0] + shift * dt, time[0] + (shift + num_pts) * dt, num=num_pts)

    return timex, wave_out, relative_ratio, cc, flipsign


def dura_amp(wave2, long=10, short=5):

    wave_abs = np.fabs(wave2) + 1e-12
    long_average = np.convolve(wave_abs, np.ones(long), 'full')/float(long)
    short_average = np.convolve(wave_abs, np.ones(short), 'full')/float(short)
    ratio1 = np.log10(short_average[:1-short]) - np.log10(long_average[:1-long])
    ratio2 = np.log10(short_average[short-1:]) - np.log10(long_average[long-1:])
    # peak_abs = np.argmax(long_average[:1-long])
    pt1 = np.argmax(ratio1[70:int(len(wave_abs)/2+50)]) + 70
    pt2 = np.argmax(ratio2[int(len(wave_abs)/2-50):1-long]) + int(len(wave_abs)/2-50)

    return pt1, pt2, ratio1, ratio2, short_average, long_average


def ellipse_directivity(azimuth, ratio, wgt, k1_itv=0.002, k2_itv=0.1, k3_itv=5.0):
    k1 = np.arange(0.0, 0.4, k1_itv, dtype=np.float64)
    k2 = np.arange(0.6, 1.5, k2_itv, dtype=np.float64)
    k3 = np.arange(0.0, 360.0, k3_itv, dtype=np.float64)
    err_min = 1000
    vr0 = 0.0
    med0 = 1.0
    dir0 = 0.0
    for vr in k1:
        for med in k2:
            for dir in k3:
                r_pred = med / (1 - vr * np.cos((azimuth - dir) / 180.0 * np.pi))
                mse = np.mean(np.square(np.multiply((r_pred - ratio), wgt)))
                mae = np.mean(np.fabs(np.multiply((r_pred - ratio), wgt)))
                if mse + mae * mae < err_min:
                    err_min = mse + mae * mae
                    vr0 = vr
                    med0 = med
                    dir0 = dir
    return vr0, med0, dir0

def p_rad_pat(strike, dip, rake, takeoff, azimuth):
    strike = strike / 180.0 * np.pi
    dip = dip / 180.0 * np.pi
    rake = rake / 180.0 * np.pi
    takeoff = takeoff / 180.0 * np.pi
    azimuth = azimuth / 180.0 * np.pi

    dd = azimuth - strike
    aa = np.sin(dd)
    bb = np.sin(2 * takeoff)
    cc = np.square(np.sin(takeoff))
    ee = np.cos(takeoff)
    ff = np.cos(rake)
    gg = np.sin(rake)
    hh = np.sin(dip) * ff * cc * np.sin(2 * dd)
    ii = np.cos(dip) * ff * bb * np.cos(dd)
    jj = np.sin(2 * dip) * gg * (np.square(ee) - cc * np.square(aa))
    kk = np.cos(2 * dip) * gg * bb * aa
    p_amp = hh - ii + jj + kk

    r = np.sin(takeoff/2) / np.cos(takeoff/2)

    return r, p_amp


def fit_spec(freq, spec, dn=0.1, df=0.01):
    # %% resample frequency in log space
    log_freq = np.squeeze(np.log10(freq))
    log_spec = np.squeeze(np.log10(spec))
    interp_f = interp1d(log_freq, log_spec, bounds_error=False, fill_value=0.)
    f = np.arange(log_freq[0], log_freq[-1], df)
    log_spec_new = interp_f(f)

    # %% grid search
    l2_min = 10000.0
    n_best = 2
    fc_best = -2
    for n in np.arange(0.0, 3.0, dn, dtype=np.float64):
        for fc in np.arange(-3.0, 0.4, df, dtype=np.float64):
            model_func = 1.0 / (1.0 + 10**((f-fc)*n))
            l2 = np.sum(np.square(log_spec_new - np.log10(model_func)))
            if l2 < l2_min:
                l2_min = l2
                n_best = n
                fc_best = 10**fc

    return n_best, fc_best


def flux_int(freq, spec, fc, n):
    model = 1.0 / (1.0 + (freq[freq > 1] / fc) ** n)
    int1 = trapezoid(np.square(freq[freq < 1] * spec[freq < 1]), freq[freq < 1], axis=-1)
    int2 = trapezoid(np.square(freq[freq > 1] * model),          freq[freq > 1], axis=-1)

    density = 3e3
    vp = 5e3
    const = np.pi**2 * 8 / (vp**5 * 15 * density)

    return (int1+int2)*const


def correct_distance(dist):

    return dist * 6371.0


def mkdir(dir_path):
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)

def get_vp_vs(depth):

    table = np.loadtxt('IASP91.txt')
    deps = table[:, 0]
    vps = table[deps < depth, 2]
    vss = table[deps < depth, 3]

    return vps[-1], vss[-1]



######## below are copied from Jiuxun's code
def write_progress(file_name, text_contents):
    with open(file_name, 'a') as f:
        f.write(text_contents)


def waveform_fft(waveform, dt):
    """ return the Fourier spectrum of the waveform
    freq, sp = waveform_fft(waveform, dt)
    """
    sp = fft(waveform)
    freq = fftfreq(waveform.size, d=dt)

    sp_positive = abs(sp[freq > 0])
    freq_positive = freq[freq > 0]
    return freq_positive, sp_positive


def downsample_series(time, series, f_downsample):

    dt = time[1] - time[0]
    # lowpass filter
    b, a = sgn.butter(4, f_downsample / 2 * 2 * dt)
    series_filt = sgn.filtfilt(b, a, series, axis=0)

    interp_f = interp1d(time, series_filt, axis=0, bounds_error=False, fill_value=0.)
    dt_new = 1 / f_downsample
    time_new = np.arange(time[0], time[-1], dt_new)
    series_downsample = interp_f(time_new)

    return time_new, series_downsample, dt_new


def randomization_noise(noise, dt, rng=np.random.default_rng(None)):
    """function to produce randomized noise by shifting the phase
    randomization_noise(noise, rng=np.random.default_rng(None))
    return randomized noise
    The input noise has to be an 3D array with (num_batch, num_time, num_channel)
    """

    s = fft(noise, axis=1)
    phase_angle_shift = (rng.random(s.shape) - 0.5) * 2 * np.pi
    # make sure the inverse transform is real
    phase_angle_shift[:, 0, :] = 0
    phase_angle_shift[:, int(s.shape[1] / 2 + 1):, :] = -1 * \
                                                        np.flip(phase_angle_shift[:, 1:int(s.shape[1] / 2), :])

    phase_shift = np.exp(np.ones(s.shape) * phase_angle_shift * 1j)

    # Here apply the phase shift in the entire domain
    # s_shifted = np.abs(s) * phase_shift

    # Instead, try only shift frequency below 10Hz
    freq = fftfreq(s.shape[1], dt)
    ii_freq = abs(freq) <= 10
    s_shifted = s.copy()
    s_shifted[:, ii_freq, :] = np.abs(s[:, ii_freq, :]) * phase_shift[:, ii_freq, :]

    noise_random = np.real(ifft(s_shifted, axis=1))
    return noise_random
