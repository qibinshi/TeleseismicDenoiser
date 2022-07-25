"""
@author: Qibin Shi
qibins@uw.edu
"""
import os
import glob
import torch
import numpy as np
import pandas as pd
import scipy.signal as sgn
import matplotlib.pyplot as plt
from distaz import DistAz
from obspy.taup import TauPyModel
from obspy import read_inventory
from obspy import read, UTCDateTime
from scipy.interpolate import interp1d
from scipy.fft import fft, fftfreq, ifft
from torch_tools import Explained_Variance_score, CCLoss


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
def process_single_event(ev, diretory=None, npts=None):
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
    pre_noise = np.zeros((36000, 3), dtype=np.double)

    evnm = str(ev.resource_id)[13:]
    evlo = ev.origins[0].longitude
    evla = ev.origins[0].latitude
    evdp = ev.origins[0].depth / 1000.0
    org_t = ev.origins[0].time
    evmg = ev.magnitudes[0].mag

    # %% Loop over stations
    for sta in glob.glob(diretory + evnm + 'stas/*xml'):
        inv = read_inventory(sta)
        stnw = inv[0].code
        stco = inv[0][0].code
        stla = inv[0][0].latitude
        stlo = inv[0][0].longitude
        stlc = inv[0][0][0].location_code
        result = DistAz(stla, stlo, evla, evlo)
        distdeg = result.getDelta()
        backazi = result.getBaz()

        try:
            st0 = read(diretory + evnm + "waves/" + stnw + "." + stco + "." + stlc + ".?H?_*")
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
        st.trim(tp - 3600.0, tp + 2000.0)

        if len(st) > 3:
            print(len(st), "channels", stco)
        if len(st) >= 3 and len(st[0].data) >= 56000 and len(st[1].data) >= 56000 and len(st[2].data) >= 56000:
            noise_amp = np.std(np.array(st[0].data)[34950:35950])
            pwave_amp = np.std(np.array(st[0].data)[36000:37000])

            if pwave_amp > (noise_amp * 25):
                for i in range(3):
                    one_pwave[:, i] = np.array(st[i].data)[26000:56000]
                    pre_noise[:, i] = np.array(st[i].data)[0:36000]

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
def process_single_event_only(ev, diretory=None, halftime=None, freq=2, rate=10, maxsnr=3, mindep=200):
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
        "trace_snr_db",
        "distance",
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

    if evdp > mindep:
        # %% Loop over stations
        for sta in glob.glob(diretory + evnm + 'stas/*xml'):
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
                st0 = read(diretory + evnm + "waves/" + stnw + "." + stco + "." + stlc + ".?H?_*")
                st = st0.copy()
            except:
                continue
            # try:
            #     st.remove_response(inventory=inv, output='VEL', pre_filt=pre_filt)
            # except:
            #     continue
            st.filter("lowpass", freq=freq)
            st.resample(rate)
            st.merge(fill_value=np.nan)
            model = TauPyModel(model="iasp91")
            arrivals = model.get_travel_times(source_depth_in_km=evdp, distance_in_degree=distdeg, phase_list=['P'])
            tp = UTCDateTime(org_t + arrivals[0].time)
            st.trim(tp - halftime, tp + halftime)

            if len(st) > 3:
                print(len(st), "channels", stco)
            if len(st) >= 3 and len(st[0].data) >= npts and len(st[1].data) >= npts and len(st[2].data) >= npts:
                noise_amp = np.std(np.array(st[0].data)[0:int(rate*(halftime-5))])
                pwave_amp = np.std(np.array(st[0].data)[int(rate*halftime):npts - 1])

                if pwave_amp < (noise_amp * maxsnr):
                    for i in range(3):
                        one_pwave[:, i] = np.array(st[i].data)[0:npts]
                    one_pwave[np.isnan(one_pwave)] = 0
                    one_pwave = (one_pwave - np.mean(one_pwave, axis=0)) / (np.std(one_pwave, axis=0) + 1e-12)
                    allpwave = np.append(allpwave, one_pwave[np.newaxis, :, :], axis=0)

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
                        "trace_snr_db": "%.3f" % (pwave_amp / (noise_amp + 1e-12)),
                        "distance": distdeg,
                        "azimuth": azimuth}, index=[0])], ignore_index=True)

    return allpwave, meta


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
    # timex = (np.arange(0, num_pts) - int((num_pts-npts)/2)) * dt

    return timex, wave_out, relative_ratio, cc, flipsign


def dura_amp(wave2, long=10, short=2):

    wave_abs = np.fabs(wave2) + 1e-12
    long_average = np.convolve(wave_abs, np.ones(long), 'full')/float(long)
    short_average = np.convolve(wave_abs, np.ones(short), 'full')/float(short)
    ratio1 = np.log10(short_average[:1-short]) - np.log10(long_average[:1-long])
    ratio2 = np.log10(short_average[short-1:]) - np.log10(long_average[long-1:])
    # peak_abs = np.argmax(long_average[:1-long])
    pt1 = np.argmax(ratio1[70:int(len(wave_abs)/2+50)]) + 70
    pt2 = np.argmax(ratio2[int(len(wave_abs)/2-50):1-long]) + int(len(wave_abs)/2-50)

    return pt1, pt2, ratio1, ratio2, short_average, long_average


def mkdir(dir_path):
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)



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


def downsample_series(time, series, f_downsampe):

    dt = time[1] - time[0]
    # lowpass filter
    b, a = sgn.butter(4, f_downsampe / 2 * 2 * dt)
    series_filt = sgn.filtfilt(b, a, series, axis=0)

    interp_f = interp1d(time, series_filt, axis=0, bounds_error=False, fill_value=0.)
    dt_new = 1 / f_downsampe
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
