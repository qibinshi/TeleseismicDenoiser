import os
import numpy as np
import scipy.signal as sgn
from scipy.fft import fft, fftfreq, ifft
from scipy.signal import stft, istft
from scipy.interpolate import interp1d
from obspy.clients.fdsn import Client

def mkdir(dir_path):
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)


def downsample_series(time, series, f_downsampe):
    """Down sample the time series given a lower sampling frequency f_downsample,
    time_new, series_downsample, dt_new = downsample_series(time, series, f_downsampe)

    The time series has been lowpass filtered (f_filter=f_downsample/2) first,
    and then downsampled through interpolation.
    """
    dt = time[1] - time[0]
    # lowpass filter
    b, a = sgn.butter(4, f_downsampe / 2 * 2 * dt)
    series_filt = sgn.filtfilt(b, a, series, axis=0)
    # downsample through interpolation
    dt_new = 1 / f_downsampe
    # time_new = np.arange(time[0], time[-1] + dt_new, dt_new)
    time_new = np.arange(time[0], time[-1], dt_new)
    # series_downsample = np.interp(time_new, time, series_filt)
    interp_f = interp1d(time, series_filt, axis=0, bounds_error=False, fill_value=0.)
    series_downsample = interp_f(time_new)

    # plt.figure()
    # plt.plot(time_noise, noise_BH1, '-k')
    # plt.plot(time, series_filt, '-r')
    # plt.plot(time_new, series_downsample, '-b')

    return time_new, series_downsample, dt_new


def waveform_fft(waveform, dt):
    """ return the Fourier spectrum of the waveform
    freq, sp = waveform_fft(waveform, dt)
    """
    sp = fft(waveform)
    freq = fftfreq(waveform.size, d=dt)

    sp_positive = abs(sp[freq > 0])
    freq_positive = freq[freq > 0]
    return freq_positive, sp_positive


def waveform_stft(waveform, dt, twin=60, toverlap=20, win_type='hann', complex=False):
    """ return the spectrogram of the waveform
    freq, time, sxx = waveform_stft(waveform, dt, twin=60, toverlap=20, win_type='hann')
    """
    fs = 1 / dt
    # apply the thresholding method in the STFT to separate the noise and signals
    f, t, sxx = stft(waveform, fs, nperseg=int(twin / dt),
                     noverlap=int(toverlap / dt), window=win_type)

    if complex:
        return f, t, sxx
    else:
        sxx_abs = abs(sxx)
        return f, t, sxx_abs


def waveform_inverse_stft(Sxx, dt, twin=60, toverlap=20, win_type='hann'):
    fs = 1 / dt
    time, waveform = istft(Sxx, fs=fs, nperseg=int(twin / dt), noverlap=int(toverlap / dt), window=win_type)
    return time, waveform



def download_station(client, network, station, channel, location, path):
    fname = path + "/" + network + "." + station + "." + location + "." + channel
    inventory = client.get_stations(network=network, station=station, channel=channel,
                                    location=location, level="response")
    inventory.write(fname + ".xml", format="STATIONXML")
    return inventory

def download_waveforms(client, network, station, channel, location, starttime, endtime, path, fname):
    tr = client.get_waveforms(network=network, station=station, channel=channel,
                              location=location, starttime=starttime - 1800, endtime=endtime + 1800,
                              attach_response=True)
    # tr.detrend("spline", order=3, dspline=1000)
    # tr.remove_response(output="VEL")
    newtr = tr.slice(starttime, endtime)  # probably fix later
    newtr.write(path + "/" + fname + ".mseed", format='MSEED')
    return newtr

def randomization_noise(noise, rng=np.random.default_rng(None)):
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

def write_progress(file_name, text_contents):
    with open(file_name, 'a') as f:
        f.write(text_contents)
