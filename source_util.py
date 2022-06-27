"""
@author: Qibin Shi
qibins@uw.edu
"""
## Codes created for moment tensor prediction
from math import sin, cos, pi
import matplotlib.pyplot as plt
import obspy
import numpy as np
import glob
from distaz import DistAz
from obspy.taup import TauPyModel
from obspy import read_inventory, read
from obspy import UTCDateTime, read_events
from obspy.imaging.beachball import beach

# convert strike/dip/rake to moment tensor #
def fm2mt(strike, dip, rake):
    epsilon = 1e-13
    deg2rad = pi / 180.
    S = strike * deg2rad
    D = dip * deg2rad
    R = rake * deg2rad

    for ang in S, D, R:
        if abs(ang) < epsilon:
            ang = 0.

    M11 = -(sin(D) * cos(R) * sin(2 * S) + sin(2 * D) * sin(R) * sin(S) ** 2)
    M22 = +(sin(D) * cos(R) * sin(2 * S) - sin(2 * D) * sin(R) * cos(S) ** 2)
    M33 = sin(2 * D) * sin(R)
    M12 = +(sin(D) * cos(R) * cos(2 * S) + sin(2 * D) * sin(R) * sin(S) * cos(S))
    M13 = -(cos(D) * cos(R) * cos(S) + cos(2 * D) * sin(R) * sin(S))
    M23 = -(cos(D) * cos(R) * sin(S) - cos(2 * D) * sin(R) * cos(S))

    #Moments = [M11, M22, M33, M12, M13, M23]
    Moments = [M33, M11, M22, M13, 0-M23, 0-M12]
    return tuple(Moments)


# plot learning curve #
def plot_curve(history, nm, swth):
    epoch = [i+1 for i in history.epoch]
    plt.figure()
    plt.xlabel('epoch')
    plt.ylabel('mean squared error')
    plt.ylim([0, 1])
    plt.plot(epoch, np.array(history.history['loss']), label='training loss')
    plt.plot(epoch, np.array(history.history['val_loss']), label='validation loss')
    if swth == 1:
        plt.plot(epoch, np.array(history.history['mse']), label='mse')
        plt.plot(epoch, np.array(history.history['val_mse']), label='val_mse')
    plt.legend()
    plt.savefig(nm+'.pdf')


# plot predicted beachballs together with ground truths #
def plot_ball(mtlabel,mtpred,nm):
    if len(mtlabel.shape) ==1:
        mtlabel = mtlabel[None,:]
    if len(mtpred.shape) ==1:
        mtpred = mtpred[None,:]
    nballs = len(mtlabel)
    plt.ylim([0,4])
    plt.xlim([-1,nballs])
    plt.text(nballs-0.3, 0.5, 'label')
    plt.text(nballs-0.3, 2.5, 'prediction')
    ax = plt.gca()
    for i in range(nballs):
        yl = 1
        yp = 3
        xl = i
        xp = i
        bl = beach(mtlabel[i], xy=(xl, yl), linewidth=1, width=0.8)
        bp = beach(mtpred[i], xy=(xp, yp), linewidth=1, width=0.8)
        ax.add_collection(bl)
        ax.add_collection(bp)
    ax.set_aspect("equal")
    plt.savefig(nm+'.pdf')


# Define shuff_split instead of using validation_split
def shuff_split(x_raw, y_raw, batch_size, vali_ratio):
    nevt = x_raw.shape[0]
    ind = np.arange(nevt)
    np.random.shuffle(ind)
    x_raw = x_raw[ind]
    y_raw = y_raw[ind]
    splitter1 = int(nevt / batch_size * vali_ratio) * batch_size
    splitter2 = int(nevt / batch_size ) * batch_size
    x_vali = x_raw[0:splitter1]
    y_vali = y_raw[0:splitter1]
    x_train = x_raw[splitter1:splitter2]
    y_train = y_raw[splitter1:splitter2]

    return x_train, y_train, x_vali, y_vali

# Trim downloaded three components for same time range
def trim_align(trace):
    startt=trace[0].stats.starttime
    if startt <trace[1].stats.starttime:
        startt=trace[1].stats.starttime
    if startt <trace[2].stats.starttime:
        startt=trace[2].stats.starttime

    endt=trace[0].stats.endtime
    if endt >trace[1].stats.endtime:
        endt=trace[1].stats.endtime
    if endt >trace[2].stats.endtime:
        endt=trace[2].stats.endtime

    return trace.trim(startt, endt)

def process_single_event(ev, diretory=None, npts=None):
    model = TauPyModel(model="iasp91")
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

    # %% Loop over stations
    for sta in glob.glob(diretory + evnm + 'stas/*xml'):
        inv = read_inventory(sta)
        stco = inv[0].code + "." + inv[0][0].code + "." + inv[0][0][0].location_code
        stlo = inv[0][0].longitude
        stla = inv[0][0].latitude
        result = DistAz(stla, stlo, evla, evlo)
        distdeg = result.getDelta()
        backazi = result.getBaz()

        try:
            st0 = read(diretory + evnm + "waves/" + stco + ".BH?_*")
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
        arrivals = model.get_travel_times(source_depth_in_km=evdp, distance_in_degree=distdeg, phase_list=['P'])
        tp = UTCDateTime(org_t + arrivals[0].time)
        st.trim(tp - 3600.0, tp + 2000.0)

        if len(st) == 3 and len(st[0].data) >= 56000 and len(st[1].data) >= 56000 and len(st[2].data) >= 56000:
            noise_amp = np.std(np.array(st[0].data)[34950:35950])
            pwave_amp = np.std(np.array(st[0].data)[36000:37000])

            if pwave_amp > (noise_amp * 25):
                for i in range(3):
                    one_pwave[:, i] = np.array(st[i].data)[26000:56000]
                    pre_noise[:, i] = np.array(st[i].data)[0:36000]

                ## %% extract low amplitudes for noise
                amplitude_series = np.sqrt(np.sum(pre_noise ** 2, axis=1))
                amplitude_median = np.nanmedian(amplitude_series)
                noise0 = pre_noise[amplitude_series < (4 * amplitude_median), :]
                ## %% make sure noise is long enough
                if noise0.shape[0] > npts / 2:
                    if noise0.shape[0] < npts:
                        noise0 = np.append(noise0, noise0, axis=0)
                    one_noise[:, :] = noise0[:npts, :]

                    ## %% normalize p wave and noise
                    one_pwave[np.isnan(one_pwave)] = 0
                    one_noise[np.isnan(one_noise)] = 0
                    one_pwave = (one_pwave - np.mean(one_pwave, axis=0)) / (np.std(one_pwave, axis=0) + 1e-12)
                    one_noise = (one_noise - np.mean(one_noise, axis=0)) / (np.std(one_noise, axis=0) + 1e-12)

                    ## %% Save the waveform pairs
                    allpwave = np.append(allpwave, one_pwave[np.newaxis, :, :], axis=0)
                    allnoise = np.append(allnoise, one_noise[np.newaxis, :, :], axis=0)

    return allpwave, allnoise