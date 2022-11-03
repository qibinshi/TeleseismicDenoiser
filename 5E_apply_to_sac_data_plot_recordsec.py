"""
Apply the model to noise data
e.g. deep earthquakes

author: Qibin Shi
"""
import os
import glob
import torch
import matplotlib
import numpy as np
import pandas as pd
from obspy.taup import TauPyModel
from obspy import read, UTCDateTime
from denoiser_util import mkdir, denoise_cc_stack
matplotlib.use('Agg')
matplotlib.rcParams.update({'font.size': 24})

strike = 165
dip = 74
rake = -10
depth = 13
mag = 6.6
stdv = 1
cmp = 1
snr = 3
dt = 0.1
npts = 1500
npts_trim = 1200
sac_dir = '/Users/qibin/Projects/Luding2022/tele_data/2022-09-05-mww66-sichuan-china/rot_dir'
preP = 75.0
time_after = 75.0
X_train = np.zeros((0, npts, 3), dtype=np.double)

meta = pd.DataFrame(columns=[
    "source_id",
    "source_depth_km",
    "source_magnitude",
    "source_strike",
    "source_dip",
    "source_rake",
    "trace_snr_db",
    "trace_stdv_2",
    "distance",
    "takeoff_phase",
    "azimuth"])

for sta in glob.glob(sac_dir + '/*.z'):

    try:
        st0 = read(sta[:-1]+'?')
        st = st0.copy()
        # st.filter("lowpass", freq=4.0)
    except:
        continue

    st.resample(10)
    if len(st) >= 3:
        tp = UTCDateTime(st[2].stats.starttime - st[2].stats.sac.b + st[2].stats.sac.t7)

        st.trim(tp - preP, tp + time_after)
        if len(st) >= 3 and len(st[0].data) >= npts and len(st[1].data) >= npts and len(st[2].data) >= npts:
            noise_amp = np.std(np.array(st[cmp].data)[:700])
            pwave_amp = np.std(np.array(st[cmp].data)[750:])

        one_wave = np.zeros((npts, 3), dtype=np.double)
        for i in range(3):
            one_wave[:, i] = np.array(st[i].data)[0:npts]
        one_wave[np.isnan(one_wave)] = 0
        scale_mean = np.mean(one_wave, axis=0, keepdims=True)
        scale_stdv = np.std(one_wave, axis=0, keepdims=True) + 1e-12
        one_wave = (one_wave - scale_mean) / scale_stdv

        X_train = np.append(X_train, one_wave[np.newaxis, :, :], axis=0)

        distance = st[2].stats.sac.gcarc
        azimuth = st[2].stats.sac.az
        mod = TauPyModel(model="iasp91")
        arrivals = mod.get_travel_times(source_depth_in_km=depth, distance_in_degree=distance, phase_list=['S'])

        meta = pd.concat([meta, pd.DataFrame(data={
            "source_id": "luding22",
            "source_depth_km": depth,
            "source_magnitude": mag,
            "source_strike": strike,
            "source_dip": dip,
            "source_rake": rake,
            "trace_snr_db": "%.3f" % (pwave_amp / (noise_amp + 1e-12)),
            "trace_stdv_2": stdv,
            "distance": distance,
            "takeoff_phase": arrivals[0].takeoff_angle,
            "azimuth": azimuth}, index=[0])], ignore_index=True)

    else:
        print('Failed to convert station: ', sta[:-4])

###########
# %% Load Denoiser
model_dir = 'Release_Middle_augmentation_P4Hz_150s'
model = torch.load(model_dir + '/Branch_Encoder_Decoder_LSTM_Model.pth', map_location=torch.device('cpu'))
model = model.module.to(torch.device('cpu'))
model.eval()

partial_func = denoise_cc_stack("luding22", meta_all=meta, X_train=X_train, model=model, fig_dir=sac_dir, dt=dt, npts=npts, npts_trim=npts_trim, normalized_stack=True, min_snr=snr, cmp=1)

print('Done\n')
