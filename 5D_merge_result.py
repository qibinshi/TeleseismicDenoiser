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
from denoiser_util import get_vp_vs
from functools import partial
from multiprocessing import Pool
from torch_tools import try_gpu
from denoiser_util import mkdir, denoise_cc_stack
matplotlib.use('Agg')
matplotlib.rcParams.update({'font.size': 24})

meta_result = pd.DataFrame(columns=[
    "source_id",
    "source_latitude_deg",
    "source_longitude_deg",
    "source_depth_km",
    "source_magnitude",
    "ts_noisy",
    "ts_clean",
    "stress_noisy",
    "stress_clean"])

fig_dir = 'Release_Middle_augmentation_P4Hz_150s/bk_Apply_releaseWDN_M5.5-8_deep100km_SNR2_azimuthBined'

csv_file = fig_dir + "/metadata_M6_deep100km_allSNR_P.csv"
meta_all = pd.read_csv(csv_file, low_memory=False)
csv_file = fig_dir + "/metadata_M55_deep100km_allSNR_P.csv"
meta_all1 = pd.read_csv(csv_file, low_memory=False)
csv_file = fig_dir + "/source_measurements.csv"
meta_src = pd.read_csv(csv_file, low_memory=False)

meta_all = pd.concat([meta_all, meta_all1], ignore_index=True)

for evid in meta_src.source_id.to_numpy():
    meta = meta_all[(meta_all.source_id == evid)]
    evla = meta.source_latitude_deg.unique()[0]
    evlo = meta.source_longitude_deg.unique()[0]
    evdp = meta.source_depth_km.unique()[0]
    evmg = meta.source_magnitude.unique()[0]

    meta = meta_src[(meta_src.source_id == evid)]
    dura_noisy = meta.duration_noisy.unique()[0]
    dura_clean = meta.duration_denoised.unique()[0]
    Es_noisy = meta.Es_noisy.unique()[0]
    Es_clean = meta.Es_denoised.unique()[0]

    mw = np.exp(0.741 + 0.21 * evmg) - 0.785
    lg_moment = (mw + 6.07) * 1.5
    moment = 10 ** lg_moment
    vp, vs, den = get_vp_vs(evdp)


    ts_clean = dura_clean * np.power(10, (19-lg_moment)/4) * vs / 4.5
    ts_noisy = dura_noisy * np.power(10, (19-lg_moment)/4) * vs / 4.5

    stress_noisy = Es_noisy * moment * 30000
    stress_clean = Es_clean * moment * 30000
    print(dura_clean, ts_clean)
    meta_result = pd.concat([meta_result, pd.DataFrame(data={
        "source_id": evid,
        "source_latitude_deg": evla,
        "source_longitude_deg": evlo,
        "source_depth_km": evdp,
        "source_magnitude": evmg,
        "ts_noisy": ts_noisy,
        "ts_clean": ts_clean,
        "stress_noisy": stress_noisy,
        "stress_clean": stress_clean}, index=[0])], ignore_index=True)

meta_result.to_csv(fig_dir + "/source_merge.csv", sep=' ', index=False)

