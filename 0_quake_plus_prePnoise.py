"""
Extract high signal-to-noise ratio teleseismic
waveforms and pre-signal noises

    Default 5000s /10Hz sampled signal and 1000s
    preP noises.

@author: Qibin Shi (qibins@uw.edu)
"""
import gc
import os
import time
import h5py
import numpy as np
import pandas as pd
from obspy import read_events
from functools import partial
from multiprocessing import Pool
from denoiser_util import process_single_event

# %% Folders of raw and reformatted data
workdir = '/data/whd01/qibin_data/raw_data_for_DenoTe/M6.0plus/2000-2021/'
datadir = '/data/whd01/qibin_data/raw_data_for_DenoTe/M6.0plus/matfiles_for_denoiser/'

# %% 5000s sampled at 10Hz
npts = 40000
allpwave = np.zeros((0, npts, 3), dtype=np.double)
allnoise = np.zeros((0, 10000, 3), dtype=np.double)
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

# %% Build a complete catalog from xml files
cat = read_events(workdir + "*.xml")
print(len(cat), "events in our catalog @_@")

# %% Create multi-processing scheme
since = time.time()
# num_proc = os.cpu_count()
num_proc = 24
pool = Pool(processes=num_proc)
print("Number of multi-processing threads: ", num_proc)
partial_func = partial(process_single_event, directory=workdir, npts=npts, noise_saved_pts=10000, noise_pts=20000, tp_after=2000.0, phase=1, cmp=0)

# %% Execute the parallel data processing
result = pool.map(partial_func, cat)
print("All threads processed. Time elapsed: %.2f s" % (time.time() - since))

# %% Merge the reformatted data from threads
for i in range(len(cat)):
    allpwave = np.append(allpwave, result[i][0], axis=0)
    allnoise = np.append(allnoise, result[i][1], axis=0)
    meta = pd.concat([meta, result[i][2]], ignore_index=True)
    print(i, 'th quake-noise pair added')

    if i == 500 or i == 1000 or i == (len(cat)-1):
        with h5py.File(datadir + 'Alldepths_snr25_2000_21_sample10_lpass4_S_preP_MP_both_BH_HH_chunk' + str(i) + '.hdf5', 'w') as f:
            f.create_dataset("pwave", data=allpwave)
            f.create_dataset("noise", data=allnoise)

        del allpwave
        del allnoise
        gc.collect()
        allpwave = np.zeros((0, npts, 3), dtype=np.double)
        allnoise = np.zeros((0, 10000, 3), dtype=np.double)

with h5py.File(datadir + 'Alldepths_snr25_2000_21_sample10_lpass4_S_preP_MP_both_BH_HH_chunk500.hdf5', 'r') as f:
    X_1 = f['pwave'][:]
    Y_1 = f['noise'][:]
with h5py.File(datadir + 'Alldepths_snr25_2000_21_sample10_lpass4_S_preP_MP_both_BH_HH_chunk1000.hdf5', 'r') as f:
    X_2 = f['pwave'][:]
    Y_2 = f['noise'][:]
with h5py.File(datadir + 'Alldepths_snr25_2000_21_sample10_lpass4_S_preP_MP_both_BH_HH_chunk' + str(len(cat)-1) + '.hdf5', 'r') as f:
    X_3 = f['pwave'][:]
    Y_3 = f['noise'][:]

X_1 = np.append(X_1, X_2, axis=0)
Y_1 = np.append(Y_1, Y_2, axis=0)
X_1 = np.append(X_1, X_3, axis=0)
Y_1 = np.append(Y_1, Y_3, axis=0)

with h5py.File(datadir + 'Alldepths_snr25_2000_21_sample10_lpass4_S_preP_MP_both_BH_HH.hdf5', 'w') as f:
    f.create_dataset("pwave", data=X_1)
    f.create_dataset("noise", data=Y_1)

meta.to_csv(datadir + "metadata_Alldepths_snr25_2000_21_sample10_lpass4_S_preP_MP_both_BH_HH_rot.csv", sep=',', index=False)
print("Total traces of data:", allpwave.shape[0])
print("Reformat is done! Time elapsed: %.2f s" % (time.time() - since))