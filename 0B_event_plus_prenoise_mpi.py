"""
Stack high signal-to-noise ratio teleseismic
waveforms with amplified pre-signal noise

@author: Qibin Shi (qibins@uw.edu)
"""
import os
import time
import h5py
import numpy as np
import pandas as pd
from obspy import read_events
from functools import partial
from multiprocessing import Pool
from denoiser_util import process_single_event

# %% 3000 seconds with 10Hz sampling
npts = 30000
allpwave = np.zeros((0, npts, 3), dtype=np.double)
allnoise = np.zeros((0, npts, 3), dtype=np.double)
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

workdir = '/mnt/DATA0/qibin_data/event_data/window_startOrg-3600/M6_2000-2021/'
datadir = '/mnt/DATA0/qibin_data/matfiles_for_denoiser/'
since = time.time()

cat = read_events(workdir + "*.xml")
print(len(cat), "events in total")

partial_func = partial(process_single_event, diretory=workdir, npts=npts)
num_proc = os.cpu_count()
pool = Pool(processes=num_proc)
print("Total number of processes: ", num_proc)

result = pool.map(partial_func, cat)

elapseT = time.time() - since
print("All processed. Time elapsed: %.2f s" % elapseT)

for i in range(len(cat)):
    allpwave = np.append(allpwave, result[i][0], axis=0)
    allnoise = np.append(allnoise, result[i][1], axis=0)
    print(i, 'th quake-noise pair added')
    meta = pd.concat([meta, result[i][2]], ignore_index=True)
    print(i, 'th metadata added')
    print('------------')

elapseT = time.time() - since
print("Added together multiprocessors. Time elapsed: %.2f s" % elapseT)
with h5py.File(datadir + 'Alldepths_snr25_2000_21_sample10_lpass2_P_preP_MP_both_BH_HH.hdf5', 'w') as f:
    f.create_dataset("pwave", data=allpwave)
    f.create_dataset("noise", data=allnoise)

meta.to_csv(datadir + "metadata_Alldepths_snr25_2000_21_sample10_lpass2_P_preP_MP_both_BH_HH.csv", sep=',', index=False)
print("Total traces of data:", allpwave.shape[0])