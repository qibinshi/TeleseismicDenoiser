"""
Extract high signal-to-noise ratio teleseismic
waveforms and pre-signal noises

    Default 3000s window and 10Hz sampling rate
    for signal and preP noises. Our raw data starts
    3600s before and 3600s after the origin time.
    So it is safe to first cut P-3600s, P+2000s

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

# %% Folders of raw and reformatted data
workdir = '/mnt/DATA0/qibin_data/event_data/window_startOrg-3600/M6_2000-2021/'
datadir = '/mnt/DATA0/qibin_data/matfiles_for_denoiser/'

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

# %% Build a complete catalog from xml files
cat = read_events(workdir + "*.xml")
print(len(cat), "events in our catalog @_@")

# %% Create multi-processing scheme
since = time.time()
num_proc = os.cpu_count()
pool = Pool(processes=num_proc)
print("Number of multi-processing threads: ", num_proc)
partial_func = partial(process_single_event, directory=workdir, npts=npts, noise_pts=36000, tp_after=2000.0)

# %% Execute the parallel data processing
result = pool.map(partial_func, cat)
print("All threads processed. Time elapsed: %.2f s" % (time.time() - since))

# %% Merge the reformatted data from threads
for i in range(len(cat)):
    allpwave = np.append(allpwave, result[i][0], axis=0)
    allnoise = np.append(allnoise, result[i][1], axis=0)
    meta = pd.concat([meta, result[i][2]], ignore_index=True)
    print(i, 'th quake-noise pair added')

print("Reformat is done! Time elapsed: %.2f s" % (time.time() - since))

# %% Saving to output folder
with h5py.File(datadir + 'Alldepths_snr25_2000_21_sample10_lpass2_P_preP_MP_both_BH_HH.hdf5', 'w') as f:
    f.create_dataset("pwave", data=allpwave)
    f.create_dataset("noise", data=allnoise)

meta.to_csv(datadir + "metadata_Alldepths_snr25_2000_21_sample10_lpass2_P_preP_MP_both_BH_HH.csv", sep=',', index=False)
print("Total traces of data:", allpwave.shape[0])
