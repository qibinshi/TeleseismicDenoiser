"""
Stack high signal-to-noise ratio teleseismic
waveforms with amplified pre-signal noise

@author: Qibin Shi (qibins@uw.edu)
"""
import os
import time
import h5py
import numpy as np
from obspy import read_events
from functools import partial
from multiprocessing import Pool
from denoiser_util import process_single_event

# %% 3000 seconds with 10Hz sampling
npts = 30000
allpwave = np.zeros((0, npts, 3), dtype=np.double)
allnoise = np.zeros((0, npts, 3), dtype=np.double)

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

elapseT = time.time() - since
print("Added together multiprocessors. Time elapsed: %.2f s" % elapseT)
with h5py.File(datadir + 'Alldepths_snr25_2000_21_sample10_lpass2_P_preP_MP1.hdf5', 'w') as f:
    f.create_dataset("pwave", data=allpwave)
    f.create_dataset("noise", data=allnoise)

print("Total traces of data:", allpwave.shape[0])