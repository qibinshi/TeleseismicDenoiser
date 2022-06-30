"""
Prepare low signal-to-noise ratio teleseismic data
e.g. deep M5 earthquakes recorded on ocean stations

@author: Qibin Shi (qibins@uw.edu)
"""
import os
import time
import h5py
import numpy as np
from functools import partial
from multiprocessing import Pool
from obspy import UTCDateTime, read_events
from denoiser_util import process_single_event_only

# %%
halftime = 300.0
samplerate = 10
freq = 2.0
npts = int(halftime*2*samplerate)
allpwave = np.zeros((0, npts, 3), dtype=np.double)

workdir = '/mnt/DATA0/qibin_data/event_data/window_startOrg/M5.5_2000-2021_depthover500/'
datadir = '/mnt/DATA0/qibin_data/matfiles_for_denoiser/'
since = time.time()

cat = read_events(workdir + "*.xml")
print(len(cat), "events in total")

partial_func = partial(process_single_event_only, diretory=workdir, halftime=halftime)
num_proc = os.cpu_count()
pool = Pool(processes=num_proc)
print("Total number of processes: ", num_proc)

result = pool.map(partial_func, cat)

elapseT = time.time() - since
print("All processed. Time elapsed: %.2f s" % elapseT)

for i in range(len(cat)):
    allpwave = np.append(allpwave, result[i], axis=0)
    print(i, 'th quake added')

elapseT = time.time() - since
print("Added together multiprocessors. Time elapsed: %.2f s" % elapseT)
with h5py.File(datadir + 'M5_deep500km_SNRmax3_2000_21_sample10_lpass2_P_mpi.hdf5', 'w') as f:
    f.create_dataset("pwave", data=allpwave)

print("Total traces of data:", allpwave.shape[0])