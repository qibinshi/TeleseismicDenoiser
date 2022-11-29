"""
Prepare low signal-to-noise ratio teleseismic data
e.g. deep M5 earthquakes recorded on ocean stations

@author: Qibin Shi (qibins@uw.edu)
"""
import os
import time
import h5py
import argparse
import numpy as np
import pandas as pd
from functools import partial
from multiprocessing import Pool
from obspy import UTCDateTime, read_events
from denoiser_util import process_single_event_only


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--phase', default='P', type=str, help='earthquake phase')
    parser.add_argument('-r', '--minsnr', default=25, type=int, help='signal-noise ratio')
    parser.add_argument('-n', '--threads', default=24, type=int, help='number of processes')
    args = parser.parse_args()
    # %% Directories of raw and reformatted data
    # workdir = '/data/whd01/qibin_data/raw_data_for_DenoTe/M5.5-6.0/2000_2021/'
    # datadir = '/data/whd01/qibin_data/raw_data_for_DenoTe/M5.5-6.0/matfiles_for_denoiser/'
    workdir = '/data/whd01/qibin_data/raw_data_for_DenoTe/M6.0plus/2000-2021/'
    datadir = '/data/whd01/qibin_data/raw_data_for_DenoTe/M6.0plus/matfiles_for_denoiser/'

    # %% Quake catalog from xml files
    cat = read_events(workdir + "*.xml")
    print(len(cat), "events in total")

    # %% Recording length
    if args.phase == 'P':
        halftime = 150.0
    else:
        halftime = 250.0

    samplerate = 10
    npts = int(halftime*2*samplerate)
    allquake = np.zeros((0, npts, 3), dtype=np.double)

    # %% Multi-processing for multi-events
    since = time.time()
    partial_func = partial(process_single_event_only,
                           directory=workdir,
                           halftime=halftime,
                           maxsnr=10000,
                           mindep=100,
                           phase=args.phase)

    num_proc = args.threads  # num_proc = os.cpu_count()
    with Pool(processes=num_proc) as pool:
        print("Total number of processes: ", num_proc)
        result = pool.map(partial_func, cat)
    print("All are processed. Time elapsed: %.2f s" % (time.time() - since))

    # %% Merge from threads
    meta = pd.DataFrame(columns=[
        "source_id",
        "source_origin_time",
        "source_latitude_deg",
        "source_longitude_deg",
        "source_depth_km",
        "source_magnitude",
        "source_strike",
        "source_dip",
        "source_rake",
        "station_network_code",
        "station_code",
        "station_location_code",
        "station_latitude_deg",
        "station_longitude_deg",
        "trace_snr_db",
        "trace_mean_0",
        "trace_stdv_0",
        "trace_mean_1",
        "trace_stdv_1",
        "trace_mean_2",
        "trace_stdv_2",
        "distance",
        "takeoff_p",
        "takeoff_phase",
        "azimuth"])

    for i in range(len(cat)):
        allquake = np.append(allquake, result[i][0], axis=0)
        print(i, 'th quake added')
        meta = pd.concat([meta, result[i][1]], ignore_index=True)
        print(i, 'th metadata added')
        print('------------')

    with h5py.File(datadir + 'M6_deep100km_allSNR_' + args.phase + '.hdf5', 'w') as f:
        f.create_dataset("quake", data=allquake)

    meta.to_csv(datadir + 'M6_deep100km_allSNR_' + args.phase + '.csv', sep=',', index=False)
    print("Total traces of data:", allquake.shape[0])
    print("All is saved! Time elapsed: %.2f s" % (time.time() - since))