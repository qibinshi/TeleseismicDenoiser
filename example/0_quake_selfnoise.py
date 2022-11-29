"""
Extract high signal-to-noise ratio teleseismic
waveforms and pre-signal noises

    P: 5000s /10Hz signal and 1000s preP noises.
    S: 4000s /10Hz signal and 150s preS noises.

@author: Qibin Shi (qibins@uw.edu)
"""
import gc
import glob
import time
import h5py
import argparse
import numpy as np
import pandas as pd
from distaz import DistAz
from functools import partial
from multiprocessing import Pool
from obspy.taup import TauPyModel
from obspy import read_events, read_inventory, read, UTCDateTime


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--phase', default='P', type=str, help='earthquake phase')
    parser.add_argument('-r', '--minsnr', default=25, type=int, help='signal-noise ratio')
    parser.add_argument('-n', '--threads', default=24, type=int, help='number of processes')
    args = parser.parse_args()
    # %% Directories of raw and reformatted data
    data_dir = '/data/whd01/qibin_data/raw_data_for_DenoTe/M6.0plus/1980-1999/'
    save_dir = '/data/whd01/qibin_data/raw_data_for_DenoTe/M6.0plus/matfiles_for_denoiser/'

    # %% Quake catalog from xml files
    cat = read_events(data_dir + "*.xml")
    print(len(cat), "events in our catalog @_@")

    # %% Recording length
    phase = args.phase
    minsnr = args.minsnr
    if phase == 'P':
        quake_len = 50000
        noise_len = 10000
    else:
        quake_len = 40000
        noise_len = 1500  # coda of P
    all_quake = np.zeros((0, quake_len, 3), dtype=np.double)
    all_noise = np.zeros((0, noise_len, 3), dtype=np.double)

    # %% Multi-processing for multi-events
    since = time.time()
    num_proc = args.threads  # num_proc = os.cpu_count()
    pool = Pool(processes=num_proc)
    print("Number of multi-processing threads: ", num_proc)
    partial_func = partial(one_quake_noise,
                           directory=data_dir,
                           noise_saved_pts=noise_len,
                           npts=quake_len,
                           phase=phase,
                           minsnr=minsnr)

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
            "station_network",
            "station_code",
            "station_location_code",
            "station_latitude_deg",
            "station_longitude_deg",
            "trace_snr_db"])

    for i in range(len(cat)):
        all_quake = np.append(all_quake, result[i][0], axis=0)
        all_noise = np.append(all_noise, result[i][1], axis=0)
        meta = pd.concat([meta, result[i][2]], ignore_index=True)
        print(i, 'th quake-noise pair added')

        # %% Save in three chunks in case of memory exploding
        if i == 500 or i == 1000 or i == (len(cat)-1):
            with h5py.File(save_dir + 'chunk' + str(i) + '.hdf5', 'w') as f:
                f.create_dataset("quake", data=all_quake)
                f.create_dataset("noise", data=all_noise)

            del all_quake
            del all_noise
            gc.collect()
            all_quake = np.zeros((0, quake_len, 3), dtype=np.double)
            all_noise = np.zeros((0, noise_len, 3), dtype=np.double)

    with h5py.File(save_dir + 'chunk500.hdf5', 'r') as f:
        q1 = f['quake'][:]
        n1 = f['noise'][:]
    with h5py.File(save_dir + 'chunk1000.hdf5', 'r') as f:
        q2 = f['quake'][:]
        n2 = f['noise'][:]
    with h5py.File(save_dir + 'chunk' + str(len(cat) - 1) + '.hdf5', 'r') as f:
        q3 = f['quake'][:]
        n3 = f['noise'][:]

    with h5py.File(save_dir + 'Alldepths_SoverCoda25_sample10_lpass4_S_TRZ.hdf5', 'r') as f:
        q0 = f['pwave'][:]
        n0 = f['noise'][:]

    q1 = np.append(q0, q1, axis=0)
    n1 = np.append(n0, n1, axis=0)

    q1 = np.append(q1, q2, axis=0)
    n1 = np.append(n1, n2, axis=0)
    q1 = np.append(q1, q3, axis=0)
    n1 = np.append(n1, n3, axis=0)

    with h5py.File(save_dir + phase + 'snr' + str(minsnr) + '_lp4_' + '1980-2021.hdf5', 'w') as f:
        f.create_dataset("quake", data=q1)
        f.create_dataset("noise", data=n1)

    meta.to_csv(save_dir + phase + 'snr' + str(minsnr) + '_lp4_' + '1980.csv', sep=',', index=False)
    print("Total traces of data:", all_quake.shape[0])
    print("All is saved! Time elapsed: %.2f s" % (time.time() - since))


def one_quake_noise(ev, directory=None, npts=50000, noise_saved_pts=10000, phase='P', minsnr=25):
    """Default 10Hz sampling rate
    Raw data is 1 hour before and after the origin time."""

    if phase == 'P':
        cmp = 2  # Z
    elif phase == 'S':
        cmp = 0  # T (after rotation)
    else:
        cmp = 1

    noise_pts = int(npts/2)

    all_quake = np.zeros((0, npts, 3), dtype=np.double)
    all_noise = np.zeros((0, noise_saved_pts, 3), dtype=np.double)
    one_quake = np.zeros((npts, 3), dtype=np.double)
    one_noise = np.zeros((noise_saved_pts, 3), dtype=np.double)
    pre_noise = np.zeros((noise_saved_pts * 2, 3), dtype=np.double)
    coda_noise = np.zeros((noise_saved_pts, 3), dtype=np.double)

    evnm = str(ev.resource_id)[13:]
    evlo = ev.origins[0].longitude
    evla = ev.origins[0].latitude
    evdp = ev.origins[0].depth / 1000.0
    org_t = ev.origins[0].time
    evmg = ev.magnitudes[0].mag

    # %% format the metadata
    meta = pd.DataFrame(columns=[
        "source_id",
        "source_origin_time",
        "source_latitude_deg",
        "source_longitude_deg",
        "source_depth_km",
        "source_magnitude",
        "station_network",
        "station_code",
        "station_location_code",
        "station_latitude_deg",
        "station_longitude_deg",
        "trace_snr_db"])

    # %% Loop over stations
    for sta in glob.glob(directory + evnm + 'stas/*xml'):
        inv = read_inventory(sta)
        stnw = inv[0].code
        stco = inv[0][0].code
        stla = inv[0][0].latitude
        stlo = inv[0][0].longitude
        stlc = inv[0][0][0].location_code
        result = DistAz(stla, stlo, evla, evlo)
        distdeg = result.getDelta()
        backazi = result.getBaz()

        try:
            st0 = read(directory + evnm + "waves/" + stnw + "." + stco + "." + stlc + ".?H?_*")
            st = st0.copy()
            st.filter("lowpass", freq=4.0)
        except:
            continue

        st.resample(10)
        st.merge(fill_value=np.nan)
        model = TauPyModel(model="iasp91")
        arrivals = model.get_travel_times(source_depth_in_km=evdp, distance_in_degree=distdeg, phase_list=[phase])
        tphase = UTCDateTime(org_t + arrivals[0].time)
        st.trim(tphase - noise_pts / 10, tphase + (npts - noise_pts) / 10)

        if len(st) >= 3 and len(st[0].data) >= npts and len(st[1].data) >= npts and len(st[2].data) >= npts:
            if phase != 'P':
                st.rotate(method="NE->RT", back_azimuth=backazi)
            noise_amp = np.std(np.array(st[cmp].data)[int(noise_pts - 1000): int(noise_pts - 100)])
            quake_amp = np.std(np.array(st[cmp].data)[int(noise_pts): int(noise_pts + 1000)])

            if quake_amp > (noise_amp * minsnr):
                for i in range(3):
                    one_quake[:, i] = np.array(st[i].data)[: npts]
                    if phase == 'P':
                        # %% extract low amplitudes for noise
                        pre_noise[:, i] = np.array(st[i].data)[int(noise_pts - 2 * noise_saved_pts):noise_pts]
                        amplitude_series = np.sqrt(np.sum(pre_noise ** 2, axis=1))
                        amplitude_median = np.nanmedian(amplitude_series)
                        noise0 = pre_noise[amplitude_series < (4 * amplitude_median), :]
                    else:
                        coda_noise[:, i] = np.array(st[i].data)[noise_pts - noise_saved_pts:noise_pts]
                        noise0 = coda_noise

                # %% make sure noise is long enough
                if noise0.shape[0] >= noise_saved_pts:
                    one_noise[:, :] = noise0[:noise_saved_pts, :]

                    # %% normalize quake and noise
                    one_quake[np.isnan(one_quake)] = 0
                    one_noise[np.isnan(one_noise)] = 0
                    one_quake = (one_quake - np.mean(one_quake, axis=0)) / (np.std(one_quake, axis=0) + 1e-12)
                    one_noise = (one_noise - np.mean(one_noise, axis=0)) / (np.std(one_quake, axis=0) + 1e-12)

                    # %% concatenate data
                    all_quake = np.append(all_quake, one_quake[np.newaxis, :, :], axis=0)
                    all_noise = np.append(all_noise, one_noise[np.newaxis, :, :], axis=0)
                    meta = pd.concat([meta, pd.DataFrame(data={
                        "source_id": evnm,
                        "source_origin_time": org_t,
                        "source_latitude_deg": "%.3f" % evla,
                        "source_longitude_deg": "%.3f" % evlo,
                        "source_depth_km": "%.3f" % evdp,
                        "source_magnitude": evmg,
                        "station_network": stnw,
                        "station_code": stco,
                        "station_location_code": stlc,
                        "station_latitude_deg": stla,
                        "station_longitude_deg": stlo,
                        "trace_snr_db": "%.3f" % (quake_amp / (noise_amp + 1e-12))}, index=[0])], ignore_index=True)

    return all_quake, all_noise, meta


if __name__ == '__main__':
    main()
