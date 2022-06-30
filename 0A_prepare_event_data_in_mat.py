"""
Use high signal-to-noise ratio teleseismic data
to compose clean earthquake wiggles in HDF5 format

@author: Qibin Shi (qibins@uw.edu)
"""
import glob
import time
import h5py
import numpy as np
from distaz import DistAz
from obspy.taup import TauPyModel
from obspy import read_inventory, read
from obspy import UTCDateTime, read_events

# %%
npts = 6000
idx_trace = 0
idx_event = 0
workdir = '/mnt/DATA0/qibin_data/event_data/M6/'
datadir = '/mnt/DATA0/qibin_data/matfiles_for_denoiser/'
model = TauPyModel(model="iasp91")
pre_filt = (0.004, 0.005, 10.0, 12.0)
allwv = np.zeros((0, npts, 3), dtype=np.double)
onewv = np.zeros((1, npts, 3), dtype=np.double)
cat = read_events(workdir + "*.xml")
print(len(cat), "events in total")

since = time.time()
## %% Loop over event catalog
for ev in cat:
    evnm = str(ev.resource_id)[13:]
    evlo = ev.origins[0].longitude
    evla = ev.origins[0].latitude
    evdp = ev.origins[0].depth / 1000.0
    org_t = ev.origins[0].time

    # %% Loop over stations
    for sta in glob.glob(workdir + evnm + 'stas/*xml'):
        inv = read_inventory(sta)
        stco = inv[0].code + "." + inv[0][0].code + "." + inv[0][0][0].location_code
        stlo = inv[0][0].longitude
        stla = inv[0][0].latitude
        result = DistAz(stla, stlo, evla, evlo)
        distdeg = result.getDelta()
        backazi = result.getBaz()

        try:
            st0 = read(workdir + evnm + "waves/" + stco + ".BH?_*")
            st = st0.copy()
        except:
            continue
        #        try:
        #            st.remove_response(inventory=inv, output='VEL', pre_filt=pre_filt)
        #        except:
        #            continue
        st.filter("lowpass", freq=2.0)
        st.resample(10)
        st.merge(fill_value=np.nan)
        arrivals = model.get_travel_times(source_depth_in_km=evdp, distance_in_degree=distdeg, phase_list=['P'])
        tp = UTCDateTime(org_t + arrivals[0].time)
        st.trim(tp - 300.0, tp + 300.0)

        if len(st) == 3 and len(st[0].data) >= npts and len(st[1].data) >= npts and len(st[2].data) >= npts:
            noise_amp = np.std(np.array(st[0].data)[0:2950])
            pwave_amp = np.std(np.array(st[0].data)[3000:npts-1])

            if pwave_amp > (noise_amp * 25):
                # st.rotate(method="NE->RT", back_azimuth=backazi)
                for i in range(3):
                    onewv[0, :, i] = np.array(st[i].data)[0:npts]
                allwv = np.append(allwv, onewv, axis=0)
                idx_trace = idx_trace + 1

    elapseT = time.time() - since
    idx_event = idx_event + 1
    print(evnm, "--------", idx_event, "events", idx_trace, "traces processed.", "Time elapsed: %.2f s" % elapseT)

with h5py.File(datadir + 'wave_Ponly_2004_18_alldepth_snr_25_sample10Hz_lowpass2Hz.hdf5', 'w') as f:
    f.create_dataset("allwv", data=allwv)

print("Total traces of data:", len(allwv))
