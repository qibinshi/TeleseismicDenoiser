import os
import glob
import numpy as np
from distaz import DistAz
from obspy.taup import TauPyModel
from obspy.clients.iris import Client
from obspy import read_inventory, read
from obspy import UTCDateTime, read_events

# %% Velocity model and prefilter
model = TauPyModel(model="iasp91")
pre_filt = (0.004, 0.005, 10.0, 12.0)

## %% Catalog
cat = read_events("*.xml")
print(len(cat), "events totally")
for ev in cat:
    evnm = str(ev.resource_id)[13:]
    evlo = ev.origins[0].longitude
    evla = ev.origins[0].latitude
    evdp = ev.origins[0].depth/1000.0
    org_t= ev.origins[0].time
    print(evnm)

    newdir = evnm+"selectTRZ"
    if not os.path.exists(newdir):
        os.makedirs(newdir)

    dis_table = np.zeros((6, 6))
    baz_table = np.zeros((6, 6))
    dazi_table = np.ones((6, 6)) * 30
    code_table = [ [ '' for i in range(6)] for i in range(6)]
    xml_table  = [ [ '' for i in range(6)] for i in range(6)]

## %% Pick 6x6 stations
    for sta in  glob.glob(evnm+'stas/*xml'):
        inv = read_inventory(sta)
        stlo = inv[0][0].longitude
        stla = inv[0][0].latitude
        stco = inv[0].code+"."+inv[0][0].code+"."+inv[0][0][0].location_code
        result = DistAz(stla, stlo, evla, evlo)
        distdeg = result.getDelta()
        azimuth = result.getAz()
        backazi = result.getBaz()
        dis_idx = round(distdeg / 10 - 3) 
        azi_idx = round(azimuth / 60)
        dazi = abs(round(azimuth - azi_idx * 60.0))
        if azi_idx >= 0 and azi_idx <=5 and dis_idx >= 0 and dis_idx <=5:
            if dazi < dazi_table[dis_idx, azi_idx]:
                dazi_table[dis_idx, azi_idx] = dazi
                baz_table[dis_idx, azi_idx] = backazi
                dis_table[dis_idx, azi_idx] = distdeg
                code_table[dis_idx][azi_idx] = stco
                xml_table[dis_idx][azi_idx] = sta

## %% Resample, trim and rotate
    for i in range(6):
        for j in range(6):
            if len(code_table[i][j]) > 0:
                stnm = evnm+"waves/"+code_table[i][j]+".BH?_*"
                st0 = read(stnm)
                st = st0.copy()
                iv = read_inventory(xml_table[i][j])
                try:
                    st.remove_response(inventory=iv, output='VEL', pre_filt=pre_filt)
                except:
                    continue
                st.filter("lowpass", freq=1.0)
                st.resample(2.0)
                arrivals = model.get_travel_times(source_depth_in_km=evdp,
                                                  distance_in_degree=dis_table[i,j],
                                                  phase_list=['P'])
                tp = UTCDateTime(org_t + arrivals[0].time)
                st.trim(tp - 100.0, tp + 2000.0)
                print(code_table[i][j], "trimed")
                st.rotate(method="NE->RT", back_azimuth=baz_table[i,j])
                for tr in st:
                    tr.write(newdir+"/"+str(i)+"_"+str(j)+"_"+ tr.id +".mseed", format="MSEED")
