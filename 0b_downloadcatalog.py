"""
@author: Qibin Shi
qibins@uw.edu
"""
# %%
from obspy import read_events
from obspy.clients.fdsn.mass_downloader import CircularDomain, \
    Restrictions, MassDownloader

cat = read_events("*.xml")
#print(len(cat))
for ev in cat:
    ev_name = str(ev.resource_id)[13:]
    strike = ev.focal_mechanisms[0].nodal_planes.nodal_plane_1.strike
    dip = ev.focal_mechanisms[0].nodal_planes.nodal_plane_1.dip
    rake = ev.focal_mechanisms[0].nodal_planes.nodal_plane_1.rake
    org_t = ev.origins[0].time
    lon = ev.origins[0].longitude
    lat = ev.origins[0].latitude
    print("Downloading-------", ev_name, strike, dip, rake)

    domain = CircularDomain(latitude=lat, longitude=lon,
                        minradius=25.0, maxradius=85.0)

    restrictions = Restrictions(
        starttime = org_t - 0,
        endtime = org_t + 3600,
        reject_channels_with_gaps=True,
        minimum_length=0.95,
        minimum_interstation_distance_in_m=200E3,
        channel_priorities=["BH[ZNE]"],
        location_priorities=["", "00"])

    mdl = MassDownloader()
    mdl.download(domain, restrictions, mseed_storage=ev_name+"waves", stationxml_storage=ev_name+"stas")
