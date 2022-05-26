import os
from obspy.clients.fdsn import Client
from obspy import UTCDateTime

if not os.path.exists("strikeslip"):
    os.makedirs("strikeslip")
if not os.path.exists("normal"):
    os.makedirs("normal")
if not os.path.exists("reverse"):
    os.makedirs("reverse")

client = Client("ISC")
starttime = UTCDateTime("2010-01-01")
endtime = UTCDateTime("2018-01-01")
cat = client.get_events(starttime=starttime,
                        endtime=endtime,
                        minmagnitude=6.0,
                        mindepth=30.0,
                        maxdepth=100.0,
                        catalog="ISC")
print(len(cat), "are found")
for ev in cat:
    if len(ev.focal_mechanisms) != 0:
#        print(str(ev.resource_id)[13:])
        try:
            strike = ev.focal_mechanisms[0].nodal_planes.nodal_plane_1.strike
            dip = ev.focal_mechanisms[0].nodal_planes.nodal_plane_1.dip 
            rake = ev.focal_mechanisms[0].nodal_planes.nodal_plane_1.rake
            if rake >= 45 and rake <= 135:
                filename = "reverse/" + str(ev.resource_id)[13:] + ".xml"
            elif rake <= -45 and rake >= -135:
                filename = "normal/" + str(ev.resource_id)[13:] + ".xml"
            else:
                filename = "strikeslip/" + str(ev.resource_id)[13:] + ".xml"
            ev.write(filename, format="QUAKEML")
        except:
            continue
