"""
@author: Qibin Shi
qibins@uw.edu
"""
## modified from distaz.c

import math

class DistAz:

    def __init__(self,  lat1,  lon1,  lat2,  lon2):

        self.stalat = lat1
        self.stalon = lon1
        self.evtlat = lat2
        self.evtlon = lon2
        if (lat1 == lat2) and (lon1 == lon2):
            delta = 0.0
            az = 0.0
            baz = 0.0
            return
        
        rad=2.*math.pi/360.0
        sph=1.0/298.257

        scolat=math.pi/2.0 - math.atan((1.-sph)*(1.-sph)*math.tan(lat1*rad))
        ecolat=math.pi/2.0 - math.atan((1.-sph)*(1.-sph)*math.tan(lat2*rad))
        slon=lon1*rad
        elon=lon2*rad
        a=math.sin(scolat)*math.cos(slon)
        b=math.sin(scolat)*math.sin(slon)
        c=math.cos(scolat)
        d=math.sin(slon)
        e=-math.cos(slon)
        g=-c*e
        h=c*d
        k=-math.sin(scolat)
        aa=math.sin(ecolat)*math.cos(elon)
        bb=math.sin(ecolat)*math.sin(elon)
        cc=math.cos(ecolat)
        dd=math.sin(elon)
        ee=-math.cos(elon)
        gg=-cc*ee
        hh=cc*dd
        kk=-math.sin(ecolat)
        delrad=math.acos(a*aa + b*bb + c*cc)
        self.delta=delrad/rad
        rhs1=(aa-d)*(aa-d)+(bb-e)*(bb-e)+cc*cc - 2.
        rhs2=(aa-g)*(aa-g)+(bb-h)*(bb-h)+(cc-k)*(cc-k) - 2.
        dbaz=math.atan2(rhs1,rhs2)
        if (dbaz<0.0):
            dbaz=dbaz+2*math.pi
        
        self.baz=dbaz/rad
        rhs1=(a-dd)*(a-dd)+(b-ee)*(b-ee)+c*c - 2.
        rhs2=(a-gg)*(a-gg)+(b-hh)*(b-hh)+(c-kk)*(c-kk) - 2.
        daz=math.atan2(rhs1,rhs2)
        if daz<0.0:
            daz=daz+2*math.pi
        
        self.az=daz/rad
        if (abs(self.baz-360.) < .00001):
            self.baz=0.0
        if (abs(self.az-360.) < .00001):
            self.az=0.0

    def getDelta(self):
        return self.delta

    def getAz(self):
        return self.az

    def getBaz(self):
        return self.baz

    def degreesToKilometers(self, degrees):
        return degrees * 111.19
    
    def kilometersToDegrees(self, kilometers):
        return kilometers / 111.19

#distaz = DistAz(0, 0, 1,1)
#print "%f  %f  %f" % (distaz.getDelta(), distaz.getAz(), distaz.getBaz())
