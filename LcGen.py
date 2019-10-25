#!/usr/bin/python

import argparse
import datetime
import dateutil.parser
import requests # installs with : pip install requests
import xml.etree.ElementTree as ElementTree
import os
import getpass

import matplotlib
matplotlib.use('Agg')

import glob
import time
from scipy import interpolate as interp
from scipy import stats
import pandas as pd
import numpy as np
from astropy.io import ascii
import sys
import pickle
import scipy.optimize as opt

from isochrones.mist import MIST_Isochrone
from isochrones.dartmouth import Dartmouth_Isochrone

from ellc import lc as ellclc

import celerite
from celerite import terms
from astropy import units as u
from astropy.constants import M_sun, L_sun, R_sun
try:
    from shocksgo.shocksgo import generate_stellar_fluxes
except:
    from shocksgo import generate_stellar_fluxes

XML_NS = {'uws':'http://www.ivoa.net/xml/UWS/v1.0', 'xlink':'http://www.w3.org/1999/xlink'}



Mjup   = 1.89852e27 #kg
Mearth = 5.9736e24 # kg
Rjup   = 71492000. # equatorial radius [m]
Rearth = 6378137.0 # equatorial radius [m]
Rsun   = 695700000.0
au     = 149597870700.0
Msun   = 1.98847e30
G      = 6.67e-11

goto='/home/hosborn'



##################################################################
#                                                                #
#                     BESANCON STAR MODELS                       #
#                                                                #
##################################################################


class UWS:

    def __init__(self, url, auth):
        self._url = url
        self._auth = auth

    def getJobsList(self):
        r = requests.get(self._url+"/jobs", auth=self._auth)
        return self._getJobInfosFromReqResult(r, "get list of jobs")

    def createJob(self):
        r = requests.post(self._url+"/jobs", auth=self._auth)
        return self._getJobInfosFromReqResult(r, "creating job")

    def getJobInfos(self, idJob):
        r = requests.get(self._url+"/jobs/"+str(idJob), auth=self._auth)
        return self._getJobInfosFromReqResult(r, "get details of a job")

    def deleteJob(self, idJob):
        r = requests.delete(self._url+"/jobs/"+str(idJob), auth=self._auth)
        return self._getJobInfosFromReqResult(r, "deleting a job")

    def runJob(self, idJob):
        r = requests.post(self._url+"/jobs/"+str(idJob)+'/phase', auth=self._auth, data={'PHASE':'RUN'})
        return self._getJobInfosFromReqResult(r, "starting a job")

    def abortJob(self, idJob):
        r = requests.post(self._url+"/jobs/"+str(idJob)+'/phase', auth=self._auth, data={'PHASE':'ABORT'})
        return self._getJobInfosFromReqResult(r, "aborting a job")

    def maxDurationJob(self, idJob, seconds):
        r = requests.post(self._url+"/jobs/"+str(idJob)+'/executionduration', auth=self._auth, data={'EXECUTIONDURATION':seconds})
        return self._getJobInfosFromReqResult(r, "setting execution duration of a job")

    def timeDestructJob(self, idJob, timeStr):
        r = requests.post(self._url+"/jobs/"+str(idJob)+'/destruction', auth=self._auth, data={'DESTRUCTION':timeStr})
        return self._getJobInfosFromReqResult(r, "setting execution duration of a job")

    def setParams(self, idJob, params):
        #if len(params) == 1:    # TODO
        #    r = requests.put(self._url+"/jobs/"+str(idJob)+'/parameters/'+params[0][0], data=params[0][1])
        #    print(r.text)
        #    return getJobInfos(idJob)#self._getJobInfosFromReqResult(r, "setting parameters of a job")
        #elif len(params) > 1:
            data={}
            for p in params:
                data[p[0]] = p[1]
            r = requests.post(self._url+"/jobs/"+str(idJob)+'/parameters', auth=self._auth, data=data)
            return self._getJobInfosFromReqResult(r, "setting parameters of a job")

    def _getJobInfosFromReqResult(self, r, phase):
        if (len(r.history) > 0):
            print("SERVER RESPONSE : "+r.history[0].text)

        if (r.status_code == 200):
            #et = ElementTree
            #ElementTree.register_namespace('uws', 'http://www.ivoa.net/xml/UWS/v1.0')
            for ns in XML_NS:
                ElementTree.register_namespace(ns, XML_NS[ns])

            try:
                return ElementTree.fromstring(r.text)
            except Exception as e:
                print("Invalid response :")
                print(r.text)
                raise e
        else:
            raise JobError(r, phase)




# Convertion into text

def xmlTextIfFound(xmlElem):
    if (xmlElem == None) or (xmlElem.text == None):
        return ''
    else:
        return xmlElem.text

def xmlDateIfFound(xmlElem):
    if (xmlElem == None) or (xmlElem.text == None):
        return '-'
    else:
        d = dateutil.parser.parse(xmlElem.text).astimezone(dateutil.tz.tzutc())
        return d.strftime("%Y-%m-%d %H:%M")


# Exceptions

class JobError(Exception):
    def __init__(self, http, when):
        self.status = http.status_code
        self.explain = http.text
        self.when = when
    def __str__(self):
        return "Error when "+self.when+" : "+str(self.status)+" "+self.explain

    
def GetBesanconCat(hemis,dtype,n,outdir='BesanconModels/'):
    print('|'+hemis+'|','|'+dtype+'|',n)
    if not os.path.isdir(outdir):
        os.system('mkdir '+outdir)
    if len(glob.glob(outdir+hemis+'_'+dtype+'_'+str(n)+'_dat*'))==0:
        user = input("Besancon Username")
        passwd = input("Besancon Password")
        
        if not os.path.isdir(outdir):
            os.system('mkdir '+outdir)

        if str(hemis)=='South':
            lats=np.arange(-54,-5.99,4.8)
            if dtype=='deep':
                longs=[250.6,255.4]
            elif dtype=='wide':
                longs=[229.0,277.0]
            else:
                print(dtype,'must be wide or deep')
        elif str(hemis)=='North':
            lats=np.arange(6,54.01,4.8)
            if dtype=='deep':
                longs=[62.6,67.4]
            elif dtype=='wide':
                longs=[41,89]
            else:
                print(dtype,'must be wide or deep')
        else:
            print(hemis,'must be North or South')
        #band_min, band_max, band_step, errBand_A, errBand_B, errBand_C
        param ={'KLEH':1,'KLEG':2,'Dist_min':0.0,'Dist_max':12,'Dist_step':0.1,
                'band_min':'-1.00, -99.00, -99.00, -99.00, -99.00, -99.00, -99.00, -99.00, -99.00',
                'mass_step':0.1,'Teff_step':50.,'logg_step':0.15,'age_step':0.1,'AlphaFe_step':0.1,'Radius_step':0.025,
                'errBand_A':'0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0',
                'errBand_B':'0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0',
                'errBand_C':'0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0',
                'errHRV_A':0,'errPM1_A':0,'errPM2_A':'0.','errTeff_A':0.,
                'errLogg_A':0.1,'errAlphaFe_A':0.02,'errMet_A':0.1,'errAge_A':0.1,'errMass_A':0.1,
                'band_step':'0.20, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00','sendmail':0}
        if dtype=='deep':
            param['band_max']='26.00, 99.00, 99.00, 99.00, 99.00, 99.00, 99.00, 99.00, 99.00'
        elif dtype=='wide':
            param['band_max']='17.00, 99.00, 99.00, 99.00, 99.00, 99.00, 99.00, 99.00, 99.00'
        param['Coor1_min']=longs[0]
        param['Coor1_max']=longs[1]
        param['Coor1_step']=4.8
        param['Coor2_min']=lats[n]
        param['Coor2_max']=lats[n+1]
        param['Coor2_step']=4.8
        param=[(p,param[p]) for p in param]

        create=True
        execdur=None
        tdest=None
        run=True
        abort=False
        delete=False
        url='https://model.obs-besancon.fr/ws'
        uws = UWS(url, (user, passwd))

        job = None
        try:
            print("Creating job")
            job = uws.createJob()
        except JobError as e:
            print(e)


        idJob = int(job.find('uws:jobId', XML_NS).text)
        job = uws.setParams(idJob, param)
        job = uws.runJob(idJob)

        waiting=True
        nwaits=0
        while waiting and nwaits<180:
            r = requests.get(uws._url+"/jobs/"+str(idJob), auth=uws._auth)
            if r.status_code==200 and '\"output-head\" xlink:href=\"' in r.text:
                waiting=False
                #head_int=r.text.find('\"output-head\" xlink:href=\"')+26
                #head_loc=r.text[head_int:head_int+79].replace('\"','').replace(' ','')
                #print("Downloading "+head_loc+" to "+outdir+hemis+'_'+dtype+'_'+str(n)+'_head.ascii')
                #os.system('wget \''+head_loc+'\' -O '+outdir+hemis+'_'+dtype+'_'+str(n)+'_head.ascii')
                res_int=r.text.find('result id=\"output\" xlink:href=')+31
                res_loc=r.text[res_int:res_int+73]
                #print("Downloading "+res_loc+" to "+outdir+hemis+'_'+dtype+'_'+str(n)+'_dat.ascii')
                #os.system('wget \''+res_loc+'\' -O '+outdir+hemis+'_'+dtype+'_'+str(n)+'_dat.ascii')
                ascii.read(res_loc).to_pandas().to_csv(outdir+hemis+'_'+dtype+'_'+str(n)+'_dat.csv')
                fout=outdir+hemis+'_'+dtype+'_'+str(n)+'_dat.csv'
                uws.deleteJob(idJob)
            elif r.status_code==404:
                #Exit
                waiting=False
                print(hemis+'_'+dtype+'_'+str(n),'has permanently failed. Exiting')
                fout=None
                raise ValueError(hemis+'_'+dtype+'_'+str(n)+' fails due to 404')
            else:
                waiting=True
                print(r.status_code,len(r.text))
                time.sleep(10)
                nwaits+=1
        if nwaits==180:
            raise ValueError("Took too long to get to Besancon server.")
        else:
            #uws.deleteJob(idJob)
            print('Downloaded file to ',outdir+hemis+'_'+dtype+'_'+str(n)+'_dat.csv')
            return fout
    else:
        print('File exists at ',outdir+hemis+'_'+dtype+'_'+str(n)+'_dat.csv')
        return glob.glob(outdir+hemis+'_'+dtype+'_'+str(n)+'_dat*')[0]



def parseBes(file):
    if type(file)==str:
        if file[-4:]=='scii':
            dat=ascii.read(file).to_pandas()
        elif file[-7:]=='.csv.gz':
            import gzip
            dat=pd.DataFrame.from_csv(gzip.open(file))
        elif file[-4:]=='.csv':
            dat=pd.DataFrame.from_csv(file)
        dat['file']=file
    elif type(file)==pd.DataFrame:
        dat=file.copy(deep=True)
    dropcols=[col for col in dat.columns if col[:3]=='err']+['HRV', 'UU', 'VV', 'WW', 'Px', 'CL', 'Pop', 'x_Gal', 'y_Gal', 'z_Gal']
    for col in dropcols:
        if col in dat.columns:
            dat.drop(col,axis=1,inplace=True)
    dat=dat.convert_objects(convert_numeric=True)
    #Removing white dwarfs:
    dat=dat.loc[dat['logg']<6.5]
    dat.loc[dat.Mass<0.1,'Mass']=0.1
    dat.loc[dat['[M/H]']<-2.5,'[M/H]']=-2.5
    dat.loc[dat['Age']<0.5,'Age']=0.5
    dat.loc[dat['Age']<0.5,'Age']=0.5
    #Plato mag:
    dat['Pmag'] = getPmag(dat['V'],dat['Teff'])

    #Some Besancon ages are too old. Adjusting:
    agelim= lambda mass:np.polyval(np.polyfit([0.87,1.45],[14.1,7],1),mass)
    ov_age_lim=dat['Age']>agelim(dat['Mass'].values)
    dat.loc[ov_age_lim,'Age']=agelim(dat.loc[ov_age_lim]['Mass'].values)

    dat['logT']=np.log10(dat['Teff'])
    dat['logL']=(4.8 - dat['Mbol'].values)/2.5
    dat['Lum']=np.power(10,dat.loc[:,'logL'])
    dat['rho']=dat.Mass/dat.Radius**3
    dat['Dist']*=1000.0

    #'V', 'B-V', 'U-B', 'V-I', 'V-K', 'mux', 'muy', 'HRV', 'UU', 'VV', 'WW',
    #'Px', 'Mv', 'CL', 'Typ', 'Teff', 'logg', 'Pop', 'Age', 'Mass', 'Mbol',
    #'Radius', '[M/H]', '[a/Fe]', 'longitude', 'latitude', 'RAJ2000',
    #'DECJ2000', 'Dist', 'x_Gal', 'y_Gal', 'z_Gal', 'Av'
    dat=dat.rename(columns={'Dist':'dist','Radius':'Rs','[M/H]':'FeH','Mass':'Ms'})
    
    return dat


##################################################################
#                                                                #
#                     PLATO-SPECIFIC SPECS                       #
#                                                                #
##################################################################

def getPmag(V,Teff):
    return V+1.238e-12*Teff**3-4.698e-8*Teff**2+5.982e-4*Teff-2.506

def getPmagRms(Pmag,Nscopes,poly=None):
    if poly is None:
        Pmags=np.array([7.66,8.16,8.66,9.16,9.66,10.16,10.66,11.16,11.66,12.16,12.66])
        noise_24=np.array([9.8,12.3,15.5,19.5,24.6,31.0,39.0,49.1,61.8,77.7,97.9])
    
        poly=np.polyfit(Pmags,np.log10(noise_24),2)
    Pnoise_24=lambda Pmag: np.polyval(poly,Pmag)
    return np.power(10,Pnoise_24(Pmag)*(Nscopes/24)**0.5),poly

def getVmagRms(V,Nscopes,poly=None):
    if poly is None:
        Vmags=np.array([8,8.5,9,9.5,10,10.5,11,11.5,12,12.5,13])
        noise_24=np.array([9.8,12.3,15.5,19.5,24.6,31.0,39.0,49.1,61.8,77.7,97.9])

        poly=np.polyfit(Vmags,np.log10(noise_24),2)
    Vnoise_24=lambda Vmag: np.polyval(poly,Vmag)
    return np.power(10,Vnoise_24(V)*(Nscopes/24)**0.5),poly

def assembleP5_new(hemisphere,npart, nmult=1,ext='',mag='Pmag',peturb_cat=False,outdir='/home/hosborn/PLATO/BesanconModels2'):
    #Assembles PLATO sample
    # hemisphere  - north or south
    # npart - number part between 0 and 9 for each of the 10 split Besancon files 
    # nmult = multiplication factor to emulate, eg, 10x the number of stars for statistical purposes.
    #
    T=2*365.25
    simarea=4.8*(4.8*10) #one 4.8deg slice of the 10x10 grid

    camerafields={24:301,18:247,12:735,6:949,'total':2232,'fast':619}

    #Dictionary as a function of height: [0,6,12,18,24]
    scopebyrow=np.array([[3.5,5.4,1,0,0],[2.2,5.5,2.5,0,0],[1.3,5.2,3.4,0,0],[0.55,5.5,4.2,0,0],[0.15,5.3,4.6,0,0],
                     [0.0,4.4,2.3,3.2,0.3],[0.0,2.6,2.8,2.8,2.0],[0.0,1.5,3.6,1.6,3],[0.0,0.8,4.4,1.0,3.9],
                     [0.0,0.4,5,0.4,4.4],[0.0,0.4,5,0.4,4.4],[0.0,0.8,4.4,1.0,3.9],[0.0,1.5,3.6,1.6,3],
                     [0.0,2.6,2.8,2.8,2.0],[0.0,4.4,2.3,3.2,0.3],[0.15,5.3,4.6,0,0],[0.55,5.5,4.2,0,0],
                     [1.3,5.2,3.4,0,0],[2.2,5.5,2.5,0,0],[3.5,5.4,1,0,0]])
    #adjusting such that area of observed region is 1.0
    scopebyrow/=np.sum(scopebyrow[:,1:])
    #adjusting such that the ratios between observed areas matches PLATO official guide
    scopebyrow[:,1:]/=np.tile(np.sum(scopebyrow[:,1:],axis=0)/(np.array([949,735,247,301])/2232),(20,1))
    '''
    newnoise= np.array([[   0,   40,  260,  200,  150,  120],
                        [   3,   10,   64,   48,   37,   29],
                        [   4,    6,   40,   30,   23,   18],
                        [   5,   10,   25,   19,   14,   11],
                        [   6,   16,   15,   12,    9,    7],
                        [   7,   25,    9,    6,    5,    4],
                        [   8,   40,   14,   10,    8,    7],
                        [   9,   65,   23,   16,   13,   11],
                        [  10, 4498,   36,   25,   21,   18],
                        [  11, 4498,   58,   41,   34,   29],
                        [  12, 4498,   97,   69,   56,   48],
                        [  13, 4498,  171,  120,   98,   86],
                        [  14, 4498,  320,  228,  187,  161],
                        [  15, 4498,  675,  478,  389,  338],
                        [  16, 4498, 1541, 1091,  893,  777],
                        [  17, 4498, 4126, 2750, 2179, 1895]])
    '''

    fovs2=np.array([[24,18,12,6,2],[301,247,735,949,619]])

    allstars=pd.DataFrame()

    starfile=GetBesanconCat(hemisphere,'wide',npart,outdir=outdir)
    stars=parseBes(starfile)
    nparts=10

    n_samples=nmult*(np.sum(scopebyrow)*camerafields['total'])/(nparts*simarea)

    #Multiplying by n_samples using normal dist:
    morestars=stars.iloc[np.random.choice(len(stars),int(np.ceil(len(stars)*n_samples*(1/nparts))),replace=True)]
    morestars['csv']=starfile

    if peturb_cat:
        for col in ['dist','Av','Pmag','logg']:
            morestars[col]*=np.random.normal(1.0,0.01,len(morestars))

        for col in ['Mv','Typ','LTef','Age','Ms','B-V','U-B','V-I','V-K','FeH','Mbol','Rs']:
            morestars[col]+=np.random.normal(0.0,0.05*np.std(stars.loc[:,col]),len(morestars))

    #Get N Telescope Coverage:
    middle=np.average([np.min(morestars['latitude'].values),np.max(morestars['latitude'].values)])

    morestars['Nscopes']=np.zeros(morestars.shape[0])
    morestars.loc[morestars['latitude']<middle,'Nscopes']=np.digitize(np.random.random(np.sum(morestars['latitude'].values<middle)),
                                                          np.cumsum(scopebyrow[npart*2])/np.sum(scopebyrow[npart*2]))*6
    morestars.loc[morestars['latitude']>=middle,'Nscopes']=np.digitize(np.random.random(np.sum(morestars['latitude'].values>=middle)),
                                                          np.cumsum(scopebyrow[npart*2+1])/np.sum(scopebyrow[npart*2+1]))*6
    morestars=morestars.drop(morestars.loc[morestars['Nscopes']==0.0].index)
    #Get RMS:
    morestars['rms_hr']=np.zeros(len(morestars))
    poly=None
    for n in np.unique(morestars.Nscopes):
        if n>0:
            rms,poly=getPmagRms(morestars.loc[morestars['Nscopes']==n,'Pmag'].values,n,poly)
            morestars.loc[morestars['Nscopes']==n,'rms_hr']=rms
    #Assuming 1yr period
    morestars['Earth_depth']=(1e6*(0.009168/morestars['Rs'].values)**2)
    morestars['Earth_dur']=0.421015/(morestars['Ms'].values/morestars['Rs'].values**3)**(1/3)
    morestars['Earth_Ntrs']=np.tile(2,len(morestars))
    #morestars['Earth_Ntrs']=np.floor(T/365.25)+(np.random.random(len(morestars))>(T/365.25-np.floor(T/365.25))).astype(int)

    #Using habitable zone period. Likely to favour smaller stars with correspondingly shorter P_HZ
    morestars['HZ_per']=365.25*np.power(10,(morestars.Mbol-4.74)/-2.5)**0.75/np.sqrt(morestars['Ms'].values)
    morestars['HZ_Ntrs']=np.zeros(len(morestars))
    #Calculating N transits. Where P<T, always have at least one. Where P>T, sometimes have one - depends on (P/T)/T
    morestars.loc[morestars['HZ_per']<=T,'HZ_Ntrs']=np.floor(T/morestars.loc[morestars['HZ_per']<=T,'HZ_per'].values)+(T/morestars.loc[morestars['HZ_per']<=T,'HZ_per']-np.floor(T/morestars.loc[morestars['HZ_per']<=T,'HZ_per'].values))
    morestars.loc[morestars['HZ_per']>T,'HZ_Ntrs']=(T/morestars.loc[morestars['HZ_per']>T,'HZ_per'].values)

    #morestars['HZ_Ntrs']=np.floor(morestars['HZ_per']/(2*365.25))+(np.random.random(len(morestars))>(morestars['HZ_per']/(2*365.25)-np.floor(morestars['HZ_per']/(2*365.25)))).astype(int)
    morestars['HZ_dur']=0.421015*np.power(10,(morestars.Mbol.values-4.74)/-2.5)/morestars['Ms'].values
    morestars['Earth_SNR']=morestars['Earth_depth'].values/(morestars['rms_hr'].values/np.sqrt(24*morestars['Earth_dur'].values*np.ceil(morestars['Earth_Ntrs'].values)))#109.1morestars['Rs']morestars['rms_hr']
    morestars['HZ_SNR']=morestars['Earth_depth'].values/(morestars['rms_hr'].values/np.sqrt(24*morestars['HZ_dur'].values*np.ceil(morestars['HZ_Ntrs'].values)))#109.1morestars['Rs']morestars['rms_hr']

    #Get Aperture size:
    #Taking TESS-scaled apertures
    #kep_ap=KepLikeApertures(morestars[mag])#ap_size=(tess_ap+np.random.random(len(morestars[mag])))*(kep_ap-tess_ap)
    morestars['ap_size']=MarchioriAperture(morestars[mag].values)
    morestars['ap_radius']=np.sqrt(morestars['ap_size'].values/np.pi)*15 #aperutre radious in arcsec, assuming 15arcsec/pix
    #np.random.randint(np.ceil(sigmoid(morestars[mag],yshift=3*1.35)),np.floor(KepLikeApSize(morestars[mag])))

    morestars['besanson_index']=morestars.index.values
    morestars.set_index(np.array([hemisphere[0]+str(npart)+ext+'_'+str(i).zfill(8) for i in np.arange(len(morestars.index.values))]),inplace=True)

    #Also includes all of P1
    p5condn=((morestars['HZ_SNR']>(7.2/4))|(morestars['Earth_SNR']>(7.2/4)))&(morestars[mag]<13)

    return morestars.loc[p5condn]

def TessLikeApertures(x,xshift=10,xmult=1.5,yshift=4.05,ymult=8.5):
    #Returns aperture size assuming PLATO is similar to TESS (and not Kepler)
    return yshift+2*ymult / (1 + np.exp((x-xshift)/xmult))


def MarchioriAperture(Pmag):
    #Returns number of pixels in mask from Marchiori et al (2019), using pixel data from figure 21c
    #     pixsize=np.polyval(np.polyfit([469,23],[4,16],1),[142,195,240,278,315,350,379,405,428])
    #     mag=np.polyval(np.polyfit([144,615],[8,12],1),[163,221,282,340,398,457,517,575,634])
    #     pix_est=np.polyfit(mag,pixsize,2)
    #     clipping between 1 and 50 pixels
    return np.clip(np.polyval(np.array([ 0.20470492, -6.06473984, 48.60059884]),Pmag),1,50)


#(p5cat,blendcat,deltamag_max_thresh=10)

def Blends_np_PLATO(wide,deep,mag='Pmag',deltamag_max_thresh=10,deltamag_min_thresh=0):
    #Gets blended stars using "deep" besancon file

    #We do not have positions in Besancon, so we should probabilisitcally assemble nearby stars at this step... How?]
    #Use the aperture area being observed (1.5**2*npix*3.98) as a fraction of the number of stars generated
    # and take a poisson-distributed n for number of stars

    # assembling nstars per degree from the deep besancon cat. Then adding the expected number of stars per deg to the wide cat
    nbins=10
    binsize=(4.8**2)/nbins
    lat_steps=np.linspace(np.min(deep.latitude)-0.01,np.max(deep.latitude)+0.01,nbins+1)
    lat_counts=np.array([np.sum(np.digitize(deep.latitude.values,lat_steps)==n) for n in range(1,nbins+1)])/binsize
    wide['lat_subbin']=np.argmin(abs(wide.latitude.values[np.newaxis,:]-lat_steps[:,np.newaxis]),axis=0)-1
    wide['n_st_perdeg']=lat_counts[wide['lat_subbin']]

    #adjusting for the assumption that blend is less bright that target:
    histbymag=np.histogram(deep[mag],50)
    n_stars_as_func_mag=np.polyfit(histbymag[1][:-1]+np.diff(histbymag[1]),np.log10(np.cumsum(histbymag[0])),3)
    wide['n_st_perdeg']*=(1-np.power(10,np.polyval(n_stars_as_func_mag,wide[mag]))/len(deep))

    #Adding blends with Poisson dist.
    #allstars['lamda']=np.clip((allstars['n_st_perdeg']-1).astype(float)*(allstars['npix']*15**2)/(3600**2),0.0,1000) # in arcsec

    wide['lamda']=np.clip(((wide['n_st_perdeg']-1).astype(float)*(wide['ap_radius']*2)**2)/(3600**2),0.0,1000) # in arcsec
    #print('lambda histogram:',np.percentile(wide['lamda'],[5,16,50,85,95]))
    wide['n_blends']=np.random.poisson(lam=wide['lamda'])
    #print(np.histogram(wide['n_blends'])[0])

    wide['has_blend']=wide['n_blends']>0

    stars_and_blends=wide.copy()
    stars_and_blends=stars_and_blends.set_index(stars_and_blends.index.values.astype(str))#making index a string
    stars_and_blends['type']=np.tile('target',stars_and_blends.shape[0])
    stars_and_blends['orb_parent']=wide.index.values
    stars_and_blends['blend_parent']=wide.index.values
    stars_and_blends['deltamag']=np.zeros(len(stars_and_blends))
    stars_and_blends['sep']=np.zeros(len(stars_and_blends))
    stars_and_blends['frac_blend_in_aperture']=np.ones(len(stars_and_blends))
    cols2take=['dist', 'Mv', 'Typ', 'Teff', 'logg', 'Age', 'Ms', 'B-V', 'U-B','V-I','V-K',
               'V', 'FeH', 'Av', 'Mbol', 'logL', 'Lum', 'Rs', 'rho','file']
    if mag not in cols2take:
                cols2take+=[mag]
    for star in stars_and_blends.loc[stars_and_blends.has_blend].iterrows():
        #Getting blended background stars for each
        chx=np.random.choice(deep.loc[deep[mag]>star[1][mag]].index.values,star[1]['n_blends'],replace=False)

        #print(deep.shape[0],star[1]['n_blends'],chx)
        new_stars = deep.loc[chx][cols2take]

        # computing separation (randomly) and deltamag
        new_stars['deltamag']=new_stars[mag]-star[1][mag]
        new_stars[mag+'_parent']=star[1][mag]
        new_stars['Nscopes']=star[1]['Nscopes']
        new_stars['lat_subbin']=star[1]['lat_subbin']
        #Dropping blends whos are brighter than the target!
        new_stars=new_stars.drop(new_stars.loc[new_stars['deltamag']<deltamag_min_thresh].index.values)

        new_stars['sep']=np.sqrt(np.random.random(new_stars.shape[0]))*(2*star[1]['ap_radius']) #distances are sqrt(random)
                
        #adding info - that these stars are blends attached to the parent star
        new_stars['type']=np.tile('blend',new_stars.shape[0])

        new_stars['blend_parent']=np.tile(star[0],new_stars.shape[0])
        #Only keeping stars with deltamag<10:
        new_stars=new_stars.loc[new_stars['deltamag']<deltamag_max_thresh,:]

        new_stars = new_stars.set_index(np.array([str(star[0])+"_"+str(ib).zfill(2) for ib in range(1,new_stars.shape[0]+1)]))
        new_stars['orb_parent']=new_stars.index.values
        new_stars['latitude']=star[1]['latitude']
        new_stars['longitude']=star[1]['longitude']
        
        new_stars['frac_blend_in_aperture']=stats.norm.cdf(new_stars['sep']-star[1]['ap_radius'],0,0.2*15)

        stars_and_blends=stars_and_blends.append(new_stars,sort=False)

    #adding logage - used later with isochrones.
    #stars_and_blends['logage']=np.log10(1e9*stars_and_blends['Age'])

    return stars_and_blends

def getKeplerGDs(Ts,logg=4.43812,FeH=0.0,v_mic=2.0,Fr='Kp',mod='PHOEN'):
    #Get Kepler Limb darkening coefficients.
    #print(label)
    arr = pd.read_fwf("tables/KeplerGDlaws.dat")

    FeHarr=np.unique(arr['Fe/H'].values)
    FeH=FeHarr[find_nearest(FeHarr,FeH)]
    v_micarr=np.unique(arr['v_mic'].values)
    v_mic=v_micarr[find_nearest(v_micarr,v_mic)]

    arr2=arr[(arr['Fe/H'].values==FeH)*(arr['v_mic'].values==v_mic)*(arr['Fr'].values==Fr)*(arr['Model'].values==mod)]
    y_0interp=interp.CloughTocher2DInterpolator(arr2[['logT','logg']].values,arr2['y_1'].values)
    if (type(Ts)==float)+(type(Ts)==int) and((Ts<50000.)*(Ts>=3500)):
        return y_0interp(np.log10(Ts),np.clip(logg,0.0,4.99))
    elif (type(logg)==float)+(type(logg)==int) and ((Ts<50000.)*(Ts>=3500)).all():
        return y_0interp(np.log10(np.clip(Ts,3500,50000)),np.tile(np.clip(logg,0.0,4.99),len(Ts)))
    else:
        return y_0interp(np.log10(np.clip(Ts,3500,50000)),np.clip(logg,0.0,4.99))


def Get_Beaming(Teff):
    beams=np.genfromtxt("tables/Beaming_with_Teff.txt")
    import scipy.interpolate as interp
    beaming=interp.interp1d(beams[:,0],beams[:,1])
    return beaming(np.clip(Teff,beams[0,0],beams[-1,0]))

def getQuadLDs(Ts,logg=4.43812,FeH=0.0,band='V',xi=2.0,method='L',model='PHOENIX'):
    if band=='Kepler':
        return getKeplerLDs_np(Ts,logg,FeH,how='2')
    else:
        #Gets Quadratic LD parameters
        #band = 'Kp', 'C', 'S1', 'S2', 'S3', 'S4', 'u', 'v', 'b', 'y', 'U', 'B', 'V', 'R', 'I', 'J', 'H', 'K', 'u*', 'g*', 'r*', 'i*', 'z*'
        #xi = Microturbulent velocity
        #Method = L (least squares) or F (flux)
        #model = ATLAS or PHOENIX

        #From Claret 2011, http://vizier.cfa.harvard.edu/viz-bin/VizieR-3?-source=J/A%2bA/529/A75/table-af
        if band=='V':
            import pickle
            dat=pickle.load(open('tables/tableab_dat_V.pickle','rb'))
        else:
            print("Loading LD table (may be slow...)")
            dat=ascii.read('tables/tableab.dat.gz').to_pandas()
            dat=dat.rename(columns={'col1':'logg','col2':'Teff','col3':'FeH','col4':'xi',
                                    'col5':'a','col6':'b','col7':'band','col8':'method','col9':'model'})
        #Removing everything except what is taken:
        dat=dat.loc[(dat['xi']==xi)&(dat['method']==method)&(dat['model']==model)&(dat['band']==band)]
        FeHarr=np.unique(dat['FeH'])
        FeH=FeHarr[np.argmin(abs(FeHarr-FeH))]
        Tlen=1 if (type(Ts)==float)+(type(Ts)==int)+(type(Ts)==np.float64) else len(Ts)
        Ts=np.array([Ts]) if Tlen==1 else Ts
        logglen=1 if type(logg)==float or type(logg)==int or type(logg)==np.float64 else len(logg)
        outarr=np.zeros((Tlen,2))
        arr2=dat.loc[dat['FeH']==FeH]
        a_interp=interp.CloughTocher2DInterpolator(np.column_stack((arr2['Teff'].values,arr2['logg'])),arr2['a'].values)
        b_interp=interp.CloughTocher2DInterpolator(np.column_stack((arr2['Teff'].values,arr2['logg'])),arr2['b'].values)
        Ts=np.clip(Ts,np.min(arr2['Teff'].values)+1,np.max(arr2['Teff'].values)-1)
        logg=np.clip(logg,np.min(arr2['logg'].values)+0.1,np.max(arr2['logg'].values)-0.1)
        if logglen>1:
            outarr=np.column_stack((a_interp(Ts,logg),b_interp(Ts,logg)))
        else:
            outarr=np.column_stack(( a_interp(Ts,np.tile(logg,len(Ts))),b_interp(Ts,np.tile(logg,len(Ts))) ))
        return outarr


def find_nearest(arr,value):
    arr = np.array(arr)
    #find nearest value in array
    idx=(np.abs(arr-value)).argmin()
    return idx

def find_nearest_2D(search_in,search_for):
    #A search for each of the closest values in dataset search_in from dataset search_for.
    return search_for[np.argmin(abs(search_in[np.newaxis,:]-search_for[:,np.newaxis]),axis=0)]

def Add_angles(binaries,all_stars,mag='Pmag'):
    #Adding 2D angles to plot to compute centroids

    all_stars.loc[all_stars.type=='target','sep']=0.0

    #Compute angle to target
    all_stars['angle_to_target']=np.tile(np.nan,all_stars.shape[0])
    all_stars['angle_to_parent']=np.random.uniform(0,np.pi*2,len(all_stars))
    
    #latiutde and longitude from parent, separation and random angle:
    all_stars['x_to_target']=np.tile(np.nan,all_stars.shape[0])
    all_stars['y_to_target']=np.tile(np.nan,all_stars.shape[0])

    all_stars.loc[all_stars.type=='target','angle_to_target']=0.0
    all_stars.loc[all_stars.type=='target','x_to_target']=0.0
    all_stars.loc[all_stars.type=='target','y_to_target']=0.0

    #Computing x and y for first order binaries
    firstorders=(all_stars.type=='blend')+(all_stars.type=='target_b')
    all_stars.loc[firstorders,'angle_to_target']=all_stars['angle_to_parent']
        
    all_stars.loc[firstorders,'x_to_target'] = all_stars.loc[firstorders,'sep']*np.sin(all_stars.loc[firstorders,'angle_to_target'])
    all_stars.loc[firstorders,'x_to_parent']=all_stars.loc[firstorders,'x_to_target']

    all_stars.loc[firstorders,'y_to_target'] = all_stars.loc[firstorders,'sep']*np.cos(all_stars.loc[firstorders,'angle_to_target'])
    all_stars.loc[firstorders,'y_to_parent']=all_stars.loc[firstorders,'y_to_target']
    
    all_stars.loc[firstorders,'latitude']=all_stars.loc[firstorders,'latitude'].values+\
                                          all_stars.loc[firstorders,'x_to_target'].values/3600
    all_stars.loc[firstorders,'longitude']=all_stars.loc[firstorders,'longitude'].values+\
                                           all_stars.loc[firstorders,'y_to_target'].values/3600


    #Taking secondary binaries to be co=incident with their parents:
    all_second_ords=(all_stars.type!='blend')*(all_stars.type!='target_b')*(all_stars.type!='target')
    ix_parents=all_stars.loc[all_second_ords]['orb_parent'].values
    all_second_ords_parent_vals=all_stars.loc[ix_parents]
    all_stars.loc[all_second_ords,'angle_to_target']=all_second_ords_parent_vals['angle_to_target'].values

    all_stars.loc[all_second_ords,'y_to_target']=all_second_ords_parent_vals['y_to_target'].values
    all_stars.loc[all_second_ords,'x_to_target']=all_second_ords_parent_vals['x_to_target'].values

    all_stars.loc[all_second_ords,'latitude']=all_stars.loc[all_second_ords,'latitude'].values+\
                                              all_second_ords_parent_vals['x_to_target'].values/3600
    all_stars.loc[all_second_ords,'longitude']=all_stars.loc[all_second_ords,'longitude'].values+\
                                               all_second_ords_parent_vals['y_to_target'].values/3600

    #Initialising distance to target
    all_stars['x_to_CoL']=np.zeros(all_stars.shape[0])
    all_stars['y_to_CoL']=np.zeros(all_stars.shape[0])

    all_stars['flux']=np.power(2.5,12-all_stars[mag])
    all_stars['flux_in_ap']=all_stars['flux']*all_stars['frac_blend_in_aperture']
    all_stars['total_flux_in_ap']=np.zeros(all_stars.shape[0])
    all_stars['prop_of_flux_in_ap']=np.zeros(all_stars.shape[0])

    #Looping targets
    for itarg in pd.unique(all_stars.blend_parent.values):
        targs=all_stars.blend_parent==itarg
        fluxtot=np.sum(all_stars.loc[targs]['flux_in_ap'])

        #Calculate position of centre of light -weighted by flux
        CoL_x=np.sum(all_stars.loc[targs]['x_to_target'])/np.sum(all_stars.loc[targs]['flux'])
        CoL_y=np.sum(all_stars.loc[targs]['y_to_target'])/np.sum(all_stars.loc[targs]['flux'])
        #Calculate new seperations from centre of light
        all_stars.loc[targs,'x_to_CoL']=CoL_x+all_stars.loc[targs]['x_to_target']
        all_stars.loc[targs,'y_to_CoL']=CoL_y+all_stars.loc[targs]['y_to_target']
        all_stars.loc[targs,'total_flux_in_ap']=fluxtot
        all_stars.loc[targs,'prop_of_flux_in_ap']=all_stars.loc[targs]['flux_in_ap']/fluxtot

    #Spot filling factor, adapted from LAMOST II (more spots on cool stars)
    all_stars['alphaS']=np.clip(np.random.normal(0.2-(np.clip(all_stars.loc[:,'Teff'].values,2500,12000)-2500)/12000,1.0-(np.clip(all_stars.loc[:,'Teff'].values,2500,12000)/12000)**0.2),0.0,0.66)

    return all_stars,binaries

def IsochronesMagic_Simple(dfstars,mag='Pmag'):
    #mist = MIST_Isochrone()
    dart= Dartmouth_Isochrone()
    dfstars['Ms']=np.clip(dfstars['Ms'],0.101,3.58)
    
    maxagebound=np.array([[-1.0,  1.01590604e+01],
                           [-9.46375839e-01,  1.01731544e+01],
                           [-8.93355705e-01,  1.01731544e+01],
                           [-8.40335570e-01,  1.01731544e+01],
                           [-7.87315436e-01,  1.01731544e+01],
                           [-7.34295302e-01,  1.01731544e+01],
                           [-6.81275168e-01,  1.01731544e+01],
                           [-6.28255034e-01,  1.01731544e+01],
                           [-5.75234899e-01,  1.01731544e+01],
                           [-5.22214765e-01,  1.01731544e+01],
                           [-4.69194631e-01,  1.01731544e+01],
                           [-4.16174497e-01,  1.01731544e+01],
                           [-3.63154362e-01,  1.01731544e+01],
                           [-3.10134228e-01,  1.01731544e+01],
                           [-2.57114094e-01,  1.01731544e+01],
                           [-2.04093960e-01,  1.01731544e+01],
                           [-1.51073826e-01,  1.01731544e+01],
                           [-9.80536913e-02,  1.01731544e+01],
                           [-4.50335570e-02,  1.01731544e+01],
                           [ 7.98657718e-03,  1.01308725e+01],
                           [ 6.10067114e-02,  1.00463087e+01],
                           [ 1.14026846e-01,  9.94765101e+00],
                           [ 1.67046980e-01,  9.83489933e+00],
                           [ 2.20067114e-01,  9.70805369e+00],
                           [ 2.73087248e-01,  9.58120805e+00],
                           [ 3.26107383e-01,  9.41208054e+00],
                           [ 3.79127517e-01,  9.24295302e+00],
                           [ 4.32147651e-01,  9.04563758e+00],
                           [ 4.85167785e-01,  8.82013423e+00],
                           [ 5.38187919e-01,  8.55234899e+00],[0.56, 8.43959732]])
    minagebound=np.array([[-1.0,  9.10201342e+00],
                           [-9.46375839e-01,  8.96107383e+00],
                           [-8.93355705e-01,  8.89060403e+00],
                           [-8.40335570e-01,  8.83422819e+00],
                           [-7.87315436e-01,  8.79194631e+00],
                           [-7.34295302e-01,  8.74966443e+00],
                           [-6.81275168e-01,  8.70738255e+00],
                           [-6.28255034e-01,  8.66510067e+00],
                           [-5.75234899e-01,  8.60872483e+00],
                           [-5.22214765e-01,  8.53825503e+00],
                           [-4.69194631e-01,  8.46778523e+00],
                           [-4.16174497e-01,  8.41140940e+00],
                           [-3.63154362e-01,  8.41140940e+00],
                           [-3.10134228e-01,  8.41140940e+00],
                           [-2.57114094e-01,  8.41140940e+00],
                           [-2.04093960e-01,  8.41140940e+00],
                           [-1.51073826e-01,  8.41140940e+00],
                           [-9.80536913e-02,  8.41140940e+00],
                           [-4.50335570e-02,  8.41140940e+00],
                           [ 7.98657718e-03,  8.41140940e+00],
                           [ 6.10067114e-02,  8.41140940e+00],
                           [ 1.14026846e-01,  8.41140940e+00],
                           [ 1.67046980e-01,  8.41140940e+00],
                           [ 2.20067114e-01,  8.41140940e+00],
                           [ 2.73087248e-01,  8.41140940e+00],
                           [ 3.26107383e-01,  8.41140940e+00],
                           [ 3.79127517e-01,  8.41140940e+00],
                           [ 4.32147651e-01,  8.41140940e+00],
                           [ 4.85167785e-01,  8.41140940e+00],
                           [ 5.38187919e-01,  8.41140940e+00],[0.56, 8.4114094 ]])
    from scipy import interpolate as interp
    minage=interp.interp1d(minagebound[:,0],minagebound[:,1])
    maxage=interp.interp1d(maxagebound[:,0],maxagebound[:,1])
    #print(minage(np.log10(np.array([0.4,0.8,0.99,1.3,2.6]))))
    #print(maxage(np.log10(np.array([0.4,0.8,0.99,1.3,2.6]))))
    #print(np.sum(dfstars['logage'].values<minage(np.log10(dfstars['Ms'].values))),'ages too small and',
    #      np.sum(dfstars['logage'].values>maxage(np.log10(dfstars['Ms'].values))),'ages too big')
    dfstars.loc[(np.isnan(dfstars['logage'].values))&(~np.isnan(dfstars['Ms'].values)),'logage']=0.5*(minage(np.log10(dfstars.loc[(np.isnan(dfstars['logage'].values))&(~np.isnan(dfstars['Ms'].values)),'Ms']))+maxage(np.log10(dfstars.loc[(np.isnan(dfstars['logage'].values))&(~np.isnan(dfstars['Ms'].values)),'Ms'])))
    
    dfnans=np.isnan(np.sum(dfstars[['Ms','logage','FeH','dist']].values,axis=1))

    dfstars.loc[(~dfnans)&(dfstars['logage'].values<minage(np.log10(dfstars['Ms'].values))),'logage']=minage(np.log10(dfstars.loc[(~dfnans)&(dfstars['logage'].values<minage(np.log10(dfstars['Ms'].values))),'Ms'].values))+0.15
    dfstars.loc[(~dfnans)&(dfstars['logage'].values>maxage(np.log10(dfstars['Ms'].values))),'logage']=maxage(np.log10(dfstars.loc[(~dfnans)&(dfstars['logage'].values>maxage(np.log10(dfstars['Ms'].values))),'Ms'].values))-0.15    
    #print(np.shape(dfnans),np.shape(dfstars[['Ms','logage','FeH','dist']].as_matrix()))
    df=dart.__call__(np.where(dfnans,0.0,dfstars['Ms'].values),
                     age=np.where(dfnans,0.0,dfstars['logage'].values),
                     feh=np.where(dfnans,0.0,dfstars['FeH'].values),
                     distance=np.where(dfnans,0.0,dfstars['dist'].values))
    df=df.set_index(dfstars.index)
    df=df.rename(columns={col:col.replace('_mag','') for col in df.columns if '_mag' in col})
    if mag=='Pmag':
        df['Pmag']=getPmag(df['V'].values,df['Teff'].values)
    df=df.rename(columns={'radius':'Rs','mass':'Ms','age':'logage'})
    print('Input nans to isochrones',np.sum(dfnans),' output nans:',np.sum(np.isnan(np.sum(df.loc[:,['Rs','logage','logg','Ms']]))))
    return df

def MaxStableSMA(M1,M2,A12):
    #Hill Sphere of secondary:
    return A12*((M2/(3*M1))**(0.3333))

def MinStableSMA(M1,M2,R1,factor=1.5):
    if type(M1)==float or type(M1)==int:
        return np.max([(factor*R1*0.00465)/(0.46*((M1/(M1+M2))**(1/3.))),R1*0.00465])
    else:
        return np.max([(factor*R1*0.00465)/(0.46*((M1/(M1+M2))**(1/3.))),R1*0.00465],axis=0)

def kepp2a(P,M1,M2=None):  #Convert Period to semi major axis. Inputs required: Period and Stellar Mass
    #Assumes Ms>>Mp. OPTION: Add Mp and calculate exactly
    G=6.67e-11
    M1=M1*Msun if np.nanmedian(M1)<400. else M1
    M2=M2*Msun if np.nanmedian(M2)<400. else M2
    P=P*86400 if np.nanmedian(P)<24000 else P
    a = ((G*(M1+M2)*P**2)/(4*(np.pi**2)))**(1./3.) if M2 is None else ((G*M1*P**2.)/(4.*(np.pi**2.)))**(1./3.)
    return a/au

def kepp2a_pl(P,M1):  #Convert Period to semi major axis. Inputs required: Period and Stellar Mass
    #Assumes Ms>>Mp. OPTION: Add Mp and calculate exactly
    G=6.67e-11
    M1=M1*Msun if np.nanmedian(M1)<400. else M1
    P=P*86400 if np.nanmedian(P)<24000 else P
    a = ((G*M1*P**2.)/(4.*(np.pi**2.)))**(1./3.)
    return a/au

def kepa2p(a,M1,M2=0):  #Convert Period to semi major axis. Inputs required: Period and Stellar Mass
    #Assumes Ms>>Mp. OPTION: Add Mp and calculate exactly
    G=6.67e-11
    M1=M1*Msun if np.nanmedian(M1)<400. else M1
    M2=M2*Msun if np.nanmedian(M2)<400. else M2
    a=a*au if np.nanmedian(a)<24000 else a
    return np.sqrt((4*(np.pi**2)*a**3)/(G*(M1+M2)))/86400.0



##################################################################
#                                                                #
#                       BINARY POPULATIONS                       #
#                                                                #
##################################################################

def sort_systems(systems,stars,maxau=100):
    # system has A,B,P,T0,ecc,omega,incl
    metaparent=systems.iloc[0]['A']
    systems=systems.sort_values('P')

    initlenstars=len(stars)
    #In binary, simple. As-is.
    #Modifying which stars are orbited in the triple+ case.
    if systems.shape[0]==2:
        #In triple, Shortest P randomly chooses
        # 3 stars...   Par - 0  ----  1   OR   Par  ----  1 - 0
        if np.random.random()>0.5:
            orb_sec=True
            stars=stars.rename({systems.iloc[0]['B']:metaparent+'_B_B',
                                systems.iloc[1]['B']:metaparent+'_B'},axis='index')
            systems.loc[systems.index.values[0],'B']=metaparent+'_B'
            systems.loc[systems.index.values[0],'A']=metaparent+'_B'
            systems.loc[systems.index.values[1],'B']=metaparent+'_B'

            if stars.loc[metaparent+'_B_B','Ms']>stars.loc[metaparent+'_B','Ms']:
                stars=stars.rename({metaparent+'_B_B':metaparent+'_Bbigger',
                                    metaparent+'_B':metaparent+'_Bsmaller'},axis='index')
                stars=stars.rename({metaparent+'_Bbigger':metaparent+'_B',
                                    metaparent+'_Bsmaller':metaparent+'_B_B'},axis='index')

            stars.loc[metaparent+'_B_B','orb_parent']=metaparent+'_B'
            stars.loc[metaparent+'_B_B','Ms_parent']=stars.loc[metaparent+'_B','Ms']
            stars.loc[metaparent+'_B_B','Rs_parent']=stars.loc[metaparent+'_B','Rs']

        else:
            orb_sec=False
            stars=stars.rename({systems.iloc[0]['B']:metaparent+'_B',
                                systems.iloc[1]['B']:metaparent+'_C'},axis='index')
            systems.loc[systems.index.values[0],'B']=metaparent+'_B'
            systems.loc[systems.index.values[1],'B']=metaparent+'_C'

    elif systems.shape[0]>=3:
        #In quad and quint, place the second-shortest per around longest P, rest around target
        if systems.shape[0]==3:
            # 4 stars...   Par - 0  ----  2 - 1
            stars=stars.rename({systems.iloc[0]['B']:metaparent+'_B',
                                systems.iloc[1]['B']:metaparent+'_C_B',
                                systems.iloc[2]['B']:metaparent+'_C'},axis='index')
            systems.loc[systems.index.values[0],'B']=metaparent+'_B'
            systems.loc[systems.index.values[1],'B']=metaparent+'_C_B'
            systems.loc[systems.index.values[1],'A']=metaparent+'_C'
            systems.loc[systems.index.values[2],'B']=metaparent+'_C'

            if stars.loc[metaparent+'_C_B','Ms']>stars.loc[metaparent+'_C','Ms']:
                #modifying binary such that biggest star is primary by switching the name from massive star to smaller one
                stars=stars.rename({metaparent+'_C_B':metaparent+'_Cbigger',
                                    metaparent+'_C':metaparent+'_Csmaller'},axis='index')
                stars=stars.rename({metaparent+'_Cbigger':metaparent+'_C',
                                    metaparent+'_Csmaller':metaparent+'_C_B'},axis='index')

            stars.loc[metaparent+'_C_B','orb_parent']=metaparent+'_C'
            stars.loc[metaparent+'_C_B','Ms_parent']=stars.loc[metaparent+'_C','Ms']
            stars.loc[metaparent+'_C_B','Rs_parent']=stars.loc[metaparent+'_C','Rs']

        elif systems.shape[0]==4:
            # 5 stars...   Par 0 2  ----  3 1 ----
            stars=stars.rename({systems.iloc[0]['B']:metaparent+'_B',
                                systems.iloc[1]['B']:metaparent+'_D_B',
                                systems.iloc[2]['B']:metaparent+'_C',
                                systems.iloc[3]['B']:metaparent+'_D'},axis='index')
            systems.loc[systems.index.values[0],'B']=metaparent+'_B'
            systems.loc[systems.index.values[1],'A']=metaparent+'_D'
            systems.loc[systems.index.values[1],'B']=metaparent+'_D_B'
            systems.loc[systems.index.values[2],'B']=metaparent+'_C'
            systems.loc[systems.index.values[3],'B']=metaparent+'_D'

            if stars.loc[metaparent+'_D_B','Ms']>stars.loc[metaparent+'_D','Ms']:
                #modifying binary such that biggest star is primary by switching the name from massive star to smaller one
                stars=stars.rename({metaparent+'_D_B':metaparent+'_Dbigger',
                                    metaparent+'_D':metaparent+'_Dsmaller'},axis='index')
                stars=stars.rename({metaparent+'_Dbigger':metaparent+'_D',
                                    metaparent+'_Dsmaller':metaparent+'_D_B'},axis='index')

            stars.loc[metaparent+'_D_B','orb_parent']=metaparent+'_D'
            stars.loc[metaparent+'_D_B','Ms_parent']=stars.loc[metaparent+'_D','Ms']
            stars.loc[metaparent+'_D_B','Rs_parent']=stars.loc[metaparent+'_D','Rs']

        elif systems.shape[0]==5:
            # 6 stars...   Par 1  ----  3 0  ----  4 2 ----
            stars=stars.rename({systems.iloc[0]['B']:metaparent+'_C_B',
                                systems.iloc[1]['B']:metaparent+'_B',
                                systems.iloc[2]['B']:metaparent+'_D_B',
                                systems.iloc[3]['B']:metaparent+'_C',
                                systems.iloc[4]['B']:metaparent+'_D'},axis='index')
            systems.loc[systems.index.values[0],'B']=metaparent+'_C_B'
            systems.loc[systems.index.values[0],'A']=metaparent+'_C'
            systems.loc[systems.index.values[1],'B']=metaparent+'_B'
            systems.loc[systems.index.values[2],'B']=metaparent+'_D_B'
            systems.loc[systems.index.values[2],'A']=metaparent+'_D'
            systems.loc[systems.index.values[3],'B']=metaparent+'_C'
            systems.loc[systems.index.values[4],'B']=metaparent+'_D'

            if stars.loc[metaparent+'_C_B','Ms']>stars.loc[metaparent+'_C','Ms']:
                #modifying binary such that biggest star is primary by switching the name from massive star to smaller one
                stars=stars.rename({metaparent+'_C_B':metaparent+'_Cbigger',
                                    metaparent+'_C':metaparent+'_Csmaller'},axis='index')
                stars=stars.rename({metaparent+'_Cbigger':metaparent+'_C',
                                    metaparent+'_Csmaller':metaparent+'_C_B'},axis='index')

            stars.loc[metaparent+'_C_B','orb_parent']=metaparent+'_C'
            stars.loc[metaparent+'_C_B','Ms_parent']=stars.loc[metaparent+'_C','Ms']
            stars.loc[metaparent+'_C_B','Rs_parent']=stars.loc[metaparent+'_C','Rs']

            if stars.loc[metaparent+'_D_B','Ms']>stars.loc[metaparent+'_D','Ms']:
                #modifying binary such that biggest star is primary by switching the name from massive star to smaller one
                stars=stars.rename({metaparent+'_D_B':metaparent+'_Dbigger',
                                    metaparent+'_D':metaparent+'_Dsmaller'},axis='index')
                stars=stars.rename({metaparent+'_Dbigger':metaparent+'_D',
                                    metaparent+'_Dsmaller':metaparent+'_D_B'},axis='index')

            stars.loc[metaparent+'_D_B','orb_parent']=metaparent+'_D'
            stars.loc[metaparent+'_D_B','Ms_parent']=stars.loc[metaparent+'_D','Ms']
            stars.loc[metaparent+'_D_B','Rs_parent']=stars.loc[metaparent+'_D','Rs']

    #stars.iloc[np.digitize(newstars.parent.values,stars.index.values),'Ms']

    systems['sma']=kepp2a(systems['P'].values,stars.loc[systems.A.values]['Ms'].values,stars.loc[systems.B.values]['Ms'].values)
    
    #MinStableSMA from maximum of hill-sphere of secondary, or the sum of the primary and secondary radii
    systems['min_sma']=np.nanmax(np.column_stack((MinStableSMA(stars.loc[systems.A.values]['Ms'].values,
                                                            stars.loc[systems.B.values]['Ms'].values,
                                                            stars.loc[systems.A.values]['Rs'].values),
                                               1.25*(Rsun/au)*(stars.loc[systems.A.values]['Rs'].values+\
                                                               stars.loc[systems.B.values]['Rs'].values)
                                              ))
                              ,axis=1)
                               
    systems['max_sma']=np.tile(maxau,systems.shape[0])
    systems['hillsph']=MaxStableSMA(stars.loc[systems['A'].values]['Ms'].values,stars.loc[systems['B'].values]['Ms'].values,systems['sma'].values)


    #Looping through each binary and modifying the *max* SMA if they are unphysical (using hill sphere). This will have the effect of constricting systems, so may artificially boost eclipses, but not too much I hope.

    if systems.shape[0]==2 and orb_sec:
        #triple, orbit around secondary
        #Min sma of B_B is hill sp of B_B
        systems.loc[systems.B==metaparent+'_B_B','max_sma']=systems.loc[systems.B==metaparent+'_B_B','hillsph'].values
    elif systems.shape[0]==2 and not orb_sec:
        #Max sma of B is sma_C - hill sp of C
        systems.loc[systems.B==metaparent+'_B','max_sma']=systems.loc[systems.B==metaparent+'_C','sma'].values-systems.loc[systems.B==metaparent+'_C','hillsph'].values

    elif systems.shape[0]==3:
        # 4 stars...   Par - 0  ----  2 - 1
        #max_sma of B is sma of C - hillsp CB
        systems.loc[systems.B==metaparent+'_B','max_sma']=systems.loc[systems.B==metaparent+'_C','sma'].values-systems.loc[systems.B==metaparent+'_C_B','hillsph'].values
        #max_sma of C_B is sma of C - hillsp of AB
        systems.loc[systems.B==metaparent+'_C_B','max_sma']=systems.loc[systems.B==metaparent+'_C','sma'].values-systems.loc[systems.B==metaparent+'_B','hillsph'].values

    elif systems.shape[0]==4:
        # 5 stars...   Par 0 2  ----  3 1 ----
        #max_sma of B is sma of C - hillsp C
        systems.loc[systems.B==metaparent+'_B','max_sma']=systems.loc[systems.B==metaparent+'_C','sma'].values-systems.loc[systems.B==metaparent+'_C','hillsph'].values
        #max_sma of C is sma of D - hillsp of D_B
        systems.loc[systems.B==metaparent+'_C','max_sma']=systems.loc[systems.B==metaparent+'_D','sma'].values-systems.loc[systems.B==metaparent+'_D_B','hillsph'].values
        #max_sma of D_B is sma of D - (hillsp C)+(hillsp B)
        systems.loc[systems.B==metaparent+'_D_B','max_sma']=systems.loc[systems.B==metaparent+'_D','sma'].values-systems.loc[systems.B==metaparent+'_C','hillsph'].values-systems.loc[systems.B==metaparent+'_B','hillsph'].values

    elif systems.shape[0]==5:
        # 6 stars...   Par 1  ----  3 0  ----  4 2 ----

        systems.loc[systems.B==metaparent+'_B','max_sma']=systems.loc[systems.B==metaparent+'_C','sma'].values-systems.loc[systems.B==metaparent+'_C_B','hillsph'].values
        #max_sma of C is sma of D - hillsp of D_B
        systems.loc[systems.B==metaparent+'_C','max_sma']=systems.loc[systems.B==metaparent+'_D','sma'].values-systems.loc[systems.B==metaparent+'_D_B','hillsph'].values
        systems.loc[systems.B==metaparent+'_C_B','max_sma']=systems.loc[systems.B==metaparent+'_D','sma'].values-systems.loc[systems.B==metaparent+'_D_B','hillsph'].values
        #max_sma of D_B is sma of D - (hillsp C)+(hillsp B)
        systems.loc[systems.B==metaparent+'_D_B','max_sma']=systems.loc[systems.B==metaparent+'_D','sma'].values-systems.loc[systems.B==metaparent+'_C','hillsph'].values-systems.loc[systems.B==metaparent+'_B','hillsph'].values

    else:
        systems.loc[:,'max_sma']=maxau

    #Finding totally unstable systems. These will be dropped at the end.
    unstable=(systems.min_sma>systems.max_sma)+pd.isnull(systems['min_sma'])+pd.isnull(systems['max_sma'])
    lenunstables=np.sum(unstable)
    
    #Re-sampling currently unstable systems uniformly between maximum and minimum:
    resampl=((systems.sma>systems.max_sma)|(systems.sma<systems.min_sma))*(~unstable)
    if np.sum(resampl)>0:
        systems.loc[resampl,'sma']=np.exp(np.random.uniform(np.log(systems.loc[resampl,'min_sma'].values),
                                                            np.log(systems.loc[resampl,'max_sma'].values)))
        systems.loc[resampl,'P']=kepa2p(systems.loc[resampl,'sma'].values,
                                        stars.loc[systems.loc[resampl,'A'].values]['Ms'].values,
                                        stars.loc[systems.loc[resampl,'B'].values]['Ms'].values)
        #Multiple systems in need of resampling might a) take high-e values from long-P rendering their close proximity to other stars unstable still... and b) are more likely to circularise. So let's set e=0
        systems.loc[resampl,'ecc']=0.0
        
    #When we do a planet-addition step, we need the stars to have min and max sma according to their nearest companions...
    systems['pl_hillsph_A']=MaxStableSMA(stars.loc[systems['A'].values]['Ms'].values,np.tile(0.001,systems.shape[0]),systems['sma'].values)
    systems['pl_hillsph_B']=MaxStableSMA(stars.loc[systems['B'].values]['Ms'].values,np.tile(0.001,systems.shape[0]),systems['sma'].values)

    if systems.shape[0]==1:
        #Binary:
        stars.loc[metaparent,'max_sma']=systems.sma.values[0]-systems.pl_hillsph_B.values[0]
        stars.loc[metaparent+'_B','max_sma']=systems.sma.values[0]-systems.pl_hillsph_A.values[0]

    elif systems.shape[0]==2 and orb_sec:
        stars.loc[metaparent,'max_sma']=systems.iloc[1]['sma']-systems.iloc[0]['sma']-systems.iloc[0]['pl_hillsph_B']
        stars.loc[metaparent+'_B','max_sma']=systems.iloc[0]['sma']-systems.iloc[0]['pl_hillsph_B']
        stars.loc[metaparent+'_B_B','max_sma']=systems.iloc[0]['pl_hillsph_B']

    elif systems.shape[0]==2 and not orb_sec:
        stars.loc[metaparent,'max_sma']=systems.iloc[0]['sma']-systems.iloc[0]['pl_hillsph_B']
        stars.loc[metaparent+'_B','max_sma']=np.min([systems.iloc[0]['pl_hillsph_B'],systems.iloc[0]['sma']-systems.iloc[1]['sma']-systems.iloc[0]['pl_hillsph_B']])
        stars.loc[metaparent+'_C','max_sma']=np.min([systems.iloc[1]['pl_hillsph_B'],systems.iloc[1]['sma']-systems.iloc[0]['sma']-systems.iloc[0]['pl_hillsph_B']])

    elif systems.shape[0]==3:
        # 4 stars...   Par - 0  ----  2 - 1
        stars.loc[metaparent,'max_sma']=systems.iloc[0]['sma']-systems.iloc[0]['pl_hillsph_B']
        stars.loc[metaparent+'_B','max_sma']=np.min([systems.iloc[0]['pl_hillsph_B'],systems.iloc[2]['sma']-systems.iloc[1]['sma']-systems.iloc[0]['sma']-systems.iloc[1]['pl_hillsph_B']])
        stars.loc[metaparent+'_C','max_sma']=systems.iloc[1]['sma']-systems.iloc[1]['pl_hillsph_B']
        stars.loc[metaparent+'_C_B','max_sma']=np.min([systems.iloc[1]['pl_hillsph_B'],systems.iloc[2]['sma']-systems.iloc[1]['sma']-systems.iloc[0]['sma']-systems.iloc[0]['pl_hillsph_B']])

    elif systems.shape[0]==4:
        # 5 stars...   Par 0 2  ----  3 1 ----
        #max_sma of B is sma of C - hillsp C
        stars.loc[metaparent,'max_sma']=systems.iloc[0]['sma']-systems.iloc[0]['pl_hillsph_B']
        stars.loc[metaparent+'_B','max_sma']=np.min([systems.iloc[0]['pl_hillsph_B'],systems.iloc[2]['sma']-systems.iloc[0]['sma']-systems.iloc[2]['pl_hillsph_B']])
        stars.loc[metaparent+'_C','max_sma']=np.min([systems.iloc[2]['pl_hillsph_B'],systems.iloc[3]['sma']-systems.iloc[2]['sma']-systems.iloc[1]['sma']-systems.iloc[1]['pl_hillsph_B']])
        stars.loc[metaparent+'_D','max_sma']=np.min([systems.iloc[3]['pl_hillsph_B'],systems.iloc[1]['sma']-systems.iloc[1]['pl_hillsph_B']])
        stars.loc[metaparent+'_D_B','max_sma']=np.min([systems.iloc[1]['pl_hillsph_B'],systems.iloc[3]['sma']-systems.iloc[2]['sma']-systems.iloc[2]['pl_hillsph_B']])

    elif systems.shape[0]==5:
        # 6 stars...   Par 1  ----  3 0  ----  4 2 ----

        stars.loc[metaparent,'max_sma']=systems.iloc[1]['sma']-systems.iloc[1]['pl_hillsph_B']
        stars.loc[metaparent+'_B','max_sma']=np.min([systems.iloc[1]['pl_hillsph_B'],systems.iloc[3]['sma']-systems.iloc[0]['sma']-systems.iloc[0]['pl_hillsph_B']])
        stars.loc[metaparent+'_C','max_sma']=np.min([systems.iloc[3]['pl_hillsph_B'],systems.iloc[0]['sma']-systems.iloc[0]['pl_hillsph_B']])
        stars.loc[metaparent+'_C_B','max_sma']=np.min([systems.iloc[0]['pl_hillsph_B'],systems.iloc[3]['sma']-systems.iloc[1]['sma']-systems.iloc[1]['pl_hillsph_B'],systems.iloc[4]['sma']-systems.iloc[3]['sma']-systems.iloc[2]['sma']-systems.iloc[0]['sma']-systems.iloc[2]['pl_hillsph_B']])
        stars.loc[metaparent+'_D','max_sma']=systems.iloc[2]['sma']-systems.iloc[2]['pl_hillsph_B']
        stars.loc[metaparent+'_D_B','max_sma']=np.min([systems.iloc[2]['pl_hillsph_B'],systems.iloc[4]['sma']-systems.iloc[3]['sma']-systems.iloc[0]['sma']-systems.iloc[0]['pl_hillsph_B']])

    if np.sum(unstable)>0:
        print("unstable:",systems.loc[unstable].index)
        for iunstable in systems.loc[unstable].iterrows():
            if iunstable[1]['B'] in systems.loc[~unstable,'A'].values:
                #unstable body has another star around it... Need to remove this too
                lenunstables+=1
                stars=stars.drop(systems.loc[systems.A==iunstable[1]['B'],'B'].values)
                systems=systems.drop(systems.loc[systems.A==iunstable[1]['B']].index.values)
        #Dropping unstable B stars:
        stars=stars.drop(systems.loc[unstable].B)
        systems=systems.drop(systems.loc[unstable].index)

    assert(len(stars)==(initlenstars-lenunstables))
    return systems, stars


def get_multis_new(stars,mag='Pmag',maxau=100):
    # Generates Multiple systems with (mostly) only Numpy
    # - stars = pandas array of input stars like the Besancon outputs.
    # - systems = pandas array of already generated multiple systems
    # - plimit_newebs = period limit for all systems at which to stop generating new ebs:
    # - p_upper_limit = period limit for each system which to not search for new ebs:
    # - mission = Which space mission are we talking about
    # - mag = Which filter band are we calculating for.

    # Determines multiplicity given an input star list
    # level determines whether binary companions are also split according to multiplicity
    stars['multiplicity']=np.polyval([4.99093097e-05,1.67277394e-01],stars.Teff.values)
    stars.loc[stars.Teff.values>=10000,'multiplicity']=np.polyval([2.42927023e-06,6.69996034e-01],stars.loc[stars.Teff.values>=10000,'Teff'].values)
    multis=np.column_stack([1.0-stars['multiplicity'],stars['multiplicity']*0.56,stars['multiplicity']*0.33,stars['multiplicity']*0.08,stars['multiplicity']*0.03])
    multis=np.cumsum(multis,axis=1)
    multi_rands=np.random.random(len(multis))

    systems=pd.DataFrame()
    nbinaries=0
    n_stars_init=stars.shape[0]
    for niter in range(4):
        newstars=pd.DataFrame() #stellar info, etc.
        newsystem=pd.DataFrame()
        #Randomly selecting those stars to insert companions (ignoring any with a short-P eb already)
        put_eb=multi_rands>multis[:,niter]
        #Adding false for those added stars appended to the df
        put_eb=np.hstack((put_eb,np.tile(False,len(stars)-len(put_eb))))

        sameys=['FeH','dist','Av','longitude','latitude','Age','blend_parent','frac_blend_in_aperture','Nscopes']

        newstars=stars.iloc[put_eb][sameys]

        newstars['Ms']=np.random.uniform(0.075,1.0)*stars.loc[put_eb,'Ms']
        newstars['Ms_parent']=stars.iloc[put_eb]['Ms'].values
        newstars['Rs_parent']=stars.iloc[put_eb]['Rs'].values
        newstars['Pmag_parent']=stars.iloc[put_eb]['Pmag'].values


        #Copying necessary info from parent
        newstars['orb_parent']=stars.loc[put_eb].index.values
        #newstars['blend_parent']=stars.loc[put_eb]

        #Making sure it is not in the planetary regime OR below the isochrones limit of 0.09Ms (95Mjup)
        newstars.loc[newstars['Ms']<0.1,'Ms']=np.random.uniform(np.tile(0.1,np.sum(newstars['Ms']<0.1)),newstars.loc[newstars['Ms']<0.1,'Ms_parent'])

        suff=['B','C','D','E','F'][niter]
        newstars=newstars.set_index(np.array([ind+'_'+suff for ind in stars.loc[put_eb].index.values]))

        newstars['type']=np.core.defchararray.add(stars.loc[put_eb,'type'].values.astype(str),np.tile('_b',np.sum(put_eb)))

        newsystem['A']=stars.iloc[put_eb].index.values
        newsystem['B']=newstars.index.values
        newsystem['P']=np.exp(np.random.normal(5.03,2.28,newstars.shape[0]))#from Raghavan
        newsystem['T0']=np.random.uniform(np.tile(0,newsystem.shape[0]),newsystem['P'])
        #Technically eccentricity could make these things unstable again. Oh well...
        newsystem['ecc']=stats.beta.cdf(np.random.random(newstars.shape[0]),
                                        a=np.tile(0.9,newstars.shape[0]),
                                        b=0.75*stats.norm.cdf(np.log10(newsystem['P']),2.0,0.85)
                                       )#est from Raghavan
        #Taking high-ecc systems at P<10 and setting to zero
        newsystem.loc[newsystem.ecc>(0.5*np.log10(newsystem.P)+0.5),'ecc']=0.0
        newsystem['omega']=np.random.random(newstars.shape[0])*2*np.pi
        newsystem['incl']=np.arccos(np.random.uniform(-1,1,newstars.shape[0]))

        #NEED AGE? ok, it is _lower_ in mass, therefore almost certainly main-sequence. Set age as "half MS lifetime" (from Ms**-2.5) or the age of the universe
        '''
        #For the moment, let's NOT do this as almost all of the input stars should work with Dartmouth isochrones
        #logages=np.log10(newstars['Age'].values
        #logages[np.isnan(logages)]=np.clip(-2.5*np.log10(newstars.loc[np.isnan(logages),'Ms'].values)+10-0.30103,5.25,10.13)
        #newstars['Age']=logages
        '''

        #Getting stellar params from isochrones:
        #print('doing ISOCHRONES magic')
        newstars['logage']=np.log10(newstars['Age'])+9
        newstars.loc[newstars['logage']<0.5]=1.001
        iso_df=IsochronesMagic_Simple(newstars,mag=mag)
        inflatedsystems=(iso_df['Rs']>1.1*newstars['Rs_parent'])+(iso_df['Pmag']<(newstars['Pmag_parent']-0.25))
        #Some problem systems where binary is far far bigger than primary, or far brighter (ie becoming the primary)... 
        #This happens because the Besancon models use a different/less generous isochrones relation. 
        # Fixing for radius by subtracting 1 from the logage (ie logage of 9.7 -> 8.7)
        # And fixing for magnitude by multiplying distance by 1.2
        if np.sum(inflatedsystems)>0:
            newstars['middle_aged']=newstars['logage'].values-1.0
            newstars.loc[newstars['middle_aged']<0.5]=1.001
            newstars['middle_distance']=newstars['dist']*1.2
            iso_df.loc[inflatedsystems]=IsochronesMagic_Simple(newstars.loc[inflatedsystems].drop(columns=['logage','dist']).rename(columns={'middle_aged':'logage','middle_distance':'dist'}),mag=mag)
            
        #We STILL have a problem... Instead let's fix the primary stars to their Isochrones-derived versions
        inflatedsystems=(iso_df['Rs']>1.05*newstars['Rs_parent'])+(iso_df['Pmag']<(newstars['Pmag_parent']-0.25))
        if np.sum(inflatedsystems)>0:
            print("new section I added to save dodgy stars... ",np.sum(inflatedsystems),"original stars to be re-formed with isochrones")
            iso_df_stars=IsochronesMagic_Simple(stars.loc[newstars.loc[inflatedsystems,'orb_parent'].values,['Ms','FeH','logage','dist']],mag=mag)
            for col in [mag,'Rs','Ms','Teff','V','logL','logg']:
                stars.loc[newstars.loc[inflatedsystems,'orb_parent'],col]=iso_df_stars[col].values
            newstars.loc[inflatedsystems,'Ms_parent']=stars.loc[newstars.loc[inflatedsystems,'orb_parent'].values,'Ms'].values
            newstars.loc[inflatedsystems,'Rs_parent']=stars.loc[newstars.loc[inflatedsystems,'orb_parent'].values,'Rs'].values
            if 'middle_aged' not in newstars.columns:
                newstars['middle_aged']=newstars['logage'].values-1.0
                newstars.loc[newstars['middle_aged']<0.5]=1.001
            newstars.loc[inflatedsystems,'logage']=newstars.loc[inflatedsystems,'middle_aged']
            iso_df.loc[inflatedsystems]=IsochronesMagic_Simple(newstars.loc[inflatedsystems],mag=mag)
        #print('pre-ischrones merge',newstars.shape,newstars.columns)
        newstars=pd.concat([newstars,iso_df.drop(['Ms','logage'],axis=1)],axis=1)#how='outer',left_index=True,right_index=True)
        newstars['deltamag']=newstars[mag].values-stars.loc[newstars.blend_parent.values,mag].values
        #print('post-ischrones merge',newstars.shape,newstars.columns)
        nbinaries+=newstars.shape[0]
        stars=stars.append(newstars)
        systems=systems.append(newsystem)

    print("N_companions",nbinaries," Ninit stars", n_stars_init)

    stars['min_sma']=MinStableSMA(stars['Ms'].values,np.tile(0.001,len(stars)),stars['Rs'].values,factor=2)
    stars['max_sma']=np.tile(100,stars.shape[0])

    systems=systems.set_index(np.arange(len(systems)))
    systems['min_sma']=np.zeros(len(systems))
    systems['max_sma']=np.tile(maxau,len(systems))
    #Looping through system parents
    allnewstars=pd.DataFrame()
    newsystems=pd.DataFrame()
    for parent in pd.unique(systems.A.values):
        #Sorting which stars orbit which and weeding out those that are unphysical:
        newsystem,newstars=sort_systems(systems.loc[systems.A==parent],stars.loc[stars.orb_parent==parent],maxau=maxau)
        newsystems=newsystems.append(newsystem)
        #dropping input rows and appending new ones:
        stars=stars.drop(stars.loc[stars.orb_parent==parent].index)
        allnewstars=allnewstars.append(newstars)
    #Some of the max_sma for planets is unstable, so set these to 0.0
    allnewstars.loc[allnewstars.max_sma.values<0,'max_sma']=0.0
    
    #Now calculating observables (don't want to do this in the loop above - numpy is quicker!
    newsystems['perihel']=newsystems['sma']*(1-newsystems['ecc'])
    newsystems['aphel']=newsystems['sma']*(1+newsystems['ecc'])
    newsystems['sep']=newsystems['sma'].values/allnewstars.loc[newsystems.A.values,'dist'].values #separation in arcsec

    newsystems['a_R']=(newsystems['sma'].values*au)/(allnewstars.loc[newsystems.A.values,'Rs'].values*Rsun)
    newsystems['a_R_parent']=(newsystems['sma'].values*au)/(allnewstars.loc[newsystems.B.values,'Rs'].values*Rsun)
    newsystems['hillsph']=MaxStableSMA(newsystems['sma'].values,allnewstars.loc[newsystems.A.values,'Ms'].values,allnewstars.loc[newsystems.B.values,'Ms'].values)

    newsystems['bpri']=abs(b_ecc_pri(newsystems['ecc'].values,newsystems['omega'].values,
                              newsystems['sma'].values,newsystems['incl'].values,
                               allnewstars.loc[newsystems['A'].values,'Rs'].values)) #impact parameter at primary eclipse
    newsystems['Rratio']=allnewstars.loc[newsystems['B'].values,'Rs'].values/allnewstars.loc[newsystems['A'].values,'Rs'].values
    newsystems['bsec']=abs(b_ecc_sec(newsystems['ecc'].values,newsystems['omega'].values,
                                newsystems['sma'].values,newsystems['incl'].values,
                                allnewstars.loc[newsystems['B'].values,'Rs'].values)) #impact parameter at primary eclipse
    newsystems['pri_ecl']=newsystems['bpri']<(1+newsystems['Rratio'])
    newsystems['sec_ecl']=newsystems['bsec']<(1+1.0/newsystems['Rratio'])

    ebtypes=newsystems.loc[(newsystems['pri_ecl'].values)|(newsystems['sec_ecl'].values),'B']

    allnewstars.loc[ebtypes,'type']=np.array([typ[:-2]+'_E' for typ in allnewstars.loc[ebtypes,'type']])

    #Getting stars not processed through the above process and concatenating.
    otherstars=stars.loc[~np.in1d(stars.orb_parent,systems.A.values)]

    print(allnewstars.shape,otherstars.shape,stars.shape,'<should sum together, minus unstable systs')

    allnewstars=otherstars.append(allnewstars)
    #converting max_sma and min_sma to periods
    allnewstars['min_per']=kepa2p(allnewstars['min_sma'].values,allnewstars.Ms.values,np.tile(0.001,len(allnewstars)))
    allnewstars['max_per']=kepa2p(allnewstars['max_sma'].values,allnewstars.Ms.values,np.tile(0.001,len(allnewstars)))

    return allnewstars,newsystems


##################################################################
#                                                                #
#                       PLANET POPULATIONS                       #
#                                                                #
##################################################################

def assemblePet(multfunc=None):
    #Getting occurrence rates and extending a bit:
    petigura=np.genfromtxt('tables/Petigura2017_ext_2.txt') #Getting petigura occurance rates from file
    #Cutting periods>500d
    
    if multfunc is not None:
        #Multifunc is multiplicitive function in P and Rp to boost planet numbers:
        for p in range(1,len(petigura[0,:])-1):
            for r in range(1,len(petigura[:,0])):
                petigura[r,p]*=multfunc(petigura[0,p],petigura[r,0])
    
    petigura=petigura[:,:15]
    petigura[1:,-1]=np.zeros(len(petigura[1:,-1]))
    
    return petigura


def GenPls(allstars,petigura):
    #Choosing positions of planets:
    rands=np.random.random((allstars.shape[0],len(petigura[:,0])-2,len(petigura[0,:])-2))

    #Forcing positions where planets might be unstable to not spawn planets there by making random number 1.0
    np.sum(np.tile(allstars.max_per[:,np.newaxis]<petigura[0,2:],(11,1,1)).swapaxes(1,0))
    np.sum(np.tile(allstars.min_per[:,np.newaxis]>petigura[0,1:-1],(11,1,1)).swapaxes(1,0))
    
    rands[np.tile(allstars.max_per[:,np.newaxis]<petigura[0,2:],(11,1,1)).swapaxes(1,0)]=1.0
    rands[np.tile(allstars.min_per[:,np.newaxis]>petigura[0,1:-1],(11,1,1)).swapaxes(1,0)]=1.0

    whr=np.where(rands<np.tile(petigura[1:-1,1:-1],(allstars.shape[0],1,1)))
    
    tran_pls_i=pd.DataFrame()

    #A is direct parent
    tran_pls_i['parent']=allstars.iloc[whr[0]].index

    #Getting orbital params
    tran_pls_i['Rp']=np.random.uniform(petigura[whr[1]+1,0],petigura[whr[1]+2,0])
    tran_pls_i['Rp']=np.clip(np.random.normal(tran_pls_i['Rp'],np.tile(0.1,tran_pls_i.shape[0])),0.5,20)
    tran_pls_i['P']=np.random.uniform(np.log(petigura[0,whr[2]+1]),np.log(petigura[0,whr[2]+2]))
    tran_pls_i['P']=np.clip(np.exp(np.random.normal(tran_pls_i['P'],np.tile(0.05,tran_pls_i.shape[0]))),0.5,1000)
    #tran_pls_i['P']=np.clip(np.exp(np.random.normal(tran_pls_i['P'],np.tile(0.05,tran_pls_i.shape[0]))),0.5,750)

    tran_pls_i['T0']=np.random.uniform(np.tile(0,tran_pls_i.shape[0]),tran_pls_i['P'])

    #Getting planet mass:
    tran_pls_i['Mp']=PlanetRtoM(tran_pls_i['Rp'])

    #\adding "planet index" - ie position of planet outwards from the star
    tran_pls_i['index_pl']=np.zeros(tran_pls_i.shape[0])
    plnames=np.array(['_b','_c','_d','_e','_f','_g','_h','_i','_j','_k'])
    limit_n_pls=10 #Limit planets to 10 per system
    for unq_id in np.unique(tran_pls_i['parent'].values):
        solarsyst=tran_pls_i.index.values[tran_pls_i['parent']==unq_id]
        if len(solarsyst)>limit_n_pls:
            booltake=np.zeros(len(solarsyst))
            booltake[np.random.choice(len(solarsyst),size=limit_n_pls,replace=False)]=1.0 #choosing 8 from those available
            booltake=booltake.astype(bool)
            tran_pls_i=tran_pls_i.drop(solarsyst[~booltake]) #drop excess planets from full df
            
            solarsyst=solarsyst[booltake] #remove excess planets from per-system list
        tran_pls_i.loc[solarsyst,'index_pl']=tran_pls_i.loc[solarsyst,'P'].argsort()
    #Making index "starname"+"_"+"planet number"
    tran_pls_i=tran_pls_i.set_index(np.core.defchararray.add(tran_pls_i['parent'].values.astype(str),plnames[tran_pls_i['index_pl'].values.astype(int)]).astype(str))

    #Eccentricity, omega and incl:
    kipping= np.random.normal([1.12,3.09,0.697,3.27],[0.1,0.3,0.2,0.34])
    tran_pls_i['ecc']=stats.beta.rvs(kipping[0],kipping[1],size=tran_pls_i.shape[0])
    tran_pls_i.loc[tran_pls_i['P']<=382.5,'ecc']=stats.beta.rvs(kipping[2],kipping[3],size=np.sum(tran_pls_i['P']<=382.5))

    tran_pls_i['omega']=np.random.random(len(tran_pls_i))*2*np.pi
    tran_pls_i['incl']=np.arccos(np.random.uniform(-1,1,len(tran_pls_i)))

    #Making 66% of multiplanet systems coplanar with i=i_b+/-06degrees or +/-0.1rad
    print("len pls: ",tran_pls_i.shape)
    for unqpar in np.unique(tran_pls_i.loc[tran_pls_i['index_pl'].values>=1,'parent']):
        ind=tran_pls_i.loc[tran_pls_i['parent'].values==unqpar].index.values
        if np.random.random()>0.33:
            tran_pls_i.loc[ind,'incl']=np.random.normal(tran_pls_i.loc[ind,'incl'].values[0],0.1,
                                                        len(ind))
    tran_pls_i.loc[:,'incl']=tran_pls_i.loc[:,'incl']%np.pi #Forcing inclinations to 0->pi

    #Merging to give stellar params
    allstars.loc[:,'planetparent']=allstars.index.values
    tran_pls_i['planetparent']=tran_pls_i['parent'].values
    print('tran_pls:',tran_pls_i.shape,'unq star parents:',len(np.unique(tran_pls_i.planetparent.values)),'stars:',allstars.shape,'unq:',len(np.unique(allstars.index.values)),"Npls with parents in allstars:",np.sum(np.in1d(tran_pls_i['planetparent'],allstars['planetparent'])),"Nstars with planets in tran_pls:",np.sum(np.in1d(allstars['planetparent'],tran_pls_i['planetparent'])) )
    #tran_combi_i=pd.concat([tran_pls_i,allstars.iloc[np.in1d(tran_pls_i['planetparent'],allstars['planetparent'])]],axis=1)
    #print(allstars.rename(columns={'parent':'starparent'}))
    #print(allstars.rename(columns={'parent':'starparent'})['planetparent'])
    #print(tran_pls_i)
    #print(tran_pls_i['planetparent'])
    colrenames={'parent':'starparent'}
    colrenames.update({col:'A_'+col for col in ['Rs','Rs_parent','Ms','Ms_parent','LD_1','LD_2',
                                            'GD_1','parent','logg','Teff','FeH']})
    tran_combi_i=pd.merge(tran_pls_i,allstars.rename(columns=colrenames),left_on='planetparent',right_index=True, validate='many_to_one')
    print('tran_pls:',tran_pls_i.shape,'tran_combi_i:',tran_combi_i.shape,'unique tran_pls index:',len(np.unique(tran_combi_i.index.values)),'unique all_stars index:',len(np.unique(allstars.index.values)))
    tran_combi_i.set_index(tran_pls_i.index.values,inplace=True)
    print('tran_combi_i:',tran_combi_i.shape)

    #right, how='inner', on=None, left_on=None, right_on=None, left_index=False, right_index=False,

    print("pls:",np.shape(tran_pls_i),"stars:",np.shape(allstars),"out:",np.shape(tran_combi_i))
    #print(tran_combi_i)

    tran_combi_i['sma']=kepp2a_pl(tran_combi_i['P'].values,tran_combi_i['A_Ms'].values)
    tran_combi_i['minP']=kepa2p(2*tran_combi_i['A_Rs'].values*0.00465233957,tran_combi_i['A_Ms'].values)

    tran_combi_i['bpri']=abs(b_ecc_pri(*tran_combi_i.loc[:,['ecc','omega','sma','incl','A_Rs']].values.swapaxes(0,1))) #impact parameter at primary eclipse
    tran_combi_i['B_Rs']=tran_combi_i['Rp']/109.2
    tran_combi_i['bsec']=abs(b_ecc_sec(*tran_combi_i.loc[:,['ecc','omega','sma','incl','B_Rs']].values.swapaxes(0,1))) #impact parameter at primary eclipse
    tran_combi_i=tran_combi_i.drop('B_Rs',axis=1)
    tran_combi_i['transits']=abs(tran_combi_i['bpri'].values)<(1+tran_combi_i['Rp']/(109.2*tran_combi_i['A_Rs']))

    #Adding type according to whether it transits or not:
    types_suf=np.tile('_pl',tran_combi_i.shape[0])
    types_suf[tran_combi_i['transits'].values]='_tpl'
    tran_combi_i['type']=np.core.defchararray.add(tran_combi_i['type'].values.astype(str),types_suf)
    #a,Rs,Ts,A
    tran_combi_i['Teff']=SurfaceT(tran_combi_i['sma'].values,tran_combi_i['A_Rs'].values,tran_combi_i['A_Teff'].values)
    tran_combi_i['albedo']=get_albedo(tran_combi_i['Teff'].values)

    #Need to redo contration with multiplicities...
    tran_combi_i['depth_pure']=np.clip((((tran_combi_i['Rp']/108.0)/tran_combi_i['A_Rs'])**2),0,1)
    tran_combi_i['depth_diluted']=np.clip(tran_combi_i['depth_pure']*tran_combi_i['prop_of_flux_in_ap'],0,1)
    tran_combi_i['Tdur']=PerToTdur(tran_combi_i['A_Rs'],tran_combi_i['depth_pure'],
                                   tran_combi_i['A_Ms'],tran_combi_i['bpri'],tran_combi_i['P'])/86400.0#days #Vperi = (sqrt(1+tran_pls_i['ecc'])),Vapo = (sqrt(1-tran_pls_i['ecc']))
    tran_combi_i['prob_trans']=((((tran_combi_i['Rp']/108.0)+tran_combi_i['A_Rs'])*Rsun)/((1-tran_pls_i['ecc']**2)*au*tran_combi_i['sma']))

    #Dropping "impossible" planets (eg those inside their stars)
    tran_combi_i=tran_combi_i.drop(tran_combi_i.loc[tran_combi_i['P'].values<tran_combi_i['minP'].values].index.values)

    tran_combi_i['GD_1']=np.zeros(tran_combi_i.shape[0])
    tran_combi_i['LD_1']=np.ones(tran_combi_i.shape[0])
    tran_combi_i['LD_2']=np.zeros(tran_combi_i.shape[0])
    tran_combi_i.loc[np.isnan(tran_combi_i['sma']),'sma']=(((86400*tran_combi_i['P'])**2*6.67e-11*(tran_combi_i['Mp']*Mearth+tran_combi_i['A_Ms']*Msun))/(4*np.pi**2))**0.3333/au
    tran_combi_i['bfac']=Get_Beaming(tran_combi_i['Teff'].values)

    #tran_combi_i['bpri']=abs(b_ecc(tran_combi_i['ecc'].values,tran_combi_i['omega'].values,
    #                            tran_combi_i['sma'].values,tran_combi_i['incl'].values,
    #                            tran_combi_i['A_Rs'].values)) #impact parameter at primary eclipse
    #tran_combi_i['bsec']=abs(b_ecc(tran_combi_i['ecc'].values,tran_combi_i['omega'].values,
    #                            tran_combi_i['sma'].values,tran_combi_i['incl'].values,
    #                            tran_combi_i['B_Rs'].values,sec=True)) #impact parameter at primary eclipse
    tran_combi_i['Rratio']=(tran_combi_i['Rp'].values*Rearth)/(tran_combi_i['A_Rs'].values*Rsun)

    tran_combi_i['pri_ecl']=tran_combi_i['bpri']<(1+tran_combi_i['Rratio'])
    tran_combi_i['sec_ecl']=tran_combi_i['bsec']<(1+1.0/tran_combi_i['Rratio'])
    tran_combi_i['sbratio']=(tran_combi_i['Teff']/tran_combi_i['A_Teff'])**4

    #Returns planets
    return tran_combi_i

def b_ecc_pri(ecc,omega,a,incl,Rs):
    #Determine impact parameter given eccentric orbit
    return (a*au)/(Rs*Rsun)*(1. - ecc**2)/(1. + ecc*np.cos(np.pi/2.0 - omega))*np.cos(incl)

def b_ecc_sec(ecc,omega,a,incl,Rs):
    return (a*au)/(Rs*Rsun)*(1. - ecc**2)/(1. + ecc*np.cos(3*np.pi/2.0 - omega))*np.cos(incl)


def CalcDur(Rs_1,Rs_2,b,incl,e,omega,P,Ms_1,Ms_2,sec=False):
    Rs_1=Rs_1*Rsun if np.nanmedian(Rs_1)<7e6 else Rs_1 #converting to m
    Rs_2=Rs_2*Rsun if np.nanmedian(Rs_2)<7e6 else Rs_2 #converting to m
    omega=(omega+np.pi)%(2*np.pi) if sec==True else omega #in radians already
    P=P*86400 if np.nanmedian(P)<24000 else P
    Ms_1=Ms_1*Msun if np.nanmedian(Ms_1)<40. else Ms_1
    Ms_2=Ms_2*Msun if np.nanmedian(Ms_2)<40. else Ms_2
    # Compute geometrical factor Z
    Z = sqrt( 1 - (b*Rs_1 * np.cos(incl))**2 / (Rs_1 + Rs_2)**2)
    # Duration
    D = 2.*Z*(Rs_1 + Rs_2)*sqrt(1. - e**2)* \
                    ( 1. + e * cos(np.pi/2.0 - omega) )**(-1.)* \
                    (P/(2. * np.pi * 6.67e-11 * (Ms_1 + Ms_2)))**(1./3.)
    return D

def PlanetRtoM(Rad):
    #From Weiss & Marcy
    Mps=np.zeros(len(Rad))
    Mps[Rad<=1.5]=((2430+3390*Rad[Rad<=1.5])*(1.3333*np.pi*(Rad[Rad<=1.5]*6371000)**3))/Mearth
    Mps[(Rad>1.5)&(Rad<=4.0)]=2.69*(Rad[(Rad>1.5)&(Rad<=4.0)]**0.93)
    import scipy.interpolate
    o=np.array([[3.99, 9.0,10.5,12.0,13.0,14.0,14.5, 15.5, 16.0],[10.0, 100.0,300.0,1000.0,1100.0,1200.0, 1350.0, 5000.0,  10000.0]])
    f= scipy.interpolate.interp1d(o[0, :],o[1, :])
    Mps[(Rad>4.)&(Rad<=16.0)]=f(Rad[(Rad>4.)&(Rad<=16.0)])
    Mps[Rad>16.0]=10000.0
    return np.random.normal(Mps,0.25*Mps)


def PerToTdur(Rs,depth,Ms,b,Period):
    #Rs in solar radii, depth in fractional flux, period in days,Ms in solar masses,
    return (2*(1+depth**0.5)*np.sqrt(1-(b/(1+depth**0.5))**2))/((2*np.pi*6.67e-11*Ms*Msun)/(Period*86400*(Rs*Rsun)**3))**(1/3.)


def SurfaceT(sma,Rs,Ts,A=0.3):
    #This function gives the surface temperature of a planet from it's Semi Major axis, Stellar Radius, Stellar Temperature and Albedo (optional)
    return (((Rs*Rsun)**2*Ts**4.*(1.-A))/(4*(sma*au)**2.))**0.25

def get_albedo(teff):
    try:
        _=len(teff)
        singlevalue=False
    except:
        singlevalue=True
        teff=np.array([teff])
    a=np.zeros(len(teff))
    #For cool planets - random albedo up to 0.7 most likely
    a[teff<500]=np.random.uniform(0,0.7,np.sum(teff<500))
    #For convective stars, low albedo, especially for BDs/giant planets
    a[(teff>=500)&(teff<=6250)]=abs(np.random.normal((teff[(teff>=500)&(teff<=6250)])/14500.0,0.1,np.sum((teff>=500)&(teff<=6250))))
    #Radiative atmospheres nearly entirely reflective.
    a[teff>=6250]=np.random.uniform(0.75,1.0,np.sum(teff>=6250))
    if singlevalue:
        return a[0]
    else:
        return a

    

##################################################################
#                                                                #
#                GENERATE LIGHTCURVE USING ELLC                  #
#                                                                #
##################################################################


def GenLC(starA,starB,times,system=None,verbose=0,nodip=False):
    #starA = star row
    #starB = secondary/planet row
    #system = binary params row (if present)
    systcols=['P', 'T0', 'ecc', 'omega', 'incl', 'sep','min_sma', 'max_sma',
              'sma', 'bpri', 'bsec', 'Rratio', 'pri_ecl', 'sec_ecl','T0_sec']

    
    if type(starA)==pd.DataFrame:
        starA=starA.iloc[0].copy()
    if type(starB)==pd.DataFrame:
        starB=starB.iloc[0].copy()
    if nodip:
        #NO DIP IN SYST OR DONT WANT TO RUN SYSTEM.
        arr={}
        arr.update({'A_'+bcol:starA[bcol] for bcol in starA.index})
        if system is not None:
            arr={key:system[key] for key in systcols}
        if starB is not None:
            arr.update({'B_'+bcol:starB[bcol] for bcol in starB.index if bcol not in systcols})
            iname=starB.name
        else:
            iname=starA.name
        #print(iname)
        return 1, pd.Series(arr,name=iname)
    if system is None:
        # PLANET CASE
        #print('pl',starA.Rs,Rsun,starB.sma,au,(starA.Rs*Rsun)/(starB.sma*au))
        #print('pl',starB.Rp,Rearth,starB.sma,au,Rearth,(starB.Rp*Rearth)/(starB.sma*au))
        try:
            arr=pd.Series()

            arr=TheoryDurationTau(arr, starA, starB, system=None)
            if starB['sec_ecl']:
                arr['T0_sec_obs']=FindT0sec(starB['T0_sec'], starB['P'], starA, starB, system=None)
            else:
                arr['T0_sec_obs']=starB['T0_sec']
            xtra_t=np.hstack((starB['T0']+np.arange(-0.75,0.751,0.025)*arr['durpri_theory'],
                              arr['T0_sec_obs']+np.arange(-0.75,0.751,0.025)*arr['dursec_theory']))
            lc=GetEllcLC(np.hstack((xtra_t,times)),starA,starB,system=None,verbose=verbose)

            if np.sum(lc[len(xtra_t):]/lc[len(xtra_t):]!=1.0)>0:
                print(lc[len(xtra_t):])
                print(np.sum(lc[len(xtra_t):]/lc[len(xtra_t):]!=1.0))
                raise ValueError()
            else:
                arr['suceeds']=True

                arr=ObsDurationTau(arr, starA, starB, None, lc[:len(xtra_t)])
                lc=lc[len(xtra_t):]
                arr['B_metatype']='PL'

        except:
            print("PL lc fails",arr)
            lc=np.tile(1.0,len(times))
            arr=pd.Series({'suceeds':False})

        arr=arr.append(starB[systcols])
        arr=arr.append(starB.rename(index={bcol:'B_'+bcol for bcol in starB.index if bcol not in systcols}))
    else:
        # ECLIPSING BINARY CASE
        print("# ECLIPSING BINARY CASE")
        if type(system)==pd.DataFrame:
            system=system.iloc[0].copy()
        #print('eb',starA.Rs,Rsun,system.sma,au,(starA.Rs*Rsun)/(system.sma*au))
        #print('eb',starB.Rs,Rsun,system.sma,au,(starB.Rs*Rsun)/(system.sma*au))
        try:
            arr=pd.Series()

            arr=TheoryDurationTau(arr, starA, starB, system)
            
            
            if system['sec_ecl']:
                arr['T0_sec_obs']=FindT0sec(system['T0_sec'],system['P'], starA,starB, system)
            else:
                arr['T0_sec_obs']=system['T0_sec']

            print(system['T0_sec'],"<old | new>",arr['T0_sec_obs'])
            xtra_t=np.hstack((system['T0']+np.arange(-0.75,0.751,0.025)*arr['durpri_theory'],
                              arr['T0_sec_obs']+np.arange(-0.75,0.751,0.025)*arr['dursec_theory']))

            lc = GetEllcLC(np.hstack((xtra_t,times)), starA, starB, system,verbose=verbose)

            if np.sum(lc[len(xtra_t):]/lc[len(xtra_t):]!=1.0)>0:
                print(lc[len(xtra_t):])
                print(len(lc[len(xtra_t):]),np.sum(lc[len(xtra_t):]/lc[len(xtra_t):]!=1.0))
                raise ValueError()
            else:
                arr['suceeds']=True
                arr['B_metatype']='EB'

                arr=ObsDurationTau(arr, starA,starB,system,lc[:len(xtra_t)])
                lc=lc[len(xtra_t):]
            
            

        except:
            print("EB lc fails:")
            print('rad/sma',(starA.Rs*Rsun)/(system.sma*au),(starB.Rs*Rsun)/(system.sma*au),
                 'sbrat',(starB.Teff/starA.Teff)**4,'incl',system['incl']*180/np.pi,
                 'f3',1.0-(starA.prop_of_flux_in_ap+starB.prop_of_flux_in_ap),'t0',system.T0,
                 'eccs',np.sqrt(system.ecc)*np.cos(system.omega),np.sqrt(system.ecc)*np.sin(system.omega), 
                 'P and sma',system.P,(system.sma*au)/(starA.Rs*Rsun),'q',(starB.Ms*Msun)/(starA.Ms*Msun),
                 'LDs',[starA.LD_1,starA.LD_2], [starB.LD_1,starB.LD_2],starA.GD_1,starB.GD_1,
                 'albedos',starA.albedo, starB.albedo,'bfacs', starA.bfac,starB.bfac)
            lc=np.tile(1.0,len(times))
            arr=pd.Series({'suceeds':False})
        print(arr)
        print(type(arr))
        
        arr=arr.append(system)
        arr=arr.append(starB.rename(index={bcol:'B_'+bcol for bcol in starB.index}))
    arr=arr.append(starA.rename(index={bcol:'A_'+bcol for bcol in starA.index}))
    arr=arr[~arr.index.duplicated(keep='first')]
    return lc,arr
                 
def GetEllcLC(times,starA,starB,system=None,verbose=0):
    if system is None:
        #PLANET case
        return ellclc(t_obs=times,
                    radius_1=(starA.Rs*Rsun)/(starB.sma*au), radius_2=(starB.Rp*Rearth)/(starB.sma*au),
                    sbratio=(starB.Teff/starA.Teff)**4, incl=starB['incl']*180/np.pi,
                    light_3=starA.prop_of_flux_in_ap**-1, t_zero=starB.T0, period=starB.P,a=(starB.sma*au)/(starA.Rs*Rsun),
                    f_c=np.sqrt(starB.ecc)*np.cos(starB.omega), f_s=np.sqrt(starB.ecc)*np.sin(starB.omega), 
                    q=(starB.Mp*Mearth)/(starA.Ms*Msun),ldc_1=[starA.LD_1,starA.LD_2], ldc_2=[1.,0.],gdc_1=starA.GD_1,
                    gdc_2=0.0, heat_1=starA.albedo, heat_2=starB.albedo, lambda_1=None,ld_1="quad",
                    ld_2="quad",grid_1='sparse',grid_2='sparse',bfac_1=starA.bfac,bfac_2=starB.bfac,
                    verbose=verbose)
    else:
          #EB case
        return ellclc(t_obs=times,
                    radius_1=(starA.Rs*Rsun)/(system.sma*au), radius_2=(starB.Rs*Rsun)/(system.sma*au),
                    sbratio=(starB.Teff/starA.Teff)**4, incl=system['incl']*180/np.pi,
                    light_3=(starA.prop_of_flux_in_ap+starB.prop_of_flux_in_ap)**-1, t_zero=system.T0,
                    f_c=np.sqrt(system.ecc)*np.cos(system.omega), f_s=np.sqrt(system.ecc)*np.sin(system.omega), 
                    period=system.P,a=(system.sma*au)/(starA.Rs*Rsun),q=(starB.Ms*Msun)/(starA.Ms*Msun),
                    ldc_1=[starA.LD_1,starA.LD_2], ldc_2=[starB.LD_1,starB.LD_2],gdc_1=starA.GD_1,gdc_2=starB.GD_1,
                    heat_1=starA.albedo, heat_2=starB.albedo, lambda_1=None,ld_1="quad",
                    ld_2="quad",grid_1='very_sparse',grid_2='very_sparse',bfac_1=starA.bfac,bfac_2=starB.bfac,
                    verbose=verbose)

def FindT0sec(t0guess, period, starA,starB,system=None):
    fluxfromt = lambda t: GetEllcLC([t],starA,starB,system)
    #print(t0guess, [fluxfromt(ti) for ti in [t0guess-0.003,t0guess,t0guess+0.003]])
    funcout=opt.minimize(fluxfromt,t0guess,bounds=(((t0guess-0.1*period,t0guess+0.1*period),)), method='TNC', tol=1e-9)
    return funcout.x[0]
                        

def TheoryDurationTau(arr,starA,starB,system):
    if system is not None:
        #Planet case
        arr['durpri_theory'] = (2./86400.)*np.sqrt( 1 - (system['sma']*au * ( 1. - system['ecc']**2 ) / ( 1 + system['ecc'] * np.cos(np.pi/2.0 - system['omega']) ) * np.cos(system['incl']))**2 / (starB['Rs']*Rsun + starA['Rs']*Rsun)**2)*(starB['Rs']*Rsun + starA['Rs']*Rsun)*np.sqrt(1. - system['ecc']**2)* \
                    ( 1. + system['ecc'] * np.cos(np.pi/2.0 - system['omega']) )**(-1.)* \
                    (system['P']*86400./(2. * np.pi * G * (starA['Ms']*Msun + starB['Ms']*Msun)))**(1./3.)
        arr['dursec_theory'] = (2./86400.)*np.sqrt( 1 - (system['sma']*au * ( 1. - system['ecc']**2 ) / ( 1 + system['ecc'] * np.cos(3*np.pi/2.0 - system['omega']) ) * np.cos(system['incl']))**2 / (starB['Rs']*Rsun + starA['Rs']*Rsun)**2)*(starB['Rs']*Rsun + starA['Rs']*Rsun)*np.sqrt(1. - system['ecc']**2)* \
                    ( 1. + system['ecc'] * np.cos(3*np.pi/2.0 - system['omega']) )**(-1.)* \
                    (system['P']*86400./(2. * np.pi * G * (starA['Ms']*Msun + starB['Ms']*Msun)))**(1./3.)
       
        durpri_flat = (2./86400.)*np.sqrt( 1 - (system['sma']*au * ( 1. - system['ecc']**2 ) / ( 1 + system['ecc'] * np.cos(np.pi/2.0 - system['omega']) ) * np.cos(system['incl']))**2 / np.clip(starA['Rs']*Rsun - starB['Rs']*Rsun,0.01,500.0*Rsun)**2)*np.clip(starA['Rs']*Rsun - starB['Rs']*Rsun,0.01,500.0*Rsun)*np.sqrt(1. - system['ecc']**2)* \
                    ( 1. + system['ecc'] * np.cos(np.pi/2.0 - system['omega']) )**(-1.)* \
                    (system['P']*86400./(2. * np.pi * G * (starA['Ms']*Msun + starB['Ms']*Msun)))**(1./3.)
        arr['taupri_theory']=1-durpri_flat/arr['durpri_theory']
        arr['taupri_theory'] = 1.0 if system['pri_ecl'] and np.isnan(arr['taupri_theory']) else arr['taupri_theory']
        
        dursec_flat = (2./86400.)*np.sqrt( 1 - (system['sma']*au * ( 1. - system['ecc']**2 ) / ( 1 + system['ecc'] * np.cos(3*np.pi/2.0 - system['omega']) ) * np.cos(system['incl']))**2 / np.clip(starA['Rs']*Rsun - starB['Rs']*Rsun,0.0,500*Rsun)**2)*np.clip(starA['Rs']*Rsun - starB['Rs']*Rsun,0.01,500*Rsun)*np.sqrt(1. - system['ecc']**2)* \
                    ( 1. + system['ecc'] * np.cos(3*np.pi/2.0 - system['omega']) )**(-1.)* \
                    (system['P']*86400./(2. * np.pi * G * (starA['Ms']*Msun + starB['Ms']*Msun)))**(1./3.)
        #print(dursec_flat)
        #print( 1 - (system['sma']*au * ( 1. - system['ecc']**2 ) / ( 1 + system['ecc'] * np.cos(3*np.pi/2.0 - system['omega']) ) * np.cos(system['incl']))**2)
        #print(np.clip(starA['Rs']*Rsun - starB['Rs']*Rsun,0.0,500*Rsun)**2)
        #print(np.clip(starA['Rs']*Rsun - starB['Rs']*Rsun,0.01,500*Rsun)*np.sqrt(1. - system['ecc']**2))
        #print(( 1. + system['ecc'] * np.cos(3*np.pi/2.0 - system['omega']) )**(-1.))
        #print((system['P']*86400./(2. * np.pi * G * (starA['Ms']*Msun + starB['Ms']*Msun)))**(1./3.))
        arr['tausec_theory']=1-dursec_flat/arr['dursec_theory']
        arr['tausec_theory'] = 1.0 if system['sec_ecl'] and np.isnan(arr['tausec_theory']) else arr['tausec_theory']
    else:
        #EB CASE
        arr['durpri_theory'] = (2./86400.)*np.sqrt( 1 - (starB['sma']*au * ( 1. - starB['ecc']**2 ) / ( 1 + starB['ecc'] * np.cos(np.pi/2.0 - starB['omega']) ) * np.cos(starB['incl']))**2 / (starB['Rs']*Rsun + starA['Rs']*Rsun)**2)*(starB['Rs']*Rsun + starA['Rs']*Rsun)*np.sqrt(1. - starB['ecc']**2)* \
                    ( 1. + starB['ecc'] * np.cos(np.pi/2.0 - starB['omega']) )**(-1.)* \
                    (starB['P']*86400./(2. * np.pi * G * (starA['Ms']*Msun + starB['Ms']*Msun)))**(1./3.)

        arr['dursec_theory'] = (2./86400.)*np.sqrt( 1 - (starB['sma']*au * ( 1. - starB['ecc']**2 ) * \
                                                         ( 1 + starB['ecc'] * np.cos(3*np.pi/2.0 - starB['omega']) )**-1 * \
                                                         np.cos(starB['incl']))**2 *\
                                                   (starB['Rs']*Rsun + starA['Rs']*Rsun)**-2) * \
                               (starB['Rs']*Rsun + starA['Rs']*Rsun) * \
                               np.sqrt(1. - starB['ecc']**2) * \
                               ( 1. + starB['ecc'] * np.cos(3*np.pi/2.0 - starB['omega']) )**(-1.) * \
                               (starB['P']*86400./(2. * np.pi * G * (starA['Ms']*Msun + starB['Ms']*Msun)))**(1./3.)
        
        durpri_flat = (2./86400.) * \
                      np.sqrt( 1 - (starB['sma']*au * ( 1. - starB['ecc']**2 ) * \
                                    ( 1 + starB['ecc'] * np.cos(np.pi/2.0 - starB['omega']) )**-1 * \
                                    np.cos(starB['incl']))**2 *
                              np.clip(starA['Rs']*Rsun - starB['Rs']*Rsun,0.0,500.0*Rsun)**-2) * \
                      np.clip(starA['Rs']*Rsun - starB['Rs']*Rsun,0.01,500.0*Rsun) * \
                      np.sqrt(1. - starB['ecc']**2) * \
                      ( 1. + starB['ecc'] * np.cos(np.pi/2.0 - starB['omega']) )**(-1.) * \
                      (starB['P']*86400./(2. * np.pi * G * (starA['Ms']*Msun + starB['Ms']*Msun)))**(1./3.)
        arr['taupri_theory']=1-durpri_flat/arr['durpri_theory']

        dursec_flat = (2./86400.)*np.sqrt( 1 - (starB['sma']*au * ( 1. - starB['ecc']**2 ) *
                                                                  ( 1 + starB['ecc'] * np.cos(3*np.pi/2.0 - starB['omega']) )**-1 *\
                                                                   np.cos(starB['incl'])
                                               )**2 *
                                           np.clip(starA['Rs']*Rsun - starB['Rs']*Rsun,0.01,500*Rsun)**-2) * \
                       np.clip(starA['Rs']*Rsun - starB['Rs']*Rsun,0.01,500*Rsun) * \
                       np.sqrt(1. - starB['ecc']**2) * \
                       ( 1. + starB['ecc'] * np.cos(3*np.pi/2.0 - starB['omega']) )**(-1.) * \
                       (starB['P']*86400./(2. * np.pi * G * (starA['Ms']*Msun + starB['Ms']*Msun)))**(1./3.)
        arr['taupri_theory'] = 1.0 if starB['pri_ecl'] and np.isnan(arr['taupri_theory']) else arr['taupri_theory']
        #print(1 - (starB['sma']*au * ( 1. - starB['ecc']**2 ) / ( 1 + starB['ecc'] * np.cos(3*np.pi/2.0 - starB['omega']) ) * np.cos(starB['incl']))**2)
        #print(np.clip(starA['Rs']*Rsun - starB['Rs']*Rsun,0.01,500*Rsun)*np.sqrt(1. - starB['ecc']**2)* \
        #            ( 1. + starB['ecc'] * np.cos(3*np.pi/2.0 - starB['omega']) )**(-1.))
        #print((starB['P']*86400./(2. * np.pi * G * (starA['Ms']*Msun + starB['Ms']*Msun)))**(1./3.))
        #print(dursec_flat)
        arr['tausec_theory']=1-dursec_flat/arr['dursec_theory']   
        arr['tausec_theory'] = 1.0 if starB['sec_ecl'] and np.isnan(arr['tausec_theory']) else arr['tausec_theory']
    return arr


def ObsDurationTau(arr, starA,starB,system,depslc):
    #Getting a quick around-transit LC and measuring real depth, duration and tau (to ~10% of dur_theoretical)
    #Extracting observed duration, egress and secondary eclipse.
    diffdepslc=np.diff(np.diff(depslc))
    arr['depthpri_obs']=1.0-np.min(depslc[:61])
    arr['durpri_obs']=0.025*(31-np.argmin(diffdepslc[:31])+np.argmin(diffdepslc[31:59]))*arr['durpri_theory']
    arr['taupri_obs']=0.025*((np.argmax(diffdepslc[:31])-np.argmin(diffdepslc[:31]))+
                            (np.argmin(diffdepslc[30:59])-np.argmax(diffdepslc[30:59])))*(arr['durpri_obs']/arr['durpri_theory'])

    arr['depthsec_obs']=1.0-np.min(depslc[61:])
    arr['dursec_obs']=0.025*(31-np.argmin(diffdepslc[61:91])+np.argmin(diffdepslc[90:]))*arr['dursec_theory']
    arr['tausec_obs']=0.025*((np.argmax(diffdepslc[61:91])-np.argmin(diffdepslc[61:91]))+(np.argmin(diffdepslc[90:])-np.argmax(diffdepslc[90:])))*(arr['dursec_theory']/arr['dursec_obs'])
    arr['lc_zoom']=','.join(depslc.astype(str))
    
    return arr

##################################################################
#                                                                #
#                STELLAR VARIABILITY FUNCTIONS                   #
#                                                                #
##################################################################


def GenQuasiPer(Q,rotper,logamp,time,err=1e-9):
    w=(np.pi*2)/rotper
    S0 = np.power(10,float(logamp)*2) / ( w * Q)
    kernel = terms.SHOTerm(log_S0=np.log(S0), log_Q=np.log(Q), log_omega0=np.log(w))

    gp = celerite.GP(kernel, mean=1.0)
    gp.compute(time, np.tile(err,len(time)))  # You always need to call compute once.
    return gp.sample(1)[0]


def ColFromTeff(teff):
    #Values from 50th degree polynomial fit to www.pas.rochester.edu/~emamajek/EEM_dwarf_UBVIJHK_colors_Teff.txt
    return np.polyval(np.array([ 6.05983799e+00, -1.25467009e+02,  1.03444427e+03, -4.24273598e+03,
        8.64992763e+03, -7.00608901e+03]),np.log10(teff))
    
def logPerFromAge(A,B,V):
    #A = Age in Myr
    if type(A)==float or type(A)==int:
        A=[A];B=[B];V=[V]
    #From Angus et al, 2014
    return np.random.normal(0.55,0.03,len(A))*np.log10(A)+np.log10(abs(np.random.normal(0.4,0.05,len(A))))+np.random.normal(0.31,0.03,len(A))*np.log10(np.power(B-V-0.45))
    #B and V are magnitudes and a, b and n are the free parameters of our model. 
    #We found a=0.40+0.3-0.05, b=0.31+0.05-0.02 and n=0.55+0.02-0.09
    
    
def logPerFromTeff(A,Teff):
    #A = Age in Myr
    if type(A)==float or type(A)==int:
        A=np.array(A);Teff=np.array(Teff)
    #From Angus et al, 2014
    #print(ColFromTeff(Teff)-0.45)
    #print(np.random.normal(0.55,0.03,len(A))*np.log10(A),
    #      np.log10(np.random.normal(0.4,0.05,len(A))),
    #      np.random.normal(0.31,0.03,len(A))*np.log10(np.clip(ColFromTeff(Teff)-0.45,0.000001,2.5)))

    return np.random.normal(0.55,0.03,len(A))*np.log10(A)+np.log10(abs(np.random.normal(0.4,0.05,len(A))))+\
           np.random.normal(0.31,0.03,len(A))*np.log10(np.clip(ColFromTeff(Teff)-0.45,0.1,2.5))

def logQfromTeff(Teff):
    #Although this is now accounted for in the LogAmp,
    # I like the current use of Q to be high for M-dwarfs and giants, so lets keep it
    
    if type(Teff)==float or type(Teff)==int or type(Teff)==np.float64:
        Teff=[Teff]
    return np.random.normal(np.polyval(np.array([ 1.42857143e-04, -1.07142857e+00]),Teff),
                                        np.polyval(np.array([ 1.33333333e-08, -6.66666667e-05,  4.80000000e-01]),Teff))
    '''
    #SCRAPPING THIS AND GOING TO A SIMPLE NORMAL DIST WITH HIGHER SCATTER AT HOT STARS
    #Getting percentage variable (from McQuillan et al)
    pc_per=np.clip(np.polyval(np.array([ 1.01428571e-07, -1.25697143e-03,  4.13501786e+00]),Teff),0.05,0.95)
    from scipy.stats import norm
    #Using PPF (inverse PDF) to find the position of the centre of a gaussian that would cause this percentage >0
    #We're actually gonna double this (and double the spread) to give great variation
    return np.power(10,np.random.normal(norm.ppf(pc_per)*0.25,np.tile(0.5,len(Teff)))-0.25)
    '''
    
def NewLogAmpFromTeff(Teff):
    #Now adjusted for fraction of observability observed
    if type(Teff)==float or type(Teff)==int or type(Teff)==np.float64:
        Teff=[Teff]
    #Roughly From Figure 3, McQuillan et al
    amps=np.polyval(np.array([ 5.00016552e-15, -1.03073565e-10, 6.71413558e-07, -1.98340003e-03, 5.92386381e+00]),
                    Teff)-6
    #-6 to go from PPM to relative flux
    amp_stds=np.clip(np.polyval(np.array([ 3.32134028e-04,-6.44271373e-01]),Teff),0.1,1.5)
    #amp_stds[Teff>7500]=0.42
    amp_stds[np.random.random(len(amp_stds))>0.8]+=0.8
    return np.clip(np.random.normal(amps,amp_stds),-15,-0.3)


def LogAmpFromTeff(Teff):
    if type(Teff)==float or type(Teff)==int or type(Teff)==np.float64:
        Teff=[Teff]
    #Roughly From Figure 3, McQuillan et al
    amps=np.polyval(np.array([ 3.16622538e-14, -6.82649385e-10,  5.18933830e-06,
                              -1.65655172e-02, 2.25532954e+01]),
                    Teff)-6
    #-6 to go from PPM to relative flux
    amp_stds=np.clip(np.polyval(np.array([ 0.00014762, -0.10119048]),Teff),0.1,1.5)
    #amp_stds[Teff>7500]=0.42
    amp_stds[np.random.random(len(amp_stds))>0.8]+=0.8
    return np.clip(np.random.normal(amps,amp_stds),-15,-0.3)

def QPlc_fromTeff(Teff,Age,time=np.arange(0,72,0.01),err=1e-9):
    #A = Age in Myr
    if type(Teff)==float or type(Teff)==int or type(Teff)==np.float64:
        Teff=np.array([float(Teff)]);Age=np.array([Age*1e3])
    else:
        Age*=1e3
    
    outlcs=[]
    q,per,amp=logQfromTeff(Teff),np.power(10,logPerFromTeff(abs(Age),Teff)),NewLogAmpFromTeff(Teff)
    #Typically Q (damping) is correlated with amplitude - pulsations are stronger than quasi-periodic behaviour.
    amp[q<=0]-=0.5*q[q<=0.0]
    amp[q>0]+=0.25*q[q>0.0]
    
    #We need to adjust for the fact that the amp =std, but the quoted amp is over the full period. Effectively (Npts per period)^-0.5
    cad=np.median(np.diff(time))
    amp=amp-0.5*np.log10(per/cad)
    
    amp=np.clip(amp,-15,-0.3)
    for irow in range(len(Teff)):
        #print('Teff',Teff[irow],'Age',Age[irow],'Q',q[irow],'Per',per[irow],'log Amp',amp[irow])
        outlcs+=[GenQuasiPer(np.power(10,q[irow]),per[irow],amp[irow],time,err=1e-9)]
    return np.vstack((outlcs)),np.column_stack((q,per,amp))


def Oscillations(M,Teff,R,time,L=None):
    # M - Stellar Mass (in Msun)
    # Teff - Stellar Teff (in K)
    # R - Stellar Radius (in Rsun)
    # time - time array - in days
    # L - Stellar Luminosity (in Lsun) - if None, calculated from others
    if type(M) is float or type(M) is np.float64:
        #print('single object')
        if L==None:
            L=R**2*(Teff/5800)**4

        fluxes = generate_stellar_fluxes((time[-1]+1.01*np.nanmedian(np.diff(time))-time[0])*u.day, M*M_sun, np.clip(Teff,4900,9800)*u.K, 
                                         np.clip(R,0.0,4*M)*R_sun, L*L_sun, cadence=np.nanmedian(np.diff(time))*u.day)
        if len(fluxes)>len(time):
            fluxes=fluxes[:len(time)]
        return fluxes[1].reshape(1, -1)
    else:
        #Arrays:
        fluxes=[]
        for nrow in range(len(M)):
            if L==None:
                iL=R[nrow]**2*(Teff[nrow]/5800)**4
            else:
                iL=L[nrow]
            #print((time[-1]-time[0])*u.day, M[nrow]*M_sun, np.clip(Teff[nrow],4900,9800)*u.K, 
            #                             np.clip(R[nrow],0.0,4*M[nrow])*R_sun, iL*L_sun, np.nanmedian(np.diff(time))*u.day)
            osc=generate_stellar_fluxes((time[-1]+np.nanmedian(np.diff(time))-time[0])*u.day, M[nrow]*M_sun, np.clip(Teff[nrow],4900,9800)*u.K, 
                                         np.clip(R[nrow],0.0,4*M[nrow])*R_sun, iL*L_sun, cadence=np.round(86400*np.nanmedian(np.diff(time)))*u.sec)
            if len(osc[1])>len(time):
                print("Ocillations produced are longer than time")
                print(len(osc[1]),len(time),len(osc[1][:len(time)]))
                osc[1]=osc[1][:len(time)]
            elif len(osc[1])<len(time):
                print("Ocillations produced are shorter than time")
                print(len(osc[1]),len(time),len(np.pad(osc[1],np.ceil(0.5*(len(osc[1])-len(time))),mode='reflect')))
                osc[1]=np.pad(osc[1],np.ceil(0.5*(len(osc[1])-len(time))),mode='reflect')
                osc[1]=osc[1][:len(time)]
                
            fluxes +=[osc[1]]
        
        #print(len(fluxes),len(fluxes[-1]))
        if len(fluxes)==1:
            return fluxes[0].reshape(1, -1)
        else:
            return np.vstack(fluxes)
    
def Noise(targ_df,time,returnall=False):
    target=targ_df.loc[targ_df.type=='target']
    
    #Correcting Age if this column is missing using logage or, in the worst case, using Target age:
    targ_df.loc[pd.isnull(targ_df['Age'])&(~pd.isnull(targ_df['logage'])),'Age']=np.power(10,targ_df.loc[pd.isnull(targ_df['Age'])&(~pd.isnull(targ_df['logage'])),'logage']-9)
    targ_df.loc[pd.isnull(targ_df['Age']),'Age']=target['Age']
    
    #Only generating variability for brightest blends:
    brightblends=targ_df.loc[targ_df.Pmag.values<(target['Pmag'].values+5)]
    oscs=np.ones((len(time),targ_df.shape[0]))
    oscs[targ_df.Pmag.values<(target['Pmag'].values+5)]=Oscillations(brightblends['Ms'].values,
                                                                     brightblends['Teff'].values,
                                                                     brightblends['Rs'].values,time)
    newcols=np.tile(np.nan,(targ_df.shape[0],5))
    rots=np.ones((len(time),targ_df.shape[0]))
    rots[targ_df.Pmag.values<(target['Pmag'].values+5)],qpcols=QPlc_fromTeff(brightblends['Teff'].values,
                                                                             brightblends['Age'].values,time)
    newcols[targ_df.Pmag.values<(target['Pmag'].values+5),0]=qpcols[:,0]
    newcols[targ_df.Pmag.values<(target['Pmag'].values+5),1]=qpcols[:,1]
    newcols[targ_df.Pmag.values<(target['Pmag'].values+5),2]=qpcols[:,2]
    #print(np.shape(newcols))
    #print(qpcols)
    #print(np.shape(oscs))
    if len(np.shape(rots))>1:
        newcols[targ_df.Pmag.values<(target['Pmag'].values+5),3]=np.std(rots,axis=1)
        newcols[targ_df.Pmag.values<(target['Pmag'].values+5),4]=np.std(oscs,axis=1)
    else:
        newcols[targ_df.Pmag.values<(target['Pmag'].values+5),3]=np.std(rots)
        newcols[targ_df.Pmag.values<(target['Pmag'].values+5),4]=np.std(oscs)
    
    #print(np.shape(oscs),np.shape(rots))
    varbly=oscs+(rots-1.0)
    #RMS is per hour, so we adjust using sqrt(cadence/hr)
    #print('rms_hr',target['rms_hr'].values/np.sqrt(np.median(np.diff(time))*24))
    #print('rms_varb',np.std(np.sum(varbly*brightblends['prop_of_flux_in_ap'].values[:,np.newaxis],axis=0)))
    #multiplying by proportion of flux from each star in aperture
    if not returnall:
        outlc=np.sum(varbly*brightblends['prop_of_flux_in_ap'].values[:,np.newaxis],axis=0)+np.random.normal(0.0,1e-6*target['rms_hr_corr'].values/np.sqrt(np.median(np.diff(time))*24),len(time))
    else:
        outlc=np.column_stack((varbly*brightblends['prop_of_flux_in_ap'].values[:,np.newaxis]))
    return outlc, newcols


def init_Gaia_Source():
    from scipy.interpolate import CloughTocher2DInterpolator as ct2d
    from scipy import interpolate as interp
    gaia=pd.DataFrame.from_csv('Gaia_Det_limits.txt',index_col=None).as_matrix()
    form=gaia[-1,0]
    gaia=gaia[:-1].astype(float)
    sourcedet_interp=interp.Rbf(gaia[:,1],gaia[:,2],gaia[:,0], function='multiquadric',smooth=0.2)

    sourcedet_interp_clipped = lambda deltamag,sep: np.clip(sourcedet_interp(deltamag,sep),0.0,1.0)
    '''
    plt.figure(figsize=(12,9))
    plt.title("Gaia companion detection")
    #fig.gca().set_xscale('log')
    #plt.yscale('log')
    plt.xlabel('Companion distance in arcsec')
    plt.ylabel('Magnitude difference')

    seps=np.linspace(0.0001,1.05,200)
    magdifs=np.linspace(0.0001,7.99,200)
    source_recovery=np.zeros((len(magdifs),len(seps)))
    for ns,s in enumerate(seps):
        for nm,m in enumerate(magdifs):
            try:
                source_recovery[nm,ns]=sourcedet_interp_clipped(s,m)
            except:
                source_recovery[nm,ns]=sourcedet_interp_clipped(s,m)
    plt.pcolor(seps, magdifs, source_recovery)
    for lev in np.unique(gaia[:,0]):
        dat2plt=gaia[gaia[:,0]==lev]
        plt.plot(dat2plt[:,1],dat2plt[:,2],'--',label=lev)
    plt.legend()
    plt.ylim(8,0)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig("Gaia_interpolated_source_detn.png")
    '''
    return sourcedet_interp_clipped
