#!/usr/bin/env python3
##################################################################
#                                                                #
#                 GENERATING LCs For PLATOSIM3                   #
#                                                                #
##################################################################
import argparse
import datetime
import dateutil.parser
import requests # installs with : pip install requests
import xml.etree.ElementTree as ElementTree
import os
import getpass

#import matplotlib
#matplotlib.use('Agg')

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

import warnings
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)

try:
    from .LcGen import *
except:
    try:
        from PlatoSim_FieldGen.LcGen import *
    except:
        try:
            from LcGen import *
        except:
            raise Exception()

import astropy.coordinates
from astropy.coordinates import SkyCoord, SkyOffsetFrame, ICRS
import astropy.units as u
from pathlib import Path

def FieldOverGap(rel_alpha, rel_delta, fieldsize, 
                 camera_offset=6.5*u.deg, FoV_square=32.5*u.deg, gapsize=111.11*15*u.arcsec, gap_orientation_axis=0):
    #Checks whether input field (and size) is over a gap
    # INPUTS:
    # - rel_alpha (field centre x coordinate in telescope frame)
    # - rel_delta (field centre y coordinate in telescope frame)
    # - fieldsize (size of the field to check)
    # - camera_offset (offset for each camera from the central telescope position in x and y)
    # - FoV_square (the field of view along the short axis of the telescope) - set to 32.5deg
    # - gapsize (the size of the gap. 111.11 pixels)
    # - gap orientation axis (along what angle are the gaps aligned - assumed to be 0)
    
    #print(rel_alpha,rel_delta)
    outarr=np.tile(False,len(rel_alpha))
    #Alpha check:
    outarr=outarr|(abs(rel_alpha-camera_offset)<0.5*(gapsize+fieldsize))
    
    #Dec Check:
    outarr=outarr|(abs(rel_delta-camera_offset)<0.5*(gapsize+fieldsize))
    
    #Checking extend limits in alpha, delta and special "corner" check
    outarr=outarr|(abs(rel_alpha)>(0.5*FoV_square+camera_offset))
    outarr=outarr|(abs(rel_delta)>(0.5*FoV_square+camera_offset))
    
    cornercut=0.15
    outarr=outarr|(np.sqrt(rel_alpha.to(u.deg).value**2+rel_delta.to(u.deg).value**2)>(1-cornercut)*np.sqrt(2)*(FoV_square.to(u.deg).value+2*gapsize.to(u.deg).value))
    
    return outarr

def Ncameras(rel_alpha, rel_delta,camera_offset=6.5*u.deg,FoV_radius=18.9*u.deg,edgecut=1*u.deg):
    #Calculates number of cameras that observe given an alpha/delta position relative to the spacecraft centre
    
    #distance from each of the cameras:
    allcams=SkyCoord([(alph,delt) for alph in [-1*camera_offset,camera_offset] for delt in [-1*camera_offset, camera_offset]])
    cam_seps=allcams.separation(SkyCoord(rel_alpha,rel_delta)).value
    
    #print(type(allcams.dec),type(rel_delta),type(FoV_radius),type(edgecut),type(allcams.ra),type(rel_alpha),type(cam_seps))
    
    obs_by_cams=((allcams.dec.to(u.deg).value-rel_delta.to(u.deg).value)<(FoV_radius.to(u.deg).value-edgecut.to(u.deg).value))*((allcams.ra.to(u.deg).value-rel_alpha.to(u.deg).value)<(FoV_radius.to(u.deg).value-edgecut.to(u.deg).value))*(cam_seps<FoV_radius.to(u.deg).value)
    return int(np.sum(obs_by_cams)*6)

def StarPositions(fieldcent_sc, scope_frame, Angle_with_RADEC=0*u.deg, starsep=7, fieldsize=100, pixsize=15*u.arcsec):
    #Generates star positions given:
    # fieldcent_sc - Field SkyCoord object
    # scope_frame - SkyCoord frame Class centred around telescope position)
    # Angle_with_RADEC - angle between the camera x-y frame and the ra-dec plane.
    # starsep - Distance (in pixels) to seperate stars apart
    # fieldsize - size of field in pixels
    # pixsize - size of pixels
    
    #Creating field frame relative to RADEC
    #field_frame = SkyOffsetFrame(origin=fieldcent_sc.transform_to(ICRS), rotation= Angle_with_RADEC)
    field_frame = SkyOffsetFrame(origin=fieldcent_sc.transform_to(astropy.coordinates.GeocentricTrueEcliptic), rotation=0)
    starsep*=pixsize #fixing starsep to distance
    fieldsize*=pixsize #fixing fieldsize to distance
    alphs=[]
    decs=[]
    for dec in np.arange((-1*fieldsize*0.5-starsep).to(u.deg).value, (fieldsize*0.5+starsep).to(u.deg).value, starsep.to(u.deg).value):
        for nalp,alp in enumerate(np.arange((-1*fieldsize*0.5-starsep).to(u.deg).value,(fieldsize*0.5+starsep).to(u.deg).value,(1/np.sqrt(2))*starsep.to(u.deg).value)):
            if nalp%2==1:
                decs+=[dec*u.deg+starsep*0.5]
            else:
                decs+=[dec*u.deg]
            alphs+=[alp*u.deg]
    starcents_wrt_field=SkyCoord(lon=alphs, lat=decs, frame=field_frame)
    return starcents_wrt_field

def InFoV(field_ra,field_dec, raPlatform,decPlatform, Q=1,fieldsize=100,group='all'):
    #INPUTS:
    # field_ra - RA of field in deg
    # field_de - DEC of field in deg
    # raPlatform - RA of platform in deg
    # decPlatform - dec of platform in deg
    # Q - quarter (srarting at 1)
    # fieldsize - size of field in pixels
    # group - either specific group (1-4) or 'all'
    
    import referenceFrames as rf
    from math import sin, cos, tan, asin, atan, radians, degrees
    import shapely.geometry as sg
    import descartes
    from scipy import constants

    focalLength = 247.52     # [mm]
    pixelSize = 18     # [µm]
    plateScale = 15     # [arcsec]
    numGroups = 4
    numCorners = 4
    ccdCodes = ["1", "2", "3", "4"]
    tiltAngles = [9.2, 9.2, 9.2, 9.2]     # [degrees]
    azimuthAngles = [45.0, 135.0, -135.0, -45.0]     # [degrees]
    fovDegrees = 18.8908
    fovPixels = fovDegrees / plateScale * constants.degree / constants.arcsec
    fovMm = focalLength * tan(radians(fovDegrees))
    sign = lambda x: (1, -1)[x < 0]
    
    raTelescope = {}
    decTelescope = {}
    # Arbitrary choice of platform pointing and solar-panel orientation

    solarPanelOrientation = 0.     # [degrees]
    raSun, decSun = rf.sunSkyCoordinatesAwayfromPlatformPointing(radians(raPlatform), radians(decPlatform), solarPanelOrientation)
    groups = range(1, 1+numGroups) if group =='all' else [group]
    for gr in groups:

        # Telescope pointing (absolute)
        ra, dec = rf.platformToTelescopePointingCoordinates(radians(raPlatform), radians(decPlatform), raSun, decSun, radians(azimuthAngles[gr-1]), radians(tiltAngles[gr-1]))     # [radians]

        # Telescope pointing w.r.t. platform pointing
        raTelescope[gr]=degrees(ra) - raPlatform     # [degrees]
        decTelescope[gr]=degrees(dec) - decPlatform  # [degrees]

    meanDist = (np.mean(np.absolute(np.array([ra for ra in raTelescope]))) + np.mean(np.absolute(np.array([dec for dec in raTelescope])))) / 2.0

    for gr in groups:

        raTelescope[gr] = sign(raTelescope[gr]) * meanDist
        decTelescope[gr] = sign(decTelescope[gr]) * meanDist
    
    observed_by={}
    for gr in groups:
        #Get the focal plane coordinates of the telescopes in the above group:
        #tscope=rf.skyToFocalPlaneCoordinates(radians(raTelescope[group] + raPlatform), radians(decTelescope[group] + decPlatform),
        #                                     radians(raPlatform), radians(decPlatform), (Q-1)*(np.pi/2),  \
        #                                     radians(tiltAngles[0]), radians(azimuthAngles[0]), 0, focalLength)
        corner_dists=[]
        fieldcen=rf.skyToFocalPlaneCoordinates(radians(field_ra), radians(field_dec), 
                                               radians(raTelescope[gr] + raPlatform), radians(decTelescope[gr] + decPlatform), (Q-1)*(np.pi/2),  \
                                               radians(tiltAngles[0]), radians(azimuthAngles[0]), 0, focalLength)

        for x_edge_mm in np.array([fieldcen[0]+0.5*fieldsize*pixelSize*1e-3,fieldcen[0]-0.5*fieldsize*pixelSize*1e-3]):
            for y_edge_mm in np.array([fieldcen[1]+0.5*fieldsize*pixelSize*1e-3,fieldcen[1]-0.5*fieldsize*pixelSize*1e-3]):
                #Compute if each the "corners" of the star field is in the FoV
                corner_dists+=[np.sqrt(x_edge_mm**2+y_edge_mm**2)]
        #Verifying if, in focal plane coordinates, the fieldcorner-to-telescope distance is less than the fov (in FP/mm)
        if np.all(np.array(corner_dists)<fovMm):
            #All corners within focal plane
            #Checking CCD edges
            ccdchecks=[]
            for ccdCode in ccdCodes:
                cornersX, cornersY = rf.computeCCDcornersInFocalPlane(ccdCode, pixelSize)
                over_gaps=[]
                for x_edge_mm in np.array([fieldcen[0]+0.5*fieldsize*pixelSize*1e-3,fieldcen[0]-0.5*fieldsize*pixelSize*1e-3]):
                    for y_edge_mm in np.array([fieldcen[1]+0.5*fieldsize*pixelSize*1e-3,fieldcen[1]-0.5*fieldsize*pixelSize*1e-3]):
                        over_gaps+=[(x_edge_mm>np.min(cornersX))&(x_edge_mm<np.max(cornersX))&\
                                    (y_edge_mm>np.min(cornersY))&(y_edge_mm<np.max(cornersY))]
                ccdchecks+=[np.all(over_gaps)]
            if np.any(ccdchecks):
                NotInFoV=False
                observed_by[gr]=['CCD'+np.array(ccdCodes)[np.array(ccdchecks)][0]]
                print(gr,"observed by "+str(gr)+'_'+np.array(ccdCodes)[np.array(ccdchecks)][0])
            else:
                observed_by[gr]=['CCD_gap']
                print(gr,"over CCD gap")
        else:
            print(gr,"not in FoV",raTelescope[gr] + raPlatform,'/',decTelescope[gr] + decPlatform,' vs ',field_ra,'/',field_dec)
            observed_by[gr]=['Outside_FoV']
    return observed_by

def GenerateSimFields(Longcen, Latcen, Nfields=60, Angle_with_RADEC=0*u.deg, FoV_square=32.5*u.deg, fieldsize=100):
    #Generates star field positions given:
    # Longcen & Latcen - central galactic coordinates of the centre of all fields
    # Nfields - number of fields
    # Angle_with_RADEC - angle between the camera x-y frame and the ra-dec plane.
    # FoV_square - angle between the camera x-y frame and the ra-dec plane.
    # fieldsize - size of subField in pixels
    
    spacecraft_cent = SkyCoord(l=Longcen, b=Latcen, frame='galactic')
    spacecraft_cent_radec=spacecraft_cent.transform_to(ICRS)
    
    scope_frame = SkyOffsetFrame(origin=spacecraft_cent.transform_to(astropy.coordinates.GeocentricTrueEcliptic))
    
    n_good_fields=0
    
    field_list=np.random.normal(0, 0.35*FoV_square.to(u.deg).value,(Nfields,2))
    
    #Looping until no fields are over gaps:
    while n_good_fields<Nfields:
        
        field_list[np.isnan(field_list[:,0])]=np.random.normal(0, 0.44*FoV_square.to(u.deg).value,(np.sum(np.isnan(field_list[:,0])),2))
        
        field_radecs= SkyCoord(lon=field_list[:,0]*u.deg, lat=field_list[:,1]*u.deg, frame=scope_frame).transform_to(ICRS)
        
        for n_field in range(len(field_list)):
            obsbty=[]
            for q in range(4):
                
                observable=InFoV(field_radecs.ra.deg[n_field], field_radecs.dec.deg[n_field],
                                 spacecraft_cent_radec.ra.deg, spacecraft_cent_radec.dec.deg,
                                 Q=q,fieldsize=200)
                obsbty+=[obs for obs in observable]
            if 'CCD_gap' in obsbty or np.all(obsbty=='Outside_FoV'):
                field_list[n_field,0]=np.nan
        #output_of_test=FieldOverGap(field_list[:,0]*u.deg, field_list[:,1]*u.deg, camera_offset, FoV_square, fieldsize, gapsize)
        #field_list[output_of_test,0]=np.nan
        n_good_fields=np.sum(~np.isnan(field_list[:,0]))
            
    fieldcent= SkyCoord(lon=field_list[:,0]*u.deg, lat=field_list[:,1]*u.deg, frame=scope_frame)
    
    #spacecraft_cent.directional_offset_by(position_angle=np.arctan(field_list[:,1]/field_list[:,0]), separation=np.sqrt(field_list[:,0]**2,field_list[:,1])*u.deg)
    #plt.plot(fieldcent.alpha.deg,fieldcent.delta.deg,'.')
    
    allstars=pd.DataFrame()
    for n,field in enumerate(fieldcent):
        stars=StarPositions(field, scope_frame, Angle_with_RADEC=Angle_with_RADEC).transform_to(scope_frame)
        
        stars_radec = stars.transform_to(ICRS)
        stars_gal = stars.transform_to('galactic')
        field_radec=field.transform_to(ICRS)
        field_gal=field.transform_to('galactic')
        df=pd.DataFrame()
        df['i']=np.arange(len(stars))
        df['Nscopes']=Ncameras(field.lon, field.lat)
        df['ra']=stars_radec.ra.deg
        df['long']=stars_gal.l.deg
        df['dec']=stars_radec.dec.deg
        df['lat']=stars_gal.b.deg
        df['longitude_scopeframe']=stars.lon.deg
        df['latitude_scopeframe']=stars.lat.deg
        df['field_cen_ra']=np.tile(field_radec.ra.deg,len(stars_radec))
        df['field_cen_long']=np.tile(field_gal.l.deg,len(stars_radec))
        df['field_cen_dec']=np.tile(field_radec.dec.deg,len(stars_radec))
        df['field_cen_lat']=np.tile(field_gal.b.deg,len(stars_radec))
        df['scope_cen_ra']=np.tile(spacecraft_cent_radec.ra.deg,len(stars_radec))
        df['scope_cen_long']=np.tile(spacecraft_cent.l.deg,len(stars_radec))
        df['scope_cen_dec']=np.tile(spacecraft_cent_radec.dec.deg,len(stars_radec))
        df['scope_cen_lat']=np.tile(spacecraft_cent.b.deg,len(stars_radec))
        df['field_ID']=np.tile(n,len(stars))
        allstars=allstars.append(df)
    
    return allstars,scope_frame


def GenDipCatPLATO(hemisphere,npart,folder='/home/hosborn/PLATO/Plato_Sims',MakeLCs=True,mag='Pmag',
                   startover=False,nmult=5,peturb_cat=False,ext=''):

    # Generate catalogue of "Dips" in PLATO.
    # hemisphere = South or North
    # npart = 0 to 8
    # MakeLCs = whether to built LCs
    # mag = base magnitude to use
    # Startover = ignore previous files
    # nmult = factor to multiply number of stars. EG to create 100 more stars than normal, put 100
    # peturb_cat = whether to add fake noise into stellar distributions

    print("Running "+hemisphere)
    
    if not os.path.isdir(folder):
        os.system('mkdir '+folder)
    #ASSEMBLING Input Catalogue:
    #Using nmult=100 to get 100x more stars and therefore a larger sample.

    #GETTING BESANCON BLENDED STARS:
    if not os.path.exists(os.path.join(folder,'PlatoSim3_'+str(npart)+'_'+hemisphere+ext+'.pickle')) or startover:
        stars_all=pd.DataFrame()
        #for npart in range(10):
         
        stars_all=assembleP5_new(hemisphere,npart,nmult=nmult,
                                 ext=ext,outdir='/home/hosborn/Plato_Simulations/BesanconModels2/')
        stars_all['npart']=npart
        stars_all=stars_all.set_index(np.array([str(npart)+str(i).zfill(6) for i in np.arange(len(stars_all))]))
        
        '''
        # Generating stars in fake PlatoSim fields:
        if hemisphere=='South':
            scope_cen=SkyCoord(253*u.deg, -30*u.deg,frame='galactic').transform_to(ICRS)
        else:
            scope_cen=SkyCoord(65*u.deg, 30*u.deg,frame='galactic').transform_to(ICRS)
                #for a Southern sky field and (l=65, b=30
            
        PlatoSim_stars,scope_frame=GenerateSimFields(scope_cen.ra, scope_cen.dec, Nfields, 0*u.deg)
        
        # Now we need to take the N nearby target stars for each field from the stars_all to assemble a catalogue
        starcat_scope = SkyCoord(stars_all.longitude*u.deg,stars_all.latitude*u.deg,frame='galactic').transform_to(ICRS)
        
        newstarcat=pd.DataFrame()
        print('fields:',len(PlatoSim_stars),'bes_stars:',len(stars_all))
        print('fields:',np.max(PlatoSim_stars['field_cen_RA']),np.min(PlatoSim_stars['field_cen_RA']),
              np.max(PlatoSim_stars['field_cen_Dec']),np.min(PlatoSim_stars['field_cen_Dec']))
        print('bes_stars:',np.max(starcat_scope.ra.deg),np.min(starcat_scope.ra.deg),
              np.max(starcat_scope.dec.deg),np.min(starcat_scope.dec.deg))
        
        for fieldid in pd.unique(PlatoSim_stars.loc[:,'field_ID']):
            allfieldstars=PlatoSim_stars.loc[PlatoSim_stars['field_ID']==fieldid]
            #print(allfieldstars)
            field=SkyCoord(allfieldstars.iloc[0]['field_cen_RA']*u.deg,
                           allfieldstars.iloc[0]['field_cen_Dec']*u.deg)
            totake=np.argsort(starcat_scope.separation(field))<len(allfieldstars)
            #print(len(allfieldstars),np.sum(totake))
            idf=stars_all.loc[totake]
            idf['RA']=allfieldstars['RA'].values
            idf['Dec']=allfieldstars['Dec'].values
            idf['latitude']=starcat_scope[totake].transform_to('galactic').b.deg
            idf['longitude']=starcat_scope[totake].transform_to('galactic').l.deg
            idf['field_ID']=fieldid
            idf['field_cen_RA']=allfieldstars.iloc[0]['field_cen_RA']
            idf['field_cen_Dec']=allfieldstars.iloc[0]['field_cen_Dec']
            newstarcat=newstarcat.append(idf)
            starcat_scope=starcat_scope[~totake]
        '''
        #Splitting up to multiprocess
        #kic=kic[int(np.ceil((part/float(nparts))*kic.shape[0])):int(np.ceil(((part+1)/float(nparts))*kic.shape[0]))]
        #print("KIC is "+str(len(kic))+" in length")
        #ASSEMBLING BESANCON CAT FOR BLENDING:

        #Initialising isochrones:
        dart = Dartmouth_Isochrone()
        _=dart.radius(1.0,np.log10(4.5e9),0.0)
        stars_w_blends=Blends_np_PLATO(stars_all,
                                         parseBes(GetBesanconCat(hemisphere,'deep',npart,
                                                                 outdir='/home/hosborn/Plato_Simulations/BesanconModels2/')),
                                         
                                       deltamag_max_thresh=10)
        
        #GETTING HIERARCHICAL BINARIES FOR BLENDED AND TARGET STARS
        stars_all, binaries = get_multis_new(stars_w_blends.loc[stars_w_blends.deltamag<10],mag='Pmag')
        print('Calculating contamination')
        #This adds random angles to contaminant stars (for centroid reasons) and calculates Centre of Light for each target

        #Initialising LDs and GDs (sorting by FeH)
        print('getting LDs and GDs')
        stars_all['LD_1']=np.zeros(stars_all.shape[0])
        stars_all['LD_2']=np.zeros(stars_all.shape[0])
        stars_all['GD_1']=np.zeros(stars_all.shape[0])
        stars_all['albedo']=get_albedo(stars_all['Teff'].values)
        nFeHs=1
        FeHbins=np.percentile(np.nan_to_num(stars_all['FeH'].values), list(np.linspace(0,100.00,nFeHs+1)))
        for nFeH in range(nFeHs):
            FeHinterval=(stars_all['FeH'].values>=FeHbins[nFeH])*(stars_all['FeH'].values<FeHbins[nFeH+1])
            lds=getQuadLDs(stars_all.loc[FeHinterval,'Teff'].values,logg=stars_all.loc[FeHinterval,'logg'],
                           FeH=FeHbins[nFeH],band='V')
            stars_all.loc[FeHinterval,'LD_1']=lds[:,0]
            stars_all.loc[FeHinterval,'LD_2']=lds[:,1]
            stars_all.loc[FeHinterval,'GD_1']=getKeplerGDs(stars_all.loc[FeHinterval,'Teff'].values,
                                                          logg=stars_all.loc[FeHinterval,'logg'],
                                                          FeH=FeHbins[nFeH],
                                                          Fr='V',mod='ATLAS')
        print('#Making sure all stars have GD as this has been a problem...',np.sum(np.isnan(stars_all['GD_1'].values)))
        print('teffs',stars_all.loc[pd.isnull(stars_all['GD_1']),'Teff'].values)
        stars_all.loc[pd.isnull(stars_all['GD_1']),'GD_1']=getKeplerGDs(stars_all.loc[pd.isnull(stars_all['GD_1']),'Teff'].values,
                                                                 logg=4.0,FeH=0.0,Fr='V',mod='ATLAS')
        
        print(np.sum(np.isnan(stars_all['GD_1'].values)),' GD nans')
        stars_all.loc[pd.isnull(stars_all['GD_1']),'GD_1'] = getKeplerGDs(np.tile(5500,len(stars_all.loc[pd.isnull(stars_all['GD_1']),'GD_1'])),
                                                                        logg=4.0,
                                                                        FeH=0.0,Fr='V',mod='ATLAS')
        print(np.sum(np.isnan(stars_all['GD_1'].values)),' GD nans')
        #Getting beaming:
        stars_all['bfac']=Get_Beaming(stars_all['Teff'].values)

        print(binaries.shape,type(binaries),binaries.iloc[0])
        stars_all,binaries=Add_angles(binaries,stars_all)

        stars_all['sim_flux_rat']=np.zeros(stars_all.shape[0])  #Calculating the flux ratio to the target
        stars_all['sim_sum_flux']=np.zeros(stars_all.shape[0])
        stars_all['sim_contam']=np.zeros(stars_all.shape[0])    #And then the ratio of "other flux" to "my flux", ie the contamination
        stars_all[mag+'_corr']=stars_all[mag].values # Calculating the new blended V-mag of the target
        for st in pd.unique(stars_all['blend_parent']):
            target=stars_all.loc[str(st)]
            target=target.iloc[0] if len(target.shape)>1 else target #De-DataFraming to a Series
            indx=stars_all['blend_parent'].values.astype(str)==str(st)
            stars_all.loc[indx,'sim_flux_rat']=np.power(2.512,np.tile(target[mag],np.sum(indx))-stars_all.loc[indx,mag])
            stars_all.loc[indx,'sim_total_flux']=np.sum(stars_all.loc[indx,'sim_flux_rat'])
            stars_all.loc[indx,'sim_contam']=stars_all.loc[indx,'sim_flux_rat']/stars_all.loc[indx,'sim_sum_flux']

            stars_all.loc[target.name,mag+'_corr']=target[mag]-2.512*np.log(np.sum(stars_all.loc[indx,'sim_flux_rat']))

        #Spot filling factor, adapted from LAMOST II (more spots on cool stars)
        stars_all['alphaS']=np.clip(np.random.normal(0.2-(np.clip(stars_all.loc[:,'Teff'].values,2500,12000)-2500)/12000,1.0-(np.clip(stars_all.loc[:,'Teff'].values,2500,12000)/12000)**0.2),0.0,0.66)


        # ASSEMBLING OCCURRENCE RATES:
        
        # Multiplies by ~10 earthlike planet occurrence rates:
        newfunc = lambda P,Rp: (np.clip(P,50,400)/50)**0.66 * (4.0/(np.clip(Rp,0.8,4.0)))**0.66
        
        petigura = assemblePet(multfunc=newfunc)

        #allstars=#STARS WITHOUT CLOSE BINARIES AND WITH DELTAMAG>5
        print('Generating planets')
        tran_pls=GenPls(stars_all.loc[stars_all.deltamag<6],petigura)

        pickle.dump([stars_all,binaries,tran_pls],open(os.path.join(folder,'PlatoSim3_'+str(npart)+'_'+hemisphere+ext+'.pickle'),'wb'))
        eblist,pllist=None,None
    else:
        print('pickle file exists')
        unpk=pickle.load(open(os.path.join(folder,'PlatoSim3_'+str(npart)+'_'+hemisphere+ext+'.pickle'),'rb'))
        if len(unpk)==3:
            stars_all,binaries,tran_pls=unpk
            eblist,pllist=None,None
        elif len(unpk)==5:
            stars_all,binaries,tran_pls,eblist,pllist=unpk
    
    if (MakeLCs)&(startover) or (MakeLCs)&(~os.path.exists(os.path.join(folder,'Dips_in_LC_'+hemisphere+'_PlatoSim3'+ext+'.csv'))):
        #2-year, 10min cadence (this is a lotta points)
        lctimes=np.arange(0,365.25*2,25./86400)#25sec cadence, 2yrs
        if eblist is None:
            binaries['bpri']=abs(b_ecc_pri(binaries['ecc'].values,binaries['omega'].values,binaries['sma'].values,
                                           binaries['incl'].values,stars_all.loc[binaries['A'].values,'Rs'].values))
            binaries['bsec']=abs(b_ecc_sec(binaries['ecc'].values,binaries['omega'].values,binaries['sma'].values,
                                           binaries['incl'].values,stars_all.loc[binaries['B'].values,'Rs'].values))

            binaries['pri_ecl']=binaries['bpri'].values<(1+binaries.Rratio.values)
            binaries['sec_ecl']=binaries['bsec'].values<(1+binaries.Rratio.values**-1)

            eblist=binaries.loc[binaries.pri_ecl|binaries.sec_ecl]
            eblist.loc[:,'T0_sec']=eblist['T0']+eblist['P']/(np.pi*2)*(np.pi+2*np.arctan((eblist['ecc']*np.cos(eblist['omega']))/(1-eblist['ecc']**2)**0.5) + (2*(1-eblist['ecc']**2)**0.5*eblist['ecc']*np.cos(eblist['omega']))/(1-eblist['ecc']**2*np.sin(eblist['omega'])**2))

            #dropping EBs which contribute <10ppm of their target's light
            total_ap_flux = stars_all.loc[eblist['A'].values,'prop_of_flux_in_ap'].values+\
                            stars_all.loc[eblist['B'].values,'prop_of_flux_in_ap'].values
            print("dropping ",np.sum(total_ap_flux<1e-5)," ultra faint stars from eblist")
            eblist=eblist.drop(eblist.loc[total_ap_flux<1e-5].index.values)
        if pllist is None:
            tran_pls['bpri']=abs(b_ecc_pri(*tran_pls.loc[:,['ecc','omega','sma','incl','A_Rs']].values.swapaxes(0,1)))
            tran_pls['bsec']=abs(b_ecc_sec(*tran_pls.loc[:,['ecc','omega','sma','incl','B_Rs']].values.swapaxes(0,1)))

            tran_pls['pri_ecl']=tran_pls['bpri'].values<(1+tran_pls.Rratio.values)
            tran_pls['sec_ecl']=tran_pls['bsec'].values<(1+tran_pls.Rratio.values**-1)
            tran_pls['Ms']=tran_pls['Mp']*(Mearth/Msun)
            tran_pls['Rs']=tran_pls['Rp']*(Rearth/Rsun)

            pllist=tran_pls.loc[tran_pls.pri_ecl|tran_pls.sec_ecl]
            pllist['T0_sec']=pllist['T0']+pllist['P']/(np.pi*2)*(np.pi+2*np.arctan((pllist['ecc']*np.cos(pllist['omega']))/(1-pllist['ecc']**2)**0.5) + (2*(1-pllist['ecc']**2)**0.5*pllist['ecc']*np.cos(pllist['omega']))/(1-pllist['ecc']**2*np.sin(pllist['omega'])**2))

            #dropping EBs which contribute <100ppm of their target's light
            print(('prop_of_flux_in_ap' in tran_pls.columns),np.median(tran_pls['prop_of_flux_in_ap'].values))
            print("dropping ",np.sum(pllist['prop_of_flux_in_ap'].values<1e-4)," ultra faint planet hosts from eblist")
            pllist=pllist.drop(pllist.loc[pllist['prop_of_flux_in_ap'].values<1e-4].index.values)

            #re-dumping as these have changed:
            pickle.dump([stars_all,binaries,tran_pls,eblist,pllist],
                        open(os.path.join(folder,'PlatoSim3_'+str(npart)+'_'+hemisphere+ext+'.pickle'),'wb'))
        ##############
        # MAKING LCS #
        ##############
        alldiparr=pd.DataFrame()
        ntargets=0;ndips=0;nnodips=0;misseddips=0;missednodips=0
        poly=None
        
        #Dropping pointless columns:     
        dropcols=['A_B', 'A_B-V', 'A_H', 'A_J', 'A_K', 'A_Kepler', 'A_Mbol',
                  'A_Ms_parent', 'A_Rs_parent','A_Typ', 'A_U-B', 'A_V',
                  'A_V-I', 'A_V-K', 'A_V_corr', 'A_W1', 'A_W2', 'A_W3', 'A_[a/Fe]',
                  'A_file', 'A_flux','A_g', 'A_i', 'A_lamda',
                  'A_mux','A_muy', 'A_z']
                
        dropcols+=['B_'+dcol[2:] for dcol in dropcols]
                
        #print(np.sum(np.isnan(stars_all.GD_1.values)),' GD nans')
        for target in pd.unique(stars_all.blend_parent.values):
            #if not os.path.exists(os.path.join(folder
            #                                   ,"Lightcurves_"+str(int(npart))+'_'+hemisphere,
            #                                   str(target).zfill(8)+'.npy')):
            starsaroundtarg=stars_all.loc[stars_all.blend_parent==target]
            
            #if len(starsaroundtarg)>10:
            #    #Let's ignore the faintest stars if we have more than 9 blends:
            #    #faint_stars=starsaroundtarg.index.values[np.argsort(starsaroundtarg[mag].values)[10:]]
            
            diparr=pd.DataFrame()
            alllcs=lctimes[:]
            allnoise=lctimes[:]

            parent_rms_ppm,poly=getPmagRms(stars_all.loc[target,'Pmag_corr'],
                                           stars_all.loc[target,'Nscopes'],
                                           poly=poly)
            #print(starsaroundtarg)
            #print(starsaroundtarg.shape)
            anydip=False
            for blend in starsaroundtarg.iterrows():
                #Looping through all stars attached to the target to generate LCs.
                dip=False
                '''
                blenddiplc=[]
                blendnoiselc=[]
                if blend[1][mag]<(stars_all.loc[target,mag]+5):
                    #Adding variability:
                    inoiselc,noisedic=Noise_Single(blend[1],lctimes)
                    allnoise=np.vstack((allnoise, -2.5*np.log10(inoiselc) ))
                else:
                    allnoise=np.vstack((allnoise, np.zeros(len(lctimes)) ))
                    noisedic={key:np.nan for key in ['q','per','amp','rotstd','oscstd']}
                '''
                if blend[0] in eblist.B.values:
                    #print(blend[0]," =EB")
                    #getting LC for binary
                    Astar=starsaroundtarg.loc[eblist.loc[eblist.B==blend[0],'A'].values].iloc[0]
                    #diplc,idiparr=GenLC(Astar,blend[1],lctimes,system=eblist.loc[eblist.B==blend[0]].iloc[0])
                    _,idiparr=GenLC(Astar,blend[1],lctimes,system=eblist.loc[eblist.B==blend[0]].iloc[0],nodip=True)
                    #idiparr['i_lc']=len(alllcs)-1
                    idiparr['target']=target
                    idiparr['dip']=True
                    idiparr['rms_hr_parent']=parent_rms_ppm
                    #for key in noisedic:
                    #    idiparr[key]=noisedic[key]
                    diparr=diparr.append(idiparr.rename(blend[0]))
                    #blenddiplc+=[diplc]
                    dip=True
                if blend[0] in pllist.planetparent.values:
                    for ipl in pllist.loc[pllist.planetparent==blend[0]].iterrows():
                        #print(ipl,'=PL')
                        #getting LC for planet
                        #diplc,idiparr=GenLC(blend[1],ipl[1],lctimes)
                        _,idiparr=GenLC(blend[1],ipl[1],lctimes,nodip=True)
                        idiparr['A']=blend[0]
                        idiparr['B']=ipl[0]
                        idiparr['target']=target
                        idiparr['dip']=True
                        idiparr['rms_hr_parent']=parent_rms_ppm
                        #idiparr['i_lc']=len(alllcs)-1
                        #for key in noisedic:
                        #    idiparr[key]=noisedic[key]
                        diparr=diparr.append(idiparr.rename(ipl[0]))
                        #blenddiplc+=[diplc]
                        dip=True
                if not dip:
                    #Creating flat LC for bright but not-dipping blend
                    _,idiparr=GenLC(blend[1],None,lctimes,nodip=True)
                    #print(idiparr)
                    #idiparr['i_lc']=len(alllcs)-1
                    #alllcs=np.vstack((alllcs,np.zeros(len(lctimes))))
                    idiparr['target']=target
                    idiparr['dip']=False
                    idiparr['rms_hr_parent']=parent_rms_ppm
                    #for key in noisedic:
                    #    idiparr[key]=noisedic[key]
                    diparr=diparr.append(idiparr.rename(blend[0]))
                else:
                    anydip=True
                #elif dip:
                    #Cleaning up some stuff
                    #diparr['target']=target
                    #diparr['depthpri_obs']=np.array(diparr['depthpri_obs']).astype(float)
                    #diparr['depthsec_obs']=np.array(diparr['depthsec_obs']).astype(float)
                    #diparr.loc[diparr.depthpri_obs>1e100,'depthpri_obs']=0.0
                    #diparr.loc[diparr.depthsec_obs>1e100,'depthsec_obs']=0.0
                    #diparr['dippri_snr']=(diparr['depthpri_obs'].values*1e6)/(diparr['rms_hr_parent']/np.sqrt(24*diparr['durpri_theory']))
                    #diparr['dipsec_snr']=(diparr['depthsec_obs'].values*1e6)/(diparr['rms_hr_parent']/np.sqrt(24*diparr['dursec_theory']))
                    
                    #print(len(blenddiplc),blenddiplc)
                    #print(np.shape(alllcs),alllcs)
                    #Transferring to deltamag
                    #alllcs=np.vstack((alllcs,
                    #                  -2.5*np.log10(1.0+np.sum(np.vstack(blenddiplc)-1.0,axis=0) ) ))
            
            #Balancing dips and no dips based on the past number of 
            
            if anydip:
                diparr.loc[diparr['target']==target,'targ_has_dip']=True
                if np.random.random()>0.5:
                    diparr.loc[diparr['target']==target,'targ_inj_noise']=True
            else:
                diparr.loc[diparr['target']==target,'targ_has_dip']=False
                if np.random.random()>0.5:
                    diparr.loc[diparr['target']==target,'targ_inj_noise']=True
            
            print(str(ntargets)+'/'+str(len(pd.unique(stars_all.blend_parent.values))))

            #Dropping columns that aren't gonna be used:
            dropcols+=[col for col in diparr if 'B_A_' in col]
            diparr=diparr.drop(columns=[col for col in dropcols if col in diparr.columns])

            #Adding location of useful files:
            #if not os.path.isdir(os.path.join(folder,'LCs_'+hemisphere+'_PlatoSim3'+ext)):
            #    os.system('mkdir '+os.path.join(folder,'LCs_'+hemisphere+'_PlatoSim3'+ext))
            #diparr['dipfile']=os.path.join(folder,'LCs_'+hemisphere+'_PlatoSim3'+ext,
            #                               str(target)+'_'+hemisphere+'_PlatoSim3'+ext+'.npz')
            diparr['picklefile']=os.path.join(folder,'PlatoSim3_'+str(npart)+'_'+hemisphere+ext+'.pickle')
            alldiparr=alldiparr.append(diparr)
            '''
            if dip and np.max(diparr['dippri_snr'])>6.0:
                if 1==1:
                    #ndips==0 or np.random.random()>(1 / (1 + np.exp(-30*(ndips/(ndips+nnodips)-0.5)))):
                    #Saving dip only if no dips, or randomly according to sigmoid function
                    #This weights probability towards unbalanced class.
                    # IE if ndips<<nnodips, P(ndips)>>P(nnodips)
                    if np.random.random()>0.5:
                        #Using NOISE
                        np.savez_compressed(os.path.join(folder,'LCs_'+hemisphere+'_PlatoSim3'+ext,
                                             str(target)+'_'+hemisphere+'_PlatoSim3'+ext+'.npz'),
                                alllcs[1:]+allnoise[1:])
                        diparr['use_noise']=True
                        diparr['has_dip']=True
                    else:
                        np.savez_compressed(os.path.join(folder,'LCs_'+hemisphere+'_PlatoSim3'+ext,
                                             str(target)+'_'+hemisphere+'_PlatoSim3'+ext+'.npz'),
                                alllcs[1:])
                        #Removing noise terms:
                        for key in ['q','per','amp','rotstd','oscstd']:
                            diparr[key]=np.nan 
                        diparr['use_noise']=False
                        diparr['has_dip']=True
                    
                    alllcs=lctimes[:]
                    allnoise=lctimes[:]
                    alldiparr=alldiparr.append(diparr)
                    
                    ndips+=1
                else:
                    misseddips+=1
            else:
                #Not detectable dip or no dip.
                if 1==1:
                    #nnodips==0 or np.random.random()>(1 / (1 + np.exp(-30*(nnodips/(ndips+nnodips)-0.5)))):
                    #Saving nodip
                    if np.random.random()>0.5:
                        #Using NOISE
                        np.savez_compressed(os.path.join(folder,'LCs_'+hemisphere+'_PlatoSim3'+ext,
                                             str(target)+'_'+hemisphere+'_PlatoSim3'+ext+'.npz'),
                                alllcs[1:]+allnoise[1:])
                        diparr['use_noise']=True
                        diparr['has_dip']=False

                    else:
                        np.savez_compressed(os.path.join(folder,'LCs_'+hemisphere+'_PlatoSim3'+ext,
                                             str(target)+'_'+hemisphere+'_PlatoSim3'+ext+'.npz'),
                                alllcs[1:])
                        diparr['use_noise']=False
                        diparr['has_dip']=False
                        #Removing noise terms:
                        for key in ['q','per','amp','rotstd','oscstd']:
                            diparr[key]=np.nan 

                    alllcs=lctimes[:]
                    allnoise=lctimes[:]
                    alldiparr=alldiparr.append(diparr)

                    nnodips+=1
                else:
                    missednodips+=1
            '''
            #if (ndips+nnodips)%100==99:
            if ntargets%250==249:
                #print("ARRAY LENGTHS: dips - ",ndips," no dips - ",nnodips," missed dips - ",misseddips," missed no dips - ",missednodips)
                #Saving:
                alldiparr.to_csv(os.path.join(folder,'AllStars_'+hemisphere+str(npart)+'_PlatoSim3'+\
                                              '_part'+str(int(np.round((ntargets)/250)))+ext+'.csv'))
                alldiparr=pd.DataFrame()

            ntargets+=1
                
        #np.save(os.path.join(folder,"Lightcurves_"+'PlatoSim3_'+hemisphere,str(target).zfill(8)+'.npy'),lcs)
        alldiparr.to_csv(os.path.join(folder,'AllStars_'+hemisphere+str(npart)+'_PlatoSim3'+\
                                      '_part-1'+ext+'.csv'))
        print("Finished Creating lightcurves - "+str(ntargets)+" targets.")

def Noise_Single(targ,time,returnall=False):    
    #Correcting Age if this column is missing using logage or, in the worst case, using Target age:
    targ['Age'] = np.power(10,targ['logage']-9) if pd.isnull(targ['Age'])&(~pd.isnull(targ['logage'])) else targ['Age']
    targ['Age'] = 3 if pd.isnull(targ['Age']) else targ['Age']
    
    #Only generating variability for brightest blends:
    #osc=Oscillations(targ['Ms'],
    #                  targ['Teff'],
    #                  targ['Rs'],time)
    rot,qpcols=QPlc_fromTeff(targ['Teff'],targ['Age'],time)
    #print(np.shape(qpcols),np.shape(rot),np.shape(time))#,np.shape(osc))
    outdic={'q':qpcols[0][0],'per':qpcols[0][1],'amp':qpcols[0][2],
            'rotstd':np.std(rot.ravel()),'oscstd':0.0}#np.std(osc.ravel())}
    
    #return osc.ravel()+(rot.ravel()-1.0),outdic
    return rot.ravel(), outdic


def SelectFieldstars(platosimstarcat, stars_all, prop_dip = 0.5):
    #Combines catalogue of "pure" dips in PLATO with catalogue of PlatoSim3 fields
    # platosimstarcat - pandas dataframe output of GenerateSimFields
    # stars_all - pandas dataframe input to my astrophysical PLATO cat of all stars infield (targ+blends+EBs)
    # prop_dip - proportion of stars with dips
    ##### prop_var - proportion of input stars to have variability
    
    lctimes=np.arange(0,365.25*2,25./86400)#25sec cadence, 2yrs
    gaia_detn_prob=init_Gaia_Source()
    
    dip_targets=stars_all.loc[(stars_all.type=='target')&(stars_all.targ_has_dip.values==True)]
    nodip_targets=stars_all.loc[(stars_all.type=='target')&(stars_all.targ_has_dip.values==False)]
    
    #Getting dips:
    newstarcat=pd.DataFrame()
    for fieldid in pd.unique(platosimstarcat.loc[:,'field_ID']):
        allfieldstars=platosimstarcat.loc[platosimstarcat['field_ID']==fieldid]
        asdips=np.zeros(len(allfieldstars))
        asdips[np.random.choice(len(allfieldstars),int(np.ceil(len(allfieldstars)*prop_dip)),replace=False)]=1.0
        asdips=asdips.astype(bool)
        
        #Taking dips and non-dips near the latitude of the field:
        dips_dist2field=abs(dip_targets.latitude.values-allfieldstars.loc[asdips,'field_cen_lat'].values[0])
        dip_targets=dip_targets.iloc[np.argsort(dips_dist2field)]
        dipdf=dip_targets.iloc[:np.sum(asdips)]
        dip_targets=dip_targets.iloc[np.sum(asdips):]
        
        nodips_dist2field=abs(nodip_targets.latitude.values-allfieldstars.loc[asdips,'field_cen_lat'].values[0])
        nodip_targets=nodip_targets.iloc[np.argsort(nodips_dist2field)]
        nodipdf=nodip_targets.iloc[:np.sum(~asdips)]
        nodip_targets=nodip_targets.iloc[np.sum(~asdips):]
        
        #print(dipstotake)
        #print(np.argsort(abs(dip_targets.latitude.values-allfieldstars.loc[asdips].iloc[0]['field_cen_lat']))<np.sum(asdips))
        #print(np.argsort(abs(nodip_targets.latitude.values-allfieldstars.loc[~asdips].iloc[0]['field_cen_lat']))<np.sum(~asdips))
        #print(dip_targets.loc[dipstotake,'latitude'])
        #print(nodipstotake)
        #print(nodip_targets.loc[nodipstotake,'latitude'])
        
        #Look at only targets and those with dips:        
        dipdf['RA']=allfieldstars.loc[asdips,'ra'].values
        dipdf['Dec']=allfieldstars.loc[asdips,'dec'].values
        dipdf['latitude_old']=dipdf.latitude.values[:]
        dipdf['latitude']=allfieldstars.loc[asdips,'lat'].values
        dipdf['longitude_old']=dipdf.longitude.values[:]
        dipdf['longitude']=allfieldstars.loc[asdips,'long'].values
       
        dipdf['field_ID']=fieldid

        #Look at only targets and those with dips:
        nodipdf['RA']=allfieldstars.loc[~asdips,'ra'].values
        nodipdf['Dec']=allfieldstars.loc[~asdips,'dec'].values
        nodipdf['latitude_old']=nodipdf.latitude.values[:]
        nodipdf['latitude']=allfieldstars.loc[~asdips,'lat'].values
        nodipdf['longitude_old']=nodipdf.longitude.values[:]
        nodipdf['longitude']=allfieldstars.loc[~asdips,'long'].values
        nodipdf['field_ID']=fieldid
        for col in ['field_cen_ra','field_cen_long','field_cen_dec','field_cen_lat','scope_cen_ra','scope_cen_dec']:
            dipdf[col]=allfieldstars.iloc[0][col]
            nodipdf[col]=allfieldstars.iloc[0][col]
        newstarcat=newstarcat.append(dipdf)
        newstarcat=newstarcat.append(nodipdf)
    return newstarcat

def CombineCats(newstarcat, stars_all, ebs_all, pls_all, hemi, outfileloc, prop_dip = 0.5,lctimes=np.arange(0,365.25*2,25/86400),num_quarts=8,overwrite=True):
    gaia_detn_prob=init_Gaia_Source()
    
    if not os.path.isdir(outfileloc):
        os.system('mkdir '+outfileloc)
    for fieldid in np.unique(newstarcat.field_ID.values):
        print("FIELD ID:",fieldid)
        fieldtargs=newstarcat.loc[newstarcat.field_ID==fieldid]
        #Making folder:

        field_file_loc=os.path.join(outfileloc,'platosimfiles_'+str(fieldid).zfill(2)+'_'+hemi)
        gt_file_loc=os.path.join(outfileloc,'platosimfiles_groundtruth_'+str(fieldid).zfill(2)+'_'+hemi)
        if not os.path.isdir(field_file_loc):
            os.system('mkdir '+field_file_loc)
            os.system('mkdir '+field_file_loc+'/lcs')
        if not os.path.isdir(gt_file_loc):
            os.system('mkdir '+gt_file_loc)
        
        if not os.path.isfile(field_file_loc+'/'+str(fieldid).zfill(2)+'_starcat.txt'):
            field_targets=open(field_file_loc+'/'+str(fieldid).zfill(2)+'_starcat.txt','w')
            field_targets.write('#RA(deg)\tDEC(deg)\tVmag\tID\tKnown\n')
        else:
            field_targets=open(field_file_loc+'/'+str(fieldid).zfill(2)+'_starcat.txt','a')
        
        if not os.path.isfile(field_file_loc+'/'+str(fieldid).zfill(2)+'_Q1_varcat.txt'):
            for Q in np.arange(num_quarts)+1:
                exec('field_varblty_q'+str(Q+1)+'=open(field_file_loc+\'/\'+str(fieldid).zfill(2)+\'_Q'+str(Q+1)+'_varcat.txt\',\'w\')')
        else:
            for Q in np.arange(num_quarts):
                exec('field_varblty_q'+str(Q+1)+'=open(field_file_loc+\'/\'+str(fieldid).zfill(2)+\'_Q'+str(Q+1)+'_varcat.txt\',\'a\')')

        #Making platosim-accessible file. Headers: # RA (deg)  DEC (deg) Vmag ID Known
        nlc=int(fieldid)*100000
        for target in fieldtargs.iterrows():
            if not os.path.isfile(os.path.join(gt_file_loc,'groundtruth_'+str(fieldid).zfill(2)+'_'+hemi+'_'+target[0]+'.csv')) or overwrite:
                diparr=pd.DataFrame()
                starsaroundtarg=stars_all.loc[stars_all.blend_parent==target[0]]
                for blend in starsaroundtarg.iterrows():
                    ID=str(fieldid)+'_'+str(blend[0])
                    dip=False
                    allblendlcs=[]
                    
                    if pd.isnull(blend[1][['x_to_target','y_to_target']]).any():
                        
                        if blend[1]['orb_parent'] in starsaroundtarg.index.values and not pd.isnull(starsaroundtarg.loc[blend[1]['orb_parent'],['x_to_target','y_to_target']]).any():
                            x2targ=starsaroundtarg.loc[blend[1]['orb_parent'],'x_to_target']
                            y2targ=starsaroundtarg.loc[blend[1]['orb_parent'],'y_to_target']
                        elif blend[1]['orb_parent'] in starsaroundtarg.index.values:
                            print(starsaroundtarg.loc[blend[1]['orb_parent'],'orb_parent'])
                            print(starsaroundtarg.loc[blend[1]['orb_parent'],'orb_parent'] in starsaroundtarg.index)
                            if not pd.isnull(starsaroundtarg.loc[starsaroundtarg.loc[blend[1]['orb_parent'],'orb_parent'],['x_to_target','y_to_target']]).any():
                                x2targ=starsaroundtarg.loc[starsaroundtarg.loc[blend[1]['orb_parent'],'orb_parent'],'x_to_target']
                                y2targ=starsaroundtarg.loc[starsaroundtarg.loc[blend[1]['orb_parent'],'orb_parent'],'y_to_target']
                            else:
                                x2targ=0
                                y2targ=0
                        else:
                            #Parent not in df... :/
                            x2targ=0
                            y2targ=0                 
                    else:
                        x2targ=blend[1]['x_to_target']
                        y2targ=blend[1]['y_to_target']
                    print(ID,x2targ,y2targ,np.sqrt(x2targ**2+y2targ**2),np.sqrt(x2targ**2+y2targ**2)<50)
                    #50arcsecs = 3.5pixels = region around
                    if np.sqrt(x2targ**2+y2targ**2)<50 and (blend[1]['Pmag']-starsaroundtarg[target]['Pmag'])<10:
                        ra=target[1]['RA']+x2targ/3600.
                        dec=target[1]['Dec']+y2targ/3600.
                        Vmag= blend[1]['Pmag']
                        #Check if blend would be spotted by gaia:
                        if blend[0]==target[0]:
                            known=True
                        elif blend[1]['Pmag']<21:
                            known=np.random.random()<gaia_detn_prob(blend[1]['Pmag']-target[1]['Pmag'],np.sqrt(blend[1]['x_to_target']**2+blend[1]['y_to_target']**2))
                        else:
                            known=False

                        #If get noise (50% random)
                        if target[1]['targ_inj_noise'] and (blend[1]['Pmag']-starsaroundtarg[target]['Pmag'])<7.5:
                            inoiselc,noisedic=Noise_Single(blend[1],lctimes)
                            inoiselc=-2.5*np.log10(inoiselc)
                            allblendlcs+=[inoiselc]
                        else:
                            noisedic={}
                            allblendlcs+=[]
                        
                        # Getting LC:
                        if blend[0] in ebs_all.B.values:
                            Astar=starsaroundtarg.loc[ebs_all.loc[ebs_all.B==blend[0],'A'].values].iloc[0]
                            diplc,idiparr=GenLC(Astar,blend[1],lctimes,system=ebs_all.loc[ebs_all.B==blend[0]].iloc[0])
                            allblendlcs+=[-2.5*np.log10(diplc)]
                            idiparr['target']=target[0]
                            idiparr['dip']=True
                            idiparr['lcloc']=field_file_loc+'/'+ID+'.txt'
                            for key in noisedic:
                                idiparr[key]=noisedic[key]

                            diparr=diparr.append(idiparr.rename(blend[0]))
                            dip=True

                        if blend[0] in pls_all.planetparent.values and (blend[1]['Pmag']-starsaroundtarg[target]['Pmag'])<5:
                            for ipl in pls_all.loc[pls_all.planetparent==blend[0]].iterrows():
                                diplc,idiparr=GenLC(blend[1],ipl[1],lctimes)
                                allblendlcs+=[-2.5*np.log10(diplc)]
                                idiparr['A']=blend[0]
                                idiparr['B']=ipl[0]
                                idiparr['target']=target
                                idiparr['dip']=True
                                for key in noisedic:
                                    idiparr[key]=noisedic[key]

                                diparr=diparr.append(idiparr.rename(ipl[0]))
                            dip=True
                        if not dip:
                            _,idiparr=GenLC(blend[1],None,lctimes,nodip=True)
                            idiparr['target']=target[0]
                            idiparr['dip']=False
                            for key in noisedic:
                                idiparr[key]=noisedic[key]
                            diparr=diparr.append(idiparr.rename(blend[0]))
                        
                        if allblendlcs!=[]:
                            #Saving
                            lc=np.column_stack((lctimes,np.sum(allblendlcs,axis=0)))
                            lc[:,0]*=86400.0
                        if allblendlcs==[] or np.max(abs(lc[:,1]))<5e-6:
                            #BASICALLY FLAT, LETS NOT WRITE THIS LC...
                            diparr['targ_lc_loc']=lcloc
                            diparr['ID_in_varcat']=-1
                            #Writing to file:
                            field_targets.write('{:.5f}\t{:.5f}\t{:.2f}\t{}\t{}\n'.format(ra,dec,Vmag,str(nlc),str(known)))
                        else:
                            #Saving variability:
                            for Q in np.arange(num_quarts)+1:
                                low=int(np.clip(np.floor(0.125*(Q-1)*len(lc[:,0])-1),0,len(lc)))
                                high=int(np.clip(np.ceil(0.125*(Q)*len(lc[:,0])+1),0,len(lc)))
                                lc2save=lc[low:high,:]
                                lcloc=field_file_loc+'/lcs/'+ID+'_Q'+str(Q)+'.txt'
                                np.savetxt(lcloc,lc2save,fmt='%.2f %.6f')#{:.2f} {:.6f}')#{:06.2f}
                                exec('field_varblty_q'+str(Q)+'.write(\'{}\\t{}\\n\'.format(nlc,lcloc))')
                                diparr['targ_lc_loc']=lcloc
                                diparr['ID_in_varcat']=nlc
                            #Writing to file:
                            field_targets.write('{:.5f}\t{:.5f}\t{:.2f}\t{}\t{}\n'.format(ra,dec,Vmag,str(nlc),str(known)))

                        nlc+=1
                    else:
                        print('too far',np.sqrt(x2targ**2+y2targ**2),np.sqrt(x2targ**2+y2targ**2),
                              np.sqrt(x2targ**2+y2targ**2),np.sqrt(x2targ**2+y2targ**2)<50)
                diparr['targ_has_dip']=target[1]['targ_has_dip']
                diparr['targ_inj_noise']=target[1]['targ_inj_noise']
                diparr.to_csv(os.path.join(gt_file_loc,'groundtruth_'+str(fieldid).zfill(2)+'_'+hemi+'_'+target[0]+'.csv'))
            else:
                print(target, " - exists")
        field_targets.close()
        for Q in range(num_quarts)+1:
            exec('field_varblty_q'+str(Q)+'.close()')

        SavePlatosim3Python(fieldid, 
                            [fieldtargs.iloc[0]['scope_cen_ra'], fieldtargs.iloc[0]['scope_cen_dec']], 
                            [fieldtargs.iloc[0]['field_cen_ra'], fieldtargs.iloc[0]['field_cen_dec']],
                            field_file_loc,num_quarts)

        #Zipping folder:
        os.system('tar -zcvf '+field_file_loc+'./tar.gz '+field_file_loc)
        #os.system('rm -r '+field_file_loc)
    
    return newstarcat

def GenP5(morestars,npart,hemisphere,mag='Pmag',siglimit=5,maglimit=13,T=730.5):
    # Generating P5 sample given some stellar catalogue with Rs, Ms, rms_hr, etc
    # Uses the expected SNR of an Earth in the HZ or at 1yr period to select stars
    #
    
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
    morestars['ap_radius']=np.sqrt(morestars['ap_size'].values/np.pi)*15
    #np.random.randint(np.ceil(sigmoid(morestars[mag],yshift=3*1.35)),np.floor(KepLikeApSize(morestars[mag])))

    morestars['besanson_index']=morestars.index.values
    morestars.set_index(np.array([hemisphere[0]+str(npart)+'_'+str(i).zfill(8) for i in np.arange(len(morestars.index.values))]),inplace=True)

    #Also includes all of P1
    p5condn=((morestars['HZ_SNR']>(siglimit))|(morestars['Earth_SNR']>(siglimit)))&(morestars[mag]<maglimit)
    
    return morestars.loc[p5condn]


def SavePlatosim3Python(fieldID, scoperadec, fieldradec, field_file_loc, num_quarts=8):
    '''
    # Saving PlatoSim3 files.
    #
    #INPUTS:
    # - fieldID
    # - scoperadec - astropy SkyCoord for the spacecraft
    # - fieldradec - astropy SkyCoord for the centre of the field
    # - field_file_loc - file location to save stuff
    # - num_quarts - number of quarters to generate.
    '''

    pyfile_contents  =  Path('PlatoSimPythonfileHeader.py').read_text()
    #yamlfile_contents = Path('PlatoSimPythonfileHeader.yaml').read_text()

    for group in np.arange(0,4)+1:
        for scope in np.arange(0,6)+1:
            for quart in np.arange(0,num_quarts)+1:
                #Check if it is indeed observed:
                observed=InFoV(fieldradec[0],fieldradec[1],
                               scoperadec[0],scoperadec[1],
                               Q=quart,group=group)
                if observed[group] not in ['CCD_gap','Outside_FoV']:
                    pyname=os.path.join(field_file_loc,fieldID + "_Q{0:1d}_group{1:1d}_camera{2:1d}_run.py".format(quart, group, scope))

                    with open(pyname,'w+') as pyfile:
                        pyfile.write(pyfile_contents.format(outputloc=field_file_loc,
                                                            fieldID=fieldID,
                                                            group=group , scope=scope, quart=quart,
                                                            raPlatform=scoperadec[0], decPlatform=scoperadec[1],
                                                            raField=fieldradec[0], decField=fieldradec[1])
                                    )
                    '''yamlname=os.path.join(field_file_loc,
                                          fieldID+'_Q'+str(quart)+'_group'+str(group)+'_camera'+str(scope)+'_run.yaml')
                    with open(yamlname,'w+') as yamlfile:
                        yamlfile.write(yamlfile_contents.format(raPlatform=scoperadec[0],decPlatform=scoperadec[1],
                                                                starcat = os.path.join(field_file_loc,fieldID+'_Q'+str(quart)+'_starcat.txt'),
                                                                varcat = os.path.join(field_file_loc,fieldID+'_Q'+str(quart)+'_varcat.txt'),
                                                                outputloc=field_file_loc)
                                   )
    '''
    #linking the inputfiles directory to the eventual file in order to import default simulations + scripts               
    if not os.path.exists(field_file_loc+'/inputfiles'):
        print('RUNNING: \"ln -s '+os.getenv("PLATO_PROJECT_HOME")+"/inputfiles "+field_file_loc+'\"')
        os.system('ln -s '+os.getenv("PLATO_PROJECT_HOME")+"/inputfiles "+field_file_loc)
    
    #linking the FieldGen directory to the eventual file in order to import the rebin_hdf5 function
    if not os.path.exists(field_file_loc+'/PlatoSim_FieldGen'):
        dirname, filename = os.path.split(os.path.abspath(__file__))
        print('RUNNING: \"ln -s '+dirname+" "+field_file_loc+'\"')
        os.system('ln -s '+dirname+" "+field_file_loc)

    #["group" , "scope", "quart", "raPlatform", "decPlatform", "raField", "decField"]

        
if __name__=="__main__":
    #GenDipCatPLATO.
    #Arguments:
    # hemisphere = North or South
    # npart = int from 0 to 9
    # folder =  output folder location 
    # nmult = fraction of target stars to simulate (can be >1)
    # startover = int 0 or 1 to start all again
    # ext = string to add to output files
    if len(sys.argv)==2:
        args=[' ']+sys.argv[1].split(' ')
    else:
        args=sys.argv
    print(args)
    print(len(args))
    if len(args)>=7:
        print('hemisphere=',args[1],'npart=',args[2],'folder=',args[3],
              'nmult=',args[4],'startover=',str(bool(int(args[5]))),'extension=',args[6])
        _=GenDipCatPLATO(hemisphere=args[1],npart=int(args[2]),folder=args[3],
                         nmult=float(args[4]),startover=bool(int(args[5])),ext=args[6])
    elif len(args)==6:
        print('hemisphere=',args[1],'npart=',args[2],'folder=',args[3],
              'nmult=',args[4],'startover=',str(bool(int(args[5]))))
        _=GenDipCatPLATO(hemisphere=args[1],npart=int(args[2]),folder=args[3],
                         nmult=float(args[4]),startover=bool(int(args[5])))
    elif len(args)==5:
        print('hemisphere=',args[1],'npart=',args[2],'folder=',args[3],'nmult=',args[4])
        _=GenDipCatPLATO(hemisphere=args[1],npart=int(args[2]),folder=args[3],nmult=float(args[4]))
    elif len(args)==4:
        _=GenDipCatPLATO(hemisphere=args[1],npart=int(args[2]),folder=args[3])
        
        
