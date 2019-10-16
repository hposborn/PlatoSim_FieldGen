##################################################################
#                                                                #
#                     COMBINING EVERYTHING                       #
#                                                                #
##################################################################
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

from LcGen import *

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def GenDipCatPLATO(hemisphere,npart,folder='/home/hosborn/PLATO/Plato_Sims',MakeLCs=True,mag='Pmag',
                   startover=True,nmult=1,peturb_cat=False,ext=''):

    # Generate catalogue of "Dips" in PLATO.
    # hemisphere = South or North
    # npart = 0 to 8
    # MakeLCs = whether to built LCs
    # mag = base magnitude to use
    # Startover = ignore previous files
    # nmult = factor to multiply number of stars. EG to create 100 more stars than normal, put 100
    # peturb_cat = whether to add fake noise into stellar distributions

    print("Running "+hemisphere+'_'+str(npart))
    
    if not os.path.isdir(folder):
        os.system('mkdir '+folder)
    #ASSEMBLING Input Catalogue:
    #Using nmult=100 to get 100x more stars and therefore a larger sample.

    #GETTING BESANCON BLENDED STARS:
    if not os.path.exists(os.path.join(folder,str(int(npart))+'_'+hemisphere+ext+'.pickle')) or startover:
        
        stars_all=assembleP5_new(hemisphere,npart,nmult=nmult,ext=ext,outdir='/home/hosborn/Plato_Simulations/BesanconModels2/')

        #Splitting up to multiprocess
        #kic=kic[int(np.ceil((part/float(nparts))*kic.shape[0])):int(np.ceil(((part+1)/float(nparts))*kic.shape[0]))]
        #print("KIC is "+str(len(kic))+" in length")
        #ASSEMBLING BESANCON CAT FOR BLENDING:

        #Initialising isochrones:
        dart = Dartmouth_Isochrone()
        _=dart.radius(1.0,np.log10(4.5e9),0.0)
        stars_all=Blends_np_PLATO(stars_all,
                                  parseBes(GetBesanconCat(hemisphere,'deep',npart,outdir='/home/hosborn/Plato_Simulations/BesanconModels2/')),
                                  deltamag_max_thresh=10)
        
        #GETTING HIERARCHICAL BINARIES FOR BLENDED AND TARGET STARS
        stars_all, binaries = get_multis_new(stars_all.loc[stars_all.deltamag<10],mag='Pmag')
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


        #ASSEMBLING OCCURRENCE RATES:
        petigura = assemblePet()

        #allstars=#STARS WITHOUT CLOSE BINARIES AND WITH DELTAMAG>5
        print('Generating planets')
        tran_pls=GenPls(stars_all.loc[stars_all.deltamag<6],petigura)

        pickle.dump([stars_all,binaries,tran_pls],open(os.path.join(folder,str(int(npart))+'_'+hemisphere+ext+'.pickle'),'wb'))
        eblist,pllist=None,None
    else:
        print('pcikle file exists')
        unpk=pickle.load(open(os.path.join(folder,str(int(npart))+'_'+hemisphere+ext+'.pickle'),'rb'))
        if len(unpk)==3:
            stars_all,binaries,tran_pls=unpk
            eblist,pllist=None,None
        elif len(unpk)==5:
            stars_all,binaries,tran_pls,eblist,pllist=unpk
    
    if (MakeLCs)*(startover) or (MakeLCs)*(~os.path.exists(os.path.join(folder,'Dips_in_LC_'+hemisphere+'_'+str(int(npart))+ext+'.csv'))):
        #2-year, 10min cadence (this is a lotta points)
        lctimes=np.arange(0,365.25*2,600./86400)#10min cadence, 2yrs
        if eblist is None:
            binaries['bpri']=abs(b_ecc_pri(binaries['ecc'].values,binaries['omega'].values,binaries['sma'].values,
                                           binaries['incl'].values,stars_all.loc[binaries['A'].values,'Rs'].values))
            binaries['bsec']=abs(b_ecc_sec(binaries['ecc'].values,binaries['omega'].values,binaries['sma'].values,
                                           binaries['incl'].values,stars_all.loc[binaries['B'].values,'Rs'].values))

            binaries['pri_ecl']=binaries['bpri'].values<(1+binaries.Rratio.values)
            binaries['sec_ecl']=binaries['bsec'].values<(1+binaries.Rratio.values**-1)

            eblist=binaries.loc[binaries.pri_ecl+binaries.sec_ecl]
            eblist['T0_sec']=eblist['T0']+eblist['P']/(np.pi*2)*(np.pi+2*np.arctan((eblist['ecc']*np.cos(eblist['omega']))/(1-eblist['ecc']**2)**0.5) + (2*(1-eblist['ecc']**2)**0.5*eblist['ecc']*np.cos(eblist['omega']))/(1-eblist['ecc']**2*np.sin(eblist['omega'])**2))

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

            pllist=tran_pls.loc[tran_pls.pri_ecl+tran_pls.sec_ecl]
            pllist['T0_sec']=pllist['T0']+pllist['P']/(np.pi*2)*(np.pi+2*np.arctan((pllist['ecc']*np.cos(pllist['omega']))/(1-pllist['ecc']**2)**0.5) + (2*(1-pllist['ecc']**2)**0.5*pllist['ecc']*np.cos(pllist['omega']))/(1-pllist['ecc']**2*np.sin(pllist['omega'])**2))

            #dropping EBs which contribute <100ppm of their target's light
            print(('prop_of_flux_in_ap' in tran_pls.columns),np.median(tran_pls['prop_of_flux_in_ap'].values))
            print("dropping ",np.sum(pllist['prop_of_flux_in_ap'].values<1e-4)," ultra faint planet hosts from eblist")
            pllist=pllist.drop(pllist.loc[pllist['prop_of_flux_in_ap'].values<1e-4].index.values)

            #re-dumping as these have changed:
            pickle.dump([stars_all,binaries,tran_pls,eblist,pllist],
                        open(os.path.join(folder,str(int(npart))+'_'+hemisphere+ext+'.pickle'),'wb'))

        alldiparr=pd.DataFrame()
        allnodiparr=pd.DataFrame()
        alllcs=lctimes[:]
        alldiplcnoise=lctimes[:]
        allnodiplcnoise=lctimes[:]
        ndips=0;nnodips=0
        poly=None
        
        #Dropping pointless columns:     
        dropcols=['A_B', 'A_B-V', 'A_DECJ2000', 'A_H', 'A_J', 'A_K', 'A_Kepler', 'A_Mbol',
                  'A_RAJ2000', 'A_Ms_parent', 'A_Rs_parent','A_Typ', 'A_U-B', 'A_V',
                  'A_V-I', 'A_V-K', 'A_V_corr', 'A_W1', 'A_W2', 'A_W3', 'A_[a/Fe]',
                  'A_file', 'A_flux','A_g', 'A_i', 'A_lamda', 'A_lat_subbin',
                  'A_mux','A_muy', 'A_z']
                
        dropcols+=['B_'+dcol[2:] for dcol in dropcols]
        
        #print(np.sum(np.isnan(stars_all.GD_1.values)),' GD nans')
        for target in pd.unique(stars_all.blend_parent.values):
            #if not os.path.exists(os.path.join(folder
            #                                   ,"Lightcurves_"+str(int(npart))+'_'+hemisphere,
            #                                   str(target).zfill(8)+'.npy')):
            starsaroundtarg=stars_all.loc[stars_all.blend_parent==target]
            
            
            
            diparr=pd.DataFrame()
            dip=False
            
            parent_rms_ppm,poly=getPmagRms(stars_all.loc[target,'Pmag_corr'],
                                           stars_all.loc[target,'Nscopes'],
                                           poly=poly)
            starsaroundtarg['rms_hr_corr']=parent_rms_ppm
            
            lcnoise,newcols=Noise(starsaroundtarg,lctimes)
            #print("Noise Columns:",starsaroundtarg.shape,np.shape(newcols))
            starsaroundtarg['rot_Q']=newcols[:,0]
            starsaroundtarg['rot_per']=newcols[:,1]
            starsaroundtarg['rot_amp']=newcols[:,2]
            starsaroundtarg['rot_std']=newcols[:,3]
            starsaroundtarg['osc_std']=newcols[:,4]
            lcs=[]
            for blend in starsaroundtarg.iterrows():
                #Looping through all stars attached to the target to generate LCs.
                if blend[0] in eblist.B.values:
                    print(blend[0]," =EB")
                    #getting LC for binary
                    Astar=starsaroundtarg.loc[eblist.loc[eblist.B==blend[0],'A']].iloc[0]
                    diplc,idiparr=GenLC(Astar,blend[1],lctimes,system=eblist.loc[eblist.B==blend[0]].iloc[0])
                    diparr=diparr.append(idiparr.rename(blend[0]))
                    lcs+=[diplc]
                    dip=True
                if blend[0] in pllist.planetparent.values:
                    for ipl in pllist.loc[pllist.planetparent==blend[0]].iterrows():
                        #getting LC for planet
                        diplc,idiparr=GenLC(blend[1],ipl[1],lctimes)
                        idiparr['A']=blend[0]
                        idiparr['B']=ipl[0]
                        diparr=diparr.append(idiparr.rename(ipl[0]))
                        lcs+=[diplc]
                        dip=True
                if np.isnan(blend[1]['GD_1']):
                    print(blend[0],'has GD nan')
                #else:
                #    print(blend[0],' not in ',pllist.planetparent.values)
            if dip:
                alllcs=np.vstack((alllcs, 1.0+np.sum(np.column_stack((lcs))-1.0,axis=1) ))
                alldiplcnoise=np.vstack((alldiplcnoise, lcnoise))
                diparr['i_lc']=len(alllcs)
                diparr['target']=target
                
                diparr['rms_hr_parent']=parent_rms_ppm
                
                diparr['depthpri_obs']=np.array(diparr['depthpri_obs']).astype(float)
                diparr['depthsec_obs']=np.array(diparr['depthsec_obs']).astype(float)
                
                diparr.loc[diparr.depthpri_obs>1e100,'depthpri_obs']=0.0
                diparr.loc[diparr.depthsec_obs>1e100,'depthsec_obs']=0.0
                diparr['dippri_snr']=(diparr['depthpri_obs'].values*1e6)/(diparr['rms_hr_parent']/np.sqrt(24*diparr['durpri_theory']))
                diparr['dipsec_snr']=(diparr['depthsec_obs'].values*1e6)/(diparr['rms_hr_parent']/np.sqrt(24*diparr['dursec_theory']))

                alldiparr=alldiparr.append(diparr.loc[:])

                if ndips%500==499:
                    print(str(ndips)+'/'+str(len(pd.unique(stars_all.blend_parent.values))))
                    #1000 objects. Need to split arrays up.
                    
                    #Dropping columns that aren't gonna be used:
                    dropcols+=[col for col in alldiparr if 'B_A_' in col]
                    alldiparr=alldiparr.drop(columns=[col for col in dropcols if col in alldiparr.columns])

                    #Adding location of useful files:
                    alldiparr['dipfile']=os.path.join(folder,'Dips_in_LC_'+hemisphere+'_'+str(int(npart))+\
                                                      '_part'+str(int(np.round(ndips/500)))+ext+'.csv')
                    alldiparr['picklefile']=os.path.join(folder,str(int(npart))+'_'+hemisphere+ext+'.pickle')
                    
                    #Saving:
                    alldiparr.to_csv(os.path.join(folder,
                                                  'Dips_in_LC_'+hemisphere+'_'+str(int(npart))+\
                                                  '_part'+str(int(np.round(ndips/500)))+ext+'.csv'))
                    np.save(os.path.join(folder,
                                         'Dips_in_LC_'+hemisphere+'_'+str(int(npart))+\
                                         '_part'+str(int(np.round(ndips/500)))+ext+'.npy'),
                            np.dstack((alllcs,alldiplcnoise)))
                    alldiparr=pd.DataFrame()
                    alllcs=lctimes[:]
                    alldiplcnoise=lctimes[:]
                ndips+=1
            if not dip:
                #Pure stellar noise - saving to 
                allnodiplcnoise=np.vstack((allnodiplcnoise, lcnoise))
                nodipdf=starsaroundtarg.loc[target]
                nodipdf['i_lc']=len(allnodiplcnoise)
                allnodiparr=allnodiparr.append(nodipdf)
                if nnodips%500==499:
                    allnodiparr=allnodiparr.drop(columns=[col[2:] for col in dropcols if col[2:] in allnodiparr.columns])

                    allnodiparr['dipfile']=os.path.join(folder,'NoDips_in_LC_'+hemisphere+'_'+str(int(npart))+\
                                                        '_part'+str(int(np.round(nnodips/500)))+ext+'.csv')
                    allnodiparr['picklefile']=os.path.join(folder,str(int(npart))+'_'+hemisphere+ext+'.pickle')
                    allnodiparr.to_csv(os.path.join(folder,
                                       'NoDips_in_LC_'+hemisphere+'_'+str(int(npart))+\
                                       '_part'+str(int(np.round(ndips/500)))+ext+'.csv'))

                    np.save(os.path.join(folder,
                                         'NoDips_in_LC_'+hemisphere+'_'+str(int(npart))+\
                                         '_part'+str(int(np.round(ndips/500)))+ext+'.npy'),
                            allnodiplcnoise)
                    allnodiparr=pd.DataFrame()
                    allnodiplcnoise=lctimes[:]
                    
                nnodips+=1
        #Saving the final versions of the dip and no-dip files:
        
        dropcols+=[col for col in alldiparr if 'B_A_' in col]
        alldiparr=alldiparr.drop(columns=[col for col in dropcols if col in alldiparr.columns])
        allnodiparr=allnodiparr.drop(columns=[col[2:] for col in dropcols if col[2:] in allnodiparr.columns])
        
        #Saving LCs:
        np.save(os.path.join(folder,'Dips_in_LC_'+hemisphere+'_'+str(int(npart))+ext+'.npy'),
                np.dstack((alllcs,alldiplcnoise)))
        #Saving dip-less LCs:
        np.save(os.path.join(folder,'NoDips_in_LC_'+hemisphere+'_'+str(int(npart))+ext+'.npy'),allnodiplcnoise)
        alldiparr['dipfile']=os.path.join(folder,'Dips_in_LC_'+hemisphere+'_'+str(int(npart))+ext+'.csv')
        alldiparr['picklefile']=os.path.join(folder,str(int(npart))+'_'+hemisphere+ext+'.pickle')
        #np.save(os.path.join(folder,"Lightcurves_"+str(int(npart))+'_'+hemisphere,str(target).zfill(8)+'.npy'),lcs)
        alldiparr.to_csv(os.path.join(folder,'Dips_in_LC_'+hemisphere+'_'+str(int(npart))+ext+'.csv'))
        allnodiparr['dipfile']=os.path.join(folder,'NoDips_in_LC_'+hemisphere+'_'+str(int(npart))+ext+'.csv')
        allnodiparr['picklefile']=os.path.join(folder,str(int(npart))+'_'+hemisphere+ext+'.pickle')
        allnodiparr.to_csv(os.path.join(folder,'NoDips_in_LC_'+hemisphere+'_'+str(int(npart))+ext+'.csv'))
        print("Finished Creating lightcurves - "+str(ndips)+" with dips and "+str(nnodips)+" without.")

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
        
        