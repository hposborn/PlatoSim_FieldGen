## Simplifying simulations.

# Generate field for PlatoSim3
# Populate with choices:
#  - choose star from Besancon "wide" cat
#  - if prob < 50% inject transiting planet(s)
#  - if prob < 20% inject background TPL/BEB
#  - if prob < 10% inject EB
#  - if prob < 50%, add realistic stellar noise
#  - if prob < 50% have blends
import numpy as np
import pandas as pd
from LcGen import *
from GenLcsForPlatoSim3 import *
import traceback
import contextlib

#Make field

def FieldGen(hemisphere='North',N_fields=1,outfileloc='/data/PLATO/Sims',
              num_quarts=8,MP=False,ext='',overwrite=False,bright_limit=6.1,
              contam_prop=0.5,var_prop=0.5):
    '''
    #
    # This is the master function for PlatoSim_FieldGen
    #
    #INPUTS:
    # - hemisphere    Hemisphere to take stars for
    # - N_fields      Number of fields to generate
    # - outfileloc:   folder location at which to save all files
    # - num_quarts:   number of quarters to generate
    # - MP:           Multiprocess? (boolean)
    # - ext:          File extension to add to every filename
    # - overwrite:    Overwrite previous files? (boolean)
    # - bright_limit: Cut stars brighter than this limit
    # - contam_prop:  Proportion of target stars to inject contaminant blends
    # - var_prop:     Proportion of target stars to inject stellar variability into
    #
    #OUTPUTS:
    # A folder in form '[outfileloc]/platosimfiles_[fieldID][ext]_[hemisphere]/'
    # A csv file for each input type (PL, EB, etc) '[fieldID][ext]_generated_[input type]s.csv'
    # A csv composed of all input types and blends '[fieldID][ext]_final_fieldcat.csv'
    # The starcat ('[fieldID][ext]_starcat.txt') and varcat ('[fieldID][ext]_Q[quarter]_varcat.txt') text files used by PlatoSim3
    # A runnable python script in the format '[fieldID][ext]_Q[quarter]_group[group]_camera[camera]_run.py'
    # An 'lcs' folder containing saved lightcurves for each star
    # Links to the "input_files" and "python" directorys in PlatoSim3 which contain necessary python scripts to import
    '''
    
    #Generating field(s)
    if hemisphere[0]=='S':
        #SOUTH
        scope_cen=SkyCoord(253*u.deg, -30*u.deg,frame='galactic')#.transform_to(ICRS)
    else:
        scope_cen=SkyCoord(65*u.deg, 30*u.deg,frame='galactic')#.transform_to(ICRS)
    stars,scope_frame = GenerateSimFields(scope_cen.l, scope_cen.b, 
                                          Nfields=N_fields, Angle_with_RADEC=0*u.deg, FoV_square=32.5*u.deg)
    #starcat_scope = SkyCoord(stars_all.longitude*u.deg,stars_all.latitude*u.deg,frame='galactic').transform_to(ICRS)
    
    for fieldid in pd.unique(stars.loc[:,'field_ID']):
        #looping through each field.
        field_file_loc=os.path.join(outfileloc,'platosimfiles_'+str(fieldid).zfill(2)+ext+'_'+hemisphere)
        print("field_file_loc:",field_file_loc)
        if not os.path.isdir(field_file_loc) or overwrite:
            if os.path.isdir(field_file_loc):
                os.system('rm -r '+field_file_loc)
            os.system('mkdir '+field_file_loc)
            os.system('mkdir '+field_file_loc+'/lcs')            
        
        labeldic={'BPL':[0.93,1.0],'EB':[0.83,0.93],'BEB':[0.70,0.83],'PL':[0.38,0.70],'NA':[0,0.38]}
        if not os.path.exists(os.path.join(field_file_loc,str(fieldid).zfill(2)+ext+'_generated_field')) or overwrite:
            allfieldstars=stars.loc[stars['field_ID']==fieldid]
            #labeldic={'BPL':[0.95,1.0],'EB':[0.825,0.95],'BEB':[0.65,0.825],'PL':[0.35,0.65],'NA':[0,0.35]}
            allfieldstars['type']='target'
            allfieldstars['system_label']=np.tile('',len(allfieldstars))
            randix=np.random.random(len(allfieldstars))
            for lab in labeldic:
                allfieldstars.loc[(randix<=labeldic[lab][1])*(randix>labeldic[lab][0]),'system_label']=lab
            #
            #for key in ['rot_Q','rot_per','rot_amp',"rot_std","osc_std"]:
            #    allfieldstars[key]=np.tile(np.nan,len(allfieldstars))
            allfieldstars['VAR']=np.tile(False,len(allfieldstars))
            allfieldstars['LC']=np.tile(False,len(allfieldstars))
            allfieldstars.to_csv(os.path.join(field_file_loc,str(fieldid).zfill(2)+ext+'_generated_field.csv'))
            field_cen_ra=np.median(allfieldstars['field_cen_ra'])
            field_cen_dec=np.median(allfieldstars['field_cen_dec'])

            BesLats=np.arange(6,54.01,4.8)
            nlat=np.argmin(abs(BesLats-np.median(allfieldstars['field_cen_lat'])))
            BesFile=parseBes(GetBesanconCat(hemisphere,'wide',nlat,
                                            outdir='/home/hosborn/Plato_Simulations/BesanconModels2/'))
            BesFile=BesFile.loc[BesFile.Pmag>bright_limit]
            if len(BesFile)<len(allfieldstars):
                #If we need more besancon stars, we can open the next file along
                newnlat=nlat-1 if nlat>5 else nlat+1
                BesFile=BesFile.append(parseBes(GetBesanconCat(hemisphere,'wide',newnlat,
                                                               outdir='/home/hosborn/Plato_Simulations/BesanconModels2/')))
            #allfieldstars=pd.concat([allfieldstars,BesFile.iloc[np.random.choice(len(BesFile),len(allfieldstars))]],axis=1)

            BesFile['rms_hr']=getPmagRms(BesFile['Pmag'],
                                           np.tile(np.median(allfieldstars['Nscopes']),len(BesFile['Pmag'])))[0]

            BesFile=GenP5(BesFile,nlat,hemisphere,siglimit=3.0) #getting the likely candidate target stars from this file
            BesFile=BesFile.drop(['Earth_SNR','Earth_depth','Earth_dur','Earth_Ntrs','HZ_dur','HZ_per','HZ_Ntrs','HZ_SNR'],axis=1)

            BesFile=StellarStuff(BesFile)
            #Randomly assigning labels (eg planet/EB/BEB/PL/nothing)

            BGstars=parseBes(GetBesanconCat(hemisphere,'deep',nlat,
                                            outdir='/home/hosborn/Plato_Simulations/BesanconModels2/'))
            BGstars=StellarStuff(BGstars.sort_values('Pmag'))
            BGstars['rms_hr']=getPmagRms(BGstars['Pmag'],
                                           np.tile(np.median(allfieldstars['Nscopes']),len(BGstars)))[0]
            print(allfieldstars.shape,allfieldstars.columns,
                  [str(lab)+":"+str(list(allfieldstars['system_label'].values).count(lab)) for lab in np.unique(allfieldstars['system_label'])])

            if MP:
                import multiprocessing as mp
                '''
                pool = mp.Pool(5)
                results = [pool.apply(getDip, args=(diptype,allfieldstars,BesFile,
                                                    BGstars,field_file_loc,num_quarts)) for diptype in ['PL','EB','BEB','BPL','NA']]
                #[pool.apply(getDip, args=(diptype,allfieldstars,BesFile,
                #                                    BGstars,field_file_loc,num_quarts)) 
                #           for diptype in ['PL','EB','BEB','BPL','NA']]
                pool.close()
                finalfieldstars = pd.concat(results)
                '''

                '''
                output=mp.Queue()
                #print("BGstars",BGstars.iloc[0],type(BGstars))
                #Generating lightcurves for all using multiprocessing

                processes = [mp.Process(target=getDip,args=[diptype,allfieldstars,BesFile,
                                                            BGstars,field_file_loc,num_quarts,])
                             for diptype in ['PL','EB','BEB','BPL','NA']]
                for p in processes:
                    p.start()
                print("joining MPs")
                for p in processes:
                    p.join()
                finalfieldstars = pd.concat([output.get() for p in processes])
                '''

                pool = mp.Pool(5)
                with contextlib.closing(mp.Pool(8)) as pool:
                    for diptype in ['PL','EB','BEB','BPL','NA']:
                        pool.apply_async(getDip, args=(diptype,allfieldstars,BesFile,BGstars,field_file_loc,str(fieldid).zfill(2)+ext,num_quarts))
                pool.join()

                finalfieldstars = pd.concat([pd.DataFrame.from_csv(os.path.join(field_file_loc,str(fieldid).zfill(2)+ext+'_generated_'+diptype.lower()+'s.csv')) for diptype in ['PL','EB','BEB','BPL','NA']])
            else:
                #Getting EBs:
                if not os.path.exists(os.path.join(field_file_loc,str(fieldid).zfill(2)+ext+'_generated_ebs.csv')):
                    ebfieldstars=getEBs(allfieldstars.loc[allfieldstars['system_label']=='EB'],BesFile,
                                    field_file_loc,num_quarts=num_quarts)
                    ebfieldstars.to_csv(os.path.join(field_file_loc,str(fieldid).zfill(2)+ext+'_generated_ebs.csv'))

                else:
                    print("Loading Generated EB stars from file")
                    ebfieldstars=pd.DataFrame.from_csv(os.path.join(field_file_loc,str(fieldid).zfill(2)+ext+'_generated_ebs.csv'))

                #Getting BPLs:
                if not os.path.exists(os.path.join(field_file_loc,str(fieldid).zfill(2)+ext+'_generated_bpls.csv')):
                    bplfieldstars=getBPLs(allfieldstars.loc[allfieldstars['system_label']=='BPL'],BesFile,
                                          BGstars,field_file_loc,num_quarts=num_quarts)
                    bplfieldstars.to_csv(os.path.join(field_file_loc,str(fieldid).zfill(2)+ext+'_generated_bpls.csv'))
                else:
                    print("Loading Generated BPL stars from file")
                    bplfieldstars=pd.DataFrame.from_csv(os.path.join(field_file_loc,str(fieldid).zfill(2)+ext+'_generated_bpls.csv'))

                #Getting BEBs:
                if not os.path.exists(os.path.join(field_file_loc,str(fieldid).zfill(2)+ext+'_generated_bebs.csv')):
                    bebfieldstars=getBEBs(allfieldstars.loc[allfieldstars['system_label']=='BEB'],BesFile,
                                          BGstars,field_file_loc,num_quarts=num_quarts)
                    bebfieldstars.to_csv(os.path.join(field_file_loc,str(fieldid).zfill(2)+ext+'_generated_bebs.csv'))
                else:
                    print("Loading Generated BEB stars from file")
                    bebfieldstars=pd.DataFrame.from_csv(os.path.join(field_file_loc,str(fieldid).zfill(2)+ext+'_generated_bebs.csv'))

                #Getting NAs (which may have variability)
                if not os.path.exists(os.path.join(field_file_loc,str(fieldid).zfill(2)+ext+'_generated_nas.csv')):
                    nafieldstars=getNAs(allfieldstars.loc[allfieldstars['system_label']=='NA'],BesFile,
                                     field_file_loc,num_quarts=num_quarts)
                    nafieldstars.to_csv(os.path.join(field_file_loc,str(fieldid).zfill(2)+ext+'_generated_nas.csv'))
                else:
                    print("Loading Generated NA stars from file")
                    nafieldstars=pd.DataFrame.from_csv(os.path.join(field_file_loc,str(fieldid).zfill(2)+ext+'_generated_nas.csv'))

                #Getting Planets: getPLs(fieldstars,stars,field_file_loc):
                if not os.path.exists(os.path.join(field_file_loc,str(fieldid).zfill(2)+ext+'_generated_pls.csv')):
                    plfieldstars=getPLs(allfieldstars.loc[allfieldstars['system_label']=='PL'],BesFile,
                                    field_file_loc,num_quarts=num_quarts)
                    plfieldstars.to_csv(os.path.join(field_file_loc,str(fieldid).zfill(2)+ext+'_generated_pls.csv'))
                else:
                    print("Loading Generated PL stars from file")
                    plfieldstars=pd.DataFrame.from_csv(os.path.join(field_file_loc,str(fieldid).zfill(2)+ext+'_generated_pls.csv'))
                #Combining cats:
                finalfieldstars=pd.concat([plfieldstars,ebfieldstars,bebfieldstars,bplfieldstars,nafieldstars])
            #Cleaning some definitions:
            finalfieldstars.loc[pd.isnull(finalfieldstars['A_Pmag']),'A_Pmag'] = finalfieldstars.loc[pd.isnull(finalfieldstars['A_Pmag']),'Pmag']
            finalfieldstars.loc[pd.isnull(finalfieldstars['AB_Pmag']),'AB_Pmag'] = finalfieldstars.loc[pd.isnull(finalfieldstars['AB_Pmag']),'A_Pmag']

            #Adding extra contaminants:
            finalfieldstars=addContams(finalfieldstars,BGstars,contam_prop=contam_prop)

            finalfieldstars.loc[finalfieldstars['type']=='target','known_blend']=True
            finalfieldstars.to_csv(os.path.join(field_file_loc,str(fieldid).zfill(2)+ext+'_final_fieldcat.csv'))
        else:
            print("Loading Generated field stars from file")
            finalfieldstars=pd.DataFrame.from_csv(os.path.join(field_file_loc,str(fieldid).zfill(2)+ext+'_final_fieldcat.csv'))

        #Saving variaibility and star files:
        if not os.path.isfile(field_file_loc+'/'+str(fieldid).zfill(2)+ext+'_starcat.txt'):
            field_targets=open(field_file_loc+'/'+str(fieldid).zfill(2)+ext+'_starcat.txt','w')
            field_targets.write('#RA(deg)\tDEC(deg)\tVmag\tID\tKnown\n')
            targs_in_list=[]
        else:
            field_targets=open(field_file_loc+'/'+str(fieldid).zfill(2)+ext+'_starcat.txt','a')
            targs_in_list=np.genfromtxt(field_targets,dtype='str')[:,3]
        
        if not os.path.isfile(field_file_loc+'/'+str(fieldid).zfill(2)+ext+'_Q1_varcat.txt'):
            for Q in np.arange(num_quarts):
                exec('field_varblty_q'+str(Q+1)+'=open(field_file_loc+\'/\'+str(fieldid).zfill(2)+ext+\'_Q'+str(Q+1)+'_varcat.txt\',\'w\')')
        else:
            for Q in np.arange(num_quarts):
                exec('field_varblty_q'+str(Q+1)+'=open(field_file_loc+\'/\'+str(fieldid).zfill(2)+ext+\'_Q'+str(Q+1)+'_varcat.txt\',\'a\')')
        for star,row in finalfieldstars.loc[finalfieldstars['system_label']!='PL'].iterrows():
            if row['type']=='target' or row['type']=='blend' and star not in targs_in_list:
                #The outpust list should only have the targets and blends and not, e.g., the planets/binary companions
                field_targets.write('{:.5f}\t{:.5f}\t{:.2f}\t{}\t{}\n'.format(row['ra'],row['dec'],row['AB_Pmag'],star,str(row['known_blend'])))
                if row['LC']:
                    for Q in np.arange(num_quarts):
                        lcloc=field_file_loc+'/lcs/'+str(star)+'_Q'+str(Q+1)+'.txt'
                        exec('field_varblty_q'+str(Q+1)+'.write(\'{}\\t{}\\n\'.format(star,lcloc))')
        #Doing the same for the planets (which may have multiple rows for the same target)
        for star in np.unique(finalfieldstars.loc[finalfieldstars['system_label']=='PL','parent'].values):
            row=finalfieldstars.loc[finalfieldstars.parent==star].iloc[0]
            if row['type']=='target' or row['type']=='blend' and star not in targs_in_list:
                #The outpust list should only have the targets and blends and not, e.g., the planets/binary companions
                field_targets.write('{:.5f}\t{:.5f}\t{:.2f}\t{}\t{}\n'.format(row['ra'],row['dec'],row['AB_Pmag'],star,str(row['known_blend'])))
                if row['LC']:
                    for Q in np.arange(num_quarts):
                        lcloc=field_file_loc+'/lcs/'+str(star)+'_Q'+str(Q+1)+'.txt'
                        exec('field_varblty_q'+str(Q+1)+'.write(\'{}\\t{}\\n\'.format(star,lcloc))')

        field_targets.close()
        for Q in np.arange(num_quarts):
            exec('field_varblty_q'+str(Q+1)+'.close()')
            
        scope_cen_radec=scope_cen.transform_to(ICRS)
        SavePlatosim3Python(str(fieldid).zfill(2)+ext, 
                            [scope_cen_radec.ra.value, scope_cen_radec.dec.value], 
                            [field_cen_ra,field_cen_dec],
                            field_file_loc,num_quarts)
    #imports() this was just to generate requirements

def StellarStuff(allfieldstars):
    #Initialising LDs and GDs (sorting by FeH)
    print('getting LDs and GDs')
    allfieldstars['LD_1']=np.zeros(allfieldstars.shape[0])
    allfieldstars['LD_2']=np.zeros(allfieldstars.shape[0])
    allfieldstars['GD_1']=np.zeros(allfieldstars.shape[0])
    
    nFeHs=1
    FeHbins=np.percentile(np.nan_to_num(allfieldstars['FeH'].values), list(np.linspace(0,100.00,nFeHs+1)))
    for nFeH in range(nFeHs):
        FeHinterval=(allfieldstars['FeH'].values>=FeHbins[nFeH])*(allfieldstars['FeH'].values<FeHbins[nFeH+1])
        lds=getQuadLDs(allfieldstars.loc[FeHinterval,'Teff'].values,logg=allfieldstars.loc[FeHinterval,'logg'],
                       FeH=FeHbins[nFeH],band='V')
        allfieldstars.loc[FeHinterval,'LD_1']=lds[:,0]
        allfieldstars.loc[FeHinterval,'LD_2']=lds[:,1]
        allfieldstars.loc[FeHinterval,'GD_1']=getKeplerGDs(allfieldstars.loc[FeHinterval,'Teff'].values,
                                                      logg=allfieldstars.loc[FeHinterval,'logg'],
                                                      FeH=FeHbins[nFeH],
                                                      Fr='V',mod='ATLAS')
   #Re-doing Gds for those with NaNs:
    allfieldstars.loc[pd.isnull(allfieldstars['GD_1']),'GD_1']=getKeplerGDs(allfieldstars.loc[pd.isnull(allfieldstars['GD_1']),'Teff'].values,logg=4.0,FeH=0.0,Fr='V',mod='ATLAS')
    allfieldstars.loc[pd.isnull(allfieldstars['GD_1']),'GD_1'] = getKeplerGDs(np.tile(5500,len(allfieldstars.loc[pd.isnull(allfieldstars['GD_1']),'GD_1'])),
                                                                    logg=4.0,
                                                                    FeH=0.0,Fr='V',mod='ATLAS')
    #Getting beaming:
    allfieldstars['bfac']=Get_Beaming(allfieldstars['Teff'].values)
    
    #Getting albedo:
    allfieldstars['albedo']=get_albedo(allfieldstars['Teff'].values)
    return allfieldstars
        
def getPLs(fieldstars,stars,field_file_loc,num_quarts=8,var_prop=0.5,SNRthresh=3.0):
    # get info for a "PL" dip - i.e. a transiting planet in the PLATO field.
    #INPUTS:
    # fieldstars - DataFrame generated with target star position info
    # stars - DataFrame from Besancon from which to take target star stellar parameters
    # field_file_loc - Location to save field files
    # num_quarts - Number of quarters to generate

    # This function gets transiting planets, given some star catologue and a dictionary of lightcurves
    #including function to multiply long-period/small planets:
    powlawfact=0.25 #(this boosts by 2x the occurrence rates of Earths)
    newfunc = lambda P,Rp: (np.clip(P,50,400)/50)**powlawfact * (4.0/(np.clip(Rp,0.8,4.0)))**powlawfact
    petigura = assemblePet(multfunc=newfunc)
    
    stars=stars.set_index(np.core.defchararray.add('PL_',stars.index.values.astype(str)))

    npls=0
    tran_pls=pd.DataFrame()
    iterations=1
    if len(stars)>50*len(fieldstars):
        stars=stars.iloc[np.random.choice(len(stars),int(50*len(fieldstars)),replace=False)]
    
    while npls<len(fieldstars):
        print("Sampling trasiting planets")
        #stars.copy()
        rands=np.random.random((len(stars),len(petigura[:,0])-2,len(petigura[0,:])-2))
        whr=np.where(rands<np.tile(petigura[1:-1,1:-1],(len(stars),1,1)))
        rename_dic={'Ms':'A_Ms','Rs':'A_Rs','Pmag':'A_Pmag','Teff':'A_Teff','logg':'A_logg','FeH':'A_FeH',
                    'LD_1':'A_LD_1','LD_2':'A_LD_2','GD_1':'A_GD_1','bfac':'A_bfac','albedo':'A_albedo'}
        tran_pls_i=pd.DataFrame()
        tran_pls_i['parent']=stars.iloc[whr[0]].index
        tran_pls_i['blend_parent']=stars.iloc[whr[0]].index
        tran_pls_i = pd.merge(tran_pls_i, stars.rename(columns=rename_dic), left_on='parent',
                              right_index=True,how='left', sort=False)
        tran_pls_i['Rs']=np.random.uniform(petigura[whr[1]+1,0],petigura[whr[1]+2,0])*(Rearth/Rsun)
        tran_pls_i['P']=np.exp(np.random.uniform(np.log(petigura[0,whr[2]+1]),np.log(petigura[0,whr[2]+2])))
        tran_pls_i['T0']=np.random.uniform(np.tile(0,tran_pls_i.shape[0]),tran_pls_i['P'])
        #Eccentricity, omega and incl:
        kipping= np.random.normal([1.12,3.09,0.697,3.27],[0.1,0.3,0.2,0.34])
        tran_pls_i['ecc']=stats.beta.rvs(kipping[0],kipping[1],size=tran_pls_i.shape[0])
        tran_pls_i.loc[tran_pls_i['P']<=382.5,'ecc']=stats.beta.rvs(kipping[2],kipping[3],size=np.sum(tran_pls_i['P'].values<=382.5))

        tran_pls_i['omega']=np.random.random(len(tran_pls_i))*2*np.pi
        #Adding dependence on number of iterations to make later runs more likely to find a solution...
        tran_pls_i['incl']=np.arccos(np.random.uniform(-1*iterations**-0.2,iterations**-0.2,len(tran_pls_i)))
        tran_pls_i['index_pl']=np.zeros(tran_pls_i.shape[0])
        
        #Adding suffixes for planets going outward in period (and cutting planets in systems >10pls)
        plnames=np.array(['_b','_c','_d','_e','_f','_g','_h','_i','_j','_k','_l'])
        for unq_id in np.unique(tran_pls_i['parent'].values):
            solarsyst=tran_pls_i.index.values[tran_pls_i['parent']==unq_id]
            tran_pls_i.loc[solarsyst,'index_pl']=tran_pls_i.loc[solarsyst,'P'].argsort()
        tran_pls_i=tran_pls_i.loc[tran_pls_i['index_pl']<=10]
        
        tran_pls_i=tran_pls_i.set_index(np.core.defchararray.add(tran_pls_i['parent'].values.astype(str),
                                                 plnames[tran_pls_i['index_pl'].values.astype(int)]).astype(str))
        
        #Randomly assigning 66% of multiplanet systems to be co-planar with first planet
        for unqpar in np.unique(tran_pls_i.loc[tran_pls_i['index_pl'].values>=1,'parent']):
            ind=tran_pls_i.loc[tran_pls_i['parent'].values==unqpar].index.values
            if np.random.random()>0.33:
                tran_pls_i.loc[ind,'incl']=np.random.normal(tran_pls_i.loc[ind,'incl'].values[0],0.1,
                                                            len(ind))
        tran_pls_i.loc[:,'incl']=tran_pls_i.loc[:,'incl']%np.pi #Forcing inclinations to 0->pi
        tran_pls_i['sma']=kepp2a_pl(tran_pls_i['P'].values,tran_pls_i['A_Ms'].values)
        tran_pls_i['minP']=kepa2p(2*tran_pls_i['A_Rs'].values*0.00465233957,tran_pls_i['A_Ms'].values)#Minimum stable orbit

        #Dropping impossible planets
        tran_pls_i=tran_pls_i.drop(tran_pls_i.loc[tran_pls_i['P'].values<tran_pls_i['minP'].values].index.values)

        tran_pls_i['depthpri_ppm']=1e6*np.clip(((tran_pls_i['Rs']/tran_pls_i['A_Rs'])**2),0,1)
        tran_pls_i['durpri'] = (2./86400.)*np.sqrt( 1 - (tran_pls_i['sma']*au * ( 1. - tran_pls_i['ecc']**2 ) / ( 1 + tran_pls_i['ecc'] * np.cos(np.pi/2.0 - tran_pls_i['omega']) ) * np.cos(tran_pls_i['incl']))**2 / (tran_pls_i['A_Rs']*Rsun + tran_pls_i['Rs']*Rsun)**2)*(tran_pls_i['Rs']*Rsun + tran_pls_i['A_Rs']*Rsun)*np.sqrt(1. - tran_pls_i['ecc']**2)* \
            ( 1. + tran_pls_i['ecc'] * np.cos(np.pi/2.0 - tran_pls_i['omega']) )**(-1.)* \
            (tran_pls_i['P']*86400./(2. * np.pi * G * (tran_pls_i['A_Ms']*Msun)))**(1./3.)

        #Dropping systems where all planets are low-SNR
        tran_pls_i['SNRpri']=tran_pls_i['depthpri_ppm']*tran_pls_i['rms_hr']*(tran_pls_i['durpri']*24*730.5/tran_pls_i['P'])**(-0.5)  
        for unqparent in pd.unique(tran_pls_i['parent']):
            allpls=tran_pls_i.loc[tran_pls_i['parent']==unqparent].index
            if np.max(tran_pls_i.loc[allpls,'SNRpri'])<SNRthresh:
                tran_pls_i=tran_pls_i.drop(allpls)        
        
        #removing those planets which do not transit:
        tran_pls_i['bpri']=abs(b_ecc_pri(*tran_pls_i.loc[:,['ecc','omega','sma','incl','A_Rs']].values.swapaxes(0,1))) #impact parameter at primary eclipse
        tran_pls_i['transits']=abs(tran_pls_i['bpri'].values)<(1+tran_pls_i['Rs']/tran_pls_i['A_Rs'])
        tran_pls_i=tran_pls_i.loc[tran_pls_i['transits']]
        
        tran_pls=tran_pls.append(tran_pls_i)
        print("Planet iterations= ",iterations,". systems with transiting planets:", len(pd.unique(tran_pls.parent)),"from",len(stars),"initial stars. Goal:",len(fieldstars))
        npls=len(tran_pls)
        iterations+=1
    
    #Appending randomly sorted planet hosts onto fieldstars df:
    unqpars=pd.unique(tran_pls['parent'].values)
    parents2field=unqpars[np.random.choice(len(unqpars),len(fieldstars),replace=False)]
    tran_pls=tran_pls.loc[np.in1d(tran_pls['parent'],parents2field)]
    fieldstars=fieldstars.set_index(parents2field)
    
    #Extra stuff to add to planets:
    tran_pls['Ms']=PlanetRtoM(tran_pls['Rs']*(Rsun/Rearth))*(Mearth/Msun)
    tran_pls['Teff']=SurfaceT(tran_pls['sma'].values,tran_pls['A_Rs'].values,tran_pls['A_Teff'].values)
    tran_pls['albedo']=get_albedo(tran_pls['Teff'].values)
   
    #Need to redo contration with multiplicities...
    tran_pls['GD_1']=np.zeros(tran_pls.shape[0])
    tran_pls['LD_1']=np.ones(tran_pls.shape[0])
    tran_pls['LD_2']=np.zeros(tran_pls.shape[0])
    tran_pls.loc[np.isnan(tran_pls['sma']),'sma']=(((86400*tran_pls['P'])**2*6.67e-11*(tran_pls['Ms']*Msun+tran_pls['A_Ms']*Msun))/(4*np.pi**2))**0.3333/au
    tran_pls['bfac']=Get_Beaming(tran_pls['Teff'].values)

    tran_pls['bpri']=abs(b_ecc_pri(tran_pls['ecc'].values,tran_pls['omega'].values,
                               tran_pls['sma'].values,tran_pls['incl'].values,
                               tran_pls['A_Rs'].values)) #impact parameter at primary eclipse
    tran_pls['bsec']=abs(b_ecc_sec(tran_pls['ecc'].values,tran_pls['omega'].values,
                               tran_pls['sma'].values,tran_pls['incl'].values,
                               tran_pls['Rs'].values)) #impact parameter at primary eclipse
    tran_pls['Rratio']=(tran_pls['Rs'].values*Rsun)/(tran_pls['A_Rs'].values*Rsun)

    tran_pls['pri_ecl']=tran_pls['bpri']<(1+tran_pls['Rratio'])
    tran_pls['sec_ecl']=tran_pls['bsec']<(1+1.0/tran_pls['Rratio'])
    
    
    tran_pls['sbratio']=(tran_pls['Teff']/tran_pls['A_Teff'])**4
    tran_pls['VAR']=np.tile(False,len(tran_pls))
    
    time=np.arange(0,365.25*2*(num_quarts/8),25/86400)
    #Getting planet lcs:
    for parid in parents2field:
        lc=[]
        for pl,row in tran_pls.loc[tran_pls.parent==parid].iterrows():
            lc+=[ellclc(t_obs=time,
                  radius_1=(row['A_Rs']*Rsun)/(row['sma']*au), radius_2=(row['Rs']*Rsun)/(row['sma']*au),
                  sbratio=(row['Teff']/row['A_Teff'])**4, incl=row['incl']*180/np.pi,
                  light_3=0.0, t_zero=row['T0'], period=row['P'],a=(row['sma']*au)/(row['A_Rs']*Rsun),
                  f_c=np.sqrt(row['ecc'])*np.cos(row['omega']), f_s=np.sqrt(row['ecc'])*np.sin(row['omega']), 
                  q=(row['Ms']*Msun)/(row['A_Ms']*Msun),ldc_1=[row['A_LD_1'],row['A_LD_2']], ldc_2=[1.,0.],gdc_1=row['A_GD_1'],
                  gdc_2=0.0, heat_1=row['A_albedo'], heat_2=row['albedo'], lambda_1=None,ld_1="quad",
                  ld_2="quad",grid_1='sparse',grid_2='sparse',bfac_1=row['A_bfac'],bfac_2=row['bfac'],
                  verbose=False)-1.0]
        lc=1+np.sum(np.vstack(lc),axis=0)
        if np.random.random()<var_prop:
            varlc,vardic=GetVar(row['Age'],row['A_Ms'],row['A_Teff'],row['A_Rs'],time)
            tran_pls.loc[tran_pls.parent==parid,'VAR']=True
            lc+=varlc
            for key in vardic:
                tran_pls.loc[tran_pls.parent==parid,key]=vardic[key]
        lc=-2.5*np.log10(lc)
        tran_pls.loc[tran_pls.parent==parid,'LC']=True
        print("writing",field_file_loc,parid)
        for Q in np.arange(num_quarts):
            low=int(np.clip(np.floor(0.125*(Q)*len(lc)-1),0,len(lc)))
            high=int(np.clip(np.ceil(0.125*(Q+1)*len(lc)+1),0,len(lc)))
            #Interpolating:
            lc_bool=InterpolateLC(lc[low:high],rms_hr=row['rms_hr']*1e-6,prec=0.05)
            
            lcloc=field_file_loc+'/lcs/'+str(parid)+'_Q'+str(Q+1)+'.txt'
            np.savetxt(lcloc,np.column_stack((time[low:high],lc[low:high]))[lc_bool],fmt='%.3f %.8f')
    
    fieldstars = fieldstars.loc[:,~fieldstars.columns.duplicated()]
    tran_pls = tran_pls.loc[:,~tran_pls.columns.duplicated()]
    
    output_df=pd.merge(tran_pls,fieldstars,left_on='blend_parent',right_index=True, how='left', sort=False)

    return output_df

def InterpolateLC(lc,rms_hr=35e-6,prec=0.005):
    # This function takes an evenly spaced lightcurve (and the photometric precision) & removes pts while maintaining the same eventual precision
    # If a point has no considerable difference before or after (>prec*rms), we mark it
    # We then remove every other marked point and repeat, raising the rms threshold by sqrt(2)
    # Typically size reductions are on the order of 100x (with a precision of 0.01)
    rms_cad=rms_hr*np.sqrt(25/3600)
    n_pts_oversampled=np.sum(abs(np.diff(lc))<prec*rms_cad)
    
    bool2keep_meta=np.tile(True,len(lc))
    #print(np.sum(bool2keep_meta))
    n_loop=0
    while n_pts_oversampled>0 and n_loop<10:
        bool2rem=(abs(np.diff(lc[bool2keep_meta][1:]))<(prec*rms_cad))&(abs(np.diff(lc[bool2keep_meta][:-1]))<(prec*rms_cad))
        bool2rem[1::2]*=False
        bool2keep=np.hstack((True,~bool2rem,True))
        
        bool2keep_meta[bool2keep_meta]=bool2keep
        rms_cad*=np.sqrt(2)
        n_pts_oversampled=np.sum(abs(np.diff(lc[bool2keep_meta]))<prec*rms_cad)
        n_loop+=1
        #print(n_loop, np.sum(bool2keep_meta),rms_cad)
    return bool2keep_meta
        
def getEBs(fieldstars, stars,field_file_loc,num_quarts=8,var_prop=0.5):
    # get info for a "EB" dip - i.e. a eclipsing binary in the PLATO field.
    #INPUTS:
    # fieldstars - DataFrame generated with target star position info
    # stars - DataFrame from Besancon from which to take target star stellar parameters
    # field_file_loc - Location to save field files
    # num_quarts - Number of quarters to generate
    nebs=0
    tran_ebs=pd.DataFrame()
    iterations=1
    
    stars=stars.set_index(np.core.defchararray.add('EB_',stars.index.values.astype(str)))
    
    if len(stars)>250*len(fieldstars):
        stars=stars.iloc[np.random.choice(len(stars),int(50*len(fieldstars)),replace=False)]
    
    while nebs<len(fieldstars):
        print("Starting while loop for EBs with ",len(stars)," stars")
        tran_ebs_i=stars.copy()
        tran_ebs_i=tran_ebs_i.rename(columns={'Ms':'A_Ms','Rs':'A_Rs','Pmag':'A_Pmag','Teff':'A_Teff',
                                              'logg':'A_logg','LD_1':'A_LD_1','LD_2':'A_LD_2',
                                              'GD_1':'A_GD_1','bfac':'A_bfac','albedo':'A_albedo',
                                              'V':'A_V','logL':'A_logL'})
        tran_ebs_i=tran_ebs_i.loc[tran_ebs_i.A_Ms>0.1]
        tran_ebs_i['Ms']=np.random.uniform(np.tile(0.1,len(tran_ebs_i)),tran_ebs_i['A_Ms'].values)
        tran_ebs_i['parent']=tran_ebs_i.index.values
        tran_ebs_i['blend_parent']=tran_ebs_i.index.values
        tran_ebs_i=tran_ebs_i.set_index(np.core.defchararray.add(tran_ebs_i['parent'].values.astype(str),
                                                             np.tile('_AB',len(tran_ebs_i))))
        tran_ebs_i['P']=np.exp(np.random.normal(5.03,2.28,len(tran_ebs_i)))#from Raghavan
        tran_ebs_i['sma']=kepp2a(tran_ebs_i['P'].values,tran_ebs_i['A_Ms'].values,tran_ebs_i['Ms'].values)
        #Only keeping systems where sma>2Rs to avoid unstable short-P systems:
        tran_ebs_i['T0']=np.random.uniform(np.tile(0,len(tran_ebs_i)),tran_ebs_i['P'])
        #Technically eccentricity could make these things unstable again. Oh well...
        tran_ebs_i['ecc']=stats.beta.cdf(np.random.random(len(tran_ebs_i)),
                                        a=np.tile(0.9,tran_ebs_i.shape[0]),
                                        b=0.75*stats.norm.cdf(np.log10(tran_ebs_i['P']),2.0,0.85)
                                       )#est from Raghavan
        #Taking high-ecc systems at P<10 and setting to zero
        tran_ebs_i.loc[tran_ebs_i.ecc>(0.5*np.log10(tran_ebs_i.P)+0.5),'ecc']=0.0
        tran_ebs_i['omega']=np.random.random(tran_ebs_i.shape[0])*2*np.pi
        tran_ebs_i['incl']=np.arccos(np.random.uniform(-1*iterations**-0.2,iterations**-0.2,len(tran_ebs_i)))
        tran_ebs_i['logage']=np.log10(tran_ebs_i['Age'])+9
        iso_df=IsochronesMagic_Simple(tran_ebs_i,mag='Pmag')
        tran_ebs_i=pd.concat([tran_ebs_i,iso_df.drop(['Ms','logage'],axis=1)],axis=1)
        #dropping systems where smaller companion is "inflated"
        inflatedsystems=(tran_ebs_i['Rs']>1.1*tran_ebs_i['A_Rs'])+(tran_ebs_i['Pmag']<(tran_ebs_i['A_Pmag']-0.25))
        tran_ebs_i=tran_ebs_i.drop(tran_ebs_i.loc[inflatedsystems].index)
        #Dropping systems where sma<1.5*(R_1+R2) - unstably close systems
        tran_ebs_i=tran_ebs_i.loc[(215.03203*tran_ebs_i['sma']>1.5*(tran_ebs_i['A_Rs']+tran_ebs_i['Rs']))]
        
        tran_ebs_i['bpri']=abs(b_ecc_pri(tran_ebs_i['ecc'].values,tran_ebs_i['omega'].values,
                                       tran_ebs_i['sma'].values,tran_ebs_i['incl'].values,
                                       tran_ebs_i['A_Rs'].values)) #impact parameter at primary eclipse
        tran_ebs_i['Rratio']=tran_ebs_i['Rs']/tran_ebs_i['A_Rs']
        tran_ebs_i['bsec']=abs(b_ecc_sec(tran_ebs_i['ecc'].values,tran_ebs_i['omega'].values,
                                    tran_ebs_i['sma'].values,tran_ebs_i['incl'].values,
                                    tran_ebs_i['Rs'])) #impact parameter at secondary eclipse (assuming Rb=0.5Ra)
        tran_ebs_i['pri_ecl']=tran_ebs_i['bpri']<(1+tran_ebs_i['Rratio'])
        tran_ebs_i['sec_ecl']=tran_ebs_i['bsec']<(1+tran_ebs_i['Rratio']**-1)

        #removing those ebs which do not transit:
        tran_ebs_i=tran_ebs_i.loc[tran_ebs_i['pri_ecl']|tran_ebs_i['sec_ecl']]
        
        tran_ebs=tran_ebs.append(tran_ebs_i)
        
        nebs=len(tran_ebs)
        print("EB / ",iterations,"iterations ", nebs," found in this run. Target:",len(fieldstars))
        iterations+=1
        
    
    #Appending randomly sorted planet hosts onto fieldstars df:
    tran_ebs=tran_ebs.iloc[np.random.choice(len(tran_ebs),len(fieldstars),replace=False)]
    #fieldstars=fieldstars.set_index(tran_ebs_hosts.index.values)

    tran_ebs['deltamag']=tran_ebs['Pmag'].values-tran_ebs['A_Pmag'].values

    tran_ebs['durpri'] = (2./86400.)*np.sqrt( 1 - (tran_ebs['sma']*au * ( 1. - tran_ebs['ecc']**2 ) / ( 1 + tran_ebs['ecc'] * np.cos(np.pi/2.0 - tran_ebs['omega']) ) * np.cos(tran_ebs['incl']))**2 / (tran_ebs['A_Rs']*Rsun + tran_ebs['Rs']*Rsun)**2)*(tran_ebs['Rs']*Rsun + tran_ebs['A_Rs']*Rsun)*np.sqrt(1. - tran_ebs['ecc']**2)* \
                ( 1. + tran_ebs['ecc'] * np.cos(np.pi/2.0 - tran_ebs['omega']) )**(-1.)* \
                (tran_ebs['P']*86400./(2. * np.pi * G * (tran_ebs['A_Ms']*Msun + tran_ebs['Ms']*Msun)))**(1./3.)
    tran_ebs['dursec'] = (2./86400.)*np.sqrt( 1 - (tran_ebs['sma']*au * ( 1. - tran_ebs['ecc']**2 ) / ( 1 + tran_ebs['ecc'] * np.cos(3*np.pi/2.0 - tran_ebs['omega']) ) * np.cos(tran_ebs['incl']))**2 / (tran_ebs['Rs']*Rsun + tran_ebs['A_Rs']*Rsun)**2)*(tran_ebs['Rs']*Rsun + tran_ebs['A_Rs']*Rsun)*np.sqrt(1. - tran_ebs['ecc']**2)* \
                ( 1. + tran_ebs['ecc'] * np.cos(3*np.pi/2.0 - tran_ebs['omega']) )**(-1.)* \
                (tran_ebs['P']*86400./(2. * np.pi * G * (tran_ebs['A_Ms']*Msun + tran_ebs['Ms']*Msun)))**(1./3.)
    
    tran_ebs=StellarStuff(tran_ebs) #Limb darkening, beaming, etc
    
    #Depths:
    tran_ebs['depthpri_ppm']=1e6*(tran_ebs['Rratio']**2/(1+tran_ebs['Rratio']**2*(tran_ebs['Teff']/tran_ebs['A_Teff'])**4))
    tran_ebs['depthsec_ppm']=1e6*(1-(1/(1+tran_ebs['Rratio']**2*(tran_ebs['Teff']/tran_ebs['A_Teff'])**4)))
    
    tran_ebs['AB_Pmag']=tran_ebs['A_Pmag'].values - \
                        2.5*np.log10(1+np.power(2.512,tran_ebs['A_Pmag'].values-tran_ebs['Pmag'].values))
    tran_ebs['rms_hr']=getPmagRms(tran_ebs['AB_Pmag'],
                                       np.tile(np.median(fieldstars['Nscopes']),len(tran_ebs)))[0]

    #SNRs:
    tran_ebs['SNRpri']=tran_ebs['depthpri_ppm']/tran_ebs['rms_hr']*(tran_ebs['durpri']*24*tran_ebs['P']/730.5)**(-0.5)
    tran_ebs['SNRsec']=tran_ebs['depthsec_ppm']/tran_ebs['rms_hr']*(tran_ebs['dursec']*24*tran_ebs['P']/730.5)**(-0.5)
    
    time=np.arange(0,365.25*2*(num_quarts/8),25/86400)
    #Getting eb lcs:
    for eb,row in tran_ebs.iterrows():
        lc=ellclc(t_obs=time,
              radius_1=(row['A_Rs']*Rsun)/(row['sma']*au), radius_2=(row['Rs']*Rsun)/(row['sma']*au),
              sbratio=(row['Teff']/row['A_Teff'])**4, incl=row['incl']*180/np.pi,
              light_3=0.0, t_zero=row['T0'], period=row['P'],a=(row['sma']*au)/(row['A_Rs']*Rsun),
              f_c=np.sqrt(row['ecc'])*np.cos(row['omega']), f_s=np.sqrt(row['ecc'])*np.sin(row['omega']), 
              q=(row['Ms']*Msun)/(row['A_Ms']*Msun),ldc_1=[row['A_LD_1'],row['A_LD_2']], ldc_2=[row['LD_1'],row['LD_2']],
              gdc_1=row['A_GD_1'],gdc_2=row['GD_1'], heat_1=row['A_albedo'], heat_2=row['albedo'],
              lambda_1=None,ld_1="quad",ld_2="quad",grid_1='sparse',grid_2='sparse',
              bfac_1=row['A_bfac'],bfac_2=row['bfac'],verbose=False)
        
        tran_ebs.loc[eb,'LC']=True
        if np.random.random()<var_prop:
            varlc,vardic=GetVar(row['Age'],row['A_Ms'],row['A_Teff'],row['A_Rs'],time)
            lc+=varlc
            tran_ebs.loc[eb,'VAR']=True
            for key in vardic:
                tran_ebs.loc[eb,key]=vardic[key]
        lc=-2.5*np.log10(lc)
        print("writing",field_file_loc,eb)
        for Q in np.arange(num_quarts):
            low=int(np.clip(np.floor(0.125*(Q)*len(lc)-1),0,len(lc)))
            high=int(np.clip(np.ceil(0.125*(Q+1)*len(lc)+1),0,len(lc)))
            #Interpolating:
            lc_bool=InterpolateLC(lc[low:high],rms_hr=row['rms_hr']*1e-6)
            lcloc=field_file_loc+'/lcs/'+str(eb)+'_Q'+str(Q+1)+'.txt'
            np.savetxt(lcloc,np.column_stack((time[low:high],lc[low:high]))[lc_bool],fmt='%.3f %.8f')
    stars = stars.loc[:,~stars.columns.duplicated()]
    tran_ebs = tran_ebs.loc[:,~tran_ebs.columns.duplicated()]
    
    fieldstars=fieldstars.set_index(tran_ebs.index.values)
    fieldstars=pd.merge(fieldstars,tran_ebs,left_index=True,right_index=True,how='left', sort=False)   

    return fieldstars

def addContams(finalfieldstars, bgstars, contam_prop=0.5):
    #Choosing stars to add contaminants around:
    finalfieldstars.loc[pd.isnull(finalfieldstars['parent']),'parent']=finalfieldstars.loc[pd.isnull(finalfieldstars['parent'])].index.values

    gaia_detn_prob=init_Gaia_Source()
    contams = finalfieldstars.loc[finalfieldstars.type=='target']
    contams = contams.iloc[np.unique(contams.parent.values,return_index=True)[1]]
    
    contams = contams.iloc[np.random.choice(len(contams),int(np.floor(len(contams)*contam_prop)),replace=False)].index.values
    finalfieldstars['N_contam']=0
    finalfieldstars.loc[contams,'N_contam']=np.clip(np.random.poisson(2,len(contams)).astype(int),1,6)
    
    all_contam_stars=pd.DataFrame()
    print(len(contams)," targets to add ",np.sum(finalfieldstars['N_contam'])," contaminants.")
    for n,target in enumerate(contams):
        targ=finalfieldstars.loc[target]
        if type(targ)==pd.DataFrame:
            targ=targ.iloc[0]
        '''if targ['system_label']=='EB':
            mag=targ['AB_Pmag']
        elif np.isnan(targ['Pmag']) and not np.isnan(targ['A_Pmag']):
            mag=targ['A_Pmag']
        elif np.isnan(targ['Pmag']) and not np.isnan(targ['A_PMag']):
            mag=targ['A_PMag']
        else:
            mag=targ['Pmag']
        '''
        
        index_limit=[np.searchsorted(bgstars.Pmag.values,targ['AB_Pmag']),np.searchsorted(bgstars.Pmag.values,10+targ['AB_Pmag'])]
        #print(mag,len(bgstars),index_limit[0],index_limit[1],int(targ['N_contam']))
        contam_stars=bgstars.iloc[np.random.randint(index_limit[0],index_limit[1],int(targ['N_contam']))]
        contam_stars['blend_parent']=target
        #RA and Dec:
        sep=np.sqrt(np.random.random(len(contam_stars)))*np.clip(1.5*targ['ap_radius'],0.0,7*15)
        angle=np.random.random(len(contam_stars))*2*np.pi
        contam_stars['ra']=targ['ra']+(sep/3600)*np.cos(angle)
        contam_stars['dec']=targ['dec']+(sep/3600)*np.sin(angle)
        contam_stars['type']='blend'
        contam_stars['system_label']='NA'
        contam_stars['known_blend']=np.random.random(len(contam_stars))<gaia_detn_prob(contam_stars['Pmag']-targ['AB_Pmag'],sep)
        contam_stars['AB_Pmag']=contam_stars['Pmag'].values
        if targ['system_label']=='PL':
            contam_stars=contam_stars.set_index(np.array([str(targ['parent'])+'_'+str(n+2).zfill(2) for n in np.arange(int(targ['N_contam']))]))
        else:
            contam_stars=contam_stars.set_index(np.array([str(targ.name)+'_'+str(n+2).zfill(2) for n in np.arange(int(targ['N_contam']))]))
        all_contam_stars=all_contam_stars.append(contam_stars)
    
    return pd.concat([finalfieldstars,all_contam_stars])

def getBEBs(fieldstars,stars,bgstars,field_file_loc,num_quarts=8,var_prop=0.5,SNRthresh=3.0):
    # get info for a "BEB" dip - i.e. a background eclpsing binary in the PLATO field.
    #INPUTS:
    # fieldstars - DataFrame generated with target star position info
    # stars - DataFrame from Besancon from which to take target star stellar parameters
    # bgstars - DataFrame from Besancon from which to take EB host star stellar parameters
    # field_file_loc - Location to save field files
    # num_quarts - Number of quarters to generate

    gaia_detn_prob=init_Gaia_Source()
    nbebs=0
    
    stars=stars.set_index(np.core.defchararray.add('BEB_',stars.index.values.astype(str)))
    bgstars=bgstars.set_index(np.core.defchararray.add('BEB_',bgstars.index.values.astype(str)))

    tran_bebs=pd.DataFrame()
    iterations=1
    while nbebs<len(fieldstars):
        print("Sampling blended EBs")
        #Choosing random star fainter than target to add the binary to:
        #Finds position of target star in bgstars (sorted by Pmag).
        #Finds position of 8mags fainter in bgstars.
        if iterations>1:
            #Making sure we dont choose the same star here:
            bgstars=bgstars.loc[~np.in1d(bgstars.index,tran_bebs['parent'].values)]
        # then takes uniform sample between them
        choice_list=np.clip(np.round(np.random.uniform(np.searchsorted(bgstars.Pmag.values,stars.Pmag.values),
                                               np.searchsorted(bgstars.Pmag.values,10+stars.Pmag.values))).astype(int),
                            0,len(bgstars)-1)
        tran_bebs_i=bgstars.iloc[choice_list].copy()
        assert(len(tran_bebs_i)==len(stars))
        tran_bebs_i=tran_bebs_i.rename(columns={'Ms':'A_Ms','Rs':'A_Rs','Pmag':'A_Pmag','Teff':'A_Teff',
                                                'logg':'A_logg','LD_1':'A_LD_1','LD_2':'A_LD_2',
                                                'GD_1':'A_GD_1','bfac':'A_bfac','albedo':'A_albedo',
                                                'V':'A_V','logL':'A_logL'})
        tran_bebs_i['Ms']=np.random.uniform(np.tile(0.1,len(tran_bebs_i)),tran_bebs_i['A_Ms'].values)
        tran_bebs_i['besancon_index']=tran_bebs_i.index.values[:]
        tran_bebs_i['blend_parent']=stars.index.values
        tran_bebs_i['type']='blend'
        tran_bebs_i['system_label']='BEB'
        
        toosmall=tran_bebs_i['Ms'].values<0.1
        tran_bebs_i.loc[toosmall,'Ms']=np.random.uniform(np.tile(0.1,np.sum(toosmall)),tran_bebs_i.loc[toosmall,'A_Ms'].values)
        tran_bebs_i=tran_bebs_i.set_index(np.core.defchararray.add(tran_bebs_i['blend_parent'].values.astype(str),
                                                             np.tile('_01_AB',len(tran_bebs_i))
                                                            ))
        tran_bebs_i['P']=np.exp(np.random.normal(5.03,2.28,len(tran_bebs_i)))#from Raghavan
        tran_bebs_i['sma']=kepp2a(tran_bebs_i['P'].values,tran_bebs_i['A_Ms'].values,tran_bebs_i['Ms'].values)

        tran_bebs_i['T0']=np.random.uniform(np.tile(0,len(tran_bebs_i)),tran_bebs_i['P'])
        #Technically eccentricity could make these things unstable again. Oh well...
        tran_bebs_i['ecc']=stats.beta.cdf(np.random.random(len(tran_bebs_i)),
                                        a=np.tile(0.9,tran_bebs_i.shape[0]),
                                        b=0.75*stats.norm.cdf(np.log10(tran_bebs_i['P']),2.0,0.85)
                                       )#est from Raghavan
        #Taking high-ecc systems at P<10 and setting to zero
        tran_bebs_i.loc[tran_bebs_i.ecc>(0.5*np.log10(tran_bebs_i.P)+0.5),'ecc']=0.0
        tran_bebs_i['omega']=np.random.random(tran_bebs_i.shape[0])*2*np.pi
        tran_bebs_i['incl']=np.arccos(np.random.uniform(-0.33*iterations**-0.2,0.33*iterations**-0.2,len(tran_bebs_i)))

        tran_bebs_i['logage']=np.log10(tran_bebs_i['Age'])+9
        iso_df=IsochronesMagic_Simple(tran_bebs_i,mag='Pmag')
        tran_bebs_i=pd.concat([tran_bebs_i,iso_df.drop(['Ms','logage'],axis=1)],axis=1)
        #dropping systems where smaller companion is "inflated"
        inflatedsystems=(iso_df['Rs']>1.1*tran_bebs_i['A_Rs'])+(iso_df['Pmag']<(tran_bebs_i['A_Pmag']-0.25))
        tran_bebs_i=tran_bebs_i.drop(tran_bebs_i.loc[inflatedsystems].index)
        #Dropping systems where sma<1.5*(R_1+R2) - unstably close systems
        tran_bebs_i=tran_bebs_i.loc[(215.03203*tran_bebs_i['sma']>1.5*(tran_bebs_i['A_Rs']+tran_bebs_i['Rs']))]
        tran_bebs_i['bpri']=abs(b_ecc_pri(tran_bebs_i['ecc'].values,tran_bebs_i['omega'].values,
                                       tran_bebs_i['sma'].values,tran_bebs_i['incl'].values,
                                       tran_bebs_i['A_Rs'].values)) #impact parameter at primary eclipse

        tran_bebs_i['bsec']=abs(b_ecc_sec(tran_bebs_i['ecc'].values,tran_bebs_i['omega'].values,
                                    tran_bebs_i['sma'].values,tran_bebs_i['incl'].values,
                                    tran_bebs_i['Rs'].values)) #impact parameter at primary eclipse assuming Rb=0.5Ra
        tran_bebs_i['Rratio']=tran_bebs_i['Rs'].values/tran_bebs_i['A_Rs'].values
        tran_bebs_i['pri_ecl']=tran_bebs_i['bpri']<(1+tran_bebs_i['Rratio'])
        tran_bebs_i['sec_ecl']=tran_bebs_i['bsec']<(1+1/tran_bebs_i['Rratio'])
                                                    
        #how='outer',left_index=True,right_index=True)
        tran_bebs_i['deltamag']=tran_bebs_i['Pmag'].values-tran_bebs_i['A_Pmag'].values
        
        #combined mag & RMS given two star's fluxes:
        tran_bebs_i['AB_Pmag']=-2.512*np.log10(np.power(2.512,-1*tran_bebs_i['Pmag'].values)+\
                                               np.power(2.512,-1*tran_bebs_i['A_Pmag'].values))
        #And with three stars fluxes:
        tran_bebs_i['XAB_Pmag']=-2.512*np.log10(np.power(2.512,-1*stars.loc[tran_bebs_i['blend_parent'],'Pmag'].values)+\
                                               np.power(2.512,-1*tran_bebs_i['AB_Pmag'].values))
        Nscopes_arr=np.tile(np.nanmedian(fieldstars['Nscopes']),len(tran_bebs_i))
        tran_bebs_i['AB_rms_hr']=getPmagRms(tran_bebs_i['AB_Pmag'],Nscopes_arr)[0]
        tran_bebs_i['XAB_rms_hr']=getPmagRms(tran_bebs_i['XAB_Pmag'],Nscopes_arr)[0]
        
        #removing those ebs which do not transit:
        tran_bebs_i=tran_bebs_i.loc[(tran_bebs_i['pri_ecl'].values)|(tran_bebs_i['sec_ecl'].values)]
        tran_bebs_i['durpri'] = (2./86400.)*np.sqrt( 1 - (tran_bebs_i['sma']*au * ( 1. - tran_bebs_i['ecc']**2 ) / ( 1 + tran_bebs_i['ecc'] * np.cos(np.pi/2.0 - tran_bebs_i['omega']) ) * np.cos(tran_bebs_i['incl']))**2 / (tran_bebs_i['A_Rs']*Rsun + tran_bebs_i['Rs']*Rsun)**2)*(tran_bebs_i['Rs']*Rsun + tran_bebs_i['A_Rs']*Rsun)*np.sqrt(1. - tran_bebs_i['ecc']**2)* \
                    ( 1. + tran_bebs_i['ecc'] * np.cos(np.pi/2.0 - tran_bebs_i['omega']) )**(-1.)* \
                    (tran_bebs_i['P']*86400./(2. * np.pi * G * (tran_bebs_i['A_Ms']*Msun + tran_bebs_i['Ms']*Msun)))**(1./3.)
        tran_bebs_i['dursec'] = (2./86400.)*np.sqrt( 1 - (tran_bebs_i['sma']*au * ( 1. - tran_bebs_i['ecc']**2 ) / ( 1 + tran_bebs_i['ecc'] * np.cos(3*np.pi/2.0 - tran_bebs_i['omega']) ) * np.cos(tran_bebs_i['incl']))**2 / (tran_bebs_i['Rs']*Rsun + tran_bebs_i['A_Rs']*Rsun)**2)*(tran_bebs_i['Rs']*Rsun + tran_bebs_i['A_Rs']*Rsun)*np.sqrt(1. - tran_bebs_i['ecc']**2)* \
                    ( 1. + tran_bebs_i['ecc'] * np.cos(3*np.pi/2.0 - tran_bebs_i['omega']) )**(-1.)* \
                    (tran_bebs_i['P']*86400./(2. * np.pi * G * (tran_bebs_i['A_Ms']*Msun + tran_bebs_i['Ms']*Msun)))**(1./3.)
        tran_bebs_i=StellarStuff(tran_bebs_i) #Limb darkening, beaming, etc
        
        #Assuming companion << target here:
        tran_bebs_i['dilution']=np.power(2.512,tran_bebs_i['XAB_Pmag'].values-tran_bebs_i['AB_Pmag'].values)

        #Depths:
        tran_bebs_i['depthpri_ppm']=1e6*tran_bebs_i['dilution']*(tran_bebs_i['Rratio']**2/(1+tran_bebs_i['Rratio']**2*(tran_bebs_i['Teff']/tran_bebs_i['A_Teff'])**4))
        tran_bebs_i['depthsec_ppm']=1e6*tran_bebs_i['dilution']*(1-(1/(1+tran_bebs_i['Rratio']**2*(tran_bebs_i['Teff']/tran_bebs_i['A_Teff'])**4)))

        #SNRs:
        tran_bebs_i['SNRpri']=tran_bebs_i['depthpri_ppm']/tran_bebs_i['XAB_rms_hr']*(tran_bebs_i['durpri']*24*tran_bebs_i['P']/730.5)**(-0.5)
        tran_bebs_i['SNRsec']=tran_bebs_i['depthsec_ppm']/tran_bebs_i['XAB_rms_hr']*(tran_bebs_i['dursec']*24*tran_bebs_i['P']/730.5)**(-0.5)
        
        #Removing low-SNR BEBs:
        tran_bebs_i=tran_bebs_i.loc[(tran_bebs_i['SNRpri'].values>4.0)|(tran_bebs_i['SNRsec'].values>SNRthresh)]

        tran_bebs=tran_bebs.append(tran_bebs_i)
        
        nbebs=len(tran_bebs)
        print("BEB / ",iterations,"iterations. Currently:",nbebs,". Target:",len(fieldstars))
        iterations+=1
    
    #Appending randomly sorted planet hosts onto fieldstars df:
    tran_bebs=tran_bebs.iloc[np.random.choice(len(tran_bebs),len(fieldstars),replace=False)]
    fieldstars['type']='target'
    fieldstars=fieldstars.set_index(tran_bebs.blend_parent.values)
    fieldstars=pd.merge(fieldstars,stars.loc[tran_bebs_i['blend_parent']],left_index=True,right_index=True,how='left',sort=False)
    #RA and Dec:
    sep=np.sqrt(np.random.random(len(tran_bebs)))*np.clip(1.5*fieldstars.loc[tran_bebs['blend_parent'].values,'ap_radius'].values,0.0,7*15)
    angle=np.random.random(len(tran_bebs))*2*np.pi
    tran_bebs['ra']=fieldstars.loc[tran_bebs['blend_parent'],'ra'].values+(sep/3600)*np.cos(angle)
    tran_bebs['dec']=fieldstars.loc[tran_bebs['blend_parent'],'dec'].values+(sep/3600)*np.sin(angle)
    
    tran_bebs['known_blend']=np.random.random(len(tran_bebs))<gaia_detn_prob(tran_bebs['AB_Pmag']-fieldstars.loc[tran_bebs['blend_parent'].values,'Pmag'].values,sep)
    
    fieldstars['VAR']=np.tile(False,len(fieldstars))
    
    time=np.arange(0,365.25*2*(num_quarts/8),25/86400)
    #Getting beb lcs:
    for beb,row in tran_bebs.iterrows():
        
        lc=ellclc(t_obs=time,
              radius_1=(row['A_Rs']*Rsun)/(row['sma']*au), radius_2=(row['Rs']*Rsun)/(row['sma']*au),
              sbratio=(row['Teff']/row['A_Teff'])**4, incl=row['incl']*180/np.pi,
              light_3=0.0, t_zero=row['T0'], period=row['P'],a=(row['sma']*au)/(row['A_Rs']*Rsun),
              f_c=np.sqrt(row['ecc'])*np.cos(row['omega']), f_s=np.sqrt(row['ecc'])*np.sin(row['omega']), 
              q=(row['Ms']*Msun)/(row['A_Ms']*Msun),ldc_1=[row['A_LD_1'],row['A_LD_2']], ldc_2=[row['LD_1'],row['LD_2']],
              gdc_1=row['A_GD_1'],gdc_2=row['GD_1'], heat_1=row['A_albedo'], heat_2=row['albedo'],
              lambda_1=None,ld_1="quad",ld_2="quad",grid_1='sparse',grid_2='sparse',
              bfac_1=row['A_bfac'],bfac_2=row['bfac'],verbose=False)
        
        #Target blend_parentlc:
        if np.random.random()<var_prop:
            lc_par,parvardic=GetVar(fieldstars.loc[row['blend_parent'],'Age'],fieldstars.loc[row['blend_parent'],'Ms'],
                          fieldstars.loc[row['blend_parent'],'Teff'],fieldstars.loc[row['blend_parent'],'Rs'],time)
            for key in parvardic:
                fieldstars.loc[row['blend_parent'],key]=parvardic[key]
            fieldstars.loc[row['blend_parent'],'VAR']=True
            fieldstars.loc[row['blend_parent'],'LC']=True
            lcloc_par=True
            lc_par=-2.5*np.log10(lc_par)
        else:
            lcloc_par=False
            fieldstars.loc[row['blend_parent'],'LC']=False
            fieldstars.loc[row['blend_parent'],'VAR']=False
        
        #EB A star lc:
        if np.random.random()<var_prop:
            varlc,vardic=GetVar(row['Age'],row['A_Ms'],row['A_Teff'],row['A_Rs'],time)
            lc+=varlc
            tran_bebs.loc[beb,'VAR']=True
            for key in vardic:
                tran_bebs.loc[beb,key]=vardic[key]
            tran_bebs.loc[beb,'LC']=True
        lc=-2.5*np.log10(lc)

        print("writing",field_file_loc,beb)
        for Q in np.arange(num_quarts):
            low=int(np.clip(np.floor(0.125*(Q)*len(lc)-1),0,len(lc)))
            high=int(np.clip(np.ceil(0.125*(Q+1)*len(lc)+1),0,len(lc)))
            #Interpolating:
            lc_bool=InterpolateLC(lc[low:high],rms_hr=row['AB_rms_hr']*1e-6)
            
            lcloc=field_file_loc+'/lcs/'+str(beb)+'_Q'+str(Q+1)+'.txt'
            np.savetxt(lcloc,np.column_stack((time[low:high],lc[low:high]))[lc_bool],fmt='%.3f %.8f')
            if lcloc_par:
                lc_bool_par=InterpolateLC(lc_par[low:high],rms_hr=fieldstars.loc[row['blend_parent'],'rms_hr']*1e-6)
                np.savetxt(field_file_loc+'/lcs/'+str(row['blend_parent'])+'_Q'+str(Q+1)+'.txt',
                           np.column_stack((time[low:high],lc_par[low:high]))[lc_bool_par],fmt='%.3f %.8f')
    
    tran_bebs = tran_bebs.loc[:,~tran_bebs.columns.duplicated()]
    fieldstars = fieldstars.loc[:,~fieldstars.columns.duplicated()]
    
    return pd.concat([fieldstars,tran_bebs])
#pd.concat([fieldstars,tran_bebs],axis=0)

def getBPLs(fieldstars,stars,bgstars,field_file_loc,num_quarts=8,var_prop=0.5,SNRthresh=3.0):
    # get info for a "BPL" dip - i.e. a background transiting planet in the PLATO field.
    #INPUTS:
    # fieldstars - DataFrame generated with target star position info
    # stars - DataFrame from Besancon from which to take target star stellar parameters
    # bgstars - DataFrame from Besancon from which to take planet host star stellar parameters
    # field_file_loc - Location to save field files
    # num_quarts - Number of quarters to generate

    gaia_detn_prob=init_Gaia_Source()

    stars=stars.set_index(np.core.defchararray.add('BPL_',stars.index.values.astype(str)))
    bgstars=bgstars.set_index(np.core.defchararray.add('BPL_',bgstars.index.values.astype(str)))

    # This function gets BG transiting planets, given some star catologue and a dictionary of lightcurves
    bgstars=bgstars.loc[bgstars.Pmag<18]
    bgstars=bgstars.sort_values('Pmag')
    
    giant_enhance = lambda P,Rp:6-1*(np.clip(abs(9-Rp),0.75,5.0))#This enhances the number of giant planets by a factor of 5
    petigura = assemblePet(giant_enhance)
    
    nbpls=0
    tran_bpls=pd.DataFrame()
    iterations=1
    while nbpls<len(fieldstars):
        print("Sampling Blended transiting planets")
        #Choosing random star fainter than target to add the binary to:
        #Finds position of target star in bgstars (sorted by Pmag).
        #Finds position of 5mags fainter in bgstars.
        # then takes uniform sample between them
        if iterations>1:
            #Making sure we dont choose the same star here:
            bgstars=bgstars.loc[~np.in1d(bgstars.index,tran_bpls['parent'].values)]
        '''
        if len(bgstars)<len(stars):
            bgstars=np.choice(
        
        # then takes uniform sample between them
        choice_list=list(np.round(np.random.uniform(np.searchsorted(bgstars.Pmag.values,stars.Pmag.values),
                                                    np.searchsorted(bgstars.Pmag.values,6+stars.Pmag.values))).astype(int))
        all_idx=np.arange(np.min(choice_list),np.max(choice_list),1)
        all_idx=all_idx[~np.in1d(all_idx,choice_list)]
        #Making sure all choices are unique:
        non_unqs=[choice for choice in choice_list if choice_list.count(choice)>1]
        print(len(non_unqs),len(all_idx))
        if len(non_unqs)>0:
            for non_unq in non_unqs:
                if choice_list.count(non_unq)>1 and len(all_idx)>0:
                    new_choice=all_idx[np.argmin(abs(non_unq-all_idx))]
                    choice_list[choice_list.index(non_unq)]=new_choice
                    all_idx=all_idx[all_idx!=new_choice]
                if len(all_idx)==0:
                    #Not enough bg stars for the number of stars...
                    choice_list[choice_list.index(non_unq)]=np.nan
        choice_list=np.array(choice_list)
        if np.sum(np.isnan(choice_list))==0:
            print(len(bgstars),np.max(choice_list))
            bgstars_i=bgstars.iloc[choice_list].copy()
            bgstars_i=bgstars_i.set_index(np.core.defchararray.add(stars.index.values.astype(str),'_01'))
            bgstars_i['blend_parent']=stars.index.values
        else:
            #Nans in choice_list... This means bgstars<stars and we have to be careful to remove some from stars list
            bgstars_i=bgstars.iloc[choice_list[~np.isnan(choice_list)]].copy()
            bgstars_i=bgstars_i.set_index(np.core.defchararray.add(stars.loc[~np.isnan(choice_list)].index.values.astype(str),'_01'))
            bgstars_i['blend_parent']=stars.loc[~np.isnan(choice_list)].index.values
        '''
        choice_list=np.clip(np.round(np.random.uniform(np.searchsorted(bgstars.Pmag.values,stars.Pmag.values),
                                               np.searchsorted(bgstars.Pmag.values,6+stars.Pmag.values))).astype(int),
                            0,len(bgstars)-1)
        bgstars_i=bgstars.iloc[choice_list]
        bgstars_i['besancon_index']=bgstars_i.index.values[:]
        bgstars_i=bgstars_i.set_index(np.core.defchararray.add(stars.index.values.astype(str),'_01'))
        assert(len(bgstars_i)==len(stars.loc[~np.isnan(choice_list)]))
        rands=np.random.random((len(bgstars_i),len(petigura[:,0])-2,len(petigura[0,:])-2))
        whr=np.where(rands<np.tile(petigura[1:-1,1:-1],(len(bgstars_i),1,1)))
        rename_dic={'Ms':'A_Ms','Rs':'A_Rs','Pmag':'A_Pmag','Teff':'A_Teff','logg':'A_logg','FeH':'A_FeH',
                    'LD_1':'A_LD_1','LD_2':'A_LD_2','GD_1':'A_GD_1','bfac':'A_bfac','albedo':'A_albedo',
                    'V':'A_V','logL':'A_logL','rms_hr':'A_rms_hr'}
        tran_bpls_i=pd.DataFrame()
        tran_bpls_i['blend_parent']=stars.iloc[whr[0]].index
        tran_bpls_i['parent']=bgstars_i.iloc[whr[0]].index
        tran_bpls_i['system_label']='BPL'
        tran_bpls_i = pd.merge(tran_bpls_i, bgstars_i.rename(columns=rename_dic), left_on='parent',
                              right_index=True, how='left', sort=False)

        #tran_bpls_i =  bgstars.rename(columns=rename_dic)
        tran_bpls_i['type']='blend'
        tran_bpls_i['Rs']=np.random.uniform(petigura[whr[1]+1,0],petigura[whr[1]+2,0])*(Rearth/Rsun)
        tran_bpls_i['Ms']=PlanetRtoM(tran_bpls_i['Rs']*(Rsun/Rearth))*(Mearth/Msun)
        tran_bpls_i['P']=np.exp(np.random.uniform(np.log(petigura[0,whr[2]+1]),np.log(petigura[0,whr[2]+2])))
        tran_bpls_i['sma']=kepp2a_pl(tran_bpls_i['P'].values,tran_bpls_i['A_Ms'].values)
        tran_bpls_i['T0']=np.random.uniform(np.tile(0,tran_bpls_i.shape[0]),tran_bpls_i['P'])
        #Eccentricity, omega and incl:
        kipping= np.random.normal([1.12,3.09,0.697,3.27],[0.1,0.3,0.2,0.34])
        tran_bpls_i['ecc']=stats.beta.rvs(kipping[0],kipping[1],size=tran_bpls_i.shape[0])
        tran_bpls_i.loc[tran_bpls_i['P']<=382.5,'ecc']=stats.beta.rvs(kipping[2],kipping[3],size=np.sum(tran_bpls_i['P'].values<=382.5))

        tran_bpls_i['omega']=np.random.random(len(tran_bpls_i))*2*np.pi
        #Adding dependence on number of iterations to make later runs more likely to find a solution...
        tran_bpls_i['incl']=np.arccos(np.random.uniform(-1*iterations**-0.2,iterations**-0.2,len(tran_bpls_i)))

        tran_bpls_i['bpri']=abs(b_ecc_pri(tran_bpls_i['ecc'].values,tran_bpls_i['omega'].values,
                                       tran_bpls_i['sma'].values,tran_bpls_i['incl'].values,
                                       tran_bpls_i['A_Rs'].values)) #impact parameter at primary eclipse
        tran_bpls_i['Rratio']=tran_bpls_i['Rs'].values/tran_bpls_i['A_Rs'].values
        tran_bpls_i['pri_ecl']=tran_bpls_i['bpri']<(1+tran_bpls_i['Rratio'])
        
        #removing those  which do not transit:
        tran_bpls_i=tran_bpls_i.loc[tran_bpls_i['pri_ecl']]
        

        tran_bpls_i['durpri'] = (2./86400.)*np.sqrt( 1 - (tran_bpls_i['sma']*au * ( 1. - tran_bpls_i['ecc']**2 ) / ( 1 + tran_bpls_i['ecc'] * np.cos(np.pi/2.0 - tran_bpls_i['omega']) ) * np.cos(tran_bpls_i['incl']))**2 / (tran_bpls_i['A_Rs']*Rsun + tran_bpls_i['Rs']*Rsun)**2)*(tran_bpls_i['Rs']*Rsun + tran_bpls_i['A_Rs']*Rsun)*np.sqrt(1. - tran_bpls_i['ecc']**2)* \
                    ( 1. + tran_bpls_i['ecc'] * np.cos(np.pi/2.0 - tran_bpls_i['omega']) )**(-1.)* \
                    (tran_bpls_i['P']*86400./(2. * np.pi * G * (tran_bpls_i['A_Ms']*Msun + tran_bpls_i['Ms']*Msun)))**(1./3.)

        tran_bpls_i['dilution']=(np.power(2.512,stars.loc[tran_bpls_i['blend_parent'],'Pmag'].values-tran_bpls_i['A_Pmag']))        
        #Removing planets orbiting within 2Rs:
        tran_bpls_i=tran_bpls_i.loc[tran_bpls_i['sma']*au>(2*tran_bpls_i['A_Rs']*Rsun)]
        
        #combined mag & RMS with target star fluxes:
        tran_bpls_i['XAB_Pmag']=-2.512*np.log10(np.power(2.512,-1*stars.loc[tran_bpls_i['blend_parent'],'Pmag'].values)+\
                                               np.power(2.512,-1*tran_bpls_i['A_Pmag'].values))
        Nscopes_arr=np.tile(np.nanmedian(fieldstars['Nscopes']),len(tran_bpls_i))
        tran_bpls_i['XAB_rms_hr']=getPmagRms(tran_bpls_i['XAB_Pmag'],Nscopes_arr)[0]

        #Depths:
        tran_bpls_i['depthpri_ppm']=1e6*tran_bpls_i['dilution']*(tran_bpls_i['Rratio']**2)
        #/(1+tran_bpls_i['Rratio']**2*(tran_bpls_i['Teff']/tran_bpls_i['A_Teff'])**4))#Dont need to include brightness

        #SNRs:
        tran_bpls_i['SNRpri']=tran_bpls_i['depthpri_ppm']/tran_bpls_i['XAB_rms_hr']*(tran_bpls_i['durpri']*24*tran_bpls_i['P']/730.5)**(-0.5)
        
        #print(np.sum(tran_bpls_i['SNRpri']>3.0),"/",len(tran_bpls_i),"have high-SNR with median of:",np.median(tran_bpls_i['SNRpri']))
        #Removing low-SNR bpls:
        tran_bpls_i=tran_bpls_i.loc[tran_bpls_i['SNRpri']>SNRthresh]

        #Removing 2nd/3rd/4th planets (which have low-SNR)
        tran_bpls_i = tran_bpls_i.sort_values('SNRpri')
        tran_bpls_i = tran_bpls_i.drop_duplicates(subset='parent', keep="first")
        
        tran_bpls_i=tran_bpls_i.set_index(np.core.defchararray.add(tran_bpls_i['parent'].values.astype(str),'_b'))

        tran_bpls=tran_bpls.append(tran_bpls_i)

        nbpls=len(tran_bpls)
        print("BPL / ",iterations,"iterations. Currently:",nbpls,". Target:",len(fieldstars))
        iterations+=1

    #Appending randomly sorted planet hosts onto fieldstars df:
    tran_bpls=tran_bpls.iloc[np.random.choice(len(tran_bpls),len(fieldstars),replace=False)]
    

    #Extra stuff to add to planets:
    tran_bpls['Ms']=PlanetRtoM(tran_bpls['Rs']*(Rsun/Rearth))*(Mearth/Msun)
    tran_bpls['Teff']=SurfaceT(tran_bpls['sma'].values,tran_bpls['A_Rs'].values,tran_bpls['A_Teff'].values)
    tran_bpls['albedo']=get_albedo(tran_bpls['Teff'].values)
   
    #Need to redo contration with multiplicities...
    tran_bpls['GD_1']=np.zeros(tran_bpls.shape[0])
    tran_bpls['LD_1']=np.ones(tran_bpls.shape[0])
    tran_bpls['LD_2']=np.zeros(tran_bpls.shape[0])
    tran_bpls.loc[np.isnan(tran_bpls['sma']),'sma']=(((86400*tran_bpls['P'])**2*6.67e-11*(tran_bpls['Ms']*Msun+tran_bpls['A_Ms']*Msun))/(4*np.pi**2))**0.3333/au
    tran_bpls['bfac']=Get_Beaming(tran_bpls['Teff'].values)

    #The non-transiting target stars are added to the fieldstars df:
    bpl_targets=stars.loc[tran_bpls.blend_parent]
    fieldstars=fieldstars.set_index(bpl_targets.index.values)
    fieldstars=pd.merge(fieldstars,bpl_targets,left_index=True,right_index=True,how='left', sort=False)
    #fieldstars=pd.concat([fieldstars,bpl_targets],axis=1)
    fieldstars['type']='target'
    #The eclipsing A stars are added to the fieldstars df:
    print("BPL:",len(tran_bpls),len(np.unique(tran_bpls.blend_parent)),len(fieldstars),fieldstars.loc[tran_bpls.blend_parent,'ap_radius'].values)
    sep=np.sqrt(np.random.random(len(tran_bpls)))*np.clip(1.5*fieldstars.loc[tran_bpls.blend_parent,'ap_radius'].values,0.0,7*15)
    angle=np.random.random(len(tran_bpls))*2*np.pi
    tran_bpls['ra']=fieldstars.loc[tran_bpls.blend_parent,'ra'].values+(sep/3600)*np.cos(angle)
    tran_bpls['dec']=fieldstars.loc[tran_bpls.blend_parent,'dec'].values+(sep/3600)*np.sin(angle)
    tran_bpls['known_blend']=np.random.random(len(tran_bpls))<gaia_detn_prob(tran_bpls['A_Pmag'].values-fieldstars.loc[tran_bpls['blend_parent'],'Pmag'].values,sep)
    
    time=np.arange(0,365.25*2*(num_quarts/8),25/86400)
    #Getting bpl lcs:
    for bpl,row in tran_bpls.iterrows():
        lc=ellclc(t_obs=time,
              radius_1=(row['A_Rs']*Rsun)/(row['sma']*au), radius_2=(row['Rs']*Rsun)/(row['sma']*au),
              sbratio=(row['Teff']/row['A_Teff'])**4, incl=row['incl']*180/np.pi,
              light_3=0.0, t_zero=row['T0'], period=row['P'],a=(row['sma']*au)/(row['A_Rs']*Rsun),
              f_c=np.sqrt(row['ecc'])*np.cos(row['omega']), f_s=np.sqrt(row['ecc'])*np.sin(row['omega']), 
              q=(row['Ms']*Msun)/(row['A_Ms']*Msun),ldc_1=[row['A_LD_1'],row['A_LD_2']], ldc_2=[0.0,1.0],
              gdc_1=row['A_GD_1'],gdc_2=0.0, heat_1=row['A_albedo'], heat_2=row['albedo'],
              lambda_1=None,ld_1="quad",ld_2="quad",grid_1='sparse',grid_2='sparse',
              bfac_1=row['A_bfac'],bfac_2=row['bfac'],verbose=False)
        
        #Planet host variability:
        if np.random.random()<var_prop:
            varlc,vardic=GetVar(row['Age'],row['A_Ms'],row['A_Teff'],row['A_Rs'],time)#Age,Ms,Teff,Rs,time
            lc+=varlc
            for key in vardic:
                tran_bpls.loc[bpl,key]=vardic[key]
            tran_bpls.loc[bpl,'VAR']=True
        tran_bpls.loc[bpl,'LC']=True
        lc=-2.5*np.log10(lc)
        
        #Target blend lc:
        if np.random.random()<var_prop:
            lc_par,parvardic=GetVar(fieldstars.loc[row['blend_parent'],'Age'],fieldstars.loc[row['blend_parent'],'Ms'],
                                    fieldstars.loc[row['blend_parent'],'Teff'],fieldstars.loc[row['blend_parent'],'Rs'],time)
            for key in parvardic:
                fieldstars.loc[row['blend_parent'],key]=parvardic[key]
            fieldstars.loc[row['blend_parent'],'VAR']=True
            
            fieldstars.loc[row['blend_parent'],'LC']=True
            lcloc_par=True
            lc_par=-2.5*np.log10(lc_par)
        else:
            lcloc_par=False
            fieldstars.loc[row['blend_parent'],'LC']=False
        
        print("writing",field_file_loc,bpl)
        for Q in np.arange(num_quarts):
            low=int(np.clip(np.floor(0.125*(Q)*len(lc)-1),0,len(lc)))
            high=int(np.clip(np.ceil(0.125*(Q+1)*len(lc)+1),0,len(lc)))
            #Interpolating:
            lc_bool=InterpolateLC(lc[low:high],rms_hr=row['A_rms_hr']*1e-6)
            lcloc=field_file_loc+'/lcs/'+str(bpl)+'_Q'+str(Q+1)+'.txt'
            np.savetxt(lcloc,np.column_stack((time[low:high],lc[low:high]))[lc_bool],fmt='%.3f %.8f')            
            if lcloc_par:
                lc_bool_par=InterpolateLC(lc_par[low:high],rms_hr=fieldstars.loc[row['blend_parent'],'rms_hr']*1e-6)
                np.savetxt(field_file_loc+'/lcs/'+str(bpl)+'_Q'+str(Q+1)+'.txt',
                           np.column_stack((time[low:high],lc_par[low:high]))[lc_bool_par],fmt='%.3f %.8f')
                lcloc_par=False
            
    return pd.concat([fieldstars,tran_bpls],axis=0)

def getNAs(fieldstars,stars,field_file_loc,num_quarts=8,var_prop=0.5):
    # get info for a "NA" dip - i.e. a dipless star in the PLATO field.
    # fieldstars - DataFrame generated with target star position info
    # stars - DataFrame from Besancon from which to take target star stellar parameters
    # field_file_loc - Location to save field files
    # num_quarts - Number of quarters to generate
    print("Getting stars without any dips")
    newstars=stars.iloc[np.random.choice(len(stars),len(fieldstars),replace=False)].copy()
    fieldstars=pd.merge(fieldstars.set_index(newstars.index.values),newstars,left_index=True,right_index=True,how='left', sort=False)
    fieldstars=fieldstars.set_index(np.core.defchararray.add('NA_',fieldstars.index.values.astype(str)))
    time=np.arange(0,365.25*2*(num_quarts/8),25/86400)
    for star,row in fieldstars.iterrows():        
        if np.random.random()<var_prop:
            lc,vardic=GetVar(row['Age'],row['Ms'],row['Teff'],row['Rs'],time)
            fieldstars.loc[star,'VAR']=True
            fieldstars.loc[star,'LC']=True
            for key in vardic:
                fieldstars.loc[star,key]=vardic[key]
            stars.loc[star,'VAR']=True
            
            print("writing",field_file_loc,star)
            for Q in np.arange(num_quarts):
                low=int(np.clip(np.floor(0.125*(Q)*len(lc)-1),0,len(lc)))
                high=int(np.clip(np.ceil(0.125*(Q+1)*len(lc)+1),0,len(lc)))
                #Interpolating:
                lc_bool=InterpolateLC(lc[low:high],rms_hr=row['rms_hr']*1e-6)
                lcloc=field_file_loc+'/lcs/'+str(star)+'_Q'+str(Q+1)+'.txt'
                np.savetxt(lcloc,np.column_stack((time[low:high],lc[low:high]))[lc_bool],fmt='%.3f %.8f')
        else:
            fieldstars.loc[star,'LC']=False
    return fieldstars
    
def getDip(diptype,fieldstars,stars,bgstars,field_file_loc,file_prefix,num_quarts=8):
    # For multiprocessing - given label, assigns the requisite getDIP function
    '''
    print("diptype:",diptype)
    print("eg fieldstars:",fieldstars.iloc[0])
    print("eg stars:",stars.iloc[0])
    print("eg bgstars:",bgstars.iloc[0])
    print("eg field_file_loc:",field_file_loc)
    print("eg num_quarts",num_quarts)
    '''
    print("MP started for",diptype)
    try:
        if len(fieldstars.loc[fieldstars['system_label'].values==diptype])==0:
            print("NO ",diptype," IN catalogue",type(diptype))
        elif os.path.exists(os.path.join(field_file_loc,file_prefix+'_generated_'+diptype.lower()+'s.csv')):
            print(os.path.join(field_file_loc,file_prefix+'_generated_'+diptype.lower()+'s.csv')," already exists")
        elif diptype=='PL':
            df = getPLs(fieldstars.loc[fieldstars['system_label'].values==diptype],stars,field_file_loc,num_quarts)
        elif diptype=='EB':
            df = getEBs(fieldstars.loc[fieldstars['system_label'].values==diptype],stars,field_file_loc,num_quarts)
        elif diptype=='BEB':
            df = getBEBs(fieldstars.loc[fieldstars['system_label'].values==diptype],stars,bgstars,field_file_loc,num_quarts)
        elif diptype=='BPL':
            df = getBPLs(fieldstars.loc[fieldstars['system_label'].values==diptype],stars,bgstars,field_file_loc,num_quarts)
        elif diptype=='NA':
            df = getNAs(fieldstars.loc[fieldstars['system_label'].values==diptype],stars,field_file_loc,num_quarts)
        if len(df)>0:
            print("saving",os.path.join(field_file_loc,file_prefix+'_generated_'+diptype.lower()+'s.csv'))
            df.to_csv(os.path.join(field_file_loc,file_prefix+'_generated_'+diptype.lower()+'s.csv'))
    except Exception as e:
        traceback.print_exc()
        print("problem when getting",diptype,":")
        print(e)
    
def GetVar(Age,Ms,Teff,Rs,time):
    # Gets variability of a star - both osscillations and rotation
    # - Age in Gyr
    # - Ms in solar masses
    # - Teff (K)
    # - Rs in solar radii
    # - time array
    
    #Getting stochastic SHM oscillations:
    oscs=Oscillations(Ms,Teff,Rs,time)[0]
    
    #Getting quasi-periodic rotation:
    rots,qpcols=QPlc_fromTeff(Teff,Age,time)
    rots=rots[0]
    if len(rots)>len(time):
        rots=rots[:len(time)]
    if len(oscs)>len(time):
        oscs=oscs[:len(time)]
    
    qpcols={'rot_Q':qpcols[0][0],'rot_per':qpcols[0][1],'rot_amp':qpcols[0][2],"rot_std":np.std(rots),"osc_std":np.std(oscs)}
    
    #print(np.shape(oscs),np.shape(rots))
    varbly=oscs+(rots-1.0)

    return varbly, qpcols


def rebin_hdf5_file(hdf5file,starcatloc,injcatloc,bin_factor=24):
    import h5py
    import tables
    #Open HDF5 page:
    platosim_output = tables.open_file(hdf5file, mode='r')

    #Access time and binning
    t=platosim_output.root.StarPositions.Time[:]
    cad=np.nanmedian(np.diff(platosim_output.root.StarPositions.Time[:]))
    idxarr=np.digitize(t,np.arange(t[0],t[-1]+0.1*cad,n_per_bin*cad))
    newt=np.array([np.average(t[idxarr==n]) for n in np.unique(idxarr)])
    imgids=np.array(list(platosim_output.root.Images._v_children.keys()))
    
    #Access star catalogue to get the stars.
    df=pd.DataFrame.from_csv(injcatloc)
    df.loc[pd.isnull(df['parent']),'parent']=df.loc[pd.isnull(df['parent'])].index.values
    df=df.drop_duplicates(subset='parent')    
    df=df.set_index(df['parent'].values)

    starcat=pd.read_table(starcatloc)
    starcat=starcat.set_index(starcat['ID'].values)
    df=pd.merge(df,starcat.rename(columns={"#RA(deg)":"PS3_RA","DEC(deg)":"PS3_Dec","Vmag":"PS3_Pmag","ID":"PS3_ID","Known":"PM3_Known"}),left_index=True,right_index=True)
    
    targets=df.loc[df['type']=='target']
    targets_in_starcat=SkyCoord(targets['PS3_RA'].values*u.deg,targets['PS3_Dec'].values*u.deg)
    stars_in_hdf5=SkyCoord(platosim_output.root.StarCatalog.RA[:]*u.deg,platosim_output.root.StarCatalog.Dec[:]*u.deg)
    idx, d2d, _ = targets_in_starcat.match_to_catalog_sky(stars_in_hdf5)
    targets['hdf5_idx_dist']=d2d.arcsec
    targets['hdf5_idx']=idx
    targets['hdf5_name']=np.tile('',len(targets))
    targets.loc[targets['hdf5_idx_dist']<0.05,'hdf5_name']=platosim_output.root.StarCatalog.starIDs[targets.loc[targets['hdf5_idx_dist']<0.05,'hdf5_idx'].values]
    targets=targets.loc[targets['hdf5_idx_dist']<0.05]
    targets['colPix_ccd']=np.array([platosim_output.root.StarCatalog['colPix'][ix] for ix in targets['hdf5_idx'].values])
    targets['rowPix_ccd']=np.array([platosim_output.root.StarCatalog['rowPix'][ix] for ix in targets['hdf5_idx'].values])
    
    targets['colPix_subField']=targets['colPix_ccd'] - platosim_output.root.InputParameters.SubField._v_attrs['ZeroPointColumn']
    targets['rowPix_subField']=targets['rowPix_ccd'] - platosim_output.root.InputParameters.SubField._v_attrs['ZeroPointRow']

    targets=targets.loc[(targets['colPix_subField']>1)*(targets['colPix_subField']<99)*(targets['rowPix_subField']>1)*(targets['rowPix_subField']<99)]

    #Saving list of targets for photometric extraction:
    np.savetxt(hdf5file.replace('.hdf5','list_targets.txt'),targets['hdf5_name'].values)

    #Open new hdf5 file and init data
    if os.path.exists(hdf5file.replace('.hdf5','_binned_imgts.hdf5')):
        os.system("rm "+hdf5file.replace('.hdf5','_binned_imgts.hdf5'))
    
    hdf5_new = h5py.File(hdf5file.replace('.hdf5','_binned_imgts.hdf5'), mode='w')
    hdf5_new.create_dataset("Time", (len(newt),), np.float32,newt)
    
    hdf5_new.create_dataset("image_start", (100,100,), np.int16, platosim_output.root.Images[imgids[0]])
    hdf5_new.create_dataset("image_end", (100,100,), np.int16, platosim_output.root.Images[imgids[-1]])
    
    img_names=[]
    for n_idx in np.unique(idxarr):
        img_names+=[str(imgids[idxarr==n_idx][0])+'-'+str(imgids[idxarr==n_idx][-1]).replace('image','')]
    string_len='S'+str(len(img_names[-1]))
    hdf5_new.create_dataset('img_names', (len(img_names),), string_len, [nstr.encode("ascii", "ignore") for nstr in img_names])

    stars_group=hdf5_new.create_group("Stars")
    print("#Loop through stars in field:")
    for i_s in targets.index.values:
        i_star=stars_group.create_group(i_s)
        #Integer position in the out.root.StarCatalog catalogue:
        i_star.attrs['StarCatalog_idx'] = targets.loc[i_s,'hdf5_idx']
        i_star.attrs['ID_int'] = targets.loc[i_s,'hdf5_name']
        i_star.attrs['ID_str'] = i_s
        i_star.attrs['RA'] = targets.loc[i_s,'ra']
        i_star.attrs['Dec'] = targets.loc[i_s,'dec']
        i_star.attrs['Vmag'] = platosim_output.root.StarCatalog['Vmag'][targets.loc[i_s,'hdf5_idx']]
        i_star.attrs['colPix'] = targets.loc[i_s,'colPix_subField']
        i_star.attrs['rowPix'] = targets.loc[i_s,'rowPix_subField']
        
        #platosim_output.root.StarCatalog['rowPix'][targets.loc[i_s,'hdf5_idx']]
        i_star.attrs['aperture_radius'] = np.clip(3.5+0.5*(9-i_star.attrs.get('Vmag')),3.5,5)
        col_in=abs(np.arange(100)-i_star.attrs.get('colPix'))<=i_star.attrs.get('aperture_radius')
        row_in=abs(np.arange(100)-i_star.attrs.get('rowPix'))<=i_star.attrs.get('aperture_radius')
        #[row,col]
        ap=np.tile(False,(100,100));ap[col_in[np.newaxis,:]*row_in[:,np.newaxis]]=True
        i_star.create_dataset('aperture',(100,100,),'bool',ap)
        i_star.attrs['in_aperture']=int(np.sum(ap)>0.0)

        i_star.attrs['row_aperture_diameter']=np.sum(row_in)
        i_star.attrs['col_aperture_diameter']=np.sum(col_in)
        
        i_star.create_dataset('imagettes',(len(newt),i_star.attrs.get('row_aperture_diameter'),i_star.attrs.get('col_aperture_diameter'),),dtype='f')
        
    for n,ix_img in enumerate(np.unique(idxarr)):
        bin_img=np.average(np.dstack([platosim_output.root.Images[imgid] for imgid in imgids[idxarr==ix_img]]),axis=2)
        for i_s in targets.index.values:
            stars_group[i_s]['imagettes'][n]=mask_nd(bin_img,stars_group[i_s]['aperture'][:,:])
        
        if n%500==0:
            print(n,"imagettes processed / ",np.max(np.unique(idxarr)))
    hdf5_new.close()
    platosim_output.close()
    return targets['hdf5_name'].values

def imports():
    # Getting all imported stuff
    import sys 
    for module in sys.modules: 
        try: 
            print(module,'==',sys.modules[module].__version__) 
        except: 
            try: 
                if  type(modules[module].version) is str: 
                    print(module,'==',sys.modules[module].version) 
                else: 
                    print(module,'==',sys.modules[module].version()) 
            except: 
                try: 
                    print(module,'==',sys.modules[module].VERSION) 
                except: 
                    pass 

if __name__=="__main__":
    #FieldGen.py
    #Arguments:
    # hemisphere = North or South
    # N_fields =  number of fields to generate 
    # ext = string to add to output files
    # folder =  output folder location 
    # num_quarts = number of quarters
    # MP - whether to use multiprocessing or not
    # overwrite = whether to overwrite previous data
    #e.g.:
    # python FieldGen.py North 1 test_oct /data/PLATO/Sims 1 True True
    
    defaults=['','North',1,'','/data/PLATO/Sims',8,False]
    if len(sys.argv)<8:
        args=sys.argv
        args=args+defaults[len(args):]
    else:
        args=sys.argv[:8]
    _=FieldGen(hemisphere=args[1],N_fields=int(args[2]),ext=args[3],outfileloc=args[4],
                num_quarts=int(args[5]),MP=args[6],overwrite=args[7])
    
    #Doing this just to print the imports for a requirements.txt file:
    
