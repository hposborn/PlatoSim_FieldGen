"""
#Usage: simQuarter.py <camaraGroupNr> <cameraNr> <quarterNr>
#
#cameraGroupNr: either 1,2,3, or 4
#cameraNr: either 1,2,3,4,5 or 6
#quarterNr: either 1,2,3,4,5,6,7 or 8
#
#Example: $ python simQuarter.py 2 5 6
"""

import os
import sys
import math

import numpy as np
from simulation import Simulation
from referenceFrames import getCCDandPixelCoordinates
from referenceFrames import platformToTelescopePointingCoordinates
from referenceFrames import sunSkyCoordinatesAwayfromPlatformPointing
from referenceFrames import CCD
 
raPlatform  = np.deg2rad({raPlatform})           # Platform right ascension pointing coordinate
decPlatform = np.deg2rad({decPlatform})           # Platform declination pointing coordinate

raCenter  = np.deg2rad({raField})         # Right ascension on which to centre the subfield
decCenter = np.deg2rad({decField})           # Declination on which to centre the subfield

numColumnsSubField = 100                 # Number of columns in the modelled sub-field [pixels]
numRowsSubField = 100                    # Number of rows in the modelled sub-field [pixels]

# Plato has 4 groups, of each 6 telescopes. Each quarter of the year, the platform
# (not the telescopes!) is rotated along its roll axis to repoint the solar panels
# towards the Sun.

# Select which camera from the arguments with which the script is called

outputDir=os.path.dirname(os.path.abspath(__file__))#Previously, this was hardcoded as: "{outputloc}"
#inputFile   = os.path.join(outputDir,"{fieldID}_Q{quart}_group{group}_camera{scope}_run.yaml")

print("group = {group}, telescope = {scope}, quarter = {quart}")

# Output will be stored in e.g. Run1_Q1_group2_camera7.hdf5

outputFilePrefix = "{fieldID}_Q{quart}_group{group}_camera{scope}"
#sim = Simulation(outputFilePrefix, inputFile)
sim = Simulation(outputFilePrefix)

sim.outputDir = outputDir

sim["General/ProjectLocation"] = outputDir

# Setting input files:
sim["Sky/VariableSourceList"] = os.path.join(outputDir,"{fieldID}_Q{quart}_varcat.txt")
sim["ObservingParameters/StarCatalogFile"] =  os.path.join(outputDir,"{fieldID}_starcat.txt")

# Set the simulation parameters that are the same for any quarter and for any telescope

sim["ObservingParameters/RApointing"] = np.rad2deg(raPlatform)
sim["ObservingParameters/DecPointing"] = np.rad2deg(decPlatform)
sim["SubField/NumColumns"] = numColumnsSubField
sim["SubField/NumRows"] = numRowsSubField

# Set the telescope group ID, this is needed for the subfield calculations later on.

sim["Telescope/GroupID"] = {group}

# Set the quarter specific parameters

sim["RandomSeeds/JitterSeed"] = 2033429158 + 100 * {quart}
sim["Platform/SolarPanelOrientation"] = math.fmod({quart} * 90., 360.)         # 0, 90, 180, and 270 degrees for Q1, Q2, Q3, and Q4

exposureTime = sim["ObservingParameters/ExposureTime"]
readoutTime, dummy = sim.getReadoutTime()
numExposures = int(365.25/4*86400/(exposureTime + readoutTime))
sim["ObservingParameters/NumExposures"] = numExposures
sim["ObservingParameters/BeginExposureNr"] = {quart} * numExposures

# Attempt to set a subfield around the specified coordinates on one of the 4 CCDs of the telescope.
# This will fail (return value == False) if the subfield is not visible by any of the 4 CCDs or
# that the subfield is too large to entirely fit on a CCD.
# If successful, the function sets the CCD and subfield parameters correctly in the 'sim' object.

isSuccessful =  sim.setSubfieldAroundCoordinates(raCenter, decCenter, numColumnsSubField, numRowsSubField, normal=True)

if isSuccessful:
    # Make sure that the following random seeds differ for each telescope and for each quarter
    # We assume a maximum of 8 quarter and 4 camera groups
    
    randomSeedOffset = 1000 * 8 * {quart} +  10 * 4 * {group} + {scope}
    
    sim["RandomSeeds/ReadOutNoiseSeed"] = 1424949740 + randomSeedOffset
    sim["RandomSeeds/PhotonNoiseSeed"]  = 1533320336 + randomSeedOffset
    sim["RandomSeeds/FlatFieldSeed"]    = 1633320381 + randomSeedOffset
    sim["RandomSeeds/DriftSeed"]        = 1733429158 + randomSeedOffset
    
    # Run the PlatoSim simulator
    # logLevel can 1 (least verbose) to 3 (most verbose)
    
    simFile = sim.run(logLevel=2)
        
    targ_ids=rebin_hdf5_file(os.path.join(outputDir,outputFilePrefix+'.hdf5'),
                              starcatloc=os.path.join(outputDir,"{fieldID}_starcat.txt"),
                              injcatloc=os.path.join(outputDir,"{fieldID}_final_fieldcat.csv"))
    
    #Running photometric Extraction:
    import platophot
    
    #np.genfromtxt(os.path.join(outputDir,outputFilePrefix+"list_targets.txt"))
    
    #Running photometric pipeline:
    platophot.photometry(inputFilePath=os.path.join(outputDir,outputFilePrefix+".hdf5"),
                        outputFilePath=os.path.join(outputDir,outputFilePrefix+"_extracted_photometry.hdf5"),
                        targetIDs=list(targ_ids.astype(int).astype(str))

    #tar-zipping hdf5 imagette file:
    os.system("tar -zcf "+os.path.join(outputDir,outputFilePrefix+"_binned_imgts.tar.gz")+\
              " "+os.path.join(outputDir,outputFilePrefix+"_binned_imgts.hdf5"))
    #tar-zipping hdf5 lightcurve file:
    os.system("tar -zcf "+os.path.join(outputDir,outputFilePrefix+"_extracted_photometry.tar.gz")+\
              " "+os.path.join(outputDir,outputFilePrefix+"_extracted_photometry.hdf5"))
    #and removing HDF5 file:
    os.system("rm "+os.path.join(outputDir,outputFilePrefix+".hdf5"))
    
else:
    print("Sub-field does not lay entirely on any of the CCDs of telescope {scope} of group {group} in quarter Q{quart}")
    