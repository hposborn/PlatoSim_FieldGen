#!/usr/bin/env python
# -*- coding: utf-8 -*- 
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
import pickle

import numpy as np
from simulation import Simulation
from referenceFrames import getCCDandPixelCoordinates
from referenceFrames import platformToTelescopePointingCoordinates
from referenceFrames import sunSkyCoordinatesAwayfromPlatformPointing
from referenceFrames import CCD
from PlatoSim_FieldGen.FieldGen import rebin_hdf5_file
 
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
Quarter_adj={quart}-1

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
sim["ObservingParameters/ExposureTime"]= 21
sim["ObservingParameters/RApointing"]  = np.rad2deg(raPlatform)
sim["ObservingParameters/DecPointing"] = np.rad2deg(decPlatform)
sim["ObservingParameters/Fluxm0"]      = 1.00179e8   # Photon flux of a V=0 G2V-star [phot/s/m^2/nm]
sim["ObservingParameters/MissionDuration"]= 6.5      # Total duration of the mission [yr], relevant for BOL->EOL degradation

sim["Sky/SkyBackground"]               = 342.            # Stellar + zodiacal background level (excl. transmission efficiency). Set negative to compute.     [phot/pix/s]
sim["Sky/IncludeVariableSources"]      = "yes"             # Include time series of variable stars or not
sim["Sky/IncludeCosmicsInSubField"]    = "no"              # Whether or not to include cosmics in the subfield on the CCD
sim["Sky/IncludeCosmicsInSmearingMap"] = "yes"             # Whether or not to include cosmics in the overscan map
sim["Sky/IncludeCosmicsInBiasMap"]     = "yes"             # Whether or not to include cosmics in the prescan map
sim["Sky/Cosmics/CosmicHitRate"]       = 10              # Cosmic hit rate [events / cm^2 / s]

'''
sim["Sky/Cosmics/TrailLength"]         = [0, 15]         # Intervalof the length of the cosmic trails [pixels]
sim["Sky/Cosmics/Intensity"]           = [2000, 40000]   # Interval of the total number of e- in a cosmic hit [e-]
'''

sim["Platform/SolarPanelOrientation"]  = math.fmod(Quarter_adj * 90., 360.)         # 0, 90, 180, and 270 degrees for Q1, Q2, Q3, and Q4
sim["Platform/UseJitter"]              = "yes"             # yes or no. If no, ignore everything below.
sim["Platform/UseJitterFromFile"]      = "no"              # yes or no. If yes, ignore RMS and timescale below.
sim["Platform/JitterYawRms"]           = 0.04            # [arcsec]
sim["Platform/JitterPitchRms"]         = 0.04            # [arcsec]
sim["Platform/JitterRollRms"]          = 0.08            # [arcsec]
sim["Platform/JitterTimeScale"]        = 3600.           # [s]
sim["Platform/JitterFileName"]         = "inputfiles/PlatoJitter_OHB.txt"


sim["Telescope/GroupID"]               = {group}        # There are four camera groups: 1, 2, 3, 4, Fast, Custom
'''
sim["Telescope/AzimuthAngle"]          = 0.0            # Azimuth angle of telescope on the platform [deg]  - used when GroupID = Custom
sim["Telescope/TiltAngle"]             = 0.0            # Tilt angle of the telescope on the platform [deg] - used when GroupID = Custom
sim["Telescope/LightCollectingArea"]   = 113.1            # Effective area of 1 telescope [cm^2]
sim["Telescope/TransmissionEfficiency/BOL"] = 0.8162          # Beginning of Life value in [0,1]
sim["Telescope/TransmissionEfficiency/EOL"] = 0.7975          # End of Life value in [0,1]
sim["Telescope/UseDrift"]              = "yes"             # yes or no. If no, ignore everything below.
sim["Telescope/UseDriftFromFile"]      = "no"              # yes or no. If yes: ignore RMS and timescale below
sim["Telescope/DriftYawRms"]           = 2.0             # RMS of thermo-elastic drift in yaw [arcsec]
sim["Telescope/DriftPitchRms"]         = 2.0             # RMS of thermo-elastic drift in pitch [arcsec]
sim["Telescope/DriftRollRms"]          = 2.0             # RMS of thermo-elastic drift in roll [arcsec]
sim["Telescope/DriftTimeScale"]        = 86400.          # Timescale of thermo-elastic drift [s]
sim["Telescope/DriftFileName"]         = "inputfiles/drift.txt"

sim["Camera/PlateScale"]               = 0.8333          # [arcsec/micron]
sim["Camera/FocalPlaneOrientation/Source"] = "ConstantValue"   # Either ConstantValue (constant in time) or FromFile (time dependent)
sim["Camera/FocalPlaneOrientation/ConstantValue"] = 0.0             # [degrees]. Used in case Source is ConstantValue.
sim["Camera/FocalPlaneOrientation/FromFile"] = "inputfiles/fporientation.txt"  # Used in case Source is FromFile. time [s] & FP orientation [deg]
sim["Camera/FocalLength/Source"]       = "ConstantValue"   # Either ConstantValue (constant in time) or FromFile (time dependent)
sim["Camera/FocalLength/ConstantValue"]= 0.24752         # Used in case Source is ConstantValue (from ZEMAX model) [m]
sim["Camera/FocalLength/FromFile"]     = "inputfiles/focallength.txt" # Used in case Source is FromFile: time [s] & focalLenght [m]
sim["Camera/ThroughputBandwidth"]      = 498             # FWHM [nm]
sim["Camera/ThroughputLambdaC"]        = 550             # Central wavelength of the throughput passband [nm]
sim["Camera/IncludeAberrationCorrection"] = "yes"             # Calculate the differential aberration correction and apply
sim["Camera/AberrationCorrection/Type"] = "differential"    # [differential, absolute]
sim["Camera/IncludeFieldDistortion"]   = "yes"             # Whether or not to include field distortion
sim["Camera/FieldDistortion/Type"]     = "Polynomial1D"    # The model for the field distortion
sim["Camera/FieldDistortion/Source"]   = "ConstantValue"   # Either ConstantValue (constant in time) or FromFile (time dependent)
sim["Camera/FieldDistortion/ConstantCoefficients"] =         [0.316257210577,  0.066373219688,  0.372589221219]      # In case Source is ConstantValue
sim["Camera/FieldDistortion/ConstantInverseCoefficients"] =  [-0.317143032936, 0.242638513347,-0.459260203502]       # In case Source is ConstantValue
sim["Camera/FieldDistortion/CoefficientsFromFile"] = "inputfiles/distortioncoefficients.txt"                   # In case Source is FromFile
sim["Camera/FieldDistortion/InverseCoefficientsFromFile"] = "inputfiles/distortioninversecoefficients.txt"            # In case Source if FromFile

sim["PSF/Model"]                        = "AnalyticNonGaussian" # PSF model: MappedGaussian, MappedFromFile, AnalyticGaussian, AnalyticNonGaussian
                                  
    # Generate the PSF from a fixed 2D Gaussian function
sim["PSF/MappedGaussian/Sigma"]   = 0.639           # Standard deviation of Gaussian PSF [pix]
sim["PSF/MappedGaussian/NumberOfPixels"] = 8               # The number of pixels in the field for which the PSF is generated
sim["PSF/MappedGaussian/ChargeDiffusionStrength"] = 0.2             # Charge diffusion strength (width of the Gaussian diffusion kernel) [pixels]
sim["PSF/MappedGaussian/IncludeChargeDiffusion"] = "yes"            # Include charge diffusion [yes or no]
sim["PSF/MappedGaussian/IncludeJitterSmoothing"] = "yes"            # Include jitter smoothing [yes or no]
sim["PSF/MappedFromFile/Filename"] = "inputfiles/psf.hdf5"
sim["PSF/MappedFromFile/DistanceToOA"] = -1               # The angular distance to the optical axis. -1 to auto-compute.               [deg]
sim["PSF/MappedFromFile/RotationAngle"] = -1               # PSF rotation angle w.r.t the x-axis of the focal plane. -1 to auto-compute. [deg]
sim["PSF/MappedFromFile/NumberOfPixels"] = 8               # The number of pixels in the field for which the PSF is generated
sim["PSF/MappedFromFile/ChargeDiffusionStrength"] = 0.2             # Charge diffusion strength (width of the Gaussian diffusion kernel) [pixels]
sim["PSF/MappedFromFile/IncludeChargeDiffusion"] = "no"              # Include charge diffusion [yes or no]
sim["PSF/MappedFromFile/IncludeJitterSmoothing"] = "no"              # Include jitter smoothing [yes or no]
sim["PSF/AnalyticGaussian/Sigma00"] = 1.0             # Stdev of Gaussian PSF in x- and y-direction at the optical axis      [pix]
sim["PSF/AnalyticGaussian/SigmaX18"] = 5.0             # Stdev of Gaussian PSF in x-direction at 18 deg from the optical axis [pix]
sim["PSF/AnalyticGaussian/SigmaY18"] = 2.0             # Stdev of Gaussian PSF in y-direction at 18 deg from the optical axis [pix]
sim["PSF/AnalyticNonGaussian/ParameterFileName"] = "inputfiles/psfallv3.txt"
sim["PSF/AnalyticNonGaussian/ChargeDiffusionStrength"] = 0.2             # Charge diffusion strength (width of the Gaussian diffusion kernel) [pixels]
sim["PSF/AnalyticNonGaussian/IncludeChargeDiffusion"] = "yes"             # Include charge diffusion [yes or no]
# Width of the analytic PSF, equal to sigma for a Gaussian PSF [pix]
sim["PSF/AnalyticNonGaussian/Sigma/Source"] = "ConstantValue"   # Either ConstantValue (constant in time) or FromFile (time dependent)
sim["PSF/AnalyticNonGaussian/Sigma/ConstantValue"] = 0.5             # Used in case Source is ConstantValue  [pix]
sim["PSF/AnalyticNonGaussian/Sigma/FromFile"] = "inputfiles/sigmaPSF.txt" # Used in case Source is FromFile: time [s] & sigma_PSF [pix]

sim["FEE/NominalOperatingTemperature"]  = 210.15          # Nominal operating temperature of the FEE [K]
sim["FEE/Temperature"]                  = "Nominal"         # Temperature fixed at the nominal operating temperature
sim["FEE/TemperatureFileName"]          = "inputfiles/feeTemperature.txt"
sim["FEE/ReadoutNoise"]                 = 32.0            # Readout noise for the FEE [e-/pixel] (same for both ADCs)
sim["FEE/Gain/RefValueLeft"]            = 0.0222          # [ADU/microV] (1 / (Total gain) = FFE gain * CCD gain = 1 / (20 e-/ADU))
sim["FEE/Gain/RefValueRight"]           = 0.0222          # [ADU/microV] (1 / (Total gain) = FFE gain * CCD gain = 1 / (20 e-/ADU))
sim["FEE/Gain/Stability"]               = -300.0e-6       # [ppm/K] = [ADU/microV/K]
sim["FEE/Gain/AllowedDifference"]       = 0.0             # Difference in gain between both ADCs [%]
sim["FEE/ElectronicOffset/RefValue"]    = 1000            # Reference value for the electronic offset [ADU/pixel]
sim["FEE/ElectronicOffset/Stability"]   = 1               # Offset stability [ADU/pixel/K]

sim["CCD/Position"]                     = 1               # one of [1, 2, 3, 4, Custom]
                                                          # (use Custom to specify the CCD position parameters yourself
#Following: only for custom CCD
sim["CCD/OriginOffsetX"]=                  0               # X Offset of CCD origin from center of focal plane [mm]  - use when CCDPosition = Custom
sim["CCD/OriginOffsetY"]=                  0               # Y Offset of CCD origin from center of focal plane [mm]  - use when CCDPosition = Custom
sim["CCD/Orientation"]=                    0               # Orientation of CCD w.r.t. focal plane orientation [deg] - use when CCDPosition = Custom
sim["CCD/NumColumns"]=                     4510            # Number of columns on the CCD [pixels]                   - use when CCDPosition = Custom
sim["CCD/NumRows"]=                        4510            # Number of rows on the CCD [pixels] (including non-exposed ones) - use when CCDPosition = Custom

sim["CCD/FirstRowExposed"]=                0               # The index of the first row that is exposed to light [pixels]
sim["CCD/PixelSize"]=                      18              # [micron]
# Brighter-fatter effect
sim["CCD/BFE/Range"]=                      2               # How far pixels can be apart and still influence each other [pixels] (use window with dimensions 2 * range + 1)
sim["CCD/BFE/p0"]=                         0.05            # Value for p0 parameter in Eq. (18) in Guyonnet et al. 2015
sim["CCD/BFE/p1"]=                         0.15            # Value for p1 parameter in Eq. (18) in Guyonnet et al. 2015
sim["CCD/BFE/RefFlux"]=                    1e6             # Reference flux for the brighter-fatter effect [e-]
sim["CCD/Gain/RefValueLeft"]=              1.80            # [microV/e-]
sim["CCD/Gain/RefValueRight"]=             1.80            # [microV/e-]
sim["CCD/Gain/Stability"]=                 -0.004          # [microV/e-/K]
sim["CCD/Gain/AllowedDifference"]=         15.0            # Difference in gain between both CCD halves [%]
sim["CCD/QuantumEfficiency/MeanQuantumEfficiency"] = 0.5985          # Mean quantum efficiency
sim["CCD/QuantumEfficiency/MeanAngleDependency"]   = 1.01            # Mean (over all pixels) of the relative efficiency due to the angle dependency of the  QE
sim["CCD/FullWellSaturation"]=             900000          # [e-/pixel]
sim["CCD/DigitalSaturation"]=              65535           # E.g. 16 bit [ADU/pixel]
sim["CCD/ReadoutNoise"]=                   38.7            # [e-/pixel]
sim["CCD/SerialTransferTime"]=             340             # Time to shift the content of the readout register by one pixel [ns]
sim["CCD/ParallelTransferTime"]=           110             # Time to shift the charges one row down in case the readout register will be read out [microsec]
sim["CCD/ParallelTransferTimeFast"]=       90              # Time to shift the charges one row down in case the readout register will not be read out [microsec]
sim["CCD/ReadoutMode/ReadoutMode"]=        "Nominal"         # Nominal (normal camera"]=full frame; fast camera"]=frame transfer) / Partial (read contiguous block of entire rows)
sim["CCD/ReadoutMode/Partial/FirstRowReadout"]=          0               # First row that will be read out by the FEE in partial-readout mode
sim["CCD/ReadoutMode/Partial/NumRowsReadout"]=           4510            # Number of rows that will be read out by the FEE, in partial-readout mode
sim["CCD/FlatfieldNoiseRMS"]=              0.010           # Flatfield noise RMS (local PRNU)
# Brightness attenuation from the optical axis to the edge of the FOV
sim["CCD/Vignetting/NaturalVignetting/ExpectedValue"]=         0.945            # Expected value of the throughput efficiency due to vignetting
# Blocking of the light due to mechanical obstruction at the edge of the FOV
sim["CCD/Vignetting/MechanicalVignetting/RadiusFOV"]=             18.8876          # Radius of the FOV [degrees]
sim["CCD/Vignetting/Polarization/ExpectedValue"]=              0.989           # Expected value (mean of all detector pixels)
sim["CCD/Contamination/ParticulateContaminationEfficiency"]=0.972    # Throughput efficiency due to particulate contamination
sim["CCD/Contamination/MolecularContaminationEfficiency"]=  0.9573   # Throughput efficiency due to molecular contamination
sim["CCD/DarkSignal/DarkCurrent"]=         1.2              # Dark current [e- / s]
sim["CCD/DarkSignal/DSNU"]=                15.0             # Dark signal non-uniformity [%]
sim["CCD/DarkSignal/Stability"]=           5.0              # Temperature stability of the dark current [[e- / K / s ]
sim["CCD/CTI/Model"]=                      "Short2013"       # The method used to generate the CTI, either "Simple" or "Short2013"
sim["CCD/CTI/Simple/MeanCTE"]=             0.99999         # Mean CTE over all CCD pixels
sim["CCD/CTI/Short2013/Beta"]=             0.37                                         # beta exponent in Short et al. (2013)
sim["CCD/CTI/Short2013/Temperature"]=      203.                                         # [K]
sim["CCD/CTI/Short2013/NumTrapSpecies"]=   4                                            # number of different trap species
sim["CCD/CTI/Short2013/TrapDensity"]=      [9.8, 3.31, 1.56, 13.24]                     # for each trap species [traps/pixel]
sim["CCD/CTI/Short2013/TrapCaptureCrossSection"]=[2.46e-20, 1.74e-22, 7.05e-23, 2.45e-23]     # for each trap species [m^2]
sim["CCD/CTI/Short2013/ReleaseTime"]=      [2.37e-4, 2.43e-2, 2.03e-3, 1.40e-1]         # for each trap species [s]
sim["CCD/NominalOperatingTemperature"]=    203.15          # Nominal operating temperature of the detector [K]
sim["CCD/Temperature"]=                    "Nominal"         # Temperature fixed at the nominal operating temperature
sim["CCD/TemperatureFileName"]=            "inputfiles/ccdTemperature.txt"
sim["CCD/IncludeFlatfield"]=               "yes"            # Include flatfield ["yes" or no]
sim["CCD/IncludeDarkSignal"]=              "yes"             # Include dark signal non-uniformity ["yes" or no]
sim["CCD/IncludeBFE"]=                     "yes"             # Include the brighter-fatter effect ["yes" or no]
sim["CCD/IncludePhotonNoise"]=             "yes"             # Include photon noise ["yes" or no]
sim["CCD/IncludeReadoutNoise"]=            "yes"             # Include readout noise ["yes" or no]
sim["CCD/IncludeCTIeffects"]=              "yes"             # Include charge transfer inefficiency effects ["yes" or no]
sim["CCD/IncludeOpenShutterSmearing"]=     "yes"             # Include trails due reading out with an open shutter ["yes" or no]
sim["CCD/IncludeQuantumEfficiency"]=       "yes"             # Include loss of throughput due to quantum efficiency
sim["CCD/IncludeNaturalVignetting"]=       "yes"             # Include brightness attenuation towards the edge of the FOV due to vignetting
sim["CCD/IncludeMechanicalVignetting"]=    "yes"             # Include blockage of incoming flux due to mechanical vignetting
sim["CCD/IncludePolarization"]=            "yes"             # Include loss of throughput due to polarisation
sim["CCD/IncludeParticulateContamination"]="yes"             # Include loss of throughput due to particulate contamination
sim["CCD/IncludeMolecularContamination"]=  "yes"             # Include loss of throughput due to molecular contamination
sim["CCD/IncludeConvolution"]=             "yes"             # Whether or not to convolve the subPixelMap with the PSF
sim["CCD/IncludeFullWellSaturation"]=      "yes"             # Whether or not full well saturation should be applied
sim["CCD/IncludeDigitalSaturation"]=       "yes"             # Whether or not digital saturation should be applied
sim["CCD/IncludeQuantisation"]=            "yes"             # Whether or not to include quantisation


#SubField stuff. Masked because we're using setSubfieldAroundCoordinates
sim["SubField/ZeroPointRow"]=              0               # Row of the (0,0) pixel of the subfield [pixels]
sim["SubField/ZeroPointColumn"]=           0               # Column of the (0,0) pixel of the subfield [pixels]
sim["SubField/NumColumns"]=                4510             # Number of columns, should be >= 8 [pixels]
sim["SubField/NumRows"]=                   4510             # Number of rows,    should be >= 8 [pixels]
sim["SubField/NumBiasPrescanRows"]=        30              # Nr of rows of the prescan strip to determine the bias [pixels]
sim["SubField/NumBiasPrescanColumns"]=     30              # Nr of columns of the prescan strip to determine the bias [pixels]
sim["SubField/NumSmearingOverscanRows"]=   30              # Nr of rows of the overscan strip to determine the smearing [pixels]
sim["SubField/SubPixels"]=                 8               # (Sqrt of) nr of subpixels per CCD pixel. Should be 2^n <= 128.



# Four groups of six camera's with their azimuth and tilt as specified
                                                                       # The fifth value is for the Fast Camera's
sim["CCD/CameraGroups/AzimuthAngle"]=      [45.0, 135.0, -135.0, -45.0, 0.0] # Azimuth angle of telescope on the platform [deg]
sim["CCD/CameraGroups/TiltAngle"]=         [9.2, 9.2, 9.2, 9.2, 0.0]         # Tilt angle of the telescope on the platform [deg]


sim["CCDPositions/OriginOffsetX"]=         [-1.3, -1.3, -1.3, -1.3]       # X Offset of CCD origin from center of focal plane [mm]
sim["CCDPositions/OriginOffsetY"]=         [82.48, 82.48, 82.48, 82.48]   # Y Offset of CCD origin from center of focal plane [mm]
sim["CCDPositions/Orientation"]=           [0, 90, 180, 270]              # Orientation of CCD w.r.t. focal plane orientation, counter-clockwise [deg]
sim["CCDPositions/NumColumns"]=            [4510, 4510, 4510, 4510]       # Number of columns on the CCD [pixels]
sim["CCDPositions/NumRows"]=               [4510, 4510, 4510, 4510]       # Number of rows on the CCD [pixels] (including non-exposed ones)
sim["CCDPositions/FirstRowForNormalCamera"]=[0, 0, 0, 0]                   # The complete CCDs are active/illuminated for the Normal Camera's
sim["CCDPositions/FirstRowForFastCamera"]= [2255, 2255, 2255, 2255]       # Only the upper half of the CCDs is active/illuminated for the Fast Camera's
'''

sim["SubField/NumColumns"]                 = numColumnsSubField
sim["SubField/NumRows"]                    = numRowsSubField

# Set the telescope group ID, this is needed for the subfield calculations later on.

sim["Telescope/GroupID"]                   = {group}

# Set the quarter specific parameters
sim["RandomSeeds/JitterSeed"]              = 2033429158 + 100 * Quarter_adj

readoutTime, dummy = sim.getReadoutTime()
#print("readoutTime",readoutTime," expTime",21, sim["ObservingParameters/ExposureTime"] )
numExposures = int(365.25/4*86400/(sim["ObservingParameters/ExposureTime"] + readoutTime))
sim["ObservingParameters/NumExposures"]    = numExposures
sim["ObservingParameters/BeginExposureNr"] = Quarter_adj * numExposures

# Attempt to set a subfield around the specified coordinates on one of the 4 CCDs of the telescope.
# This will fail (return value == False) if the subfield is not visible by any of the 4 CCDs or
# that the subfield is too large to entirely fit on a CCD.
# If successful, the function sets the CCD and subfield parameters correctly in the 'sim' object.

isSuccessful =  sim.setSubfieldAroundCoordinates(raCenter, decCenter, numColumnsSubField, numRowsSubField, normal=True)

if isSuccessful:
    
    # Make sure that the following random seeds differ for each telescope and for each quarter
    # We assume a maximum of 8 quarter and 4 camera groups
    
    randomSeedOffset = 1000 * 8 * Quarter_adj +  10 * 4 * {group} + {scope}
    
    sim["RandomSeeds/ReadOutNoiseSeed"]    = 1424949740 + randomSeedOffset
    sim["RandomSeeds/PhotonNoiseSeed"]     = 1533320336 + randomSeedOffset
    sim["RandomSeeds/FlatFieldSeed"]       = 1633320381 + randomSeedOffset
    sim["RandomSeeds/CosmicSeed"]          = 1234567457 + randomSeedOffset
    sim["RandomSeeds/DriftSeed"]           = 1733429158 + randomSeedOffset
    sim["RandomSeeds/DarkSignalSeed"]      = 1803517191 + randomSeedOffset
    # Run the PlatoSim simulator
    # logLevel can 1 (least verbose) to 3 (most verbose)
    
    if not os.path.exists(os.path.join(outputDir,outputFilePrefix+'.hdf5')):
        simFile = sim.run(logLevel=2)
    else:
        print("simFile alrady exists")
    if not os.path.exists(os.path.join(outputDir,outputFilePrefix+'_binned_imgts.hdf5')) or not os.path.exists(os.path.join(outputDir,outputFilePrefix+'list_targets.pickle')):
        print("Binning PlatoSim3 output into imagettes")
        targ_ids=rebin_hdf5_file(os.path.join(outputDir,outputFilePrefix+'.hdf5'),
                              starcatloc=os.path.join(outputDir,"{fieldID}_starcat.txt"),
                              injcatloc=os.path.join(outputDir,"{fieldID}_final_fieldcat.csv"))
        print(targ_ids[:5],type(targ_ids[0]))
    else:
        print("Loading target IDs from file")
        #targ_ids=np.genfromtxt(os.path.join(outputDir,outputFilePrefix+"list_targets.txt"))
        targ_ids=pickle.load(open(os.path.join(outputDir,outputFilePrefix+"list_targets.pickle"),"rb"))
        print("targs",targ_ids[:5],type(targ_ids[0]))
    #Running photometric Extraction:
    import platophot
    
    #Running photometric pipeline:
    platophot.photometry(inputFilePath=os.path.join(outputDir,outputFilePrefix+".hdf5"),
                        outputFilePath=os.path.join(outputDir,outputFilePrefix+"_extracted_photometry.hdf5"),
                        targetIDs=[np.string_(str(int(targ_ids[n]))) for n in range(len(targ_ids))]
                        )
                        #list(targ_ids.astype(int).astype(str)) )

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
    
