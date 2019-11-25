# PlatoSim FieldGen

A python tool to produce [PlatoSim3](http://ivs-kuleuven.github.io/PlatoSim3/) fields and the requisite run files needed to use PlatoSim to generate fields, extract lightcurves/imagettes, etc.

## To install:

First, a virtual environment, either with venv or conda, is probably necessary. 

Second, PlatoSim3 should be installed, specially with the "develop" branch which allows photometric extraction. Details for installing PlatoSim3 from git can be found [here](http://ivs-kuleuven.github.io/PlatoSim3/DownloadUpdateBuild.html).

Clone the directory:
`git clone https://github.com/hposborn/PlatoSim_FieldGen`

Next, pip install the requirements file:
`pip install -t requirements.txt`

There may be some special treatment here (e.g. to install the development version of [shocksgo](https://github.com/bmorris3/shocksgo).)


## Using FieldGen
Effectively, FieldGen.py can be run like any python script, or installed and individual parts run separately. 

In it's default mode, it generates randomly-placed 100x100 pixel fields in PLATO's north or southern viewing zones, each with stars set at 7-pixel increments, and with a set fraction of system types across the target stars (PLs:32%, EBs:7%, BEBs:13%, BPLs:7%), plus a set variability fractions (50%), and fraction of stars with contaminants (50%).

The simplest way is to run `python FieldGen.py` with the following arguments:

* hemisphere - string = "North" or "South"
* N_fields   - integer = number of fields to generate
* ext        - string  = string to add to output files
* folder     - string  = output folder location 
* num_quarts - integer = number of quarters
* MP         - boolean = whether to use multiprocessing or not
* overwrite  - boolean = whether to overwrite previous data

For example:
`python FieldGen.py North 1 _test /data/PLATO/Sims 1 True True` 
generates 1 northern field in a new folder in `/data/PLATO/Sims` with the file string '\_test' added to it. Lightcurves for a single quarter are generated. It uses multiprocessing and overwrites previous files.

## Explanation

A field of star positions is generated using astropy and PlatoSim's in house functions.

For each of the target types, a Besancon catalogue of stars is used to choose parent stars, a planet or binary population is injected (using modified occurrence rates) and those targets with significant transiting signals are chosen from to form the input population. Variability and contaminant stars (again from Besancon stellar catalogues) are then generated. Finally lightcurves are generated for each target and PlatoSim3-compatible stellar catalogues created. A PlatoSim3 runfile is finally made which will enable the PlatoSim3 field creation, imagette extraction and binning (to 10min), and photometric extraction of lightcurves for each input target star.

For full documentation see [this document](https://github.com/hposborn/PlatoSim_FieldGen-Document/blob/master/main.pdf) or [this flowchart](flowchart.pdf).
