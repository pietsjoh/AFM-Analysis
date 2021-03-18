# AFM_analysis

## General information
This code aims to analyze AFM measurements. As a user, the only file to worry about is run_afm.py. Configure the available options and run the file (for Example:       python run_afm.py).

## Requirements
- Python 3 (tested on 3.9.1)
- numpy
- scipy
- matplotlib
- pandas

## Directory structure
#### (pathToDirectory = None [if not None, should be the relative path to the head dir of NP7593])
- HEAD
    - AFM_analysis
    - dataAFM
        - NP7593
            - NP7593_1mum_Mitte.xyz
            - NP7594_1mum_rand.xyz
        - NP7594
            -NP7594_1mum_Mitte.xyz



## To Do
- maybe rethink directory structure
- compare density, height, fwhm and aspectratio with experiments
