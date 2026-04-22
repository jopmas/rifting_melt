#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 15:08:50 2026

@author: jobueno
"""

from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.offsetbox import AnchoredText

from scipy.interpolate import RegularGridInterpolator

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os, gc, json, sys
import xarray as xr
path = os.getcwd().split('/')
machine_path = f'/{path[1]}/{path[2]}' #cat the /home/user/ or /Users/user from system using path
path_to_functions = f"{machine_path}/opt/rifting_melt"
sys.path.append(os.path.abspath(path_to_functions))
# from easyRo_class import easyRo
if '' in sys.path:
    sys.path.remove('')
from functions.mandyoc_class import MandyocScen

msc = MandyocScen(f"/Users/joao_macedo/Desktop/RFT_Tp1400oC",
                            variables=['temperature','pressure','strain_rate','velocity'],
                            chunks_vars={'x':'auto','z':'auto','time':'auto'})
msc.load_mainParticles(name="particles_trajectories.nc",chunks={'id':'auto'},filter_air=True)
msc.correctZcoord()
msc.ylimits = [-50e3, 5e3]

msc.selectParticles_bycoords(replace_original=True)
#msc.selectParticles_bylayers([4,  5,6,7,  11,8,9,10],tsel=tsel, replace_original=True)

msc.fieldToParticle('temperature',select_original=True,replace_original=True)
msc.fieldToParticle('pressure',select_original=True,replace_original=True)
#msc.fieldToParticle('strain_rate',select_original=True,replace_original=True)  
#msc.fieldToParticle('velocity',select_original=True,replace_original=True)

#msc.classify_ParticlesRange(domain_intervals=sides, 
                                    #tsel=tsel, replace_original=True)
#msc.classify_ParticlesRange(domain_intervals=scen, tsel=tsel, replace_original=True)

msc.original_particles.to_netcdf(f"{msc.path}/particles_PTt.nc")

del msc
gc.collect()