#### EXOSIMS Integration Time Adjusted Completeness Integration Test Script
# Written by: Dean Keithly

import os
#from projectedEllipse import *
import EXOSIMS.MissionSim
#import matplotlib.pyplot as plt
#import matplotlib
#from mpl_toolkits.mplot3d import Axes3D
#import numpy.random as random
#import time
from astropy import constants as const
import astropy.units as u
#from EXOSIMS.util.deltaMag import deltaMag
#from EXOSIMS.util.planet_star_separation import planet_star_separation
#import itertools
#import datetime
#import re


#### Instantiate EXOSIMS Object
folder_load = os.path.normpath(os.path.expandvars('$HOME/Documents/exosims/exodetbox/exodetbox/scripts'))
filename = 'HabEx_CKL2_PPKL2.json'
filename = 'WFIRSTcycle6core.json'
filename = 'HabEx_CBrownKL_PPBrownKL_IACintegrationScript.json'
#filename = 'HabEx_CSAG13_PPSAG13_compSubtypeHighEccen.json'
scriptfile = os.path.join(folder_load,filename)
sim = EXOSIMS.MissionSim.MissionSim(scriptfile=scriptfile,nopar=True)

