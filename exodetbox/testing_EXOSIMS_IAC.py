#### EXOSIMS Integration Time Adjusted Completeness Integration Test Script
# Written by: Dean Keithly

import os
import EXOSIMS.MissionSim
from astropy import constants as const
import astropy.units as u

#### Instantiate EXOSIMS Object
folder_load = os.path.normpath(os.path.expandvars('$HOME/Documents/exosims/exodetbox/exodetbox/scripts'))
filename = 'HabEx_CKL2_PPKL2.json'
filename = 'WFIRSTcycle6core.json'
filename = 'HabEx_CSAG13_PPSAG13_IACtest.json'
scriptfile = os.path.join(folder_load,filename)
sim = EXOSIMS.MissionSim.MissionSim(cachedir=None,**{"erange": [0,0.35],"arange":[0.09, 12.4],"Rprange":[0.5, 11.6],"scaleOrbits": False,"constrainOrbits": True,"whichPlanetPhaseFunction": "quasiLambertPhaseFunction",\
            "scienceInstruments": [{"name": "imagingEMCCD","QE": 0.9,"optics": 0.28,"FoV": 0.75,"pixelNumber": 105,"pixelSize": 13e-6,"sread": 0,"idark": 3e-5,"CIC": 1.3e-3,"texp": 100,"ENF": 1,"PCeff": 0.75}],\
            "starlightSuppressionSystems": [{"name": "VVC500","lam": 500,"IWA": 0.045,"OWA": 2.127,"ohTime": 0.1,"BW": 0.20,"optics": 0.95,"optics_comment": "contamination",
            "core_platescale": 0.1,"occ_trans": 0.9,
            "core_thruput": 0.9,           
            "core_mean_intensity": 0.9,
            "occ_trans_local": 0.9,
            "core_thruput_local": 1,           
            "core_mean_intensity_local": 1}],
            "modules":{"PlanetPopulation": "SAG13","StarCatalog": "EXOCAT1","OpticalSystem": "Nemati","ZodiacalLight": "Stark","BackgroundSources": "GalaxiesFaintStars",\
            "PlanetPhysicalModel": "Forecaster","Observatory": "WFIRSTObservatoryL2","TimeKeeping": " ","PostProcessing": " ","Completeness": "IntegrationTimeAdjustedCompleteness","TargetList": " ",\
            "SimulatedUniverse": "SAG13Universe","SurveySimulation": " ","SurveyEnsemble": " "}})
IWA_HabEx = 0.045*u.arcsec #taken from a Habex Script in units of mas
#IWA2=0.150*u.arcsec #Suggested by dmitry as analahous to WFIRST
OWA_HabEx = 6.*u.arcsec #from the HabEx Standards Team Final Report
smin = 10.*u.pc.to('AU')*IWA_HabEx.to('rad').value
smax = 10.*u.pc.to('AU')*OWA_HabEx.to('rad').value
dMag=25.
IAC = sim.Completeness.comp_calc(smin, smax, dMag, subpop=-2, tmax=0.,starMass=const.M_sun, IACbool=True)
print(IAC)

